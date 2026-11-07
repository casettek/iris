use std::path::Path;
use std::sync::Arc;

use anyhow::Result;
use prefill_prepare_aux::input::{ActivationSequence, PleLayer, PleLayerInputs};

use crate::artifact_io::{with_main_sequence_scope, with_stage_sequence_scope};
use crate::cache::{
    materialization_key_from_stage_files, materialize_with_cache, CachedInputs,
    MaterializationCache, MaterializationCacheKey,
};
use crate::ple::{self, Loading, SelectedWeights, StaticWeights};
use inference_kernels::kernels::prefill_prepare_aux::{
    run_prefill_prepare_aux_direct, PrefillPrepareAuxDirectInputs,
};
use raster_core::input::ExternalEncoding;

pub struct Inputs {
    component_root: raster::BytesPage,
    pub embedded: ActivationSequence,
    layer: Weights,
}

enum Weights {
    Eager(Arc<PleLayer>),
    Selective(SelectedWeights),
}

impl Inputs {
    /// Construct the eager comparison path from already materialized weights.
    pub fn eager(embedded: ActivationSequence, layer: Arc<PleLayer>) -> Self {
        Self {
            component_root: model_identity::stream::digest(layer.as_ref()),
            embedded,
            layer: Weights::Eager(layer),
        }
    }
}

pub fn load_inputs_from_args() -> Result<Inputs> {
    // Match Raster's CLI input context, including inline JSON documents.
    let args: Vec<_> = std::env::args().collect();
    let arg = |name| {
        args.iter()
            .position(|s| s == name)
            .and_then(|i| args.get(i + 1))
            .map(String::as_str)
    };
    let mode = Loading::from_env()?;
    with_main_sequence_scope(|| {
        load_inputs_from_initialized_runtime(
            &CachedInputs::new(),
            None,
            None,
            mode,
            arg("--input"),
            arg("--input-manifest"),
        )
    })
}

pub fn load_inputs_from_paths(
    input: &Path,
    input_manifest: &Path,
    cached_inputs: &CachedInputs,
) -> Result<Inputs> {
    load_inputs_from_paths_with_cache(input, input_manifest, cached_inputs, None)
}

pub fn load_inputs_from_paths_with_cache(
    input: &Path,
    input_manifest: &Path,
    cached_inputs: &CachedInputs,
    materialization_cache: Option<&MaterializationCache>,
) -> Result<Inputs> {
    let mode = Loading::from_env()?;
    let layer_key = match mode {
        Loading::Selective => {
            materialization_key_from_stage_files::<StaticWeights>(input, input_manifest, "layer")?
        }
        Loading::Eager => {
            materialization_key_from_stage_files::<PleLayer>(input, input_manifest, "layer")?
        }
    };
    with_stage_sequence_scope(input, input_manifest, || {
        load_inputs_from_initialized_runtime(
            cached_inputs,
            materialization_cache,
            layer_key,
            mode,
            input.to_str(),
            input_manifest.to_str(),
        )
    })
}

pub fn run_direct(inputs: &Inputs) -> Result<PleLayerInputs> {
    let mut output = match &inputs.layer {
        Weights::Eager(layer) => run_prefill_prepare_aux_direct(PrefillPrepareAuxDirectInputs {
            embedded: &inputs.embedded,
            layer,
        }),
        Weights::Selective(layer) => ple::run(&inputs.embedded, layer),
    }?;
    output.model = model_identity::stream::verify(
        output.model,
        &inputs.component_root,
        model_identity::Role::Auxiliary(output.layer_idx),
    );
    Ok(output)
}

fn load_inputs_from_initialized_runtime(
    cached_inputs: &CachedInputs,
    materialization_cache: Option<&MaterializationCache>,
    layer_key: Option<MaterializationCacheKey>,
    mode: Loading,
    input: Option<&str>,
    input_manifest: Option<&str>,
) -> Result<Inputs> {
    let binding = raster::start_program(&[
        raster::entry_argument_spec::<ActivationSequence>("embedded"),
        raster::entry_argument_spec::<PleLayer>("layer"),
        raster::entry_argument_spec::<model_identity::sparse::SparseWitness>("witness"),
    ])?;
    let component_root;
    let mut embedded = match cached_inputs.get("embedded") {
        Some(value) => value.as_aux_activation_sequence()?,
        None => raster::materialize_auth_return(raster::entry_argument_auth_ref::<
            ActivationSequence,
        >(binding.reference.clone(), "embedded")),
    };
    let layer = if mode == Loading::Selective
        && binding
            .arguments
            .iter()
            .all(|arg| arg.encoding == ExternalEncoding::Raster)
    {
        let identity_source = crate::verified_weights::VerifiedSelections::new(
            raster_runtime::EntryArgumentsBinding {
                reference: binding.reference.clone(),
                arguments: binding.arguments.clone(),
            },
            input,
            input_manifest,
        )?;
        let selected = ple::load(
            binding,
            input,
            input_manifest,
            &embedded,
            materialization_cache,
            layer_key,
        )?;
        let (root, valid) = selected.identity(&identity_source)?;
        component_root = root;
        embedded.model.require(valid);
        Weights::Selective(selected)
    } else {
        // Postcard cannot supply indexed selection witnesses. Keep the old
        // materializer, with its own type key even for mixed-encoding inputs.
        let layer_key = layer_key.map(|mut key| {
            key.type_name = std::any::type_name::<PleLayer>();
            key
        });
        let witness =
            raster::materialize_auth_return(raster::entry_argument_auth_ref::<
                model_identity::sparse::SparseWitness,
            >(binding.reference.clone(), "witness"));
        let layer = materialize_with_cache(materialization_cache, layer_key, || {
            raster::materialize_auth_return(raster::entry_argument_auth_ref::<PleLayer>(
                binding.reference,
                "layer",
            ))
        })?;
        component_root = ple::ple_identity_root(
            &layer.params,
            &layer.projection,
            layer.embeddings.byte_len,
            layer.embeddings.page_size,
            &witness.header,
        );
        for row in embedded.rows.iter() {
            let offset = prefill_prepare_aux::ple_row_offset(
                row.token_id,
                layer.params.ple_width,
                layer.embeddings.byte_len,
            );
            let index = offset / layer.embeddings.page_size;
            embedded.model.require(
                layer
                    .embeddings
                    .pages
                    .as_slice()
                    .get(index as usize)
                    .zip(witness.paths.as_slice().get(index as usize))
                    .map(|(page, path)| {
                        model_identity::sparse::verify(
                            &witness.header.root,
                            witness.header.count,
                            index,
                            path,
                            &model_identity::stream::digest(page),
                        )
                    })
                    .unwrap_or(false),
            );
        }
        Weights::Eager(layer)
    };
    Ok(Inputs {
        component_root,
        embedded,
        layer,
    })
}
