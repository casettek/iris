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
    match &inputs.layer {
        Weights::Eager(layer) => run_prefill_prepare_aux_direct(PrefillPrepareAuxDirectInputs {
            embedded: &inputs.embedded,
            layer,
        }),
        Weights::Selective(layer) => ple::run(&inputs.embedded, layer),
    }
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
    ])?;
    let embedded = match cached_inputs.get("embedded") {
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
        Weights::Selective(ple::load(
            binding,
            input,
            input_manifest,
            &embedded,
            materialization_cache,
            layer_key,
        )?)
    } else {
        // Postcard cannot supply indexed selection witnesses. Keep the old
        // materializer, with its own type key even for mixed-encoding inputs.
        let layer_key = layer_key.map(|mut key| {
            key.type_name = std::any::type_name::<PleLayer>();
            key
        });
        Weights::Eager(materialize_with_cache(
            materialization_cache,
            layer_key,
            || {
                raster::materialize_auth_return(raster::entry_argument_auth_ref::<PleLayer>(
                    binding.reference,
                    "layer",
                ))
            },
        )?)
    };
    Ok(Inputs { embedded, layer })
}
