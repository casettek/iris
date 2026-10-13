use std::path::Path;
use std::sync::Arc;

use anyhow::Result;
use prefill_prepare_aux::input::{ActivationSequence, PleLayer, PleLayerInputs};

use crate::artifact_io::{with_main_sequence_scope, with_stage_sequence_scope};
use crate::cache::{
    materialization_key_from_stage_files, materialize_with_cache, CachedInputs,
    MaterializationCache, MaterializationCacheKey,
};
use inference_kernels::kernels::prefill_prepare_aux::{
    run_prefill_prepare_aux_direct, PrefillPrepareAuxDirectInputs,
};

pub struct Inputs {
    pub embedded: ActivationSequence,
    pub layer: Arc<PleLayer>,
}

pub fn load_inputs_from_args() -> Result<Inputs> {
    with_main_sequence_scope(|| {
        load_inputs_from_initialized_runtime(&CachedInputs::new(), None, None)
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
    let layer_key =
        materialization_key_from_stage_files::<PleLayer>(input, input_manifest, "layer")?;
    with_stage_sequence_scope(input, input_manifest, || {
        load_inputs_from_initialized_runtime(cached_inputs, materialization_cache, layer_key)
    })
}

pub fn run_direct(inputs: &Inputs) -> Result<PleLayerInputs> {
    run_prefill_prepare_aux_direct(PrefillPrepareAuxDirectInputs {
        embedded: &inputs.embedded,
        layer: &inputs.layer,
    })
}

fn load_inputs_from_initialized_runtime(
    cached_inputs: &CachedInputs,
    materialization_cache: Option<&MaterializationCache>,
    layer_key: Option<MaterializationCacheKey>,
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
    let layer = materialize_with_cache(materialization_cache, layer_key, || {
        raster::materialize_auth_return(raster::entry_argument_auth_ref::<PleLayer>(
            binding.reference,
            "layer",
        ))
    })?;
    Ok(Inputs { embedded, layer })
}
