use std::path::Path;
use std::sync::Arc;

use anyhow::Result;
use prefill_finalize::input::{ActivationSequence, FinalHead, PrefillLogits};

use crate::artifact_io::{with_main_sequence_scope, with_stage_sequence_scope};
use crate::cache::{
    materialization_key_from_stage_files, materialize_with_cache, CachedInputs,
    MaterializationCache, MaterializationCacheKey,
};
use inference_kernels::kernels::prefill_finalize::{
    run_prefill_finalize_direct, PrefillFinalizeDirectInputs,
};

pub struct Inputs {
    pub activations: ActivationSequence,
    pub head: Arc<FinalHead>,
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
    let head_key =
        materialization_key_from_stage_files::<FinalHead>(input, input_manifest, "head")?;
    with_stage_sequence_scope(input, input_manifest, || {
        load_inputs_from_initialized_runtime(cached_inputs, materialization_cache, head_key)
    })
}

pub fn run_direct(inputs: &Inputs) -> Result<PrefillLogits> {
    run_prefill_finalize_direct(PrefillFinalizeDirectInputs {
        activations: &inputs.activations,
        head: &inputs.head,
    })
}

fn load_inputs_from_initialized_runtime(
    cached_inputs: &CachedInputs,
    materialization_cache: Option<&MaterializationCache>,
    head_key: Option<MaterializationCacheKey>,
) -> Result<Inputs> {
    let binding = raster::start_program(&[
        raster::entry_argument_spec::<ActivationSequence>("activations"),
        raster::entry_argument_spec::<FinalHead>("head"),
    ])?;
    let activations = match cached_inputs.get("activations") {
        Some(value) => value.as_finalize_activation_sequence()?,
        None => raster::materialize_auth_return(raster::entry_argument_auth_ref::<
            ActivationSequence,
        >(binding.reference.clone(), "activations")),
    };
    let head = materialize_with_cache(materialization_cache, head_key, || {
        raster::materialize_auth_return(raster::entry_argument_auth_ref::<FinalHead>(
            binding.reference,
            "head",
        ))
    })?;
    Ok(Inputs { activations, head })
}
