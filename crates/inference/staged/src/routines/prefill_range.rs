use std::path::Path;
use std::sync::Arc;

use anyhow::Result;
use prefill_range::input::{ActivationSequence, PleLayerInputs, TransformerLayer};

use crate::artifact_io::{with_main_sequence_scope, with_stage_sequence_scope};
use crate::cache::{
    materialization_key_from_stage_files, materialize_with_cache, CachedInputs,
    MaterializationCache, MaterializationCacheKey,
};

pub use inference_kernels::kernels::prefill_range::{
    run_prefill_range_direct, run_prefill_range_direct_with_weight_cache, PrefillRangeDirectInputs,
    PrefillRangeWeightCache,
};

pub struct Inputs {
    pub activations: ActivationSequence,
    pub layer: Arc<TransformerLayer>,
    pub layer_cache_key: Option<MaterializationCacheKey>,
    pub prior_kv: ActivationSequence,
    pub donor_a_kv: ActivationSequence,
    pub donor_b_kv: ActivationSequence,
    pub ple: PleLayerInputs,
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
        materialization_key_from_stage_files::<TransformerLayer>(input, input_manifest, "layer")?;
    with_stage_sequence_scope(input, input_manifest, || {
        load_inputs_from_initialized_runtime(cached_inputs, materialization_cache, layer_key)
    })
}

pub fn run_direct(inputs: &Inputs) -> Result<ActivationSequence> {
    run_direct_with_weight_cache(inputs, None)
}

pub fn run_direct_with_weight_cache(
    inputs: &Inputs,
    weight_cache: Option<&PrefillRangeWeightCache>,
) -> Result<ActivationSequence> {
    run_prefill_range_direct_with_weight_cache(
        PrefillRangeDirectInputs {
            activations: &inputs.activations,
            layer: &inputs.layer,
            layer_cache_key: inputs.layer_cache_key.as_ref(),
            prior_kv: &inputs.prior_kv,
            donor_a_kv: &inputs.donor_a_kv,
            donor_b_kv: &inputs.donor_b_kv,
            ple: &inputs.ple,
        },
        weight_cache,
    )
}

fn load_inputs_from_initialized_runtime(
    cached_inputs: &CachedInputs,
    materialization_cache: Option<&MaterializationCache>,
    layer_key: Option<MaterializationCacheKey>,
) -> Result<Inputs> {
    let binding = raster::start_program(&[
        raster::entry_argument_spec::<ActivationSequence>("activations"),
        raster::entry_argument_spec::<TransformerLayer>("layer"),
        raster::entry_argument_spec::<ActivationSequence>("prior_kv"),
        raster::entry_argument_spec::<ActivationSequence>("donor_a_kv"),
        raster::entry_argument_spec::<ActivationSequence>("donor_b_kv"),
        raster::entry_argument_spec::<PleLayerInputs>("ple"),
    ])?;

    let activations = match cached_inputs.get("activations") {
        Some(value) => value.as_range_activation_sequence()?,
        None => raster::materialize_auth_return(raster::entry_argument_auth_ref::<
            ActivationSequence,
        >(binding.reference.clone(), "activations")),
    };
    let layer = materialize_with_cache(materialization_cache, layer_key.clone(), || {
        raster::materialize_auth_return(raster::entry_argument_auth_ref::<TransformerLayer>(
            binding.reference.clone(),
            "layer",
        ))
    })?;
    let prior_kv = match cached_inputs.get("prior_kv") {
        Some(value) => value.as_range_activation_sequence()?,
        None => raster::materialize_auth_return(raster::entry_argument_auth_ref::<
            ActivationSequence,
        >(binding.reference.clone(), "prior_kv")),
    };
    let donor_a_kv = match cached_inputs.get("donor_a_kv") {
        Some(value) => value.as_range_activation_sequence()?,
        None => raster::materialize_auth_return(raster::entry_argument_auth_ref::<
            ActivationSequence,
        >(binding.reference.clone(), "donor_a_kv")),
    };
    let donor_b_kv = match cached_inputs.get("donor_b_kv") {
        Some(value) => value.as_range_activation_sequence()?,
        None => raster::materialize_auth_return(raster::entry_argument_auth_ref::<
            ActivationSequence,
        >(binding.reference.clone(), "donor_b_kv")),
    };
    let ple = match cached_inputs.get("ple") {
        Some(value) => value.as_range_ple_inputs()?,
        None => raster::materialize_auth_return(raster::entry_argument_auth_ref::<PleLayerInputs>(
            binding.reference,
            "ple",
        )),
    };

    Ok(Inputs {
        activations,
        layer,
        layer_cache_key: layer_key,
        prior_kv,
        donor_a_kv,
        donor_b_kv,
        ple,
    })
}
