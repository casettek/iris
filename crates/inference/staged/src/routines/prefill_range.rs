use std::path::Path;
use std::sync::{Arc, OnceLock};

use anyhow::Result;
use inference_kernels::weight_budget::RetainedWeightBytes;
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

/// One loaded copy of a transformer layer.
struct Loaded {
    layer: TransformerLayer,
    root: OnceLock<raster::BytesPage>,
}

impl RetainedWeightBytes for Loaded {
    fn retained_weight_bytes(&self) -> usize {
        // The root is 32 bytes beside tens of megabytes of weights.
        self.layer.retained_weight_bytes() + 64
    }
}

/// A transformer layer as loaded, with the model-identity root of its bytes.
///
/// Every layer stage binds the layer it ran to the model the claim names by
/// that root, which takes a sweep of the whole layer. The root is a function
/// of the bytes alone, so it is computed on first use and kept with this copy.
/// A layer the run retains across ranges is swept once; a layer that is loaded
/// again is a new copy and is swept again.
#[derive(Clone)]
pub struct LoadedLayer(Arc<Loaded>);

impl LoadedLayer {
    /// The model-identity root of this copy's bytes.
    pub fn root(&self) -> &raster::BytesPage {
        self.0
            .root
            .get_or_init(|| model_identity::stream::digest(&self.0.layer))
    }
}

impl From<TransformerLayer> for LoadedLayer {
    fn from(layer: TransformerLayer) -> Self {
        Self(Arc::new(Loaded {
            layer,
            root: OnceLock::new(),
        }))
    }
}

impl std::ops::Deref for LoadedLayer {
    type Target = TransformerLayer;

    fn deref(&self) -> &TransformerLayer {
        &self.0.layer
    }
}

pub struct Inputs {
    pub activations: ActivationSequence,
    pub layer: LoadedLayer,
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
    let layer_key = materialization_key_from_stage_files::<Loaded>(input, input_manifest, "layer")?;
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
    let mut output = run_prefill_range_direct_with_weight_cache(
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
    )?;
    output.model = model_identity::stream::verify(
        output.model,
        inputs.layer.root(),
        model_identity::Role::Transformer(inputs.layer.params.layer_idx),
    );
    output.model.require(
        !(inputs.prior_kv.model.is_origin()
            || inputs.prior_kv.model.producer == model_identity::context::EMBEDDING)
            || inputs.prior_kv.kv.is_empty(),
    );
    Ok(output)
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
    let layer = LoadedLayer(materialize_with_cache(
        materialization_cache,
        layer_key.clone(),
        || Loaded {
            layer: raster::materialize_auth_return(raster::entry_argument_auth_ref::<
                TransformerLayer,
            >(
                binding.reference.clone(), "layer"
            )),
            root: OnceLock::new(),
        },
    )?);
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
