//! The `layer-mistral` stage's native routine: one `mistral-dense` layer over
//! one range.

use std::path::Path;
use std::sync::{Arc, OnceLock};

use anyhow::Result;
use inference_kernels::mistral::stage::{run_layer_stage, LayerStageInputs};
use inference_kernels::weight_budget::RetainedWeightBytes;
use layer_mistral::input::TransformerLayer;
// The pipeline's activation type, which this stage defines as every stage does.
use prefill_range::input::ActivationSequence;

use crate::artifact_io::{with_main_sequence_scope, with_stage_sequence_scope};
use crate::cache::{
    materialization_key_from_stage_files, materialize_with_cache, CachedInputs,
    MaterializationCache, MaterializationCacheKey,
};

/// One loaded copy of a layer, with the model-identity root of its bytes. The
/// root takes a sweep of the whole layer, so it is computed on first use and
/// kept with the copy.
struct Loaded {
    layer: TransformerLayer,
    root: OnceLock<raster::BytesPage>,
}

impl RetainedWeightBytes for Loaded {
    fn retained_weight_bytes(&self) -> usize {
        self.layer.retained_weight_bytes() + 64
    }
}

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

impl std::ops::Deref for LoadedLayer {
    type Target = TransformerLayer;

    fn deref(&self) -> &TransformerLayer {
        &self.0.layer
    }
}

pub struct Inputs {
    pub activations: ActivationSequence,
    pub layer: LoadedLayer,
    pub prior_kv: ActivationSequence,
}

pub fn load_inputs_from_args() -> Result<Inputs> {
    with_main_sequence_scope(|| {
        load_inputs_from_initialized_runtime(&CachedInputs::new(), None, None)
    })
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
    let mut output = run_layer_stage(LayerStageInputs {
        activations: &inputs.activations,
        layer: &inputs.layer,
        prior_kv: &inputs.prior_kv,
    })?;
    output.model = model_identity::stream::verify(
        output.model,
        inputs.layer.root(),
        model_identity::Role::Transformer(inputs.layer.params.layer_idx),
    );
    // The first range of the text inherits no cache.
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
    ])?;

    let sequence = |name: &str| -> Result<ActivationSequence> {
        match cached_inputs.get(name) {
            Some(value) => value.as_range_activation_sequence(),
            None => Ok(raster::materialize_auth_return(
                raster::entry_argument_auth_ref::<ActivationSequence>(
                    binding.reference.clone(),
                    name,
                ),
            )),
        }
    };
    let activations = sequence("activations")?;
    let prior_kv = sequence("prior_kv")?;
    let layer = LoadedLayer(materialize_with_cache(
        materialization_cache,
        layer_key,
        || Loaded {
            layer: raster::materialize_auth_return(raster::entry_argument_auth_ref::<
                TransformerLayer,
            >(
                binding.reference.clone(), "layer"
            )),
            root: OnceLock::new(),
        },
    )?);

    Ok(Inputs {
        activations,
        layer,
        prior_kv,
    })
}
