use crate::{
    artifact_io::{with_main_sequence_scope, with_stage_sequence_scope},
    cache::{
        materialization_key_from_stage_files, materialize_with_cache, CachedInputs,
        MaterializationCache, MaterializationCacheKey,
    },
    head::{self, PagedHead, StaticHead},
    verified_weights::{input_documents_from_args, raster_inputs},
};
use anyhow::Result;
use inference_kernels::kernels::prefill_finalize::{
    run_prefill_finalize_direct, PrefillFinalizeDirectInputs,
};
use prefill_finalize::input::{ActivationSequence, FinalHead, PrefillLogits};
use std::{path::Path, sync::Arc};

pub struct Inputs {
    pub activations: ActivationSequence,
    head: Weights,
}
enum Weights {
    Eager(Arc<FinalHead>),
    Paged(PagedHead),
}
impl Inputs {
    /// Already-materialized comparison path; indexed Raster inputs use paged projection.
    pub fn eager(activations: ActivationSequence, head: Arc<FinalHead>) -> Self {
        Self {
            activations,
            head: Weights::Eager(head),
        }
    }
}
pub fn load_inputs_from_args() -> Result<Inputs> {
    let (input, manifest) = input_documents_from_args();
    with_main_sequence_scope(|| {
        load_inputs_from_initialized_runtime(
            &CachedInputs::new(),
            None,
            None,
            input.as_deref(),
            manifest.as_deref(),
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
    let key = materialization_key_from_stage_files::<StaticHead>(input, input_manifest, "head")?;
    with_stage_sequence_scope(input, input_manifest, || {
        load_inputs_from_initialized_runtime(
            cached_inputs,
            materialization_cache,
            key,
            input.to_str(),
            input_manifest.to_str(),
        )
    })
}
pub fn run_direct(inputs: &Inputs) -> Result<PrefillLogits> {
    match &inputs.head {
        Weights::Eager(head) => run_prefill_finalize_direct(PrefillFinalizeDirectInputs {
            activations: &inputs.activations,
            head,
        }),
        Weights::Paged(head) => head::run(&inputs.activations, head),
    }
}
fn load_inputs_from_initialized_runtime(
    cached_inputs: &CachedInputs,
    materialization_cache: Option<&MaterializationCache>,
    key: Option<MaterializationCacheKey>,
    input: Option<&str>,
    manifest: Option<&str>,
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
    let head = if raster_inputs(&binding) {
        Weights::Paged(head::load(
            binding,
            input,
            manifest,
            materialization_cache,
            key,
        )?)
    } else {
        let key = key.map(|mut key| {
            key.type_name = std::any::type_name::<FinalHead>();
            key
        });
        Weights::Eager(materialize_with_cache(materialization_cache, key, || {
            raster::materialize_auth_return(raster::entry_argument_auth_ref::<FinalHead>(
                binding.reference,
                "head",
            ))
        })?)
    };
    Ok(Inputs { activations, head })
}
