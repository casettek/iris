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
use inference_kernels::kernels::sequence_check::{
    check_positions, decode_edge, first_mismatch, run_sequence_check_direct, CheckHeadParams,
    SequenceCheckDirectInputs,
};
use prefill_finalize::input::unpack_i32s;
use sequence_check::input::{
    ActivationSequence, ClaimedOutput, DecodeEdge, FinalHead, PromptTokenization,
};
use std::{path::Path, sync::Arc};

pub struct Inputs {
    pub activations: ActivationSequence,
    pub prompt: PromptTokenization,
    pub claimed: ClaimedOutput,
    head: Weights,
}
enum Weights {
    Eager(Arc<FinalHead>),
    Paged(PagedHead),
}
impl Inputs {
    /// Already-materialized comparison path; indexed Raster inputs use the paged head.
    pub fn eager(
        activations: ActivationSequence,
        prompt: PromptTokenization,
        claimed: ClaimedOutput,
        head: Arc<FinalHead>,
    ) -> Self {
        Self {
            activations,
            prompt,
            claimed,
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
pub fn run_direct(inputs: &Inputs) -> Result<DecodeEdge> {
    let edge = match &inputs.head {
        Weights::Eager(head) => run_sequence_check_direct(SequenceCheckDirectInputs {
            activations: &inputs.activations,
            prompt: &inputs.prompt,
            claimed: &inputs.claimed,
            head,
        })?,
        Weights::Paged(head) => run_paged(inputs, head)?,
    };
    Ok(edge)
}
fn run_paged(inputs: &Inputs, head: &PagedHead) -> Result<DecodeEdge> {
    let params = head.params();
    let norm_weights = unpack_i32s(&params.norm_weights).map_err(anyhow::Error::msg)?;
    let positions = check_positions(
        &inputs.activations,
        &inputs.prompt,
        &inputs.claimed,
        &CheckHeadParams {
            hidden_size: params.hidden_size,
            norm_eps: params.norm_eps,
            softcap: params.softcap,
            norm_weights: &norm_weights,
        },
    )?;
    let normalized = positions
        .iter()
        .map(|position| position.normalized.as_slice())
        .collect::<Vec<_>>();
    let selections = head::select_positions(&normalized, head)?;
    if let Some(position) = first_mismatch(&positions, &selections) {
        println!(
            "sequence check: claimed token at position {position} is not the greedy selection"
        );
    }
    decode_edge(&positions, &selections)
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
        raster::entry_argument_spec::<PromptTokenization>("prompt"),
        raster::entry_argument_spec::<ClaimedOutput>("claimed"),
        raster::entry_argument_spec::<FinalHead>("head"),
    ])?;
    let activations = match cached_inputs.get("activations") {
        Some(value) => value.as_check_activation_sequence()?,
        None => raster::materialize_auth_return(raster::entry_argument_auth_ref::<
            ActivationSequence,
        >(binding.reference.clone(), "activations")),
    };
    let prompt = match cached_inputs.get("prompt") {
        Some(value) => value.as_check_prompt()?,
        None => raster::materialize_auth_return(raster::entry_argument_auth_ref::<
            PromptTokenization,
        >(binding.reference.clone(), "prompt")),
    };
    let claimed = raster::materialize_auth_return(
        raster::entry_argument_auth_ref::<ClaimedOutput>(binding.reference.clone(), "claimed"),
    );
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
    Ok(Inputs {
        activations,
        prompt,
        claimed,
        head,
    })
}
