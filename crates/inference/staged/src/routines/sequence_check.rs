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
use inference_kernels::kernels::finalize::{claim_record, output_record, FinalizeInputs};
use inference_kernels::kernels::sequence_check::{
    check_frame, check_positions, decode_edge, run_sequence_check_direct, scorable,
    CheckHeadParams, SequenceCheckDirectInputs,
};
use sequence_check::input::{
    unpack_i32s, ActivationSequence, DecodeEdge, DecoderTable, DecoderWitness, FinalHead,
};
use sequence_check::prompt::{PromptContext, PromptState};
use std::{path::Path, sync::Arc};

pub struct Inputs {
    component_root: raster::BytesPage,
    pub activations: ActivationSequence,
    pub previous: ActivationSequence,
    pub prior: DecodeEdge,
    /// The last prompt range's context: the range limit and the prompt's length.
    pub context: PromptContext,
    head: Weights,
    /// What the output record is stated from: the committed decoder table,
    /// its membership paths and the model descriptor.
    pub decoder: Arc<DecoderTable>,
    pub descriptor: raster::BytesPage,
    pub witness: Arc<DecoderWitness>,
}
enum Weights {
    Eager(Arc<FinalHead>),
    Paged(PagedHead),
}
impl Inputs {
    /// Already-materialized comparison path; indexed Raster inputs use the paged head.
    #[allow(clippy::too_many_arguments)]
    pub fn eager(
        activations: ActivationSequence,
        previous: ActivationSequence,
        prior: DecodeEdge,
        head: Arc<FinalHead>,
        context: PromptContext,
        decoder: Arc<DecoderTable>,
        descriptor: raster::BytesPage,
        witness: Arc<DecoderWitness>,
    ) -> Self {
        Self {
            component_root: model_identity::stream::digest(head.as_ref()),
            activations,
            previous,
            prior,
            context,
            head: Weights::Eager(head),
            decoder,
            descriptor,
            witness,
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
    let mut edge = match &inputs.head {
        Weights::Eager(head) => run_sequence_check_direct(SequenceCheckDirectInputs {
            activations: &inputs.activations,
            previous: &inputs.previous,
            prior: &inputs.prior,
            head,
            context: &inputs.context,
        })?,
        Weights::Paged(head) => run_paged(inputs, head)?,
    };
    let mut checked_model = inputs.activations.model.clone();
    checked_model.validate_rows(
        inputs.activations.start_position,
        inputs.activations.rows.len() as u64,
    );
    edge.model = checked_model.head(
        &inputs.previous.model,
        &inputs.context.model,
        &inputs.prior.model,
    );
    edge.model = model_identity::stream::verify(
        edge.model,
        &inputs.component_root,
        model_identity::Role::Head,
    );
    edge.model
        .require(!inputs.prior.model.is_origin() || inputs.prior.generated_token_ids.is_empty());
    // The transcript so far, finalized. It reads the edge's model context,
    // so it is stated last.
    edge.record = output_record(
        &edge,
        &FinalizeInputs {
            decoder: &inputs.decoder,
            context: &inputs.context,
            descriptor: &inputs.descriptor,
            witness: &inputs.witness,
        },
    )?;
    edge.claim = claim_record(&edge);
    Ok(edge)
}
fn run_paged(inputs: &Inputs, head: &PagedHead) -> Result<DecodeEdge> {
    let params = head.params();
    let norm_weights = unpack_i32s(&params.norm_weights).map_err(anyhow::Error::msg)?;
    let positions = check_positions(
        &inputs.activations,
        &inputs.previous,
        &CheckHeadParams {
            hidden_size: params.hidden_size,
            norm_eps: params.norm_eps,
            softcap: params.softcap,
            norm_weights: &norm_weights,
        },
    )?;
    let selections = head::select_positions(&scorable(&positions), head)?;
    let mut edge = decode_edge(
        &positions,
        &selections,
        &inputs.prior,
        &check_frame(&inputs.activations, &inputs.prior, &inputs.context),
    )?;
    if edge.claim_mismatches > inputs.prior.claim_mismatches {
        println!(
            "sequence check: {} claimed token(s) are not the greedy selection; first at position {}",
            edge.claim_mismatches, edge.first_claim_mismatch_position
        );
    }
    let mut checked_model = inputs.activations.model.clone();
    checked_model.validate_rows(
        inputs.activations.start_position,
        inputs.activations.rows.len() as u64,
    );
    edge.model = checked_model.head(
        &inputs.previous.model,
        &inputs.context.model,
        &inputs.prior.model,
    );
    Ok(edge)
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
        raster::entry_argument_spec::<ActivationSequence>("previous"),
        raster::entry_argument_spec::<DecodeEdge>("prior"),
        raster::entry_argument_spec::<FinalHead>("head"),
        raster::entry_argument_spec::<PromptState>("prompt"),
        raster::entry_argument_spec::<DecoderTable>("decoder"),
        raster::entry_argument_spec::<raster::BytesPage>("descriptor"),
        raster::entry_argument_spec::<DecoderWitness>("witness"),
    ])?;
    let component_root = model_identity::stream::page(
        &binding
            .arguments
            .iter()
            .find(|v| v.name == "head")
            .expect("declared component input")
            .commitment,
    );
    let activations = match cached_inputs.get("activations") {
        Some(value) => value.as_check_activation_sequence()?,
        None => raster::materialize_auth_return(raster::entry_argument_auth_ref::<
            ActivationSequence,
        >(binding.reference.clone(), "activations")),
    };
    let previous = match cached_inputs.get("previous") {
        Some(value) => value.as_check_activation_sequence()?,
        None => raster::materialize_auth_return(raster::entry_argument_auth_ref::<
            ActivationSequence,
        >(binding.reference.clone(), "previous")),
    };
    let prior = match cached_inputs.get("prior") {
        Some(value) => value.as_check_prior()?,
        None => raster::materialize_auth_return(raster::entry_argument_auth_ref::<DecodeEdge>(
            binding.reference.clone(),
            "prior",
        )),
    };
    let context = match cached_inputs.get("prompt") {
        Some(value) => value.as_check_prompt_context()?,
        None => {
            raster::materialize_auth_return(raster::entry_argument_auth_ref::<PromptState>(
                binding.reference.clone(),
                "prompt",
            ))
            .context
        }
    };
    let decoder = Arc::new(raster::materialize_auth_return(
        raster::entry_argument_auth_ref::<DecoderTable>(binding.reference.clone(), "decoder"),
    ));
    let descriptor = raster::materialize_auth_return(raster::entry_argument_auth_ref::<
        raster::BytesPage,
    >(binding.reference.clone(), "descriptor"));
    // Never reuse a validation flag from CachedInputs: check the supplied
    // witness against the bytes being consumed on every execution.
    let witness = Arc::new(raster::materialize_auth_return(
        raster::entry_argument_auth_ref::<DecoderWitness>(binding.reference.clone(), "witness"),
    ));
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
        component_root,
        activations,
        previous,
        prior,
        context,
        head,
        decoder,
        descriptor,
        witness,
    })
}
