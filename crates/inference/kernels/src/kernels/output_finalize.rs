use anyhow::Result;
use output_finalize::input::{DecodeEdge, DecoderTable, GeneratedOutput};
use output_finalize::prompt::PromptContext;
use output_finalize::{output_fault_of, sha256_hex};
use raster::List;

pub struct OutputFinalizeDirectInputs<'a> {
    pub edge: &'a DecodeEdge,
    pub decoder: &'a DecoderTable,
    pub context: &'a PromptContext,
}

/// This stage's copy of the prompt context: the stages share its source, not a
/// Rust type.
pub fn prompt_context(source: &prompt_prepare::input::PromptContext) -> PromptContext {
    PromptContext {
        input_text_sha256: source.input_text_sha256.clone(),
        model_id: source.model_id.clone(),
        raw_prompt: source.raw_prompt,
        max_new_tokens: source.max_new_tokens,
        range_size: source.range_size,
        next_piece: source.next_piece,
        prompt_tokens: source.prompt_tokens,
        complete: source.complete,
        fault: source.fault,
        fault_position: source.fault_position,
    }
}

pub fn run_output_finalize_direct(
    inputs: OutputFinalizeDirectInputs<'_>,
) -> Result<GeneratedOutput> {
    let mut state = output_finalize::initial_finalize_state();
    for token_id in inputs.edge.generated_token_ids.iter().copied() {
        let token = inputs
            .decoder
            .tokens
            .get(token_id as usize)
            .ok_or_else(|| anyhow::anyhow!("decoder table has no token id {token_id}"))?;
        state = output_finalize::advance_finalize_state(state, token_id, token.clone());
    }
    let state = output_finalize::finished_state(state);
    let output_fault = output_fault_of(&state, inputs.edge.has_selected, inputs.context);
    let context = inputs.context;
    Ok(GeneratedOutput {
        generated_token_count: state.count,
        generated_token_ids: List::from(
            inputs
                .edge
                .generated_token_ids
                .iter()
                .copied()
                .collect::<Vec<_>>(),
        ),
        generated_token_ids_sha256: sha256_hex(state.json.as_bytes()),
        output_text_sha256: sha256_hex(state.text.as_bytes()),
        generated_text: state.text,
        stop_reason: if state.stopped {
            String::from("eos")
        } else {
            String::from("max_new_tokens")
        },
        claim_mismatches: inputs.edge.claim_mismatches,
        first_claim_mismatch_position: inputs.edge.first_claim_mismatch_position,
        check_fault: inputs.edge.check_fault,
        check_fault_position: inputs.edge.check_fault_position,
        output_fault,
        input_text_sha256: context.input_text_sha256.clone(),
        model_id: context.model_id.clone(),
        raw_prompt: context.raw_prompt,
        max_new_tokens: context.max_new_tokens,
        range_size: context.range_size,
        prompt_tokens: context.prompt_tokens,
        prompt_complete: context.complete,
        prompt_fault: context.fault,
        prompt_fault_position: context.fault_position,
    })
}
