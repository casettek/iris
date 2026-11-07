//! Generated output finalization sequences.

use raster::prelude::*;

use output_finalize::input::*;
use output_finalize::prompt::{PromptContext, PromptState};
use output_finalize::*;

/// Resolves one generated token by its authorized ID and advances the bounded
/// detokenization/commitment state.
#[sequence(kind = recur)]
fn decode_generated_token(
    input: RecurSequenceInput<u32>,
    state: RecurSequenceState<FinalizeState>,
    tokens: List<DecoderToken>,
    paths: List<DecoderProof>,
    descriptor: BytesPage,
    model_id: String,
    token_count: u64,
) -> RecurSequenceState<FinalizeState> {
    let token_id = into_ref!(input);
    let token = select!(DecoderToken, tokens[token_id]);
    let proof = select!(DecoderProof, paths[token_id]);
    call!(
        consume_authenticated_decoder,
        state,
        descriptor,
        model_id,
        token_id,
        token_count,
        proof,
        token
    )
}

/// `prompt` is the last prompt range's output: its context is what the final
/// output says the chain consumed, and under which configuration.
#[sequence]
fn main(
    edge: DecodeEdge,
    decoder: DecoderTable,
    prompt: PromptState,
    descriptor: BytesPage,
    witness: DecoderWitness,
) -> GeneratedOutput {
    let context = select!(PromptContext, prompt.context);
    let model_id = select!(String, context.clone().model_id);
    let token_count = select!(u64, witness.clone().token_count);
    let paths = select!(List<DecoderProof>, witness.paths);
    let edge_model = select!(ModelContext, edge.clone().model);
    let has_selected = select!(bool, edge.clone().has_selected);
    let ids = select!(List<u32>, edge.clone().generated_token_ids);
    let ids_for_decode = clone!(ids);
    let tokens = select!(List<DecoderToken>, decoder.tokens);

    let state = call!(
        begin_authenticated_finalize,
        descriptor.clone(),
        model_id.clone()
    );
    let state = call!(
        verify_final_model_context,
        state,
        edge_model,
        context.clone(),
        descriptor.clone()
    );
    let state = call_recur_seq!(
        sequence = decode_generated_token,
        input = ids_for_decode,
        state = state,
        args = (tokens, paths, descriptor, model_id, token_count)
    );
    let state = call!(finish_finalize_state, state);

    let claim_mismatches = select!(u32, edge.clone().claim_mismatches);
    let first_claim_mismatch = select!(u32, edge.clone().first_claim_mismatch_position);
    let check_fault = select!(u32, edge.clone().check_fault);
    let check_fault_position = select!(u32, edge.check_fault_position);
    let draft = call!(
        begin_generated_output,
        state,
        has_selected,
        claim_mismatches,
        first_claim_mismatch,
        check_fault,
        check_fault_position,
        context
    );
    let output = call_recur!(
        tile = copy_output_token_ids,
        input = ids,
        chunk = 64,
        output = draft,
        args = ()
    );
    raster::println!("generated output → {:?}", output);
    output
}
