//! Output finalization, natively: the record a check states with its edge,
//! and direct inference's output.
use anyhow::Result;
use sequence_check::input::{
    ClaimRecord, DecodeEdge, DecoderTable, DecoderWitness, OutputRecord,
};
use sequence_check::prompt::PromptContext;

/// What finalization reads besides the edge.
pub struct FinalizeInputs<'a> {
    pub decoder: &'a DecoderTable,
    /// The last prompt range's context: what the chain consumed, and how.
    pub context: &'a PromptContext,
    pub descriptor: &'a raster::BytesPage,
    pub witness: &'a DecoderWitness,
}

/// The record of the transcript `edge` holds, as the check stage states it:
/// every id decoded against the committed decoder table, with what the
/// transcript breaks. Only the schedule's last claimed range gets a record
/// without a model fault.
pub fn output_record(edge: &DecodeEdge, inputs: &FinalizeInputs<'_>) -> Result<OutputRecord> {
    let mut state =
        sequence_check::authenticated_finalize_state(inputs.descriptor, &inputs.context.model_id);
    state.model_fault |=
        sequence_check::final_context_fault(&edge.model, inputs.context, inputs.descriptor);
    sequence_check::begin_final_history(
        &mut state,
        &edge.model,
        edge.has_selected,
        edge.decode_position,
        edge.token_id,
    );
    for token_id in edge.generated_token_ids.iter().copied() {
        let token = inputs
            .decoder
            .tokens
            .get(token_id as usize)
            .ok_or_else(|| anyhow::anyhow!("decoder table has no token id {token_id}"))?;
        let siblings = inputs
            .witness
            .paths
            .get(token_id as usize)
            .ok_or_else(|| anyhow::anyhow!("decoder witness has no token id {token_id}"))?;
        state = sequence_check::advance_authenticated_finalize(
            state,
            inputs.descriptor,
            &inputs.context.model_id,
            token_id,
            inputs.witness.token_count,
            &siblings.siblings,
            token.clone(),
        );
    }
    Ok(sequence_check::output_record(
        sequence_check::finished_state(state),
        edge.has_selected,
        inputs.context,
    ))
}

/// Direct inference's final edge with its output record: byte for byte what
/// an honest claim's last check publishes.
pub fn run_output_finalize_direct(
    edge: &DecodeEdge,
    inputs: &FinalizeInputs<'_>,
) -> Result<DecodeEdge> {
    let mut edge = edge.clone();
    edge.record = output_record(&edge, inputs)?;
    edge.claim = claim_record(&edge);
    Ok(edge)
}

/// The claim record of an edge whose output record is already stated, as the
/// check stage states it when it opens the edge.
pub fn claim_record(edge: &DecodeEdge) -> ClaimRecord {
    sequence_check::claim_record(&edge.record, edge.claim_mismatches, edge.check_fault)
}

/// Output discovery for direct-only imports without a staged commitment package.
/// Explicitly marks the record unverified; this path cannot authorize a claim.
pub fn run_output_finalize_discovery(
    edge: &DecodeEdge,
    decoder: &DecoderTable,
    context: &PromptContext,
) -> Result<DecodeEdge> {
    let mut state = sequence_check::initial_finalize_state();
    state.model_fault = 1;
    for &token_id in edge.generated_token_ids.iter() {
        let token = decoder
            .tokens
            .get(token_id as usize)
            .ok_or_else(|| anyhow::anyhow!("decoder table has no token id {token_id}"))?;
        state = sequence_check::advance_finalize_state(state, token_id, token.clone());
    }
    let mut edge = edge.clone();
    edge.record = sequence_check::output_record(
        sequence_check::finished_state(state),
        edge.has_selected,
        context,
    );
    edge.claim = claim_record(&edge);
    Ok(edge)
}
