//! `sequence-check`: sequences, one stage per range of claimed tokens.
//!
//! ```text
//!   previous.rows ──recur (chunk = 256)──▶ last row ───────────┐
//!   activations.rows ──recur (chunk = 64)──▶ append ───────────┴▶ predicting rows
//!   claimed.token_ids ──recur seq──▶ [ step · predicting[i] · normalise · head pages ──recur──▶ argmax · record ]
//!   errors     ──recur──▶ summary ─┐
//!   selections ──recur──▶ summary ─┤
//!   prompt.context ────────────────┼▶ check fault ─┐
//!   prior scalars ─────────────────┘               │
//!   prior.claim_mismatches ────────────────────────┴▶ edge scalars + claim verdict + fault
//!   prior.generated_token_ids ──recur──▶ copy ──▶ selections ──recur──▶ copy ──▶ DecodeEdge
//! ```

use raster::prelude::*;

use sequence_check::input::*;
use sequence_check::prompt::{PromptContext, PromptState};
use sequence_check::*;

/// One generated position: locate the row that predicts it, norm it, sweep
/// the head into an argmax, and record the selection beside the claim.
#[sequence(kind = recur)]
fn check_position(
    input: RecurSequenceInput<ActivationRow>,
    state: RecurSequenceState<CheckCursor>,
    output: RecurSequenceOutput<CheckedPositions>,
    rows: List<ActivationRow>,
    pages: List<BytesPage>,
    params: FinalHeadParams,
) -> (
    RecurSequenceState<CheckCursor>,
    RecurSequenceOutput<CheckedPositions>,
) {
    let token = call!(checked_row_token, input);
    let step = call!(begin_check_step, state, token);
    let row_index = select!(u32, step.clone().row_index);
    let row = select!(ActivationRow, rows[row_index]);
    let normalized = call!(normalize_position, row, params.clone());
    let seed = call!(begin_argmax);
    let best = call_recur!(
        tile = score_head_page,
        input = pages,
        state = seed,
        args = (normalized, params)
    );
    let output = call!(record_selection, output, step.clone(), best);
    let state = call!(advance_check_cursor, step);
    (state, output)
}

/// Resolves one token of the transcript by its authorized id and advances the
/// bounded detokenization and commitment state.
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

/// Teacher-forced check entrypoint for one range of claimed tokens.
///
/// `activations` is the last transformer layer's output for this range;
/// `previous` is its output for the range before, whose last row predicts
/// this range's first token; each activation row carries its consumed ID; `prior`
/// is the previous range's check (or the empty edge); `head` is
/// the committed output head; `prompt` is the last prompt range's output, whose
/// context says how many tokens the prompt became and how large a range may be.
///
/// The check also finalizes the transcript so far: `decoder` is the committed
/// decoder table, `witness` its membership paths and `descriptor` the model's.
/// The record of the last claimed range is the claim's output.
#[sequence]
fn main(
    activations: ActivationSequence,
    previous: ActivationSequence,
    prior: DecodeEdge,
    head: FinalHead,
    prompt: PromptState,
    decoder: DecoderTable,
    descriptor: BytesPage,
    witness: DecoderWitness,
) -> Result<DecodeEdge> {
    let input_model = select!(ModelContext, activations.clone().model);
    let checked_rows = select!(List<ActivationRow>, activations.clone().rows);
    let zero = call!(identity_zero_count);
    let row_count = call_recur!(
        tile = binding_count_rows,
        input = checked_rows,
        chunk = 64,
        state = zero,
        args = ()
    );
    let input_start = select!(u32, activations.clone().start_position);
    let input_model = call!(binding_validate_rows, input_model, input_start, row_count);

    let previous_model = select!(ModelContext, previous.clone().model);
    let prompt_model = select!(ModelContext, prompt.clone().context.model);
    let prior_model = select!(ModelContext, prior.clone().model);
    let model = call!(
        head_model_context,
        input_model,
        previous_model,
        prompt_model,
        prior_model.clone()
    );
    let prior_ids = select!(List<u32>, prior.clone().generated_token_ids);
    let zero = call!(identity_zero_count);
    let count = call_recur!(
        tile = identity_count_tokens,
        input = prior_ids,
        chunk = 256,
        state = zero,
        args = ()
    );
    let model = call!(identity_prior_guard, model, prior_model, count);
    let root = call_seq!(identity_root, head.clone());
    let model = call!(identity_verify, model, root);
    let params = select!(FinalHeadParams, head.clone().params);
    let params = call!(validate_final_head, params)?;

    let previous_rows = select!(List<ActivationRow>, previous.rows);
    let tail_seed = call!(begin_previous_tail);
    // 64 rows a call: a row is `hidden_size · 4` bytes, and reading 256 of
    // them at width 2560 was the heaviest tile call in the chain.
    let tail = call_recur!(
        tile = keep_last_row,
        input = previous_rows,
        chunk = 64,
        state = tail_seed,
        args = ()
    );
    let opened = call!(begin_predicting_rows, tail);
    let range_rows = select!(List<ActivationRow>, activations.clone().rows);
    // 16 rows a call: a row copied into the output costs about five times a
    // row read.
    let predicting = call_recur!(
        tile = append_predicting_rows,
        input = range_rows.clone(),
        chunk = 16,
        output = opened,
        args = ()
    );

    let start_position = select!(u32, activations.start_position);
    let cursor = call!(begin_check_cursor, start_position.clone());
    let rows = select!(List<ActivationRow>, predicting.rows);
    let pages = select!(List<BytesPage>, head.projection.pages);
    let checked = call_recur_seq!(
        sequence = check_position,
        input = range_rows,
        state = cursor,
        output,
        args = (rows, pages, params)
    );

    let errors = select!(List<PositionError>, checked.clone().errors);
    let error_seed = call!(begin_error_summary);
    let error_summary = call_recur!(
        tile = summarise_errors,
        input = errors,
        state = error_seed,
        args = ()
    );

    let selections = select!(List<PositionSelection>, checked.selections);
    let summary_seed = call!(begin_selection_summary);
    let summary = call_recur!(
        tile = summarise_selections,
        input = selections.clone(),
        chunk = 64,
        state = summary_seed,
        args = ()
    );

    let prior_has_selected = select!(bool, prior.clone().has_selected);
    let prior_position = select!(u32, prior.clone().decode_position);
    let prior_fault = select!(u32, prior.clone().check_fault);
    let prior_fault_position = select!(u32, prior.clone().check_fault_position);
    let range_size = select!(u32, prompt.clone().context.range_size);
    let prompt_tokens = select!(u32, prompt.clone().context.prompt_tokens);
    let frame = call!(
        begin_check_frame,
        start_position,
        prior_has_selected,
        prior_position,
        prior_fault,
        prior_fault_position,
        range_size,
        prompt_tokens
    );
    let fault = call!(find_check_fault, summary.clone(), error_summary, frame);

    // The transcript so far: the earlier ranges' ids, then this range's
    // selections. It is built before the edge because the record is stated
    // over it, and an edge's own fields are set once, when it is opened.
    let prior_mismatches = select!(u32, prior.clone().claim_mismatches);
    let prior_first_mismatch = select!(u32, prior.clone().first_claim_mismatch_position);
    let prior_ids = select!(List<u32>, prior.generated_token_ids);
    let carried = call_recur!(
        tile = copy_prior_token_ids,
        input = prior_ids,
        chunk = 64,
        output,
        args = ()
    );
    let transcript = call_recur!(
        tile = copy_selected_token_ids,
        input = selections,
        chunk = 64,
        output = carried,
        args = ()
    );
    let ids = select!(List<u32>, transcript.ids);

    // Finalize it: decode every id against the committed decoder table, and
    // state what the transcript breaks. The final step belongs after the
    // schedule's last check, so an earlier range's record carries a fault
    // and only the last range's can be a claim's output.
    let context = select!(PromptContext, prompt.context);
    let model_id = select!(String, context.clone().model_id);
    let token_count = select!(u64, witness.clone().token_count);
    let paths = select!(List<DecoderProof>, witness.paths);
    let tokens = select!(List<DecoderToken>, decoder.tokens);
    let has_selected = select!(bool, summary.clone().has_selected);
    let position = select!(u32, summary.clone().decode_position);
    let token = select!(u32, summary.clone().token_id);
    let state = call!(
        begin_authenticated_finalize,
        descriptor.clone(),
        model_id.clone()
    );
    let state = call!(
        verify_final_model_context,
        state,
        model.clone(),
        context.clone(),
        descriptor.clone(),
        has_selected.clone(),
        position,
        token
    );
    let state = call_recur_seq!(
        sequence = decode_generated_token,
        input = ids.clone(),
        state = state,
        args = (tokens, paths, descriptor, model_id, token_count)
    );
    let record = call!(finish_output_record, state, has_selected, context);

    // The verdict accumulates across ranges, so the last check's edge
    // covers every claimed token.
    let base = call!(
        begin_decode_edge,
        summary,
        prior_mismatches,
        prior_first_mismatch,
        fault,
        model,
        record
    );
    let edge = call_recur!(
        tile = copy_transcript,
        input = ids,
        chunk = 64,
        output = base,
        args = ()
    );
    raster::println!("sequence check → {:?}", edge);
    Ok(edge)
}

#[sequence]
fn identity_bytes_48_960(value: Bytes<48_960>) -> BytesPage {
    let len = select!(u64, value.clone().byte_len);
    let size = select!(u64, value.clone().page_size);
    let pages = select!(List<BytesPage>, value.pages);
    let seed = call!(identity_empty);
    let folded = call_recur!(tile = identity_page, input = pages, state = seed, args = ());
    let root = call!(identity_finish, folded);
    call!(identity_bytes, len, size, root)
}
#[sequence]
fn identity_root(value: FinalHead) -> BytesPage {
    let params = select!(FinalHeadParams, value.clone().params);
    let params = call!(identity_hash_finalheadparams, params);
    let projection = select!(Bytes<48_960>, value.clone().projection);
    let projection = call_seq!(identity_bytes_48_960, projection);
    call!(identity_component, params, projection)
}
