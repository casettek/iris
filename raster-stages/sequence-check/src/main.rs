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
use sequence_check::prompt::PromptState;
use sequence_check::*;

/// One generated position: locate the row that predicts it, norm it, sweep
/// the head into an argmax, and record the selection beside the claim.
#[sequence(kind = recur)]
fn check_position(
    input: RecurSequenceInput<u32>,
    state: RecurSequenceState<CheckCursor>,
    output: RecurSequenceOutput<CheckedPositions>,
    rows: List<ActivationRow>,
    pages: List<BytesPage>,
    params: FinalHeadParams,
) -> (
    RecurSequenceState<CheckCursor>,
    RecurSequenceOutput<CheckedPositions>,
) {
    let step = call!(begin_check_step, state, input);
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

/// Teacher-forced check entrypoint for one range of claimed tokens.
///
/// `activations` is the last transformer layer's output for this range;
/// `previous` is its output for the range before, whose last row predicts
/// this range's first token; `claimed` is this range's claimed ids; `prior`
/// is the previous range's check (or the empty `decode_init` edge); `head` is
/// the committed output head; `prompt` is the last prompt range's output, whose
/// context says how many tokens the prompt became and how large a range may be.
#[sequence]
fn main(
    activations: ActivationSequence,
    previous: ActivationSequence,
    claimed: ClaimedOutput,
    prior: DecodeEdge,
    head: FinalHead,
    prompt: PromptState,
) -> Result<DecodeEdge> {
    let params = select!(FinalHeadParams, head.clone().params);
    let params = call!(validate_final_head, params)?;

    let previous_rows = select!(List<ActivationRow>, previous.rows);
    let tail_seed = call!(begin_previous_tail);
    let tail = call_recur!(
        tile = keep_last_row,
        input = previous_rows,
        chunk = 256,
        state = tail_seed,
        args = ()
    );
    let opened = call!(begin_predicting_rows, tail);
    let range_rows = select!(List<ActivationRow>, activations.clone().rows);
    let predicting = call_recur!(
        tile = append_predicting_rows,
        input = range_rows,
        chunk = 64,
        output = opened,
        args = ()
    );

    let start_position = select!(u32, activations.start_position);
    let cursor = call!(begin_check_cursor, start_position.clone());
    let rows = select!(List<ActivationRow>, predicting.rows);
    let pages = select!(List<BytesPage>, head.projection.pages);
    let claimed_ids = select!(List<u32>, claimed.token_ids);
    let checked = call_recur_seq!(
        sequence = check_position,
        input = claimed_ids,
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
    let prompt_tokens = select!(u32, prompt.context.prompt_tokens);
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

    // The verdict accumulates across ranges, so the last check's edge, the
    // one `output_finalize` publishes, covers every claimed token.
    let prior_mismatches = select!(u32, prior.clone().claim_mismatches);
    let prior_first_mismatch = select!(u32, prior.clone().first_claim_mismatch_position);
    // Push-only sites deriving in turn from the opened edge: the earlier
    // ranges' transcript, then this range's selections.
    let base = call!(
        begin_decode_edge,
        summary,
        prior_mismatches,
        prior_first_mismatch,
        fault
    );
    let prior_ids = select!(List<u32>, prior.generated_token_ids);
    let carried = call_recur!(
        tile = copy_prior_token_ids,
        input = prior_ids,
        chunk = 64,
        output = base,
        args = ()
    );
    let edge = call_recur!(
        tile = copy_selected_token_ids,
        input = selections,
        chunk = 64,
        output = carried,
        args = ()
    );
    raster::println!("sequence check → {:?}", edge);
    Ok(edge)
}
