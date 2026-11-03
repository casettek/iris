//! `sequence-check`: sequences.
//!
//! ```text
//!   prompt.token_ids ──recur (chunk = 256)──▶ prompt_len ──▶ cursor at row prompt_len - 1
//!   claimed.token_ids ──recur seq──▶ [ step · rows[i] · normalise · head pages ──recur──▶ argmax · record ]
//!   selections ──recur──▶ summary ──▶ edge scalars
//!   selections ──recur──▶ copy ids ──▶ DecodeEdge
//!   errors     ──recur──▶ summarise ──▶ assert none
//! ```

use raster::prelude::*;

use sequence_check::input::*;
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

/// Teacher-forced check entrypoint.
///
/// `activations` is bound by the chain to the last transformer layer's output
/// over `prompt ‖ claimed`; `prompt` to `prompt_prepare`; `claimed` and `head`
/// are committed externals.
#[sequence]
fn main(
    activations: ActivationSequence,
    prompt: PromptTokenization,
    claimed: ClaimedOutput,
    head: FinalHead,
) -> Result<DecodeEdge> {
    let params = select!(FinalHeadParams, head.clone().params);
    let params = call!(validate_final_head, params)?;

    let prompt_ids = select!(List<u32>, prompt.token_ids);
    let count_seed = call!(begin_token_count);
    let prompt_count = call_recur!(
        tile = count_token_chunk,
        input = prompt_ids,
        chunk = 256,
        state = count_seed,
        args = ()
    );
    let prompt_len = select!(u32, prompt_count.count);
    let start_position = select!(u32, activations.clone().start_position);
    let cursor = call!(begin_check_cursor, prompt_len, start_position)?;

    let rows = select!(List<ActivationRow>, activations.rows);
    let pages = select!(List<BytesPage>, head.projection.pages);
    let claimed_ids = select!(List<u32>, claimed.token_ids);
    let checked = call_recur_seq!(
        sequence = check_position,
        input = claimed_ids,
        state = cursor,
        output,
        args = (rows, pages, params)
    );

    let errors = select!(List<String>, checked.clone().errors);
    let error_seed = call!(begin_error_summary);
    let error_summary = call_recur!(
        tile = summarise_errors,
        input = errors,
        state = error_seed,
        args = ()
    );
    let error_count = select!(u32, error_summary.clone().count);
    let first_error = select!(String, error_summary.first);

    let selections = select!(List<PositionSelection>, checked.selections);
    let summary_seed = call!(begin_selection_summary);
    let summary = call_recur!(
        tile = summarise_selections,
        input = selections.clone(),
        chunk = 64,
        state = summary_seed,
        args = ()
    );
    let summary = call!(finish_check, summary, error_count, first_error)?;

    let base = call!(begin_decode_edge, summary);
    let edge = call_recur!(
        tile = copy_selected_token_ids,
        input = selections,
        chunk = 64,
        output = base,
        args = ()
    );
    raster::println!("sequence check → {:?}", edge);
    Ok(edge)
}
