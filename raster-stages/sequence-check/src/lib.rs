//! `sequence-check`: tiles.
//!
//! One stage per range of claimed tokens.
//!
//! ```text
//!   predicting = [previous range's last row] ‖ this range's rows
//!   for each claimed token k in this range:
//!     row      = predicting[k]                     (teacher-forced: row i sees ids[..=i])
//!     normed   = rms_norm(row, w_final)
//!     selected = argmax_t softcap(normed · projection[t])   (first maximum wins)
//!   DecodeEdge {
//!     last selection,
//!     claim_mismatches    = earlier ranges' + #{k : selected_k ≠ claimed_k},
//!     generated_token_ids = earlier ranges' ‖ every selection,
//!   }
//! ```
//!
//! The scores and the tie-break are `prefill-finalize` followed by
//! `decode-select-token`, fused so the vocabulary-wide logit list is never
//! built: each head page folds straight into a running argmax, in page order,
//! so a strictly greater score replaces the incumbent exactly as it would
//! scanning the materialized list.
//!
//! The stage re-derives the greedy continuation rather than asserting the
//! claimed one, and commits the comparison as the edge's claim verdict. An
//! honest claim's output is byte-identical to the autoregressive chain's final
//! decode edge. A claim whose tokens are not the greedy continuation, run
//! honestly, publishes a non-zero verdict; one that publishes a clean verdict
//! anyway has doctored this stage, and a challenge replays it here.
//!
//! `#![no_std]` so the same tiles compile into RISC0 replay guests.

#![no_std]

extern crate alloc;

use alloc::string::String;
use raster::prelude::*;

pub mod input;

use input::*;

/// Guards the head external before any of it is used.
#[tile(kind = iter, description = "Validate the output head's declared shapes")]
pub fn validate_final_head(params: FinalHeadParams) -> Result<FinalHeadParams> {
    if params.hidden_size == 0 {
        return Err(String::from("output head requires a non-zero hidden_size"));
    }
    if params.norm_eps < 0 {
        return Err(String::from("rms-norm epsilon must be non-negative"));
    }
    if params.softcap < 0 {
        return Err(String::from(
            "final logit softcap must be non-negative (0 disables it)",
        ));
    }
    let expected = params.hidden_size as usize * 4;
    if params.norm_weights.len() != expected {
        return Err(alloc::format!(
            "final norm weights have {} bytes, expected {expected}",
            params.norm_weights.len()
        ));
    }
    Ok(params)
}

/// The previous range's row-count fold: nothing counted.
#[tile(kind = iter, description = "Open the previous range's row count")]
pub fn begin_token_count() -> TokenCount {
    TokenCount { count: 0 }
}

/// Counts one chunk of the previous range's rows.
#[tile(kind = recur, description = "Count a chunk of the previous range's rows")]
pub fn count_row_chunk(
    input: RecurInput<Block<ActivationRow>>,
    state: RecurState<TokenCount>,
) -> RecurState<TokenCount> {
    let mut state = state;
    state.count += input.into_value().len() as u32;
    state
}

/// Index of the previous range's last row, which predicts this range's first
/// claimed token.
#[tile(kind = iter, description = "Index of the previous range's last row")]
pub fn last_row_index(previous_rows: TokenCount) -> Result<u32> {
    previous_rows.count.checked_sub(1).ok_or_else(|| {
        String::from("teacher-forced check requires a non-empty range before the claimed tokens")
    })
}

/// Opens the predicting rows with the previous range's last row.
#[tile(kind = iter, description = "Open the predicting rows at the range boundary")]
pub fn begin_predicting_rows(boundary: ActivationRow) -> Draft<PredictingRows> {
    let mut output = Draft::<PredictingRows>::new();
    output.rows().push(boundary);
    output
}

/// Appends a chunk of this range's rows after the boundary row.
#[tile(kind = recur, description = "Append a chunk of this range's rows to the predicting rows")]
pub fn append_predicting_rows(
    input: RecurInput<Block<ActivationRow>>,
    output: RecurOutput<PredictingRows>,
) -> RecurOutput<PredictingRows> {
    let mut output = output;
    for row in input.into_value() {
        output.rows().push(row);
    }
    output
}

/// Points the cursor at the boundary row, which predicts the token at this
/// range's start position.
#[tile(kind = iter, description = "Open the check cursor at the range boundary")]
pub fn begin_check_cursor(start_position: u32) -> CheckCursor {
    CheckCursor {
        row_index: 0,
        start_position,
    }
}

/// Pairs the cursor's row with the claimed token it is meant to predict.
#[tile(kind = iter, description = "Begin checking one generated position")]
pub fn begin_check_step(cursor: CheckCursor, claimed_token_id: u32) -> CheckStep {
    CheckStep {
        row_index: cursor.row_index,
        decode_position: cursor.start_position + cursor.row_index,
        claimed_token_id,
        start_position: cursor.start_position,
    }
}

/// Moves the cursor to the row that predicts the next generated token.
#[tile(kind = iter, description = "Advance the check cursor")]
pub fn advance_check_cursor(step: CheckStep) -> CheckCursor {
    CheckCursor {
        row_index: step.row_index + 1,
        start_position: step.start_position,
    }
}

/// Applies the final RMS norm to the predicting row.
///
/// A failure is recorded rather than raised: a recur sequence cannot
/// propagate an error, so it travels to the post-sweep assertion.
#[tile(kind = iter, description = "Normalise one predicting row")]
pub fn normalize_position(row: ActivationRow, params: FinalHeadParams) -> NormalizedPosition {
    let failed = |error: String| NormalizedPosition {
        values: pack_i32s(&[]),
        error,
    };
    let mut values = match unpack_i32s(&row.values) {
        Ok(values) => values,
        Err(error) => return failed(error),
    };
    if values.len() != params.hidden_size as usize {
        return failed(alloc::format!(
            "predicting row has {} values, expected hidden_size {}",
            values.len(),
            params.hidden_size
        ));
    }
    let weights = match unpack_i32s(&params.norm_weights) {
        Ok(weights) => weights,
        Err(error) => return failed(error),
    };
    if let Err(error) = rms_norm(&mut values, &weights, params.norm_eps) {
        return failed(error);
    }
    NormalizedPosition {
        values: pack_i32s(&values),
        error: String::new(),
    }
}

/// The argmax fold's opening state: no value seen yet.
#[tile(kind = iter, description = "Open the argmax with no value seen")]
pub fn begin_argmax() -> ArgmaxState {
    ArgmaxState {
        has_value: false,
        token_id: 0,
        value: 0,
        error_count: 0,
        first_error: String::new(),
    }
}

fn record_argmax_error(state: &mut ArgmaxState, error: String) {
    if state.error_count == 0 {
        state.first_error = error;
    }
    state.error_count += 1;
}

/// Scores one page of the vocabulary and folds it into the running argmax.
///
/// Rows are visited in token-id order and only a strictly greater score
/// replaces the incumbent, so the first maximum keeps the slot — the
/// `decode-select-token` tie-break over the same scores.
#[tile(
    kind = recur,
    description = "Score one page of the vocabulary into the argmax",
    estimated_cycles = 800_000
)]
pub fn score_head_page(
    input: RecurInput<BytesPage>,
    state: RecurState<ArgmaxState>,
    normalized: NormalizedPosition,
    params: FinalHeadParams,
) -> RecurState<ArgmaxState> {
    let mut state = state;
    if !normalized.error.is_empty() {
        record_argmax_error(&mut state, normalized.error);
        return state;
    }
    let position = match unpack_i32s(&normalized.values) {
        Ok(values) => values,
        Err(error) => {
            record_argmax_error(&mut state, error);
            return state;
        }
    };
    let hidden = params.hidden_size as usize;
    let page = input.into_value();
    let stride = match det_num::q8::row_bytes(hidden) {
        Some(stride) if page.len() % stride == 0 => stride,
        _ => {
            record_argmax_error(
                &mut state,
                alloc::format!(
                    "projection page at offset {} is not a whole number of hidden-width Q8_0 rows",
                    page.offset()
                ),
            );
            return state;
        }
    };
    if page.offset() % stride as u64 != 0 {
        record_argmax_error(
            &mut state,
            String::from("projection page offset is not row-aligned"),
        );
        return state;
    }
    let first_id = (page.offset() / stride as u64) as u32;
    for (local, row) in page.as_slice().chunks_exact(stride).enumerate() {
        let token_id = first_id + local as u32;
        match det_num::q8::dot_row(&position, row) {
            Ok(score) => {
                let value = softcap(score, params.softcap);
                if !state.has_value || value > state.value {
                    state.has_value = true;
                    state.token_id = token_id;
                    state.value = value;
                }
            }
            Err(error) => {
                record_argmax_error(&mut state, alloc::format!("token {token_id}: {error}"))
            }
        }
    }
    state
}

/// Records the selection at one generated position, or why there is none.
#[tile(kind = iter, description = "Record one position's greedy selection")]
pub fn record_selection(
    output: Draft<CheckedPositions>,
    step: CheckStep,
    best: ArgmaxState,
) -> Draft<CheckedPositions> {
    let mut output = output;
    if best.error_count > 0 {
        output.errors().push(alloc::format!(
            "position {}: {} vocabulary row(s) failed to score; first: {}",
            step.decode_position,
            best.error_count,
            best.first_error
        ));
    } else if !best.has_value {
        output.errors().push(alloc::format!(
            "position {}: the head scored no token",
            step.decode_position
        ));
    } else {
        output.selections().push(PositionSelection {
            decode_position: step.decode_position,
            token_id: best.token_id,
            value: best.value,
            claimed_token_id: step.claimed_token_id,
        });
    }
    output
}

/// The selection summary's opening state: nothing selected.
#[tile(kind = iter, description = "Open the selection summary")]
pub fn begin_selection_summary() -> SelectionSummary {
    SelectionSummary::default()
}

/// Folds a chunk of selections: keeps the last, counts claim disagreements.
#[tile(kind = recur, description = "Summarise a chunk of selections")]
pub fn summarise_selections(
    input: RecurInput<Block<PositionSelection>>,
    state: RecurState<SelectionSummary>,
) -> RecurState<SelectionSummary> {
    let mut state = state;
    for selection in input.into_value() {
        if selection.token_id != selection.claimed_token_id {
            if state.mismatches == 0 {
                state.first_mismatch_position = selection.decode_position;
            }
            state.mismatches += 1;
        }
        state.count += 1;
        state.has_selected = true;
        state.decode_position = selection.decode_position;
        state.token_id = selection.token_id;
        state.value = selection.value;
    }
    state
}

/// The failure fold's opening state: nothing counted.
#[tile(kind = iter, description = "Open the failure summary with nothing counted")]
pub fn begin_error_summary() -> ErrorSummary {
    ErrorSummary {
        count: 0,
        first: String::new(),
    }
}

/// Folds the recorded failures into a count plus the first message.
#[tile(kind = recur, description = "Summarise the positions this check could not score")]
pub fn summarise_errors(
    input: RecurInput<String>,
    state: RecurState<ErrorSummary>,
) -> RecurState<ErrorSummary> {
    let mut state = state;
    let message = input.into_value();
    if state.count == 0 {
        state.first = message;
    }
    state.count += 1;
    state
}

/// Fails the program when a position went unscored or nothing was claimed.
///
/// A disagreement with the claim is not a failure: it is published in the
/// output's claim verdict, which is what makes it disputable. An aborting
/// stage has no output to hash or replay.
#[tile(kind = iter, description = "Reject an incomplete teacher-forced check")]
pub fn finish_check(
    summary: SelectionSummary,
    error_count: u32,
    first_error: String,
) -> Result<SelectionSummary> {
    if error_count != 0 {
        return Err(alloc::format!(
            "{error_count} generated position(s) failed to score; first: {first_error}"
        ));
    }
    if !summary.has_selected {
        return Err(String::from(
            "teacher-forced check requires at least one claimed output token",
        ));
    }
    if summary.mismatches != 0 {
        raster::println!(
            "sequence check: {} of {} claimed token(s) are not the greedy selection; first at position {}",
            summary.mismatches,
            summary.count,
            summary.first_mismatch_position
        );
    }
    Ok(summary)
}

/// Opens the output edge with the last selection's scalars and the claim
/// verdict: the earlier ranges' disagreements plus this range's, keeping the
/// earliest position.
///
/// The transcript list stays empty here: the copy sites below derive from
/// this object and only push — the earlier ranges' ids, then this range's.
#[tile(kind = iter, description = "Open the checked decode edge")]
pub fn begin_decode_edge(
    summary: SelectionSummary,
    prior_mismatches: u32,
    prior_first_mismatch_position: u32,
) -> Draft<DecodeEdge> {
    let mut output = Draft::<DecodeEdge>::new();
    output.has_selected().set(summary.has_selected);
    output.decode_position().set(summary.decode_position);
    output.token_id().set(summary.token_id);
    output.value().set(summary.value);
    // Saturating: any non-zero count rejects the claim, and a saturated count
    // stays non-zero.
    output
        .claim_mismatches()
        .set(prior_mismatches.saturating_add(summary.mismatches));
    output
        .first_claim_mismatch_position()
        .set(if prior_mismatches > 0 {
            prior_first_mismatch_position
        } else {
            summary.first_mismatch_position
        });
    output
}

/// Copies a bounded chunk of the earlier ranges' transcript into the edge.
#[tile(kind = recur, description = "Copy earlier ranges' selected token ids")]
pub fn copy_prior_token_ids(
    input: RecurInput<Block<u32>>,
    output: RecurOutput<DecodeEdge>,
) -> RecurOutput<DecodeEdge> {
    let mut output = output;
    for token_id in input.into_value() {
        output.generated_token_ids().push(token_id);
    }
    output
}

/// Appends a bounded chunk of this range's selected token ids to the edge.
#[tile(kind = recur, description = "Copy selected token ids into the edge")]
pub fn copy_selected_token_ids(
    input: RecurInput<Block<PositionSelection>>,
    output: RecurOutput<DecodeEdge>,
) -> RecurOutput<DecodeEdge> {
    let mut output = output;
    for selection in input.into_value() {
        output.generated_token_ids().push(selection.token_id);
    }
    output
}
