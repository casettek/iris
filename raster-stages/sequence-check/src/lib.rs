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
//!   transcript = earlier ranges' ids ‖ every selection
//!   record     = transcript decoded and committed, with what it breaks
//!   DecodeEdge {
//!     last selection,
//!     claim_mismatches    = earlier ranges' + #{k : selected_k ≠ claimed_k},
//!     check_fault         = the earliest range's fault, else this range's,
//!     record,
//!     claim               = statement digest, text hash, one verdict over all of the above,
//!     generated_token_ids = transcript,
//!   }
//! ```
//!
//! The scores and the tie-break are direct generation's logits and greedy
//! selection, fused so the vocabulary-wide logit list is never built: each head
//! page folds straight into a running argmax, in page order, so a strictly
//! greater score replaces the incumbent exactly as it would scanning the
//! materialized list.
//!
//! The stage re-derives the greedy continuation rather than asserting the
//! claimed one, and commits the comparison as the edge's claim verdict. An
//! honest claim's output is byte-identical to direct generation's final edge. A
//! claim whose tokens are not the greedy continuation, run
//! honestly, publishes a non-zero verdict; one that publishes a clean verdict
//! anyway has doctored this stage, and a challenge replays it here.
//!
//! A check that cannot stand behind the edge finishes all the same and
//! publishes `check_fault`: a position with no predicting row, a range with no
//! tokens or too many, or a range that does not continue the sequence. The
//! range limit and the prompt's token count come from the prompt context.
//!
//! `#![no_std]` so the same tiles compile into RISC0 replay guests.

#![no_std]

extern crate alloc;

use alloc::string::String;
use raster::prelude::*;

pub mod finalize;
pub mod finalize_model;
pub mod input;
pub mod prompt;

pub use finalize::*;
pub use finalize_model::*;

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

/// The previous range's tail fold: no row seen yet.
#[tile(kind = iter, description = "Open the previous range's tail")]
pub fn begin_previous_tail() -> PreviousTail {
    PreviousTail { last: empty_row() }
}

/// Keeps the last row of one chunk of the previous range.
#[tile(kind = recur, description = "Keep the last row of a chunk of the previous range")]
pub fn keep_last_row(
    input: RecurInput<Block<ActivationRow>>,
    state: RecurState<PreviousTail>,
) -> RecurState<PreviousTail> {
    let mut state = state;
    if let Some(row) = input.into_value().into_iter().last() {
        state.last = row;
    }
    state
}

/// Opens the predicting rows with the previous range's last row, which
/// predicts this range's first claimed token.
#[tile(kind = iter, description = "Open the predicting rows at the range boundary")]
pub fn begin_predicting_rows(tail: PreviousTail) -> Draft<PredictingRows> {
    let mut output = Draft::<PredictingRows>::new();
    output.rows().push(tail.last);
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
/// replaces the incumbent, so the first maximum keeps the slot.
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
        output.errors().push(PositionError {
            decode_position: step.decode_position,
            message: alloc::format!(
                "{} vocabulary row(s) failed to score; first: {}",
                best.error_count,
                best.first_error
            ),
        });
    } else if !best.has_value {
        output.errors().push(PositionError {
            decode_position: step.decode_position,
            message: String::from("the head scored no token"),
        });
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
        observe_selection(&mut state, &selection);
    }
    state
}

/// Folds one selection into the summary. Called from [`summarise_selections`]
/// and from the native kernel, so both publish the same scalars.
pub fn observe_selection(state: &mut SelectionSummary, selection: &PositionSelection) {
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

/// The failure fold's opening state: nothing counted.
#[tile(kind = iter, description = "Open the failure summary with nothing counted")]
pub fn begin_error_summary() -> ErrorSummary {
    ErrorSummary::default()
}

/// Folds the recorded failures into a count plus the first message.
#[tile(kind = recur, description = "Summarise the positions this check could not score")]
pub fn summarise_errors(
    input: RecurInput<PositionError>,
    state: RecurState<ErrorSummary>,
) -> RecurState<ErrorSummary> {
    let mut state = state;
    let error = input.into_value();
    if state.count == 0 {
        state.first_position = error.decode_position;
        state.first = error.message;
    }
    state.count += 1;
    state
}

/// Gathers the scalars this check's fault is established from.
#[tile(kind = iter, description = "Gather what this check's fault depends on")]
pub fn begin_check_frame(
    start_position: u32,
    prior_has_selected: bool,
    prior_decode_position: u32,
    prior_fault: u32,
    prior_fault_position: u32,
    range_size: u32,
    prompt_tokens: u32,
) -> CheckFrame {
    CheckFrame {
        start_position,
        range_size,
        prompt_tokens,
        prior_has_selected,
        prior_decode_position,
        prior_fault,
        prior_fault_position,
    }
}

/// Establishes what this check publishes as its fault.
///
/// Neither a fault nor a disagreement with the claim stops the stage: an
/// aborting stage has no output to hash or replay, so both are published.
#[tile(kind = iter, description = "Establish the fault this check publishes")]
pub fn find_check_fault(
    summary: SelectionSummary,
    errors: ErrorSummary,
    frame: CheckFrame,
) -> CheckFault {
    if summary.mismatches != 0 {
        raster::println!(
            "sequence check: {} of {} claimed token(s) are not the greedy selection; first at position {}",
            summary.mismatches,
            summary.count,
            summary.first_mismatch_position
        );
    }
    if errors.count != 0 {
        raster::println!(
            "sequence check: {} generated position(s) could not be scored; first at position {}: {}",
            errors.count,
            errors.first_position,
            errors.first
        );
    }
    check_fault_of(&summary, &errors, &frame)
}

/// The fault [`find_check_fault`] publishes. An earlier range's fault stands;
/// otherwise this range's first, in the order the range is checked. Every
/// position is either selected or recorded as an error, so together they count
/// the claimed tokens.
pub fn check_fault_of(
    summary: &SelectionSummary,
    errors: &ErrorSummary,
    frame: &CheckFrame,
) -> CheckFault {
    let at_start = |fault| CheckFault {
        fault,
        position: frame.start_position,
    };
    if frame.prior_fault != CHECK_FAULT_NONE {
        return CheckFault {
            fault: frame.prior_fault,
            position: frame.prior_fault_position,
        };
    }
    let claimed = summary.count.saturating_add(errors.count);
    if claimed == 0 {
        return at_start(CHECK_FAULT_NO_CLAIMED_TOKENS);
    }
    if claimed > frame.range_size {
        return at_start(CHECK_FAULT_RANGE_TOO_LARGE);
    }
    let expected_start = if frame.prior_has_selected {
        frame.prior_decode_position.saturating_add(1)
    } else {
        frame.prompt_tokens
    };
    if frame.start_position != expected_start {
        return at_start(CHECK_FAULT_DISCONTINUOUS);
    }
    if errors.count != 0 {
        return CheckFault {
            fault: CHECK_FAULT_UNSCORED,
            position: errors.first_position,
        };
    }
    CheckFault::default()
}

/// Opens the output edge with the last selection's scalars and the claim
/// verdict: the earlier ranges' disagreements plus this range's, keeping the
/// earliest position. It states the claim record here, where the record and
/// the accumulated verdict are both in hand.
///
/// The transcript list stays empty here: the copy site below derives from
/// this object and only pushes the transcript's ids.
#[tile(kind = iter, description = "Open the checked decode edge")]
pub fn begin_decode_edge(
    summary: SelectionSummary,
    prior_mismatches: u32,
    prior_first_mismatch_position: u32,
    fault: CheckFault,
    model: ModelContext,
    record: OutputRecord,
) -> Draft<DecodeEdge> {
    // Saturating: any non-zero count rejects the claim, and a saturated count
    // stays non-zero.
    let claim_mismatches = prior_mismatches.saturating_add(summary.mismatches);
    let mut output = Draft::<DecodeEdge>::new();
    output
        .claim()
        .set(claim_record(&record, claim_mismatches, fault.fault));
    output.model().set(model);
    output.record().set(record);
    output.has_selected().set(summary.has_selected);
    output.decode_position().set(summary.decode_position);
    output.token_id().set(summary.token_id);
    output.value().set(summary.value);
    output.claim_mismatches().set(claim_mismatches);
    output
        .first_claim_mismatch_position()
        .set(if prior_mismatches > 0 {
            prior_first_mismatch_position
        } else {
            summary.first_mismatch_position
        });
    output.check_fault().set(fault.fault);
    output.check_fault_position().set(fault.position);
    output
}

/// Copies a bounded chunk of the earlier ranges' ids into the transcript.
#[tile(kind = recur, description = "Copy earlier ranges' selected token ids")]
pub fn copy_prior_token_ids(
    input: RecurInput<Block<u32>>,
    output: RecurOutput<Transcript>,
) -> RecurOutput<Transcript> {
    let mut output = output;
    for token_id in input.into_value() {
        output.ids().push(token_id);
    }
    output
}

/// Appends a bounded chunk of this range's selected token ids to the
/// transcript.
#[tile(kind = recur, description = "Copy selected token ids into the transcript")]
pub fn copy_selected_token_ids(
    input: RecurInput<Block<PositionSelection>>,
    output: RecurOutput<Transcript>,
) -> RecurOutput<Transcript> {
    let mut output = output;
    for selection in input.into_value() {
        output.ids().push(selection.token_id);
    }
    output
}

/// Copies a bounded chunk of the transcript into the edge.
#[tile(kind = recur, description = "Copy the transcript into the edge")]
pub fn copy_transcript(
    input: RecurInput<Block<u32>>,
    output: RecurOutput<DecodeEdge>,
) -> RecurOutput<DecodeEdge> {
    let mut output = output;
    for token_id in input.into_value() {
        output.generated_token_ids().push(token_id);
    }
    output
}

#[tile(kind = iter, description = "Check completed model lineage before head scoring")]
pub fn head_model_context(
    model: ModelContext,
    previous: ModelContext,
    prompt: ModelContext,
    prior: ModelContext,
) -> ModelContext {
    model.head(&previous, &prompt, &prior)
}

pub mod identity;
pub use identity::*;

/// The claim is the token that actually entered the transformer pipeline.
#[tile(kind = iter, description = "Read the consumed token from its activation row")]
pub fn checked_row_token(row: ActivationRow) -> u32 {
    row.token_id
}

#[tile(kind=recur)]
pub fn binding_count_rows(
    input: RecurInput<Block<ActivationRow>>,
    state: RecurState<u64>,
) -> RecurState<u64> {
    RecurState::new(*state + input.into_value().len() as u64)
}
#[tile(kind=iter)]
pub fn binding_validate_rows(mut model: ModelContext, start: u32, count: u64) -> ModelContext {
    model.validate_rows(start, count);
    model
}
