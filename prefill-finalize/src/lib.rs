//! Phase 5 — `prefill-finalize`: tiles.
//!
//! ```text
//!   position = last row of the prompt's activations
//!   normed   = rms_norm(position, w_final)
//!   logit[t] = softcap(normed · projection[t])
//! ```
//!
//! The projection is a paged byte region; each replay unit scores one page of
//! vocabulary rows. `token_id` is `page.offset() / (hidden * 4) + local_row`.
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

/// Empty last-position state, for the sequence to bind before the fold.
#[tile(kind = iter, description = "Empty final-position accumulator")]
pub fn empty_final_position() -> FinalPosition {
    FinalPosition {
        found: false,
        token_id: 0,
        token_count: 0,
        values: pack_i32s(&[]),
    }
}

/// Walks the prompt to its last position, counting as it goes.
#[tile(kind = recur, description = "Track the prompt's final position")]
pub fn track_final_position(
    input: RecurInput<ActivationRow>,
    state: RecurState<FinalPosition>,
) -> RecurState<FinalPosition> {
    let mut state = state;
    let row = input.into_value();
    state.found = true;
    state.token_id = row.token_id;
    state.token_count += 1;
    state.values = row.values;
    state
}

/// Applies the final RMS norm to the position every logit is scored against.
#[tile(kind = iter, description = "Normalise the final position")]
pub fn normalize_final_position(
    position: FinalPosition,
    params: FinalHeadParams,
) -> Result<NormalizedPosition> {
    if !position.found {
        return Err(String::from(
            "prefill produced no activation rows; there is no final position to score",
        ));
    }
    let mut values = unpack_i32s(&position.values)?;
    if values.len() != params.hidden_size as usize {
        return Err(alloc::format!(
            "final position has {} values, expected hidden_size {}",
            values.len(),
            params.hidden_size
        ));
    }
    rms_norm(
        &mut values,
        &unpack_i32s(&params.norm_weights)?,
        params.norm_eps,
    )?;
    Ok(NormalizedPosition {
        values: pack_i32s(&values),
    })
}

/// Seeds the output draft with the position decoding resumes from.
#[tile(kind = iter, description = "Open the logits draft at the decode position")]
pub fn begin_logits(token_count: u32, start_position: u32) -> Draft<PrefillLogits> {
    let mut output = Draft::<PrefillLogits>::new();
    // Absolute, not a count. `token_count` alone is the position of the next
    // token only when the stage started at zero; a decode stage sees one row
    // and would otherwise report position 1 no matter how far in it is.
    output
        .decode_position()
        .set(start_position + token_count);
    output
}

/// Scores one page of the vocabulary: a dot product per row, then softcap.
#[tile(
    kind = recur,
    description = "Score one page of the vocabulary",
    estimated_cycles = 800_000
)]
pub fn project_logit_page(
    input: RecurInput<BytesPage>,
    output: RecurOutput<PrefillLogits>,
    normalized: NormalizedPosition,
    params: FinalHeadParams,
) -> RecurOutput<PrefillLogits> {
    let mut output = output;
    let position = match unpack_i32s(&normalized.values) {
        Ok(values) => values,
        Err(error) => {
            output.errors().push(error);
            return output;
        }
    };
    let hidden = params.hidden_size as usize;
    if position.len() != hidden {
        output.errors().push(alloc::format!(
            "normalised position has {} values, expected {hidden}",
            position.len()
        ));
        return output;
    }

    let page = input.into_value();
    let weights = match unpack_i32s(&page) {
        Ok(values) => values,
        Err(error) => {
            output.errors().push(error);
            return output;
        }
    };
    if hidden == 0 || weights.len() % hidden != 0 {
        output.errors().push(alloc::format!(
            "projection page at offset {} is not a whole number of hidden-width rows",
            page.offset()
        ));
        return output;
    }
    let stride = hidden as u64 * 4;
    if page.offset() % stride != 0 {
        output.errors().push(String::from(
            "projection page offset is not row-aligned",
        ));
        return output;
    }
    let first_id = (page.offset() / stride) as u32;
    for (local, row) in weights.chunks_exact(hidden).enumerate() {
        let token_id = first_id + local as u32;
        match dot(&position, row) {
            Ok(value) => output.logits().push(LogitEntry {
                token_id,
                value: softcap(value, params.softcap),
            }),
            Err(error) => output
                .errors()
                .push(alloc::format!("token {token_id}: {error}")),
        }
    }
    output
}

/// The failure fold's opening state: nothing counted.
///
/// A stored seed, so the fold's state chain starts at a committed value
/// rather than a literal nothing pins.
#[tile(kind = iter, description = "Open the failure summary with nothing counted")]
pub fn begin_error_summary() -> ErrorSummary {
    ErrorSummary {
        count: 0,
        first: String::new(),
    }
}

/// Folds the recorded failures into a count plus the first message.
#[tile(kind = recur, description = "Summarise the logits this head could not score")]
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

/// Fails the program when any vocabulary row went unscored.
#[tile(kind = iter, description = "Reject an incomplete logit vector")]
pub fn assert_logits_complete(error_count: u32, first_error: String) -> Result<u32> {
    if error_count == 0 {
        Ok(error_count)
    } else {
        Err(alloc::format!(
            "{error_count} vocabulary row(s) failed to score; first: {first_error}"
        ))
    }
}
