//! Phase 2 — `input-embedding`: tiles.
//!
//! Gathers one embedding row per prompt token id. The table is a paged byte
//! region of `Q8_0` rows, addressed by `token_id · hidden_size / 32 · 34` and
//! dequantized on the way out; an out-of-range id aborts (a list has no
//! non-membership proof).
//!
//! `#![no_std]` so the same tiles compile into RISC0 replay guests.

#![no_std]

extern crate alloc;

use alloc::string::String;
use det_num::ops::mul_sat;
use det_num::Act;
use raster::prelude::*;

pub mod input;

use input::*;

/// Materializes one prompt token id so later steps can each read it.
#[tile(kind = iter, description = "Turn a prompt token id into a reusable binding")]
pub fn begin_row_lookup(token_id: u32) -> u32 {
    token_id
}

/// `token_id · hidden_size / 32 · 34` — the byte offset of that `Q8_0` row in
/// the table.
#[tile(kind = iter, description = "Byte offset of an embedding row")]
pub fn embedding_byte_offset(token_id: u32, hidden_size: u32) -> u64 {
    token_id as u64 * (hidden_size as u64 / 32 * 34)
}

/// Page index containing a byte offset.
#[tile(kind = iter, description = "Page containing a byte offset", estimated_cycles = 200)]
pub fn page_of(byte_offset: u64, page_size: u64) -> u64 {
    if page_size == 0 {
        0
    } else {
        byte_offset / page_size
    }
}

/// Opens the row-count fold over the previous range's embedding.
#[tile(kind = iter, description = "Open the previous range's row count")]
pub fn begin_row_count() -> RowCount {
    RowCount { count: 0 }
}

/// Counts one chunk of the previous range's rows.
#[tile(kind = recur, description = "Count a chunk of the previous range's rows")]
pub fn count_row_chunk(
    input: RecurInput<Block<ActivationRow>>,
    state: RecurState<RowCount>,
) -> RecurState<RowCount> {
    let mut state = state;
    state.count += input.into_value().len() as u32;
    state
}

/// Where this range starts: right after the previous range's last row.
///
/// The previous range is `range-origin` (start 0, no rows) for the first
/// range, so positions are never chosen by run preparation: they follow from
/// how many ids the in-chain tokenizer produced for every earlier range.
#[tile(kind = iter, description = "Start position after the previous range")]
pub fn range_start(previous_start: u32, previous_rows: RowCount) -> Result<u32> {
    previous_start
        .checked_add(previous_rows.count)
        .ok_or_else(|| String::from("range start position overflows u32"))
}

/// Stamps this range's starting position onto its activation sequence.
///
/// Every consumer of an `ActivationSequence` reads `start_position` to place
/// its tokens: RoPE and sliding-window visibility are defined over absolute
/// positions. Set-once, so it happens here rather than inside the per-token
/// append.
#[tile(kind = iter, description = "Open this range's activation sequence at its start position")]
pub fn begin_prompt_activations(start_position: u32) -> Draft<ActivationSequence> {
    let mut output = Draft::<ActivationSequence>::new();
    output.start_position().set(start_position);
    output
}

/// Extracts the row at `byte_off` from its page and appends it, or records
/// a miss when the page does not contain a full row of `hidden_size`.
#[tile(kind = iter, description = "Append one gathered activation row")]
pub fn append_activation_row(
    output: Draft<ActivationSequence>,
    token_id: u32,
    page: BytesPage,
    byte_off: u64,
    hidden_size: u32,
    embedding_scale: i32,
) -> Draft<ActivationSequence> {
    let mut output = output;
    match q8_row_at(&page, byte_off, hidden_size) {
        // Gemma scales the embedding by sqrt(hidden) before layer 0. Applied
        // here, with the canonical multiply, because the row is only ever read
        // once — and because an unscaled activation is not detectably wrong
        // downstream, it just quietly produces a different token.
        Ok(values) => {
            let scale = Act::from_bits(embedding_scale);
            let scaled: alloc::vec::Vec<i32> = values
                .iter()
                .map(|bits| mul_sat(Act::from_bits(*bits), scale).to_bits())
                .collect();
            output.rows().push(ActivationRow {
                token_id,
                values: pack_i32s(&scaled),
            })
        }
        Err(_) => output.errors().push(alloc::format!(
            "token {token_id} has no embedding row of the declared width"
        )),
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
#[tile(kind = recur, description = "Summarise the rows this stage could not produce")]
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

/// Fails the program when any prompt token went unembedded.
#[tile(kind = iter, description = "Reject a run that could not embed every token")]
pub fn assert_all_tokens_embedded(error_count: u32, first_error: String) -> Result<u32> {
    if error_count == 0 {
        Ok(error_count)
    } else {
        Err(alloc::format!(
            "{error_count} prompt token(s) could not be embedded; first: {first_error}"
        ))
    }
}
