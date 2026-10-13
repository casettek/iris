//! Phase 7 — `decode-embed`: tiles.
//!
//! Gathers the embedding row for **one** generated token. Same table, same
//! addressing and the same canonical scale as `input-embedding` — the only
//! difference is that there is one token rather than a prompt, so the sequence
//! is straight-line instead of a recur, and the sequence starts partway in
//! rather than at zero.
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

/// Refuses to embed the empty edge used for a zero-token generation.
#[tile(kind = iter, description = "Require a selected decode token")]
pub fn require_selected_token(has_selected: bool) -> Result<bool> {
    if has_selected {
        Ok(has_selected)
    } else {
        Err(String::from(
            "decode-embed received the empty decode edge with no selected token",
        ))
    }
}

/// `token_id * hidden_size * 4` — the byte offset of that row in the table.
#[tile(kind = iter, description = "Byte offset of an embedding row")]
pub fn embedding_byte_offset(token_id: u32, hidden_size: u32) -> u64 {
    token_id as u64 * hidden_size as u64 * 4
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

/// Stamps this decode step's absolute position onto the activation sequence.
///
/// The mirror of `input-embedding`'s `begin_prompt_activations`, which sets
/// zero. Here it is `selected.decode_position`, which `prefill-finalize`
/// already computed as `start_position + token_count` — the position after
/// everything scored so far, which is exactly where the token it selected
/// belongs. Every layer downstream reads this to place its token for RoPE and
/// for window visibility, so getting it wrong is a wrong answer rather than an
/// error.
#[tile(kind = iter, description = "Open the decode step's activation sequence at its position")]
pub fn begin_decode_activations(
    output: Draft<ActivationSequence>,
    decode_position: u32,
) -> Draft<ActivationSequence> {
    let mut output = output;
    output.start_position().set(decode_position);
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
    match unpack_i32s_at(&page, byte_off, hidden_size) {
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
