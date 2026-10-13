//! Finalizes the generated token transcript into IDs, a reference-compatible
//! commitment, and decoded text.

#![no_std]

extern crate alloc;

use alloc::string::{String, ToString};
use alloc::vec::Vec;
use raster::prelude::*;
use sha2::{Digest, Sha256};

pub mod input;

use input::*;

#[tile(kind = iter, description = "Initialize generated output finalization")]
pub fn begin_finalize_state() -> FinalizeState {
    initial_finalize_state()
}

/// The empty state, as a plain function — see [`advance_finalize_state`].
pub fn initial_finalize_state() -> FinalizeState {
    FinalizeState {
        count: 0,
        json: String::from("["),
        text: String::new(),
        pending_bytes: BytesPage::__from_parts(0, 0, Vec::new()),
        stopped: false,
    }
}

fn hex_nibble(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

fn byte_fallback(token: &str) -> Option<u8> {
    let bytes = token.as_bytes();
    if bytes.len() != 6 || &bytes[..3] != b"<0x" || bytes[5] != b'>' {
        return None;
    }
    Some(hex_nibble(bytes[3])? * 16 + hex_nibble(bytes[4])?)
}

fn flush_pending(state: &mut FinalizeState) {
    if state.pending_bytes.as_slice().is_empty() {
        return;
    }
    state
        .text
        .push_str(&String::from_utf8_lossy(state.pending_bytes.as_slice()));
    state.pending_bytes = BytesPage::__from_parts(0, 0, Vec::new());
}

/// Adds one token to the JSON commitment stream and decoder state.
#[tile(kind = iter, description = "Consume one generated token")]
pub fn consume_decoder_token(
    state: FinalizeState,
    token_id: u32,
    token: DecoderToken,
) -> FinalizeState {
    advance_finalize_state(state, token_id, token)
}

/// The step [`consume_decoder_token`] is, as a plain function.
///
/// Called from the tile body, so it is inside that tile's image id and replays
/// in the guest with it. Split out because a tile can only be entered from a
/// sequence, and the terminal-token branch is worth pinning with a test rather
/// than with a chain run long enough for a real model to close its turn.
pub fn advance_finalize_state(
    state: FinalizeState,
    token_id: u32,
    token: DecoderToken,
) -> FinalizeState {
    let mut state = state;
    if state.count > 0 {
        state.json.push(',');
    }
    state.json.push_str(&token_id.to_string());
    state.count = state.count.saturating_add(1);

    // Past the end of the answer. The repeat count in the chain manifest is a
    // static unroll, so decoding runs to it whatever the model emitted; those
    // ids stay in the transcript and its commitment, but the decoded text ends
    // where the model ended.
    if state.stopped {
        return state;
    }
    if token.terminal {
        flush_pending(&mut state);
        state.stopped = true;
        return state;
    }
    if token.special {
        flush_pending(&mut state);
        return state;
    }
    if let Some(byte) = byte_fallback(&token.token) {
        let mut pending = state.pending_bytes.as_slice().to_vec();
        pending.push(byte);
        state.pending_bytes = BytesPage::__from_parts(0, 0, pending);
        return state;
    }

    flush_pending(&mut state);
    state.text.push_str(&token.token.replace('\u{2581}', " "));
    state
}

/// Flushes byte fallback and hashes exactly the JSON token-id representation
/// used by the reference's `build_output_decode_commitment`.
#[tile(kind = iter, description = "Finish generated text and token commitment")]
pub fn finish_finalize_state(state: FinalizeState) -> FinalizeState {
    let mut state = state;
    flush_pending(&mut state);
    state.json.push(']');
    state
}

#[tile(kind = iter, description = "Validate decode edge transcript state")]
pub fn validate_decode_edge(has_selected: bool, count: u32) -> Result<u32> {
    if has_selected != (count > 0) {
        return Err(alloc::format!(
            "decode edge selection flag {has_selected} disagrees with generated token count {count}"
        ));
    }
    Ok(count)
}

#[tile(kind = iter, description = "Open final generated output")]
pub fn begin_generated_output(
    output: Draft<GeneratedOutput>,
    state: FinalizeState,
) -> Draft<GeneratedOutput> {
    let mut output = output;
    let digest = Sha256::digest(state.json.as_bytes());
    output.generated_token_count().set(state.count);
    output
        .generated_token_ids_sha256()
        .set(alloc::format!("{digest:x}"));
    output.generated_text().set(state.text);
    output.stop_reason().set(if state.stopped {
        String::from("eos")
    } else {
        String::from("max_new_tokens")
    });
    output
}

#[tile(kind = recur, description = "Copy generated token ids to final output")]
pub fn copy_output_token_ids(
    input: RecurInput<Block<u32>>,
    output: RecurOutput<GeneratedOutput>,
) -> RecurOutput<GeneratedOutput> {
    let mut output = output;
    for token_id in input.into_value() {
        output.generated_token_ids().push(token_id);
    }
    output
}
