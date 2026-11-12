//! Finalization: the transcript so far as token ids, decoded text and the
//! commitments to both, with what it breaks and the context the chain ran
//! under. Every check publishes this record with its edge; the last range's
//! is the claim's.
//!
//! This was the `output-finalize` stage. It is three steps a token, far too
//! little to be a stage of its own, and the check before it has the edge it
//! reads.

use alloc::string::{String, ToString};
use alloc::vec::Vec;
use raster::prelude::*;
use sha2::{Digest, Sha256};

use crate::finalize_model::finish_final_history;
use crate::input::*;
use crate::prompt::PromptContext;

/// The empty state, as a plain function — see [`advance_finalize_state`].
pub fn initial_finalize_state() -> FinalizeState {
    FinalizeState {
        history: ModelContext::default(),
        expected_token: 0,
        last_token: 0,
        model_fault: 0,
        count: 0,
        json: String::from("["),
        text: String::new(),
        pending_bytes: BytesPage::__from_parts(0, 0, Vec::new()),
        stopped: false,
        past_terminal: false,
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

/// Adds one authenticated token to the JSON commitment stream and decoder state.
pub fn advance_finalize_state(
    state: FinalizeState,
    token_id: u32,
    token: DecoderToken,
) -> FinalizeState {
    let mut state = state;
    if !state.history.is_origin() {
        state.history.append_consumed_token(token_id);
    }
    state.last_token = token_id;
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
        state.past_terminal = true;
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

/// Flushes byte fallback and closes the JSON token-id representation that
/// the reference's `build_output_decode_commitment` hashes.
pub fn finished_state(state: FinalizeState) -> FinalizeState {
    let mut state = state;
    finish_final_history(&mut state);
    flush_pending(&mut state);
    state.json.push(']');
    state
}

/// What the finished transcript breaks, or [`OUTPUT_FAULT_NONE`].
pub fn output_fault_of(state: &FinalizeState, has_selected: bool, context: &PromptContext) -> u32 {
    if has_selected != (state.count > 0) {
        return OUTPUT_FAULT_EDGE_DISAGREES;
    }
    if state.count == 0 {
        return OUTPUT_FAULT_NO_TOKENS;
    }
    if state.count > context.max_new_tokens {
        return OUTPUT_FAULT_OVER_BUDGET;
    }
    if state.past_terminal {
        return OUTPUT_FAULT_PAST_EOS;
    }
    if state.count < context.max_new_tokens && !state.stopped {
        return OUTPUT_FAULT_SHORT_WITHOUT_EOS;
    }
    OUTPUT_FAULT_NONE
}

/// Lowercase hex SHA-256 of these exact bytes.
pub fn sha256_hex(bytes: &[u8]) -> String {
    alloc::format!("{:x}", Sha256::digest(bytes))
}

/// The record of a finished transcript: its commitments, what it breaks, and
/// the context the chain ran under.
pub fn output_record(
    state: FinalizeState,
    has_selected: bool,
    context: &PromptContext,
) -> OutputRecord {
    OutputRecord {
        model_fault: state.model_fault,
        output_fault: output_fault_of(&state, has_selected, context),
        output_text_sha256: sha256_hex(state.text.as_bytes()),
        generated_token_count: state.count,
        generated_token_ids_sha256: sha256_hex(state.json.as_bytes()),
        generated_text: state.text,
        stop_reason: if state.stopped {
            String::from("eos")
        } else {
            String::from("max_new_tokens")
        },
        input_text_sha256: context.input_text_sha256.clone(),
        model_id: context.model_id.clone(),
        raw_prompt: context.raw_prompt,
        max_new_tokens: context.max_new_tokens,
        range_size: context.range_size,
        prompt_tokens: context.prompt_tokens,
        prompt_complete: context.complete,
        prompt_fault: context.fault,
        prompt_fault_position: context.fault_position,
    }
}

/// Finishes the transcript and states its record.
#[tile(kind = iter, description = "Finish the transcript and state its output record")]
pub fn finish_output_record(
    state: FinalizeState,
    has_selected: bool,
    context: PromptContext,
) -> OutputRecord {
    output_record(finished_state(state), has_selected, &context)
}
