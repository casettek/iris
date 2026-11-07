//! Finalizes the generated token transcript into IDs, a reference-compatible
//! commitment, and decoded text.

#![no_std]

extern crate alloc;

use alloc::string::{String, ToString};
use alloc::vec::Vec;
use raster::prelude::*;
use sha2::{Digest, Sha256};

pub mod input;
pub mod model;
pub use model::*;
pub mod prompt;

use input::*;
use prompt::PromptContext;

/// The empty state, as a plain function — see [`advance_finalize_state`].
pub fn initial_finalize_state() -> FinalizeState {
    FinalizeState {
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

/// Flushes byte fallback and hashes exactly the JSON token-id representation
/// used by the reference's `build_output_decode_commitment`.
#[tile(kind = iter, description = "Finish generated text and token commitment")]
pub fn finish_finalize_state(state: FinalizeState) -> FinalizeState {
    finished_state(state)
}

/// The step [`finish_finalize_state`] is, as a plain function, for the native
/// kernel.
pub fn finished_state(state: FinalizeState) -> FinalizeState {
    let mut state = state;
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

/// Opens the final output: the finished transcript, the commitments to it, the
/// verdict and fault carried from the edge, and the context the chain ran under.
#[tile(kind = iter, description = "Open final generated output")]
pub fn begin_generated_output(
    state: FinalizeState,
    has_selected: bool,
    claim_mismatches: u32,
    first_claim_mismatch_position: u32,
    check_fault: u32,
    check_fault_position: u32,
    context: PromptContext,
) -> Draft<GeneratedOutput> {
    let mut output = Draft::<GeneratedOutput>::new();
    output.model_fault().set(state.model_fault);
    output
        .output_fault()
        .set(output_fault_of(&state, has_selected, &context));
    output
        .output_text_sha256()
        .set(sha256_hex(state.text.as_bytes()));
    output.generated_token_count().set(state.count);
    output
        .generated_token_ids_sha256()
        .set(sha256_hex(state.json.as_bytes()));
    output.generated_text().set(state.text);
    output.stop_reason().set(if state.stopped {
        String::from("eos")
    } else {
        String::from("max_new_tokens")
    });
    output.claim_mismatches().set(claim_mismatches);
    output
        .first_claim_mismatch_position()
        .set(first_claim_mismatch_position);
    output.check_fault().set(check_fault);
    output.check_fault_position().set(check_fault_position);
    output.input_text_sha256().set(context.input_text_sha256);
    output.model_id().set(context.model_id);
    output.raw_prompt().set(context.raw_prompt);
    output.max_new_tokens().set(context.max_new_tokens);
    output.range_size().set(context.range_size);
    output.prompt_tokens().set(context.prompt_tokens);
    output.prompt_complete().set(context.complete);
    output.prompt_fault().set(context.fault);
    output.prompt_fault_position().set(context.fault_position);
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
