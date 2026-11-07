//! Boundary and work types for generated-token finalization.

pub use model_identity::ModelContext;
extern crate alloc;

use alloc::string::String;
use raster::{BytesPage, List};
use serde::{Deserialize, Serialize};

/// MUST match the output of `decode-init` and `sequence-check`.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct DecodeEdge {
    pub model: ModelContext,
    pub has_selected: bool,
    pub decode_position: u32,
    pub token_id: u32,
    pub value: i32,
    /// The teacher-forced claim verdict, published as-is in the output.
    pub claim_mismatches: u32,
    pub first_claim_mismatch_position: u32,
    /// The teacher-forced check's own fault, published as-is in the output.
    pub check_fault: u32,
    pub check_fault_position: u32,
    pub generated_token_ids: List<u32>,
}

/// One dense decoder-table row. Its list position is its token id.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct DecoderToken {
    pub token: String,
    pub special: bool,
    /// In the model's `eos_token_id` set. The chain's decode loop is statically
    /// expanded to a fixed repeat count, so it cannot stop early; this is what
    /// lets the finalized output report where the answer actually ended.
    pub terminal: bool,
}

/// Committed tokenizer data needed by output finalization.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct DecoderTable {
    pub tokens: List<DecoderToken>,
}

/// Bounded loop state. `pending_bytes` only holds one byte-fallback run. The
/// text and JSON fields grow with the requested generation bound; the chain's
/// static repeat count pins that bound.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct FinalizeState {
    pub history: ModelContext,
    pub expected_token: u32,
    pub last_token: u32,
    pub model_fault: u32,
    pub count: u32,
    pub json: String,
    pub text: String,
    pub pending_bytes: BytesPage,
    /// Set by the first terminal token. Everything after it is still counted
    /// and committed — it was generated — but it is not the model's answer, so
    /// it stops contributing to `text`.
    pub stopped: bool,
    /// A token followed the first terminal token.
    pub past_terminal: bool,
}

/// Final authorized inference output.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct GeneratedOutput {
    /// Nonzero when consumed model data failed authentication.
    pub model_fault: u32,
    pub generated_token_count: u32,
    pub generated_token_ids: List<u32>,
    pub generated_token_ids_sha256: String,
    pub generated_text: String,
    pub stop_reason: String,
    /// How many generated positions were given a claimed token that is not
    /// the greedy selection there. A claim stands only if this is zero.
    pub claim_mismatches: u32,
    /// Absolute sequence position of the first disagreement, or zero when
    /// there is none.
    pub first_claim_mismatch_position: u32,
    /// The teacher-forced check's fault, or zero. See `CHECK_FAULT_*` in
    /// `sequence-check`.
    pub check_fault: u32,
    pub check_fault_position: u32,
    /// SHA-256 of the exact UTF-8 bytes of `generated_text`.
    pub output_text_sha256: String,
    /// What the transcript breaks, or zero: one of the `OUTPUT_FAULT_*` codes.
    pub output_fault: u32,
    /// SHA-256 of the exact text the chain consumed, before chat formatting.
    pub input_text_sha256: String,
    /// The identifier of the committed model the chain was configured with.
    pub model_id: String,
    pub raw_prompt: bool,
    pub max_new_tokens: u32,
    pub range_size: u32,
    /// How many tokens the prompt became.
    pub prompt_tokens: u32,
    /// The prompt ranges covered the whole prepared prompt, in order.
    pub prompt_complete: bool,
    /// What failed in prompt preparation, or zero. See `FAULT_*` in the shared
    /// prompt source.
    pub prompt_fault: u32,
    pub prompt_fault_position: u64,
}

/// The transcript breaks no rule.
pub const OUTPUT_FAULT_NONE: u32 = 0;
/// The run produced no tokens.
pub const OUTPUT_FAULT_NO_TOKENS: u32 = 1;
/// The run produced more tokens than its budget.
pub const OUTPUT_FAULT_OVER_BUDGET: u32 = 2;
/// The run stopped short of its budget without reaching a terminal token.
pub const OUTPUT_FAULT_SHORT_WITHOUT_EOS: u32 = 3;
/// Tokens continue after a terminal token.
pub const OUTPUT_FAULT_PAST_EOS: u32 = 4;
/// The edge's selection flag disagrees with its transcript.
pub const OUTPUT_FAULT_EDGE_DISAGREES: u32 = 5;

/// What a published `output_fault` means.
pub fn output_fault_description(fault: u32) -> &'static str {
    match fault {
        OUTPUT_FAULT_NONE => "no fault",
        OUTPUT_FAULT_NO_TOKENS => "the output has no tokens",
        OUTPUT_FAULT_OVER_BUDGET => "the output has more tokens than the run allows",
        OUTPUT_FAULT_SHORT_WITHOUT_EOS => {
            "the output stops short of the token budget without reaching EOS"
        }
        OUTPUT_FAULT_PAST_EOS => "the output continues past EOS",
        OUTPUT_FAULT_EDGE_DISAGREES => "the decode edge's selection flag disagrees with its tokens",
        _ => "an unknown fault",
    }
}

/// Reusable membership paths; separate from the decoder's identity.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct DecoderWitness {
    pub token_count: u64,
    pub paths: List<DecoderProof>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct DecoderProof {
    pub siblings: BytesPage,
}
