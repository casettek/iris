//! The `claim-result` stage's input: the last check's output.
//!
//! `DecodeEdge`, `ClaimRecord` and `OutputRecord` are `sequence-check`'s,
//! duplicated field for field, because a chain binds by structural
//! commitment and not by Rust type name. A test in `staged-infer` holds the
//! two to the same commitment. This stage reads two things of the edge: the
//! generated text, and the claim record.

pub use model_identity::ModelContext;
extern crate alloc;

use alloc::string::String;
use raster::{BytesPage, List};
use serde::{Deserialize, Serialize};

/// A check's output. See `sequence-check` for what each field is.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct DecodeEdge {
    pub model: ModelContext,
    pub has_selected: bool,
    pub decode_position: u32,
    pub token_id: u32,
    pub value: i32,
    /// The claim verdict so far: how many claimed tokens, in this range and
    /// every earlier one, are not the greedy selection at their position.
    /// Zero for an honest claim.
    pub claim_mismatches: u32,
    /// Absolute sequence position of the first disagreement, or zero when
    /// there is none.
    pub first_claim_mismatch_position: u32,
    /// Why the check could not stand behind this edge: one of the
    /// `CHECK_FAULT_*` codes, the earliest found.
    pub check_fault: u32,
    /// Absolute sequence position of that fault, or zero when there is none.
    pub check_fault_position: u32,
    /// The transcript so far as an output: see [`OutputRecord`].
    pub record: OutputRecord,
    /// What a contract reads of this edge: see [`ClaimRecord`].
    pub claim: ClaimRecord,
    pub generated_token_ids: List<u32>,
}

/// What a contract reads of a claim: one statement, one output commitment and
/// one verdict. Three fixed-size values, so a contract holds them without
/// parsing the record or the transcript.
///
/// Every check states it beside its record, from that record and the edge's
/// own verdict. Only the schedule's last claimed range can state a valid one:
/// an earlier range's record carries a model fault.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct ClaimRecord {
    /// `Statement::digest()` over the input text hash, the model id, the raw
    /// prompt flag, the token budget and the range size the record names: 32
    /// bytes, all zero when either id is not a 32-byte hash.
    pub statement: BytesPage,
    /// SHA-256 of the exact UTF-8 bytes of the generated text: 32 bytes.
    pub output_text_sha256: BytesPage,
    /// The chain found nothing wrong with its own run: the prompt is complete
    /// and without a fault, consumed model data authenticated, every claimed
    /// token is the greedy selection, and the check and the transcript broke
    /// no rule. See [`crate::claim_record`].
    pub valid: bool,
}

/// No statement, no output and no verdict: what an edge holds before a check
/// has stated its claim. Written out because a page has no default value; no
/// tile reads it.
impl Default for ClaimRecord {
    fn default() -> Self {
        Self {
            statement: BytesPage::__from_parts(0, 0, alloc::vec![0; 32]),
            output_text_sha256: BytesPage::__from_parts(0, 0, alloc::vec![0; 32]),
            valid: false,
        }
    }
}

/// The transcript so far, finalized: its commitments, its decoded text, what
/// it breaks, and the context the chain ran under.
///
/// Every check states it. Only the last claimed range's is a claim's output:
/// an earlier range's has `model_fault` set, because the schedule puts the
/// final step after the last check and nowhere else.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct OutputRecord {
    /// Nonzero when consumed model data failed authentication, or when this
    /// is not the record of the schedule's last claimed range.
    pub model_fault: u32,
    pub generated_token_count: u32,
    pub generated_token_ids_sha256: String,
    pub generated_text: String,
    pub stop_reason: String,
    /// SHA-256 of the exact UTF-8 bytes of `generated_text`.
    pub output_text_sha256: String,
    /// What the transcript breaks, or zero: one of the `OUTPUT_FAULT_*` codes.
    pub output_fault: u32,
    /// SHA-256 of the exact text the chain consumed, before chat formatting.
    pub input_text_sha256: String,
    /// The source the text was built from: see `source_id` in the shared prompt
    /// source. 64 zeros for a text taken as given.
    pub source_sha256: String,
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
