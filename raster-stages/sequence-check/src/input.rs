//! Shared Rastered data types and deterministic numerics for the
//! `sequence-check` stage.
//!
//! Boundary types are duplicated field-for-field from the stages they link
//! to — `ActivationSequence` from `prefill-range`, `PromptTokenization` from
//! `prompt-prepare`, `DecodeEdge` from `decode-init` and `output-finalize` —
//! because the chain binds by structural commitment, not by Rust type name.
//! The scoring numerics delegate to `det_num`, so a score is bit-identical to
//! direct generation's score of the same activation row.

pub use model_identity::ModelContext;
extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;
use det_num::ops::{mac_bits, requantize, rms_norm_in_place, softcap_act};
use det_num::{Acc, Act, Wgt};
use raster::{Bytes, BytesPage, List};
use serde::{Deserialize, Serialize};

/// Replay-unit page size for the logits projection: whole `Q8_0` rows of
/// `hidden_size / 32 · 34` bytes for every width this chain is imported with
/// (128 → 136, 1536 → 1632, 2560 → 2720). MUST match the page size the
/// importer writes the head external with.
pub const PAGE_SIZE: u64 = 48_960;

// ---------------------------------------------------------------------------
// Boundary types
// ---------------------------------------------------------------------------

/// One sequence position: the token that produced it and its activation row.
///
/// MUST match `ActivationRow` in every stage that carries activations.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct ActivationRow {
    pub token_id: u32,
    pub values: BytesPage,
}

/// Key/value side of one token. MUST match `KeyRow` in `prefill-range`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct KeyRow {
    pub position: u32,
    pub k: BytesPage,
    pub v: BytesPage,
}

/// The last transformer layer's output over `prompt ‖ claimed_output`.
///
/// MUST match `ActivationSequence` in `prefill-range`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct ActivationSequence {
    pub model: ModelContext,
    pub rows: List<ActivationRow>,
    pub errors: List<String>,
    pub kv: List<KeyRow>,
    pub start_position: u32,
}

/// The claim's output token ids. MUST match `ClaimedOutput` in `input-embedding`.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct ClaimedOutput {
    pub token_ids: List<u32>,
}

/// The output head's scalar parameters plus the final-norm vector.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct FinalHeadParams {
    pub hidden_size: u32,
    pub norm_eps: i64,
    /// `0` disables softcapping; otherwise logits become
    /// `softcap · tanh(logit / softcap)`.
    pub softcap: i32,
    pub norm_weights: BytesPage,
}

/// The committed output head.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct FinalHead {
    pub params: FinalHeadParams,
    /// `Q8_0` rows, one per vocabulary entry.
    #[page_size = 48_960]
    pub projection: Bytes<48_960>,
}

/// The stage's authorized output: the greedy selection at every generated
/// position, in the shape `output-finalize` consumes.
///
/// MUST match `DecodeEdge` in `decode-init` and `output-finalize`. For an honest
/// claim it is byte-identical to direct generation's final edge.
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
    pub generated_token_ids: List<u32>,
}

// ---------------------------------------------------------------------------
// This stage's own types
// ---------------------------------------------------------------------------

/// Loop-carried tail of the previous range: its last row so far.
///
/// One row, replaced rather than grown, and folded a chunk at a time. Starts as
/// [`empty_row`], so a previous range with no rows leaves that row in place
/// and the position it should have predicted is recorded as unscored: indexing
/// an empty list would abort with no output.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct PreviousTail {
    pub last: ActivationRow,
}

/// The rows that predict this range's claimed tokens, in order: the previous
/// range's last row, then this range's rows.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct PredictingRows {
    pub rows: List<ActivationRow>,
}

/// Loop-carried index of the row that predicts the next claimed token.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct CheckCursor {
    /// Index into the predicting rows.
    pub row_index: u32,
    /// Absolute position of this range's first token.
    pub start_position: u32,
}

/// One generated position about to be checked.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct CheckStep {
    pub row_index: u32,
    /// Absolute position of the token this row predicts — the
    /// `decode_position` direct generation reports for it.
    pub decode_position: u32,
    pub claimed_token_id: u32,
    pub start_position: u32,
}

/// The predicting row after the output norm, or why it could not be normed.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct NormalizedPosition {
    pub values: BytesPage,
    pub error: String,
}

/// Loop-carried argmax over the head, plus the first scoring failure.
///
/// Scalar-small, which is what recur `state` requires.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct ArgmaxState {
    pub has_value: bool,
    pub token_id: u32,
    pub value: i32,
    pub error_count: u32,
    pub first_error: String,
}

/// The greedy selection at one generated position.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct PositionSelection {
    pub decode_position: u32,
    pub token_id: u32,
    pub value: i32,
    pub claimed_token_id: u32,
}

/// A generated position that could not be scored, and why.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct PositionError {
    pub decode_position: u32,
    pub message: String,
}

/// Every generated position's selection, in order, plus scoring failures.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct CheckedPositions {
    pub selections: List<PositionSelection>,
    pub errors: List<PositionError>,
}

/// Loop-carried summary of the selections: the last one, and how many
/// disagreed with the claim.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct SelectionSummary {
    pub count: u32,
    pub mismatches: u32,
    pub first_mismatch_position: u32,
    pub has_selected: bool,
    pub decode_position: u32,
    pub token_id: u32,
    pub value: i32,
}

/// Loop-carried summary of failed positions: how many, and the first one.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct ErrorSummary {
    pub count: u32,
    pub first_position: u32,
    pub first: String,
}

/// The fault this check publishes, carried or newly found.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct CheckFault {
    pub fault: u32,
    pub position: u32,
}

/// What this range's check establishes its fault from.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct CheckFrame {
    /// Absolute position of this range's first token.
    pub start_position: u32,
    /// The run's limit on one range, from the prompt context.
    pub range_size: u32,
    /// How many tokens the prompt became, from the prompt context.
    pub prompt_tokens: u32,
    pub prior_has_selected: bool,
    pub prior_decode_position: u32,
    pub prior_fault: u32,
    pub prior_fault_position: u32,
}

/// No check failed.
pub const CHECK_FAULT_NONE: u32 = 0;
/// A generated position has no selection: its predicting row is missing or malformed.
pub const CHECK_FAULT_UNSCORED: u32 = 1;
/// The range claims no tokens.
pub const CHECK_FAULT_NO_CLAIMED_TOKENS: u32 = 2;
/// The range claims more tokens than the run's range size allows.
pub const CHECK_FAULT_RANGE_TOO_LARGE: u32 = 3;
/// The range does not start where the prompt or the previous range ended.
pub const CHECK_FAULT_DISCONTINUOUS: u32 = 4;

/// What a published `check_fault` means.
pub fn check_fault_description(fault: u32) -> &'static str {
    match fault {
        CHECK_FAULT_NONE => "no fault",
        CHECK_FAULT_UNSCORED => "a generated position could not be scored",
        CHECK_FAULT_NO_CLAIMED_TOKENS => "a range claims no tokens",
        CHECK_FAULT_RANGE_TOO_LARGE => "a range claims more tokens than the range size allows",
        CHECK_FAULT_DISCONTINUOUS => {
            "a range does not start where the prompt or the previous range ended"
        }
        _ => "an unknown fault",
    }
}

/// The row a previous range with no rows leaves behind. It has no values, so
/// normalising it records why the position it should predict went unscored.
pub fn empty_row() -> ActivationRow {
    ActivationRow {
        token_id: 0,
        values: pack_i32s(&[]),
    }
}

// ---------------------------------------------------------------------------
// Packing and Q16.16 kernels
// ---------------------------------------------------------------------------

pub fn pack_i32s(values: &[i32]) -> BytesPage {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    BytesPage::__from_parts(0, 0, bytes)
}

pub fn unpack_i32s(page: &BytesPage) -> core::result::Result<Vec<i32>, String> {
    let bytes = page.as_slice();
    if bytes.len() % 4 != 0 {
        return Err(alloc::format!(
            "page length {} is not i32-aligned",
            bytes.len()
        ));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| i32::from_le_bytes(chunk.try_into().expect("chunks_exact(4)")))
        .collect())
}

/// Q16.16 dot product: full Q32.32 accumulation, one requantize.
pub fn dot(a: &[i32], b: &[i32]) -> core::result::Result<i32, String> {
    if a.len() != b.len() {
        return Err(alloc::format!(
            "dot width mismatch: {} vs {}",
            a.len(),
            b.len()
        ));
    }
    let mut acc: i64 = 0;
    for (left, right) in a.iter().zip(b) {
        acc = mac_bits(acc, *left, *right);
    }
    Ok(requantize(Acc::from_bits(acc)).to_bits())
}

/// RMS normalisation through the canonical kernel. `eps` is `Acc` bits.
pub fn rms_norm(row: &mut [i32], weights: &[i32], eps: i64) -> core::result::Result<(), String> {
    if row.len() != weights.len() {
        return Err(alloc::format!(
            "rms norm width mismatch: {} values vs {} weights",
            row.len(),
            weights.len()
        ));
    }
    if row.is_empty() {
        return Err(String::from("rms norm requires at least one value"));
    }
    if eps < 0 {
        return Err(String::from("rms norm epsilon must be non-negative"));
    }
    let mut lanes: Vec<Act> = row.iter().map(|b| Act::from_bits(*b)).collect();
    let weight: Vec<Wgt> = weights.iter().map(|b| Wgt::from_bits(*b)).collect();
    rms_norm_in_place(&mut lanes, &weight, Acc::from_bits(eps));
    for (slot, value) in row.iter_mut().zip(lanes.iter()) {
        *slot = value.to_bits();
    }
    Ok(())
}

/// Gemma's final-logit softcapping.
pub fn softcap(value: i32, cap: i32) -> i32 {
    if cap <= 0 {
        return value;
    }
    softcap_act(Act::from_bits(value), Act::from_bits(cap)).to_bits()
}
