//! Shared Rastered data types and deterministic numerics for the
//! `sequence-check` stage.
//!
//! Boundary types are duplicated field-for-field from the stages they link
//! to — `ActivationSequence` from `prefill-range`, `PromptTokenization` from
//! `prompt-prepare`, `FinalHead` from `prefill-finalize`, `DecodeEdge` from
//! `output-finalize` — because the chain binds by structural commitment, not
//! by Rust type name. The scoring numerics delegate to `det_num` exactly as
//! `prefill-finalize` does, so a teacher-forced score is bit-identical to the
//! decode-path score of the same activation row.

extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;
use det_num::ops::{mac_bits, requantize, rms_norm_in_place, softcap_act};
use det_num::{Acc, Act, Wgt};
use raster::{Bytes, BytesPage, List};
use serde::{Deserialize, Serialize};

/// Replay-unit page size for the logits projection. MUST match
/// `prefill-finalize`: both stages read the same committed head external.
pub const PAGE_SIZE: u64 = 52_224;

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
///
/// MUST match `FinalHeadParams` in `prefill-finalize`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct FinalHeadParams {
    pub hidden_size: u32,
    pub norm_eps: i64,
    /// `0` disables softcapping; otherwise logits become
    /// `softcap · tanh(logit / softcap)`.
    pub softcap: i32,
    pub norm_weights: BytesPage,
}

/// The committed output head. MUST match `FinalHead` in `prefill-finalize`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct FinalHead {
    pub params: FinalHeadParams,
    /// `Q8_0` rows, one per vocabulary entry.
    #[page_size = 52_224]
    pub projection: Bytes<52_224>,
}

/// The stage's authorized output: the greedy selection at every generated
/// position, in the shape `output-finalize` consumes.
///
/// MUST match `DecodeEdge` in `decode-select-token` and `output-finalize`.
/// For an honest claim it is byte-identical to the autoregressive chain's
/// final `decode.edge`.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct DecodeEdge {
    pub has_selected: bool,
    pub decode_position: u32,
    pub token_id: u32,
    pub value: i32,
    /// The claim verdict so far: how many claimed tokens, in this range and
    /// every earlier one, are not the greedy selection at their position.
    /// Zero for an honest claim, as on every autoregressive edge.
    pub claim_mismatches: u32,
    /// Absolute sequence position of the first disagreement, or zero when
    /// there is none.
    pub first_claim_mismatch_position: u32,
    pub generated_token_ids: List<u32>,
}

// ---------------------------------------------------------------------------
// This stage's own types
// ---------------------------------------------------------------------------

/// Loop-carried count of the previous range's rows.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct TokenCount {
    pub count: u32,
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
    /// `decode_position` the autoregressive chain reports for it.
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

/// Every generated position's selection, in order, plus scoring failures.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct CheckedPositions {
    pub selections: List<PositionSelection>,
    pub errors: List<String>,
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

/// Loop-carried summary of failed positions: how many, and the first message.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct ErrorSummary {
    pub count: u32,
    pub first: String,
}

// ---------------------------------------------------------------------------
// Packing and Q16.16 kernels — the `prefill-finalize` contracts
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
