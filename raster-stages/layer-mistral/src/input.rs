//! Shared Rastered data types and deterministic numerics for the
//! `layer-mistral` stage.
//!
//! Q16.16 fixed point throughout (`1.0 == 1 << 16`), integer-only so every
//! tile replays bit-identically in the zkVM.

pub use model_identity::ModelContext;
extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;
use det_num::ops::{mul_sat, rms_norm_in_place};
use det_num::{Acc, Act, Wgt};
use raster::{Bytes, BytesPage, List};
use serde::{Deserialize, Serialize};

/// Replay-unit page size for the `Q8_0` matrices: whole rows of
/// `in / 32 · 34` bytes for every width this family is imported with. MUST
/// match the importer's `Q8_PAGE_SIZE`.
pub const Q8_PAGE_SIZE: u64 = 195_840;

// ---------------------------------------------------------------------------
// The pipeline's activation type — field-for-field identical in every stage
// that passes activations on.
// ---------------------------------------------------------------------------

/// One position: the token that produced it and its activation row.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct ActivationRow {
    pub token_id: u32,
    pub values: BytesPage,
}

/// A range's activations, in order.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct ActivationSequence {
    pub model: ModelContext,
    pub rows: List<ActivationRow>,
    pub errors: List<String>,
    /// This layer's K/V cache after this stage: everything it was handed as
    /// `prior_kv`, plus one row per token this stage processed. The same layer
    /// takes it as `prior_kv` for the next range.
    pub kv: List<KeyRow>,
    /// Absolute position of `rows[0]` in the full sequence.
    pub start_position: u32,
}

// ---------------------------------------------------------------------------
// This stage's own types
// ---------------------------------------------------------------------------

/// What a layer computes with besides its matrices. Materializable — legal
/// `args`. Weight matrices live on [`TransformerLayer`] as `Bytes` regions.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct LayerParams {
    pub layer_idx: u32,
    pub hidden_size: u32,
    pub ffn_size: u32,
    pub num_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub norm_eps: i64,
    /// `1 / sqrt(head_dim)` in Q0.32, applied to every attention score.
    pub attention_scale_q32: u32,
    /// Positions from here on are refused: past it the model scales its
    /// queries by a function of the position that this program does not have.
    pub max_positions: u32,
    /// One Q32.32 angle per position for each consecutive pair of a head, as
    /// little-endian `i64`s: `head_dim / 2` of them.
    pub rope_inverse_frequencies: BytesPage,
    /// The norm before attention.
    pub norm_attention: BytesPage,
    /// The norm before the feed-forward block.
    pub norm_feed_forward: BytesPage,
}

/// One transformer layer's committed external.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct TransformerLayer {
    pub params: LayerParams,
    // Projection and MLP matrices: `Q8_0` rows, `in / 32 · 34` bytes each.
    #[page_size = 195_840]
    pub w_q: Bytes<195_840>,
    #[page_size = 195_840]
    pub w_k: Bytes<195_840>,
    #[page_size = 195_840]
    pub w_v: Bytes<195_840>,
    #[page_size = 195_840]
    pub w_o: Bytes<195_840>,
    #[page_size = 195_840]
    pub w_gate: Bytes<195_840>,
    #[page_size = 195_840]
    pub w_up: Bytes<195_840>,
    #[page_size = 195_840]
    pub w_down: Bytes<195_840>,
}

/// Key/value side of one token, for the attention scan.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct KeyRow {
    pub position: u32,
    pub k: BytesPage,
    pub v: BytesPage,
}

/// Query side of one token, plus the residual it will be added back to.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct QueryRow {
    /// Absolute position in the full sequence.
    pub position: u32,
    /// Index of this token within this stage: `position - start_position`.
    /// Pass 1's `keys` hold one entry per token this stage processed, so they
    /// are indexed by `local`; an out-of-range dynamic index aborts the run.
    pub local: u32,
    pub token_id: u32,
    pub q: BytesPage,
    pub residual: BytesPage,
}

/// Pass 1's loop-carried position pair.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct TokenCursor {
    pub position: u32,
    pub local: u32,
}

/// Pass 1's result: keys and queries as separate lists so the key scan does
/// not rematerialize query-side fields.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct KvSequence {
    pub keys: List<KeyRow>,
    pub queries: List<QueryRow>,
    pub errors: List<String>,
}

/// RMS-normalised activation, ready for Q/K/V matvecs.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct QkvPrep {
    pub token_id: u32,
    pub residual: BytesPage,
    pub normed: BytesPage,
    pub error: String,
}

/// Matvec accumulator: packed output vector plus an error if a page failed.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct ProjAccum {
    pub values: BytesPage,
    pub error: String,
}

/// One query's attention scores, key-major (`scores[key * num_heads + head]`),
/// addressed by the key's position so the order keys arrive in does not matter.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct ScoreAccum {
    pub scores: BytesPage,
    /// How many positions have been scored. Fewer than `count` at the end
    /// means a key the query needs was missing.
    pub filled: u32,
    /// Positions this query sees: every one up to its own.
    pub count: u32,
    pub error: String,
}

/// Softmax weights, same layout and count as the scores they came from.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct WeightVec {
    pub weights: BytesPage,
    pub count: u32,
    pub error: String,
}

/// Weighted-sum accumulator: one `i64` per (head, lane), requantized once.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct CtxAccum {
    pub acc: BytesPage,
    pub error: String,
}

/// A packed vector that may carry a failure from an earlier step.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct PackedVec {
    pub values: BytesPage,
    pub error: String,
}

/// Loop-carried summary of failed rows.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct ErrorSummary {
    pub count: u32,
    pub first: String,
}

/// Bounded verifier for the ordered cache positions a range inherits.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct CacheCoverage {
    pub next: u32,
    pub end: u32,
    pub fault: u32,
}

// ---------------------------------------------------------------------------
// Packing
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

pub fn pack_i64s(values: &[i64]) -> BytesPage {
    let mut bytes = Vec::with_capacity(values.len() * 8);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    BytesPage::__from_parts(0, 0, bytes)
}

pub fn unpack_i64s(page: &BytesPage) -> core::result::Result<Vec<i64>, String> {
    let bytes = page.as_slice();
    if bytes.len() % 8 != 0 {
        return Err(alloc::format!(
            "page length {} is not i64-aligned",
            bytes.len()
        ));
    }
    Ok(bytes
        .chunks_exact(8)
        .map(|chunk| i64::from_le_bytes(chunk.try_into().expect("chunks_exact(8)")))
        .collect())
}

// ---------------------------------------------------------------------------
// Q16.16 kernels
// ---------------------------------------------------------------------------

/// Canonical multiply: widen, then requantize ties-to-even and saturate.
pub fn mul(a: i32, b: i32) -> i32 {
    mul_sat(Act::from_bits(a), Act::from_bits(b)).to_bits()
}

pub fn add_row(dst: &mut [i32], src: &[i32]) -> core::result::Result<(), String> {
    if dst.len() != src.len() {
        return Err(alloc::format!(
            "row add width mismatch: {} vs {}",
            dst.len(),
            src.len()
        ));
    }
    for (value, other) in dst.iter_mut().zip(src) {
        *value = value.saturating_add(*other);
    }
    Ok(())
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
    let mut lanes: Vec<Act> = row.iter().map(|bits| Act::from_bits(*bits)).collect();
    let weight: Vec<Wgt> = weights.iter().map(|bits| Wgt::from_bits(*bits)).collect();
    rms_norm_in_place(&mut lanes, &weight, Acc::from_bits(eps));
    for (slot, value) in row.iter_mut().zip(lanes.iter()) {
        *slot = value.to_bits();
    }
    Ok(())
}
