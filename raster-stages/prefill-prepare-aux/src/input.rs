//! Shared Rastered data types and deterministic numerics for the
//! `prefill-prepare-aux` stage.
//!
//! Q16.16 fixed-point throughout. Tiles must replay bit-identically inside the
//! zkVM, so floats are out — see SKILL.md §3.

extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;
use raster::{Bytes, BytesPage, List};
use det_num::ops::{mac_bits, mul_sat, requantize, rms_norm_in_place};
use det_num::{Acc, Act, Wgt};
use serde::{Deserialize, Serialize};

/// Fractional bits in the Q16.16 representation.
pub const FRAC_BITS: u32 = 16;

/// One unit (`1.0`) in Q16.16.
pub const ONE: i32 = 1 << FRAC_BITS;

/// Replay-unit page size for PLE tables and the projection matrix.
pub const PAGE_SIZE: u64 = 196_608;

// ---------------------------------------------------------------------------
// Types crossing the stage boundary
// ---------------------------------------------------------------------------

/// One prompt position, from `input_embedding`.
///
/// MUST match `ActivationRow` in `input-embedding` and `prefill-range`
/// field-for-field.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct ActivationRow {
    pub token_id: u32,
    pub values: BytesPage,
}

/// MUST match `ActivationSequence` in `input-embedding` and `prefill-range`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct ActivationSequence {
    pub rows: List<ActivationRow>,
    pub errors: List<String>,
    /// This layer's K/V cache after this stage: everything it was handed as
    /// `prior_kv`, pruned to the sliding window, plus one row per token this
    /// stage processed.
    ///
    /// Every layer publishes it, and it is read by two different consumers: a
    /// later *sharing* layer attends over its donor's cache in the same step,
    /// and the *same* layer one decode step later takes it as `prior_kv`. A
    /// layer that is neither a donor nor decoded again still pays to publish
    /// it, which is the price of one uniform program across all 35 stages.
    pub kv: List<KeyRow>,
    /// Absolute position of `rows[0]` in the full sequence — 0 for the prompt,
    /// `prompt_len + t` for decode step `t`.
    ///
    /// Positions cannot be recovered by counting: a decode stage sees one row
    /// but sits at position `prompt_len + t`, and RoPE and sliding-window
    /// visibility are both defined over the absolute value. Carrying it
    /// explicitly is what lets one program serve prefill and decode.
    pub start_position: u32,
}

/// Scalars plus the PLE projection norm. Materializable — legal `args`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct PleLayerParams {
    pub layer_idx: u32,
    pub hidden_size: u32,
    pub ple_width: u32,
    pub embedding_scale: i32,
    pub projection_scalar: i32,
    pub input_scale: i32,
    pub norm_eps: i64,
    pub norm_weights: BytesPage,
}

/// This layer's committed external.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct PleLayer {
    pub params: PleLayerParams,
    #[page_size = 196_608]
    pub embeddings: Bytes<196_608>,
    #[page_size = 196_608]
    pub projection: Bytes<196_608>,
}

/// One token's work in flight.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct PleTask {
    pub token_id: u32,
    pub activation: BytesPage,
}

/// Matvec accumulator: a packed output vector, plus an error if a page failed.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct ProjAccum {
    pub values: BytesPage,
    pub error: String,
}

/// The projected + normalised half of one row, between the two compute tiles.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct ProjectedRow {
    pub values: BytesPage,
    pub error: String,
}

/// Loop-carried summary of the rows that failed.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct ErrorSummary {
    pub count: u32,
    pub first: String,
}

/// Key/value side of one token — the layer's KV cache entry.
///
/// MUST match `KeyRow` in every stage that carries activations: Gemma 3n's last
/// 20 layers attend over an earlier layer's cache, so the cache has to survive a
/// stage boundary, and a chain binds by structural commitment.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct KeyRow {
    pub position: u32,
    pub k: BytesPage,
    pub v: BytesPage,
}

/// One token's per-layer embedding.
///
/// A wrapper rather than a bare `BytesPage` on purpose: `select!(BytesPage, xs[i])`
/// treats its target as a *paged region* and rewrites the path to `xs.pages[i]`,
/// which is wrong for a `List<BytesPage>`. Naming a struct keeps the index
/// meaning what it says.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct PleRow {
    pub values: BytesPage,
}

/// The stage's authorized output.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct PleLayerInputs {
    pub layer_idx: u32,
    pub rows: List<PleRow>,
    pub errors: List<String>,
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

pub fn unpack_i32s_at(
    page: &BytesPage,
    byte_off: u64,
    n_values: u32,
) -> core::result::Result<Vec<i32>, String> {
    let local = byte_off
        .checked_sub(page.offset())
        .ok_or_else(|| String::from("row offset is before this page"))? as usize;
    let len = n_values as usize * 4;
    let bytes = page.as_slice().get(local..local + len).ok_or_else(|| {
        alloc::format!(
            "row at byte {byte_off} spans past this page (local {local}, need {len}, have {})",
            page.len()
        )
    })?;
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| i32::from_le_bytes(chunk.try_into().expect("chunks_exact(4)")))
        .collect())
}

// ---------------------------------------------------------------------------
// Q16.16 kernels
// ---------------------------------------------------------------------------

fn clamp_i64(value: i64) -> i32 {
    if value > i32::MAX as i64 {
        i32::MAX
    } else if value < i32::MIN as i64 {
        i32::MIN
    } else {
        value as i32
    }
}

pub fn mul(a: i32, b: i32) -> i32 {
    // Canonical multiply: widen, then requantize ties-to-even and saturate.
    // The previous version truncated with `>> FRAC_BITS`, which rounds toward
    // negative infinity and differs from the reference on half the inputs.
    mul_sat(Act::from_bits(a), Act::from_bits(b)).to_bits()
}

pub fn add_sat(a: i32, b: i32) -> i32 {
    a.saturating_add(b)
}

pub fn scale_row(row: &mut [i32], scalar: i32) {
    for value in row.iter_mut() {
        *value = mul(*value, scalar);
    }
}


pub fn dot(a: &[i32], b: &[i32]) -> core::result::Result<i32, String> {
    if a.len() != b.len() {
        return Err(alloc::format!(
            "dot width mismatch: {} vs {}",
            a.len(),
            b.len()
        ));
    }
    // Accumulate the *full* Q32.32 products and requantize once, which is the
    // canonical MAC contract. The previous version shifted each product down to
    // Q16.16 before adding, truncating on every term and rounding differently at
    // the end — invisible per element, decisive over a 1536-wide row.
    let mut acc: i64 = 0;
    for (left, right) in a.iter().zip(b) {
        acc = mac_bits(acc, *left, *right);
    }
    Ok(requantize(Acc::from_bits(acc)).to_bits())
}

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

    // Delegates to the canonical kernel rather than reimplementing it. The
    // hand-rolled version this replaces differed from the reference in four
    // ways at once — Q16.16 instead of Q32.32 accumulation, truncating instead
    // of ties-to-even division, dividing by the RMS instead of multiplying by
    // its reciprocal, and applying the weight before the scale rather than
    // after. None of that is visible in the output; it just produces a
    // different activation. `eps` is `Acc` (Q32.32) bits, which is also what
    // the old encoding got wrong: 1e-6 in Q16.16 rounds to zero.
    let mut lanes: alloc::vec::Vec<Act> = row.iter().map(|b| Act::from_bits(*b)).collect();
    let weight: alloc::vec::Vec<Wgt> = weights.iter().map(|b| Wgt::from_bits(*b)).collect();
    rms_norm_in_place(&mut lanes, &weight, Acc::from_bits(eps));
    for (slot, value) in row.iter_mut().zip(lanes.iter()) {
        *slot = value.to_bits();
    }
    Ok(())
}
