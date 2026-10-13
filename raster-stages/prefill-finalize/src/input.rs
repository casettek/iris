//! Shared Rastered data types and deterministic numerics for the
//! `prefill-finalize` stage.
//!
//! Q16.16 fixed point throughout, integer-only so every tile replays
//! bit-identically in the zkVM (SKILL.md §3). The kernels mirror the shape of
//! `det_kernels::det_hidden_to_logits`, not its quantisation.

extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;
use raster::{Bytes, BytesPage, List};
use det_num::ops::{div_act, mac_bits, mul_sat, requantize, rms_norm_in_place, softcap_act};
use det_num::{Acc, Act, Wgt};
use serde::{Deserialize, Serialize};

/// Fractional bits in the Q16.16 representation.
pub const FRAC_BITS: u32 = 16;

/// One unit (`1.0`) in Q16.16.
pub const ONE: i32 = 1 << FRAC_BITS;

/// Replay-unit page size for the logits projection.
pub const PAGE_SIZE: u64 = 196_608;

// ---------------------------------------------------------------------------
// The pipeline's activation type — field-for-field identical in every stage
// that passes activations on, because the chain links by structural
// commitment.
// ---------------------------------------------------------------------------

/// One prompt position: the token that produced it and its activation row.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct ActivationRow {
    pub token_id: u32,
    pub values: BytesPage,
}

/// A prompt's activations, in prompt order.
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

// ---------------------------------------------------------------------------
// This stage's own types
// ---------------------------------------------------------------------------

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

/// The stage's committed external: the final norm plus the logits projection
/// (vocab × hidden, row-major).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct FinalHead {
    pub params: FinalHeadParams,
    #[page_size = 196_608]
    pub projection: Bytes<196_608>,
}

/// Loop-carried result of walking the prompt to its last position.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct FinalPosition {
    pub found: bool,
    pub token_id: u32,
    pub token_count: u32,
    pub values: BytesPage,
}

/// The final position after the output norm — the vector every logit is a dot
/// product against.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct NormalizedPosition {
    pub values: BytesPage,
}

/// One scored token.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct LogitEntry {
    pub token_id: u32,
    pub value: i32,
}

/// Loop-carried summary of failed rows: how many, and the first message.
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

/// The stage's authorized output: the prompt's next-token scores.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct PrefillLogits {
    pub decode_position: u32,
    pub logits: List<LogitEntry>,
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

/// Q16.16 multiply, saturating.
pub fn mul(a: i32, b: i32) -> i32 {
    // Canonical multiply: widen, then requantize ties-to-even and saturate.
    // The previous version truncated with `>> FRAC_BITS`, which rounds toward
    // negative infinity and differs from the reference on half the inputs.
    mul_sat(Act::from_bits(a), Act::from_bits(b)).to_bits()
}

/// Q16.16 divide, saturating. Dividing by zero yields zero rather than
/// trapping: a tile must not panic on staged data.
pub fn div(a: i32, b: i32) -> i32 {
    if b == 0 {
        return 0;
    }
    // Canonical divide: ties-to-even, not the truncating shift-and-divide this
    // replaces.
    div_act(Act::from_bits(a), Act::from_bits(b)).to_bits()
}

/// Integer square root: the largest `r` with `r*r <= value`. Deterministic on
/// every target, which `f64::sqrt` would not be.

/// `e^x` in Q16.16, via `e^x = 2^(x·log2 e)`.
pub fn exp(x: i32) -> i32 {
    const LOG2E: i32 = 94548;
    const A1: i32 = 45426;
    const A2: i32 = 15743;
    const A3: i32 = 3634;

    if x <= -(20 * ONE) {
        return 0;
    }
    if x >= 20 * ONE {
        return i32::MAX;
    }

    let y = mul(x, LOG2E);
    let int_part = y >> FRAC_BITS;
    let frac = y - (int_part << FRAC_BITS);
    let poly = ONE + mul(frac, A1 + mul(frac, A2 + mul(frac, A3)));

    if int_part >= 0 {
        clamp_i64((poly as i64) << int_part)
    } else {
        let shift = (-int_part) as u32;
        if shift >= 31 {
            0
        } else {
            poly >> shift
        }
    }
}

/// Q16.16 dot product, accumulated in i64.
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

/// RMS normalisation: `x[i] = x[i] / rms(x) · weight[i]`.
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

/// `tanh(x)` in Q16.16, from [`exp`].
pub fn tanh(x: i32) -> i32 {
    let e2x = exp(x.saturating_mul(2));
    div(e2x.saturating_sub(ONE), e2x.saturating_add(ONE))
}

/// Gemma's final-logit softcapping.
pub fn softcap(value: i32, cap: i32) -> i32 {
    if cap <= 0 {
        return value;
    }
    // `cap * tanh(value / cap)` through the canonical kernel — the divide, the
    // tanh and the multiply all have contracts the hand-rolled version did not
    // reproduce.
    softcap_act(Act::from_bits(value), Act::from_bits(cap)).to_bits()
}
