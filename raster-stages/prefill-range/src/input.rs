//! Shared Rastered data types and deterministic numerics for the
//! `prefill-range` stage.
//!
//! Q16.16 fixed point throughout (`1.0 == 1 << 16`), integer-only so every
//! tile replays bit-identically in the zkVM (SKILL.md §3).

extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;
use raster::{Bytes, BytesPage, List};
use det_num::ops::{div_act, gelu_pytorch_tanh_act, mac_bits, mul_sat, requantize, rms_norm_in_place, value_rms_norm_in_place};
use det_num::{Acc, Act, Wgt};
use serde::{Deserialize, Serialize};

/// Fractional bits in the Q16.16 representation.
pub const FRAC_BITS: u32 = 16;

/// One unit (`1.0`) in Q16.16.
pub const ONE: i32 = 1 << FRAC_BITS;

/// Replay-unit page size for every weight matrix in this program.
pub const PAGE_SIZE: u64 = 196_608;

// ---------------------------------------------------------------------------
// The pipeline's activation type — field-for-field identical in every stage
// that passes activations on.
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

/// One layer's per-layer embeddings, one row per prompt token.
///
/// MUST match `PleLayerInputs` in `prefill-prepare-aux` — the chain binds
/// `prefill_range_lN` to `prefill_prepare_aux_lN` by structural commitment.
/// `rows` is in prompt order, so a token's row is a dynamic index on its
/// position.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct PleLayerInputs {
    pub layer_idx: u32,
    pub rows: List<PleRow>,
    pub errors: List<String>,
}

// ---------------------------------------------------------------------------
// This stage's own types
// ---------------------------------------------------------------------------

/// Scalars plus the four (plus Q/K) norm vectors. Materializable — legal `args`.
/// Weight matrices live on [`TransformerLayer`] as `Bytes` regions.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct LayerParams {
    pub layer_idx: u32,
    pub hidden_size: u32,
    pub ffn_size: u32,
    pub num_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    /// `0` = full attention; otherwise a key is visible only while
    /// `query.position - key.position < sliding_window`.
    pub sliding_window: u32,
    /// `1/sqrt(head_dim)`, applied to attention scores.
    pub attn_scale: i32,
    /// Applied to the layer's output row; `0` means "not present".
    pub layer_scalar: i32,
    pub norm_eps: i64,
    /// RoPE base as `Acc` (Q32.32) bits — 10 000 on sliding layers, 1 000 000 on
    /// full-attention ones, from `config.json`'s `rope_parameters`.
    pub rope_base: i64,
    /// How many of a head's lanes rotate. The whole head on a sliding layer;
    /// `head_dim × partial_rotary_factor` (a quarter) on a full-attention one,
    /// which is what makes Gemma's "partial rotary" partial.
    pub rotary_dim: u32,
    /// Denominator dimension of the frequency ladder — always `head_dim`, and
    /// *not* the same as `rotary_dim` once the rotation is partial.
    pub rope_freq_base_dim: u32,
    /// Layer this one borrows K/V from, or `-1` when it computes its own.
    ///
    /// Gemma 3n shares KV across its last `num_kv_shared_layers` layers: a
    /// sharing layer projects only Q and attends over the donor's cache, so it
    /// never runs `w_k`/`w_v` at all. `-1` rather than `Option` because the
    /// value crosses a tile boundary and must stay a plain scalar.
    pub kv_donor_layer: i32,
    /// The two donor layers carried uniformly by every stage. Values are model
    /// facts committed with the layer parameters; unused slots are negative
    /// values other than `-1`, which is reserved for a layer's own cache.
    pub donor_a_layer: i32,
    pub donor_b_layer: i32,
    pub norm_input: BytesPage,
    pub norm_post_attn: BytesPage,
    pub norm_pre_ffw: BytesPage,
    pub norm_post_ffw: BytesPage,
    pub q_norm: BytesPage,
    pub k_norm: BytesPage,
    /// Width of this layer's per-layer-embedding vector (`hidden_size_per_layer_input`).
    pub ple_width: u32,
    /// Post-projection norm for the PLE block. **Hidden-width, not `ple_width`** —
    /// it normalises after `per_layer_projection` has mapped back up to hidden.
    pub ple_post_norm: BytesPage,
}

/// One transformer layer's committed external.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct TransformerLayer {
    pub params: LayerParams,
    #[page_size = 196_608]
    pub w_q: Bytes<196_608>,
    #[page_size = 196_608]
    pub w_k: Bytes<196_608>,
    #[page_size = 196_608]
    pub w_v: Bytes<196_608>,
    #[page_size = 196_608]
    pub w_o: Bytes<196_608>,
    #[page_size = 196_608]
    pub w_gate: Bytes<196_608>,
    #[page_size = 196_608]
    pub w_up: Bytes<196_608>,
    #[page_size = 196_608]
    pub w_down: Bytes<196_608>,
    /// PLE gate: hidden -> ple_width.
    #[page_size = 196_608]
    pub ple_input_gate: Bytes<196_608>,
    /// PLE projection back up: ple_width -> hidden.
    #[page_size = 196_608]
    pub ple_layer_projection: Bytes<196_608>,
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
    /// Absolute position in the full sequence — RoPE and window visibility.
    pub position: u32,
    /// Index of this token *within this stage*: `position - start_position`.
    ///
    /// Not the same number as `position` once a stage starts partway through
    /// the sequence, and the distinction is load-bearing. `ple.rows` and pass
    /// 1's `keys` are both built by this stage and hold one entry per token it
    /// processed, so they are indexed by `local`. Indexing them by `position`
    /// works only while a stage starts at zero, and an out-of-range dynamic
    /// index aborts the run with no output rather than committing an error.
    pub local: u32,
    pub token_id: u32,
    pub q: BytesPage,
    pub residual: BytesPage,
}

/// Pass 1's loop-carried position pair.
///
/// Scalar-small, which is what recur `state` requires: it is re-committed on
/// every iteration. Both fields advance in lockstep; they differ by the
/// stage's `start_position`, and carrying both beats recomputing either.
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

/// Per-query attention scores, key-major (`scores[key * num_heads + head]`).
///
/// Key-major so a pass can append one key's scores for every head with a single
/// extend; the softmax reads each head with a stride. Only *visible* keys are
/// appended, matching the reference, which softmaxes over the window rather than
/// masking a full-length row.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct ScoreAccum {
    /// Dense `window_len × num_heads` scores, addressed by the key's position
    /// rather than by the order the sweep reached it.
    ///
    /// Positional because arrival order is no longer trustworthy: the sweeps
    /// visit three separate key lists, and a decode stage's inherited cache is
    /// pruned to the sliding window. Addressing by `key.position -
    /// window_start` also makes the carrier a *fixed* size — the previous
    /// append-shaped version grew by `num_heads` per visible key, so the
    /// `2 · N · |T|` a recur `state` pays over `N` iterations grew with it
    /// (`docs/issues/append-shaped-accumulators.md` §4).
    pub scores: BytesPage,
    /// How many slots have been written. Equal to `window_len` exactly when the
    /// window is complete; anything less means a key the window needs was
    /// missing, which is a committed error rather than a silent mis-attention.
    pub filled: u32,
    /// Number of visible positions for this query: `min(position + 1,
    /// sliding_window)`, or `position + 1` on a full-attention layer. Known
    /// before any key is seen, which is what makes the dense array allocatable
    /// up front.
    pub window_len: u32,
    pub error: String,
}

/// Softmax weights, same layout and count as the scores they came from.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct WeightVec {
    pub weights: BytesPage,
    pub count: u32,
    pub error: String,
}

/// Weighted-sum accumulator: one `i64` per (head, lane).
///
/// `i64` because the reference accumulates the whole weighted sum in `Acc`
/// precision and requantizes exactly once at the end; accumulating in `Act`
/// would round on every key.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct CtxAccum {
    pub acc: BytesPage,
    pub error: String,
}

/// Packs `i64` accumulators little-endian.
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
        return Err(alloc::format!("page length {} is not i64-aligned", bytes.len()));
    }
    Ok(bytes
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes(c.try_into().expect("chunks_exact(8)")))
        .collect())
}

/// Streaming softmax state — one running max, sum and weighted accumulator
/// per head.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct AttnState {
    pub started: bool,
    pub max: BytesPage,
    pub sum: BytesPage,
    pub acc: BytesPage,
    pub error: String,
}

/// A packed vector that may carry a failure from an earlier step.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct PackedVec {
    pub values: BytesPage,
    pub error: String,
}

/// Weightless RMS norm, applied per head to V.
///
/// Gemma 3n normalises values as well as queries and keys — `q_norm`/`k_norm`
/// carry learned weights, this one does not. Omitting it is invisible: the
/// attention still produces a plausible context, just not the model's.
pub fn value_rms_norm(row: &mut [i32], eps: i64) -> core::result::Result<(), String> {
    if row.is_empty() {
        return Err(String::from("value rms norm requires at least one value"));
    }
    if eps < 0 {
        return Err(String::from("value rms norm epsilon must be non-negative"));
    }
    let mut lanes: alloc::vec::Vec<Act> = row.iter().map(|b| Act::from_bits(*b)).collect();
    value_rms_norm_in_place(&mut lanes, Acc::from_bits(eps));
    for (slot, value) in row.iter_mut().zip(lanes.iter()) {
        *slot = value.to_bits();
    }
    Ok(())
}

/// Loop-carried summary of failed rows.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct ErrorSummary {
    pub count: u32,
    pub first: String,
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

pub fn mul(a: i32, b: i32) -> i32 {
    // Canonical multiply: widen, then requantize ties-to-even and saturate.
    // The previous version truncated with `>> FRAC_BITS`, which rounds toward
    // negative infinity and differs from the reference on half the inputs.
    mul_sat(Act::from_bits(a), Act::from_bits(b)).to_bits()
}

pub fn div(a: i32, b: i32) -> i32 {
    if b == 0 {
        return 0;
    }
    // Canonical divide: ties-to-even, not the truncating shift-and-divide this
    // replaces.
    div_act(Act::from_bits(a), Act::from_bits(b)).to_bits()
}

pub fn add_sat(a: i32, b: i32) -> i32 {
    a.saturating_add(b)
}


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

pub fn gelu(x: i32) -> i32 {
    // The model declares `hidden_activation: "gelu_pytorch_tanh"`. What this
    // replaced was a *sigmoid* approximation — a different function, not a
    // different rounding — feeding every MLP and the PLE gate.
    gelu_pytorch_tanh_act(Act::from_bits(x)).to_bits()
}

pub fn scale_row(row: &mut [i32], scalar: i32) {
    for value in row.iter_mut() {
        *value = mul(*value, scalar);
    }
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
        *value = add_sat(*value, *other);
    }
    Ok(())
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
