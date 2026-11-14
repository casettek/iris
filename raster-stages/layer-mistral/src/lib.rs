//! `layer-mistral`: tiles.
//!
//! One `mistral-dense` transformer layer over a range of tokens:
//!
//! ```text
//!   a   = x + W_o · attention(rope(W_q n), rope(W_k n), W_v n)     n = rms_norm(x)
//!   out = a + W_down · (silu(W_gate m) * (W_up m))                  m = rms_norm(a)
//! ```
//!
//! Every layer attends to the whole prefix, takes no per-layer input and
//! borrows no other layer's cache. Weight matrices are paged byte regions and
//! each matvec is a recur over `.pages`. Attention is two passes: pass 1 drafts
//! K/V, pass 2 attends. The layer loop lives in the chain manifest.
//!
//! The steps have the shape of `prefill-range`'s, which is Gemma's layer. The
//! arithmetic that is this family's own is `det-num`'s `mistral` module.
//!
//! `#![no_std]` so the same tiles compile into RISC0 replay guests.

#![no_std]

extern crate alloc;

use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use det_num::mistral::{rotate_consecutive_pairs, scaled_score, silu};
use det_num::ops::{attention_softmax, mac_bits, requantize};
use det_num::{Acc, Act};
use raster::prelude::*;

pub mod input;

use input::*;

/// Guards scalar shapes, the rotation table and norm-vector widths. Matrix
/// byte lengths are checked separately against the `Bytes` regions.
#[tile(kind = iter, description = "Validate this layer's declared shapes")]
pub fn validate_layer_params(params: LayerParams) -> Result<LayerParams> {
    check_layer_params(&params)?;
    Ok(params)
}

/// What [`validate_layer_params`] checks, for the native path too.
pub fn check_layer_params(params: &LayerParams) -> core::result::Result<(), String> {
    let hidden = params.hidden_size as usize;
    let heads = params.num_heads as usize;
    let kv_heads = params.num_kv_heads as usize;
    let head_dim = params.head_dim as usize;
    if hidden == 0 || params.ffn_size == 0 {
        return Err(String::from(
            "transformer layer requires non-zero hidden_size and ffn_size",
        ));
    }
    if heads == 0 || kv_heads == 0 || head_dim == 0 {
        return Err(String::from(
            "transformer layer requires non-zero num_heads, num_kv_heads and head_dim",
        ));
    }
    if heads % kv_heads != 0 {
        return Err(alloc::format!(
            "layer {}: {heads} query heads do not group evenly over {kv_heads} kv heads",
            params.layer_idx
        ));
    }
    if head_dim % 2 != 0 {
        return Err(alloc::format!(
            "layer {}: head_dim {head_dim} must be even",
            params.layer_idx
        ));
    }
    let frequencies = unpack_i64s(&params.rope_inverse_frequencies)?;
    if frequencies.len() != head_dim / 2 {
        return Err(alloc::format!(
            "layer {}: {} rotation frequencies for head_dim {head_dim}",
            params.layer_idx,
            frequencies.len()
        ));
    }
    if frequencies
        .iter()
        .any(|frequency| !(0..=1_i64 << 32).contains(frequency))
    {
        return Err(alloc::format!(
            "layer {}: a rotation frequency is outside 0..=1",
            params.layer_idx
        ));
    }
    if params.norm_eps < 0 {
        return Err(String::from("rms-norm epsilon must be non-negative"));
    }
    for (name, page) in [
        ("norm_attention", &params.norm_attention),
        ("norm_feed_forward", &params.norm_feed_forward),
    ] {
        if page.len() != hidden * 4 {
            return Err(alloc::format!(
                "layer {} {name} has {} bytes, expected {} ({hidden} values)",
                params.layer_idx,
                page.len(),
                hidden * 4
            ));
        }
    }
    Ok(())
}

#[tile(kind = iter, description = "Query projection width")]
pub fn q_out_len(params: LayerParams) -> u32 {
    params.num_heads * params.head_dim
}

#[tile(kind = iter, description = "Key/value projection width")]
pub fn kv_out_len(params: LayerParams) -> u32 {
    params.num_kv_heads * params.head_dim
}

#[tile(kind = iter, description = "Reject a Q8_0 weight region of the wrong byte length")]
pub fn assert_matrix_bytes(byte_len: u64, out_len: u32, in_width: u32) -> Result<u32> {
    let row = det_num::q8::row_bytes(in_width as usize)
        .ok_or_else(|| alloc::format!("matrix width {in_width} is not whole Q8_0 blocks"))?;
    let expected = out_len as u64 * row as u64;
    if byte_len == expected {
        Ok(0)
    } else {
        Err(alloc::format!(
            "matrix has {byte_len} bytes, expected {expected} ({out_len} x {in_width} Q8_0)"
        ))
    }
}

/// Opens pass 1's cursor at the stage's own starting position. `local` counts
/// within this stage and always starts at zero.
#[tile(kind = iter, description = "Open the token cursor at the stage's start position")]
pub fn begin_cursor(start_position: u32) -> TokenCursor {
    TokenCursor {
        position: start_position,
        local: 0,
    }
}

#[tile(kind = iter, description = "Advance the token cursor")]
pub fn advance_cursor(cursor: TokenCursor) -> TokenCursor {
    TokenCursor {
        position: cursor.position.saturating_add(1),
        local: cursor.local.saturating_add(1),
    }
}

/// Stamps the stage's context and start position onto its output before
/// anything is appended: both are set once.
#[tile(kind = iter, description = "Open this stage's activation output")]
pub fn begin_layer_output(start_position: u32, model: ModelContext) -> Draft<ActivationSequence> {
    let mut output = Draft::<ActivationSequence>::new();
    output.model().set(model);
    output.start_position().set(start_position);
    output
}

/// Copies one chunk of the inherited cache into this stage's published cache.
/// Every layer attends to the whole prefix, so every key is kept.
#[tile(
    kind = recur,
    description = "Carry one chunk of the inherited K/V cache into this stage's output",
    estimated_cycles = 400_000
)]
pub fn carry_cached_key(
    input: RecurInput<Block<KeyRow>>,
    output: RecurOutput<ActivationSequence>,
) -> RecurOutput<ActivationSequence> {
    let mut output = output;
    for key in input.into_value().into_vec() {
        output.kv().push(key);
    }
    output
}

#[tile(kind = iter, description = "Zero a matvec accumulator")]
pub fn zero_accum(out_len: u32) -> ProjAccum {
    ProjAccum {
        values: pack_i32s(&vec![0; out_len as usize]),
        error: String::new(),
    }
}

#[tile(
    kind = recur,
    description = "MAC one Q8_0 weight page into the output accumulator",
    estimated_cycles = 800_000
)]
pub fn mac_q8_weight_page(
    input: RecurInput<BytesPage>,
    state: RecurState<ProjAccum>,
    row: BytesPage,
    in_width: u32,
) -> RecurState<ProjAccum> {
    let mut state = state;
    if !state.error.is_empty() {
        return state;
    }
    let page = input.into_value();
    if let Err(error) = mac_q8_page(&mut state, &page, &row, in_width, core::convert::identity) {
        state.error = error;
    }
    state
}

/// The gate projection of the feed-forward block: one weight page's rows, each
/// put through SiLU as it is written. The activation is spread over the
/// matvec's pages this way. As one tile over the whole vector it cost eight
/// times the layer's next heaviest step.
#[tile(
    kind = recur,
    description = "MAC one Q8_0 gate weight page and apply SiLU to its rows",
    estimated_cycles = 5_500_000
)]
pub fn mac_q8_gate_page(
    input: RecurInput<BytesPage>,
    state: RecurState<ProjAccum>,
    row: BytesPage,
    in_width: u32,
) -> RecurState<ProjAccum> {
    let mut state = state;
    if !state.error.is_empty() {
        return state;
    }
    let page = input.into_value();
    if let Err(error) = mac_q8_page(&mut state, &page, &row, in_width, silu) {
        state.error = error;
    }
    state
}

/// Writes the output rows one weight page holds, each as `finish` of its dot
/// product with `row`.
fn mac_q8_page(
    state: &mut ProjAccum,
    page: &BytesPage,
    row: &BytesPage,
    in_width: u32,
    finish: fn(i32) -> i32,
) -> core::result::Result<(), String> {
    let row = unpack_i32s(row)?;
    if row.len() != in_width as usize {
        return Err(alloc::format!(
            "activation has {} values, expected in_width {in_width}",
            row.len()
        ));
    }
    let stride = det_num::q8::row_bytes(row.len())
        .ok_or_else(|| alloc::format!("matvec in_width {in_width} is not whole Q8_0 blocks"))?;
    if page.len() % stride != 0 {
        return Err(String::from(
            "weight page does not contain whole output rows",
        ));
    }
    if page.offset() % stride as u64 != 0 {
        return Err(String::from("weight page offset is not row-aligned"));
    }
    let mut out = unpack_i32s(&state.values)?;
    let start_row = (page.offset() / stride as u64) as usize;
    for (local, weights) in page.as_slice().chunks_exact(stride).enumerate() {
        let out_idx = start_row + local;
        if out_idx >= out.len() {
            return Err(String::from("weight page writes past the accumulator"));
        }
        out[out_idx] = finish(det_num::q8::dot_row(&row, weights)?);
    }
    state.values = pack_i32s(&out);
    Ok(())
}

// ---------------------------------------------------------------------------
// Pass 1 — projections
// ---------------------------------------------------------------------------

/// The RMS-normalised page on a Q/K/V prep. `select!(BytesPage, …)` is only
/// for indexing a `Bytes` region.
#[tile(kind = iter, description = "Normalised page of a Q/K/V prep")]
pub fn qkv_normed(prep: QkvPrep) -> BytesPage {
    prep.normed
}

/// The packed values of a vector that may carry an earlier error.
#[tile(kind = iter, description = "Values page of a packed vector")]
pub fn packed_values(vec: PackedVec) -> BytesPage {
    vec.values
}

/// RMS-normalise one activation row for the Q/K/V matvecs.
#[tile(kind = iter, description = "RMS-normalise one token for Q/K/V")]
pub fn begin_qkv(row: ActivationRow, params: LayerParams) -> QkvPrep {
    match begin_qkv_inner(&row, &params) {
        Ok((residual, normed)) => QkvPrep {
            token_id: row.token_id,
            residual: pack_i32s(&residual),
            normed: pack_i32s(&normed),
            error: String::new(),
        },
        Err(error) => QkvPrep {
            token_id: row.token_id,
            residual: pack_i32s(&[]),
            normed: pack_i32s(&vec![0; params.hidden_size as usize]),
            error,
        },
    }
}

fn begin_qkv_inner(
    row: &ActivationRow,
    params: &LayerParams,
) -> core::result::Result<(Vec<i32>, Vec<i32>), String> {
    let residual = unpack_i32s(&row.values)?;
    let hidden = params.hidden_size as usize;
    if residual.len() != hidden {
        return Err(alloc::format!(
            "activation row has {} values, expected hidden_size {hidden}",
            residual.len()
        ));
    }
    let mut normed = residual.clone();
    rms_norm(
        &mut normed,
        &unpack_i32s(&params.norm_attention)?,
        params.norm_eps,
    )?;
    Ok((residual, normed))
}

/// Rotates the query and key heads to the token's position, then appends the
/// key and query rows.
#[tile(kind = iter, description = "Finish Q/K/V and append key and query rows")]
pub fn finish_qkv(
    output: Draft<KvSequence>,
    cursor: TokenCursor,
    prep: QkvPrep,
    q: ProjAccum,
    k: ProjAccum,
    v: ProjAccum,
    params: LayerParams,
) -> Draft<KvSequence> {
    let mut output = output;
    let position = cursor.position;
    match finish_qkv_inner(&cursor, &prep, &q, &k, &v, &params) {
        Ok((query, key)) => {
            output.queries().push(query);
            output.keys().push(key);
        }
        Err(error) => output
            .errors()
            .push(alloc::format!("token at position {position}: {error}")),
    }
    output
}

fn finish_qkv_inner(
    cursor: &TokenCursor,
    prep: &QkvPrep,
    q: &ProjAccum,
    k: &ProjAccum,
    v: &ProjAccum,
    params: &LayerParams,
) -> core::result::Result<(QueryRow, KeyRow), String> {
    for error in [&prep.error, &q.error, &k.error, &v.error] {
        if !error.is_empty() {
            return Err(error.clone());
        }
    }
    let position = cursor.position;
    if position >= params.max_positions {
        return Err(alloc::format!(
            "the position is past the {} this model is built for",
            params.max_positions
        ));
    }
    let mut q = unpack_i32s(&q.values)?;
    let mut k = unpack_i32s(&k.values)?;
    let head_dim = params.head_dim as usize;
    let frequencies = unpack_i64s(&params.rope_inverse_frequencies)?;
    // Values are not rotated: the rotation is of the query and key space.
    for head in q.chunks_mut(head_dim).chain(k.chunks_mut(head_dim)) {
        rotate_consecutive_pairs(head, &frequencies, position);
    }
    Ok((
        QueryRow {
            position,
            local: cursor.local,
            token_id: prep.token_id,
            q: pack_i32s(&q),
            residual: prep.residual.clone(),
        },
        KeyRow {
            position,
            k: pack_i32s(&k),
            v: v.values.clone(),
        },
    ))
}

// ---------------------------------------------------------------------------
// Pass 2 — attention and the feed-forward block
//
// Per (head, query): every visible score, then one softmax over them, then a
// weighted sum accumulated in `Acc` and requantized once. A streaming softmax
// would round differently at every step, so there are two sweeps of the keys.
// ---------------------------------------------------------------------------

/// Materializes one query so the steps that follow can each read it.
#[tile(kind = iter, description = "Open one token's attention query")]
pub fn begin_attention(row: QueryRow) -> QueryRow {
    row
}

#[tile(kind = iter, description = "Zero the dense attention scores")]
pub fn zero_scores(query: QueryRow, params: LayerParams) -> ScoreAccum {
    let count = query.position.saturating_add(1);
    ScoreAccum {
        scores: pack_i32s(&vec![0; count as usize * params.num_heads as usize]),
        filled: 0,
        count,
        error: String::new(),
    }
}

/// Scores one chunk of keys against the query, for every head. A key later
/// than the query is not visible to it.
#[tile(
    kind = recur,
    description = "Score one chunk of keys against the query, for every head",
    estimated_cycles = 1_900_000
)]
pub fn score_key(
    input: RecurInput<Block<KeyRow>>,
    state: RecurState<ScoreAccum>,
    query: QueryRow,
    params: LayerParams,
) -> RecurState<ScoreAccum> {
    let mut state = state;
    if !state.error.is_empty() {
        return state;
    }
    let keys = input.into_value();
    if let Err(error) = push_scores(&mut state, &query, keys.as_slice(), &params) {
        state.error = error;
    }
    state
}

/// The lanes of the key head a query head attends to.
fn kv_lanes(head: usize, params: &LayerParams) -> core::ops::Range<usize> {
    let head_dim = params.head_dim as usize;
    let group = (params.num_heads / params.num_kv_heads.max(1)).max(1) as usize;
    (head / group) * head_dim..(head / group + 1) * head_dim
}

fn push_scores(
    state: &mut ScoreAccum,
    query: &QueryRow,
    keys: &[KeyRow],
    params: &LayerParams,
) -> core::result::Result<(), String> {
    let head_dim = params.head_dim as usize;
    let heads = params.num_heads as usize;
    let kv_width = params.num_kv_heads as usize * head_dim;
    let q = unpack_i32s(&query.q)?;
    if q.len() != heads * head_dim {
        return Err(String::from("query is not num_heads * head_dim wide"));
    }
    let mut scores = unpack_i32s(&state.scores)?;
    let mut filled = state.filled;
    for key in keys {
        if key.position > query.position {
            continue;
        }
        // Addressed by position, not by arrival: the keys come on two lists.
        let slot = key.position as usize;
        if slot >= state.count as usize {
            return Err(alloc::format!(
                "key at position {} falls outside {} positions",
                key.position,
                state.count
            ));
        }
        let k = unpack_i32s(&key.k)?;
        if k.len() != kv_width {
            return Err(String::from("key is not num_kv_heads * head_dim wide"));
        }
        for head in 0..heads {
            scores[slot * heads + head] = scaled_score(
                &q[head * head_dim..(head + 1) * head_dim],
                &k[kv_lanes(head, params)],
                params.attention_scale_q32,
            );
        }
        filled += 1;
    }
    state.scores = pack_i32s(&scores);
    state.filled = filled;
    Ok(())
}

/// One softmax per head over that head's scores.
#[tile(kind = iter, description = "Softmax the per-head score lists")]
pub fn softmax_scores(state: ScoreAccum, params: LayerParams) -> WeightVec {
    match softmax_scores_inner(&state, &params) {
        Ok(values) => WeightVec {
            weights: pack_i32s(&values),
            count: state.count,
            error: String::new(),
        },
        Err(error) => WeightVec {
            weights: pack_i32s(&[]),
            count: 0,
            error,
        },
    }
}

fn softmax_scores_inner(
    state: &ScoreAccum,
    params: &LayerParams,
) -> core::result::Result<Vec<i32>, String> {
    if !state.error.is_empty() {
        return Err(state.error.clone());
    }
    // Every position up to the query's must have been scored. A short count
    // means a key never reached the sweep; softmaxing over the zeros left in
    // its place would give a plausible distribution and the wrong token.
    if state.filled != state.count {
        return Err(alloc::format!(
            "attention has {} of {} positions scored",
            state.filled,
            state.count
        ));
    }
    let heads = params.num_heads as usize;
    let count = state.count as usize;
    let scores = unpack_i32s(&state.scores)?;
    if count == 0 || scores.len() != heads * count {
        return Err(alloc::format!(
            "score list has {} entries, expected {}",
            scores.len(),
            heads * count
        ));
    }
    let mut out = scores.clone();
    let mut logits: Vec<Act> = Vec::with_capacity(count);
    for head in 0..heads {
        logits.clear();
        for key in 0..count {
            logits.push(Act::from_bits(scores[key * heads + head]));
        }
        let weights = attention_softmax(&logits);
        for (key, weight) in weights.iter().enumerate() {
            out[key * heads + head] = weight.to_bits();
        }
    }
    Ok(out)
}

#[tile(kind = iter, description = "Zero a weighted-sum accumulator")]
pub fn zero_context(params: LayerParams) -> CtxAccum {
    CtxAccum {
        acc: pack_i64s(&vec![0; (params.num_heads * params.head_dim) as usize]),
        error: String::new(),
    }
}

/// Folds one chunk of visible keys' value rows into the context, weighted.
#[tile(
    kind = recur,
    description = "Accumulate one chunk of weighted value rows",
    estimated_cycles = 1_900_000
)]
pub fn accumulate_context(
    input: RecurInput<Block<KeyRow>>,
    state: RecurState<CtxAccum>,
    query: QueryRow,
    weights: WeightVec,
    params: LayerParams,
) -> RecurState<CtxAccum> {
    let mut state = state;
    if !state.error.is_empty() {
        return state;
    }
    let keys = input.into_value();
    if let Err(error) = accumulate_weighted(&mut state, keys.as_slice(), &weights, &query, &params)
    {
        state.error = error;
    }
    state
}

fn accumulate_weighted(
    state: &mut CtxAccum,
    keys: &[KeyRow],
    weights: &WeightVec,
    query: &QueryRow,
    params: &LayerParams,
) -> core::result::Result<(), String> {
    if !weights.error.is_empty() {
        return Err(weights.error.clone());
    }
    let head_dim = params.head_dim as usize;
    let heads = params.num_heads as usize;
    let kv_width = params.num_kv_heads as usize * head_dim;
    let w = unpack_i32s(&weights.weights)?;
    let mut acc = unpack_i64s(&state.acc)?;
    if acc.len() != heads * head_dim {
        return Err(String::from("context is not num_heads * head_dim wide"));
    }
    for key in keys {
        if key.position > query.position {
            continue;
        }
        let slot = key.position as usize;
        if slot >= weights.count as usize || (slot + 1) * heads > w.len() {
            return Err(String::from("visible key falls outside the weight list"));
        }
        let v = unpack_i32s(&key.v)?;
        if v.len() != kv_width {
            return Err(String::from("value is not num_kv_heads * head_dim wide"));
        }
        for head in 0..heads {
            let weight = w[slot * heads + head];
            for (dst, value) in acc[head * head_dim..(head + 1) * head_dim]
                .iter_mut()
                .zip(&v[kv_lanes(head, params)])
            {
                *dst = mac_bits(*dst, weight, *value);
            }
        }
    }
    state.acc = pack_i64s(&acc);
    Ok(())
}

/// Requantizes the accumulated context, once.
#[tile(kind = iter, description = "Requantize the attention context")]
pub fn finish_context(state: CtxAccum) -> PackedVec {
    if !state.error.is_empty() {
        return PackedVec {
            values: pack_i32s(&[]),
            error: state.error,
        };
    }
    match unpack_i64s(&state.acc) {
        Ok(acc) => PackedVec {
            values: pack_i32s(
                &acc.iter()
                    .map(|bits| requantize(Acc::from_bits(*bits)).to_bits())
                    .collect::<Vec<i32>>(),
            ),
            error: String::new(),
        },
        Err(error) => PackedVec {
            values: pack_i32s(&[]),
            error,
        },
    }
}

/// The residual add after the output projection.
#[tile(kind = iter, description = "Add residual after the output projection")]
pub fn attn_residual(query: QueryRow, context: PackedVec, attn_proj: ProjAccum) -> PackedVec {
    let attended = (|| {
        for error in [&context.error, &attn_proj.error] {
            if !error.is_empty() {
                return Err(error.clone());
            }
        }
        let mut attended = unpack_i32s(&attn_proj.values)?;
        add_row(&mut attended, &unpack_i32s(&query.residual)?)?;
        Ok(attended)
    })();
    match attended {
        Ok(values) => PackedVec {
            values: pack_i32s(&values),
            error: String::new(),
        },
        Err(error) => PackedVec {
            values: pack_i32s(&[]),
            error: alloc::format!("token at position {}: {error}", query.position),
        },
    }
}

/// The norm before the feed-forward block: the vector the gate and up
/// matvecs consume.
#[tile(kind = iter, description = "RMS-normalise before the feed-forward block")]
pub fn pre_ff_norm(attn: PackedVec, params: LayerParams) -> PackedVec {
    let normed = (|| {
        if !attn.error.is_empty() {
            return Err(attn.error.clone());
        }
        let mut normed = unpack_i32s(&attn.values)?;
        rms_norm(
            &mut normed,
            &unpack_i32s(&params.norm_feed_forward)?,
            params.norm_eps,
        )?;
        Ok(normed)
    })();
    match normed {
        Ok(values) => PackedVec {
            values: pack_i32s(&values),
            error: String::new(),
        },
        Err(error) => PackedVec {
            values: pack_i32s(&vec![0; params.hidden_size as usize]),
            error,
        },
    }
}

/// `silu(gate) ⊙ up`, where `gate` went through SiLU as it was projected
/// ([`mac_q8_gate_page`]).
#[tile(kind = iter, description = "Elementwise product of the gated and up projections")]
pub fn mul_gated(gate: ProjAccum, up: ProjAccum) -> PackedVec {
    let gated = (|| {
        for error in [&gate.error, &up.error] {
            if !error.is_empty() {
                return Err(error.clone());
            }
        }
        let mut gate = unpack_i32s(&gate.values)?;
        let up = unpack_i32s(&up.values)?;
        if gate.len() != up.len() {
            return Err(alloc::format!(
                "gate/up width mismatch: {} vs {}",
                gate.len(),
                up.len()
            ));
        }
        for (value, up_value) in gate.iter_mut().zip(&up) {
            *value = mul(*value, *up_value);
        }
        Ok(gate)
    })();
    match gated {
        Ok(values) => PackedVec {
            values: pack_i32s(&values),
            error: String::new(),
        },
        Err(error) => PackedVec {
            values: pack_i32s(&[]),
            error,
        },
    }
}

/// The residual add after the feed-forward block, which closes the layer:
/// appends the token's output row and its own cache entry.
#[tile(kind = iter, description = "Add the feed-forward residual and close the layer")]
pub fn finish_layer(
    output: Draft<ActivationSequence>,
    query: QueryRow,
    residual: PackedVec,
    ff_in: PackedVec,
    ff: ProjAccum,
    own_key: KeyRow,
) -> Draft<ActivationSequence> {
    let mut output = output;
    let row = (|| {
        for error in [&residual.error, &ff_in.error, &ff.error] {
            if !error.is_empty() {
                return Err(error.clone());
            }
        }
        let mut out = unpack_i32s(&ff.values)?;
        add_row(&mut out, &unpack_i32s(&residual.values)?)?;
        Ok(out)
    })();
    match row {
        Ok(values) => output.rows().push(ActivationRow {
            token_id: query.token_id,
            values: pack_i32s(&values),
        }),
        Err(error) => output.errors().push(alloc::format!(
            "token at position {}: {error}",
            query.position
        )),
    }
    output.kv().push(own_key);
    output
}

// ---------------------------------------------------------------------------
// Failure accounting
// ---------------------------------------------------------------------------

/// The failure fold's opening state: a stored seed, so the fold's state chain
/// starts at a committed value.
#[tile(kind = iter, description = "Open the failure summary with nothing counted")]
pub fn begin_error_summary() -> ErrorSummary {
    ErrorSummary {
        count: 0,
        first: String::new(),
    }
}

#[tile(kind = recur, description = "Summarise the rows this layer could not produce")]
pub fn summarise_errors(
    input: RecurInput<String>,
    state: RecurState<ErrorSummary>,
) -> RecurState<ErrorSummary> {
    let mut state = state;
    let message = input.into_value();
    if state.count == 0 {
        state.first = message;
    }
    state.count += 1;
    state
}

#[tile(kind = iter, description = "Reject an incomplete layer output")]
pub fn assert_layer_complete(
    projection_errors: u32,
    first_projection_error: String,
    attention_errors: u32,
    first_attention_error: String,
) -> Result<u32> {
    if projection_errors > 0 {
        return Err(alloc::format!(
            "{projection_errors} row(s) failed projection; first: {first_projection_error}"
        ));
    }
    if attention_errors > 0 {
        return Err(alloc::format!(
            "{attention_errors} row(s) failed attention/MLP; first: {first_attention_error}"
        ));
    }
    Ok(0)
}

// ---------------------------------------------------------------------------
// The stage's place in the chain
// ---------------------------------------------------------------------------

#[tile(kind = iter, description = "Check model and producer context of a transformer layer")]
pub fn layer_model_context(
    model: ModelContext,
    prior: ModelContext,
    params: LayerParams,
) -> ModelContext {
    model.plain_layer(params.layer_idx, &prior)
}

pub mod identity;
pub use identity::*;

#[tile(kind = recur)]
pub fn binding_count_rows(
    input: RecurInput<Block<ActivationRow>>,
    state: RecurState<u64>,
) -> RecurState<u64> {
    RecurState::new(*state + input.into_value().len() as u64)
}

#[tile(kind = iter)]
pub fn binding_validate_rows(mut model: ModelContext, start: u32, count: u64) -> ModelContext {
    model.validate_rows(start, count);
    model
}

/// Passes the range's rows on only when the range is the one this stage's
/// context names.
#[tile(kind = recur)]
pub fn binding_copy_rows(
    input: RecurInput<Block<ActivationRow>>,
    output: RecurOutput<ActivationSequence>,
    model: ModelContext,
) -> RecurOutput<ActivationSequence> {
    let mut output = output;
    if model.execution_fault == 0 || model.execution.as_slice().is_empty() {
        for row in input.into_value() {
            output.rows().push(row);
        }
    }
    output
}

#[tile(kind = iter)]
pub fn begin_cache_coverage(context: ModelContext) -> CacheCoverage {
    cache_coverage(&context)
}

/// The positions the inherited cache must hold: every one before the range.
pub fn cache_coverage(context: &ModelContext) -> CacheCoverage {
    if context.is_origin() || context.producer == model_identity::context::EMBEDDING {
        return CacheCoverage {
            next: 0,
            end: 0,
            fault: 0,
        };
    }
    match context.execution_context() {
        Some(e) => CacheCoverage {
            next: 0,
            end: e.end.position,
            fault: 0,
        },
        None => CacheCoverage {
            next: 0,
            end: 0,
            fault: 1,
        },
    }
}

pub fn consume_cache_position(state: &mut CacheCoverage, position: u32) {
    if position != state.next || state.next >= state.end {
        state.fault = 1;
    }
    state.next = state.next.saturating_add(1);
}

#[tile(kind = recur)]
pub fn check_cache_positions(
    input: RecurInput<Block<KeyRow>>,
    state: RecurState<CacheCoverage>,
) -> RecurState<CacheCoverage> {
    let mut state = (*state).clone();
    for key in input.into_value() {
        consume_cache_position(&mut state, key.position);
    }
    RecurState::new(state)
}

#[tile(kind = iter)]
pub fn finish_cache_coverage(model: ModelContext, coverage: CacheCoverage) -> ModelContext {
    validate_cache_coverage(model, coverage)
}

pub fn validate_cache_coverage(mut model: ModelContext, coverage: CacheCoverage) -> ModelContext {
    if !model.execution.as_slice().is_empty() {
        model.require_execution(coverage.fault == 0 && coverage.next == coverage.end);
    }
    model
}
