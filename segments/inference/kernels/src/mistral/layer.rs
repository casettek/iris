//! One `mistral-dense` layer over a range of tokens.
//!
//! ```text
//!   a   = x + W_o · attention(rope(W_q n), rope(W_k n), W_v n)     n = rms_norm(x)
//!   out = a + W_down · (silu(W_gate m) * (W_up m))                  m = rms_norm(a)
//! ```
//!
//! Every layer attends to the whole prefix. A query head's score with a key
//! head is their product times `1 / sqrt(head_dim)`.

use anyhow::{bail, ensure, Result};
use det_num::ops::attention_softmax;
use det_num::Act;
use rayon::prelude::*;

use super::num::{rotate_consecutive_pairs, scaled_score, silu};
use crate::tensor::{
    linear_slab_from_source, mac_weighted_value, pack_i32_page, requantize_acc, unpack_page_i32s,
    MatrixSource, Slab,
};
use ::prefill_range::input::{add_row, mul, rms_norm, ActivationRow, ActivationSequence, KeyRow};

/// What a layer computes with besides its matrices.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerParams {
    pub layer_idx: u32,
    pub hidden_size: u32,
    pub ffn_size: u32,
    pub num_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub norm_eps: i64,
    pub attention_scale_q32: u32,
    /// One Q32.32 angle per position for each consecutive pair of a head.
    pub rope_inverse_frequencies: Vec<i64>,
    /// Positions from here on are refused.
    pub max_positions: u32,
    pub norm_attention: Vec<i32>,
    pub norm_feed_forward: Vec<i32>,
}

/// One layer's matrices, wherever their bytes live.
pub struct LayerMatrices<'a, Q: MatrixSource> {
    pub w_q: &'a Q,
    pub w_k: &'a Q,
    pub w_v: &'a Q,
    pub w_o: &'a Q,
    pub w_gate: &'a Q,
    pub w_up: &'a Q,
    pub w_down: &'a Q,
}

pub fn validate_layer_params(params: &LayerParams) -> Result<()> {
    let hidden = params.hidden_size as usize;
    let heads = params.num_heads as usize;
    let kv_heads = params.num_kv_heads as usize;
    let head_dim = params.head_dim as usize;
    ensure!(
        hidden > 0 && params.ffn_size > 0 && heads > 0 && kv_heads > 0 && head_dim > 0,
        "layer {} has a zero dimension",
        params.layer_idx
    );
    ensure!(
        heads % kv_heads == 0,
        "layer {}: {heads} query heads do not group evenly over {kv_heads} kv heads",
        params.layer_idx
    );
    ensure!(
        head_dim % 2 == 0 && params.rope_inverse_frequencies.len() == head_dim / 2,
        "layer {}: {} rotation frequencies for head_dim {head_dim}",
        params.layer_idx,
        params.rope_inverse_frequencies.len()
    );
    ensure!(
        params
            .rope_inverse_frequencies
            .iter()
            .all(|frequency| (0..=1_i64 << 32).contains(frequency)),
        "layer {}: a rotation frequency is outside 0..=1",
        params.layer_idx
    );
    ensure!(
        params.norm_eps >= 0,
        "rms-norm epsilon must be non-negative"
    );
    ensure!(
        params.norm_attention.len() == hidden && params.norm_feed_forward.len() == hidden,
        "layer {}: norm weights are not hidden_size wide",
        params.layer_idx
    );
    Ok(())
}

/// Tokens taken through the feed-forward block together. Bounds the working
/// memory of a pass over a long prompt.
const CHUNK_ROWS: usize = 256;

/// The keys and values of positions `0..`, one entry per position.
struct Keys {
    k: Vec<Vec<i32>>,
    v: Vec<Vec<i32>>,
}

/// The activation rows and keys one layer produces for one range, or `None`
/// for a range with no rows. `prior_keys` are the keys of every earlier
/// position, in order.
pub fn layer_rows_and_keys<Q: MatrixSource>(
    activations: &ActivationSequence,
    prior_keys: &[KeyRow],
    weights: &LayerMatrices<'_, Q>,
    params: &LayerParams,
) -> Result<Option<(Vec<ActivationRow>, Vec<KeyRow>)>> {
    validate_layer_params(params)?;
    let hidden = params.hidden_size as usize;
    let count = activations.rows.len();
    if count == 0 {
        return Ok(None);
    }
    let start = activations.start_position;
    let end = start as u64 + count as u64;
    ensure!(
        end <= u64::from(params.max_positions),
        "layer {}: position {} is past the {} this model is built for",
        params.layer_idx,
        end - 1,
        params.max_positions
    );

    let mut token_ids = Vec::with_capacity(count);
    let mut rows = Vec::with_capacity(count);
    for (index, row) in activations.rows.iter().enumerate() {
        let values = unpack_page_i32s(&row.values)?;
        ensure!(
            values.len() == hidden,
            "activation row {index} has {} values, expected hidden_size {hidden}",
            values.len()
        );
        token_ids.push(row.token_id);
        rows.push(values);
    }
    let input = Slab::from_rows(rows)?;

    // Queries, keys and values of this range.
    let normed = map_rows(count, |index| {
        let mut row = input.row(index).to_vec();
        rms_norm(&mut row, &params.norm_attention, params.norm_eps).map_err(anyhow::Error::msg)?;
        Ok(row)
    })?;
    let mut q = linear_slab_from_source(&normed, weights.w_q)?;
    let mut k = linear_slab_from_source(&normed, weights.w_k)?;
    let v = linear_slab_from_source(&normed, weights.w_v)?;
    let heads = params.num_heads as usize;
    let kv_heads = params.num_kv_heads as usize;
    let head_dim = params.head_dim as usize;
    ensure!(
        q.cols() == heads * head_dim && k.cols() == kv_heads * head_dim && v.cols() == k.cols(),
        "layer {}: Q/K/V widths {} {} {} do not match its heads",
        params.layer_idx,
        q.cols(),
        k.cols(),
        v.cols()
    );
    for slab in [&mut q, &mut k] {
        let cols = slab.cols();
        let rotate = |(index, row): (usize, &mut [i32])| {
            for head in row.chunks_mut(head_dim) {
                rotate_consecutive_pairs(
                    head,
                    &params.rope_inverse_frequencies,
                    start + index as u32,
                );
            }
        };
        if crate::tensor::use_parallel() {
            slab.as_flat_mut()
                .par_chunks_mut(cols)
                .enumerate()
                .for_each(rotate);
        } else {
            slab.as_flat_mut()
                .chunks_mut(cols)
                .enumerate()
                .for_each(rotate);
        }
    }

    // Every key a query of this range may see: the prefix's, then its own.
    let mut keys = Keys {
        k: Vec::with_capacity(prior_keys.len() + count),
        v: Vec::with_capacity(prior_keys.len() + count),
    };
    for (index, key) in prior_keys.iter().enumerate() {
        ensure!(
            key.position as usize == index,
            "layer {}: key {index} is for position {}",
            params.layer_idx,
            key.position
        );
        keys.k.push(unpack_page_i32s(&key.k)?);
        keys.v.push(unpack_page_i32s(&key.v)?);
    }
    ensure!(
        keys.k.len() == start as usize,
        "layer {}: {} keys before position {start}",
        params.layer_idx,
        keys.k.len()
    );
    let mut own_keys = Vec::with_capacity(count);
    for index in 0..count {
        own_keys.push(KeyRow {
            position: start + index as u32,
            k: pack_i32_page(k.row(index)),
            v: pack_i32_page(v.row(index)),
        });
        keys.k.push(k.row(index).to_vec());
        keys.v.push(v.row(index).to_vec());
    }
    for (index, (key, value)) in keys.k.iter().zip(&keys.v).enumerate() {
        if key.len() != kv_heads * head_dim || value.len() != kv_heads * head_dim {
            bail!(
                "layer {}: key at position {index} is {} and {} wide",
                params.layer_idx,
                key.len(),
                value.len()
            );
        }
    }

    let mut output = Vec::with_capacity(count);
    for chunk_start in (0..count).step_by(CHUNK_ROWS) {
        let chunk = chunk_start..(chunk_start + CHUNK_ROWS).min(count);
        let size = chunk.len();
        let contexts = map_rows(size, |offset| {
            let index = chunk.start + offset;
            attention_context(q.row(index), start as usize + index, &keys, params)
        })?;
        let projected = linear_slab_from_source(&contexts, weights.w_o)?;
        let attended = map_rows(size, |offset| {
            let mut row = projected.row(offset).to_vec();
            add_row(&mut row, input.row(chunk.start + offset)).map_err(anyhow::Error::msg)?;
            Ok(row)
        })?;
        let ff_in = map_rows(size, |offset| {
            let mut row = attended.row(offset).to_vec();
            rms_norm(&mut row, &params.norm_feed_forward, params.norm_eps)
                .map_err(anyhow::Error::msg)?;
            Ok(row)
        })?;
        let gate = linear_slab_from_source(&ff_in, weights.w_gate)?;
        let up = linear_slab_from_source(&ff_in, weights.w_up)?;
        let gated = map_rows(size, |offset| {
            Ok(gate
                .row(offset)
                .iter()
                .zip(up.row(offset))
                .map(|(gate, up)| mul(silu(*gate), *up))
                .collect())
        })?;
        let ff = linear_slab_from_source(&gated, weights.w_down)?;
        for offset in 0..size {
            let mut values = ff.row(offset).to_vec();
            add_row(&mut values, attended.row(offset)).map_err(anyhow::Error::msg)?;
            output.push(ActivationRow {
                token_id: token_ids[chunk.start + offset],
                values: pack_i32_page(&values),
            });
        }
    }
    Ok(Some((output, own_keys)))
}

/// One query's attention output before `W_o`: for each query head, the
/// softmax-weighted sum of its key head's values over positions `0..=position`.
fn attention_context(
    query: &[i32],
    position: usize,
    keys: &Keys,
    params: &LayerParams,
) -> Result<Vec<i32>> {
    let heads = params.num_heads as usize;
    let kv_heads = params.num_kv_heads as usize;
    let head_dim = params.head_dim as usize;
    let group = heads / kv_heads;
    let visible = position + 1;
    ensure!(
        keys.k.len() >= visible,
        "position {position} has {} keys to attend to",
        keys.k.len()
    );
    let mut context = vec![0_i32; heads * head_dim];
    let mut logits = Vec::with_capacity(visible);
    for head in 0..heads {
        let kv = (head / group) * head_dim..(head / group + 1) * head_dim;
        let q_head = &query[head * head_dim..(head + 1) * head_dim];
        logits.clear();
        for key in &keys.k[..visible] {
            logits.push(Act::from_bits(scaled_score(
                q_head,
                &key[kv.clone()],
                params.attention_scale_q32,
            )));
        }
        let weights = attention_softmax(&logits);
        let mut acc = vec![0_i64; head_dim];
        for (weight, value) in weights.iter().zip(&keys.v[..visible]) {
            for (slot, lane) in acc.iter_mut().zip(&value[kv.clone()]) {
                mac_weighted_value(slot, weight.to_bits(), *lane);
            }
        }
        for (slot, acc) in context[head * head_dim..(head + 1) * head_dim]
            .iter_mut()
            .zip(acc)
        {
            *slot = requantize_acc(acc);
        }
    }
    Ok(context)
}

/// One row per token from a per-token step, in token order.
fn map_rows(count: usize, step: impl Fn(usize) -> Result<Vec<i32>> + Sync) -> Result<Slab> {
    let rows = if crate::tensor::use_parallel() {
        (0..count)
            .into_par_iter()
            .map(&step)
            .collect::<Result<Vec<_>>>()?
    } else {
        (0..count).map(&step).collect::<Result<Vec<_>>>()?
    };
    Slab::from_rows(rows)
}
