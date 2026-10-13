use std::collections::{BTreeMap, VecDeque};
use std::sync::{Arc, Mutex};

use anyhow::{bail, Result};
use det_num::ops::{attention_softmax, rope_rotate_pairs_in_place};
use det_num::{Acc, Act};
use raster::List;
use rayon::prelude::*;

use crate::tensor::{
    dot_bits, linear_slab, mac_weighted_value, pack_i32_page, requantize_acc, unpack_page_i32s,
    Matrix, Slab,
};
use crate::MaterializationCacheKey;
use ::prefill_range::input::{
    add_row, gelu, rms_norm, scale_row, value_rms_norm, ActivationRow, ActivationSequence, KeyRow,
    LayerParams, PleLayerInputs, PleRow, TransformerLayer,
};

pub struct PrefillRangeDirectInputs<'a> {
    pub activations: &'a ActivationSequence,
    pub layer: &'a TransformerLayer,
    pub layer_cache_key: Option<&'a MaterializationCacheKey>,
    pub prior_kv: &'a ActivationSequence,
    pub donor_a_kv: &'a ActivationSequence,
    pub donor_b_kv: &'a ActivationSequence,
    pub ple: &'a PleLayerInputs,
}

pub struct PrefillRangeWeightCache {
    state: Mutex<PrefillRangeWeightCacheState>,
    capacity: usize,
}

#[derive(Default)]
struct PrefillRangeWeightCacheState {
    weights: BTreeMap<PreparedLayerWeightCacheKey, Arc<LayerWeights>>,
    order: VecDeque<PreparedLayerWeightCacheKey>,
}

impl Default for PrefillRangeWeightCache {
    fn default() -> Self {
        let capacity = std::env::var("STAGED_INFER_PREFILL_WEIGHT_CACHE_ENTRIES")
            .or_else(|_| std::env::var("DIRECT_NATIVE_PREFILL_WEIGHT_CACHE_ENTRIES"))
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(0);
        Self {
            state: Mutex::new(PrefillRangeWeightCacheState::default()),
            capacity,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct LayerWeightCacheKey {
    layer_idx: u32,
    hidden_size: u32,
    ffn_size: u32,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    ple_width: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct PreparedLayerWeightCacheKey {
    materialization: MaterializationCacheKey,
    params: LayerWeightCacheKey,
}

pub fn run_prefill_range_direct(
    inputs: PrefillRangeDirectInputs<'_>,
) -> Result<ActivationSequence> {
    run_prefill_range_direct_with_weight_cache(inputs, None)
}

pub fn run_prefill_range_direct_with_weight_cache(
    inputs: PrefillRangeDirectInputs<'_>,
    weight_cache: Option<&PrefillRangeWeightCache>,
) -> Result<ActivationSequence> {
    let params = validate_layer_params(&inputs.layer.params)?;
    let weights = match (weight_cache, inputs.layer_cache_key) {
        (Some(cache), Some(key)) => cache.get_or_prepare(inputs.layer, &params, key)?,
        _ => Arc::new(LayerWeights::from_layer(inputs.layer, &params)?),
    };
    let input_rows = activation_rows(inputs.activations, params.hidden_size as usize)?;

    if input_rows.rows() == 0 {
        return Ok(ActivationSequence {
            rows: List::new(),
            errors: List::new(),
            kv: List::new(),
            start_position: inputs.activations.start_position,
        });
    }

    let (queries, own_keys) = project_tokens(
        &input_rows,
        &weights,
        &params,
        inputs.activations.start_position,
    )?;
    let ple_rows = inputs.ple.rows.as_slice();
    let output_rows = attend_tokens(
        &queries,
        inputs.prior_kv.kv.as_slice(),
        &own_keys,
        inputs.donor_a_kv.kv.as_slice(),
        inputs.donor_b_kv.kv.as_slice(),
        ple_rows,
        &weights,
        &params,
    )?;
    let output_kv = output_kv(
        inputs.prior_kv.kv.as_slice(),
        &own_keys,
        inputs.activations.start_position,
        params.sliding_window,
    );

    Ok(ActivationSequence {
        rows: List::from(output_rows),
        errors: List::new(),
        kv: List::from(output_kv),
        start_position: inputs.activations.start_position,
    })
}

impl PrefillRangeWeightCache {
    #[cfg(test)]
    fn with_capacity(capacity: usize) -> Self {
        Self {
            state: Mutex::new(PrefillRangeWeightCacheState::default()),
            capacity,
        }
    }

    fn get_or_prepare(
        &self,
        layer: &TransformerLayer,
        params: &LayerParams,
        materialization_key: &MaterializationCacheKey,
    ) -> Result<Arc<LayerWeights>> {
        if self.capacity == 0 {
            return Ok(Arc::new(LayerWeights::from_layer(layer, params)?));
        }

        let key = PreparedLayerWeightCacheKey {
            materialization: materialization_key.clone(),
            params: LayerWeightCacheKey::from(params),
        };
        if let Some(weights) = self.state.lock().unwrap().weights.get(&key).cloned() {
            return Ok(weights);
        }
        let weights = Arc::new(LayerWeights::from_layer(layer, params)?);
        let mut state = self.state.lock().unwrap();
        if let Some(existing) = state.weights.get(&key).cloned() {
            return Ok(existing);
        }
        if state.weights.len() >= self.capacity {
            if let Some(evicted) = state.order.pop_front() {
                state.weights.remove(&evicted);
            }
        }
        state.order.push_back(key.clone());
        state.weights.insert(key, weights.clone());
        Ok(weights)
    }
}

impl From<&LayerParams> for LayerWeightCacheKey {
    fn from(params: &LayerParams) -> Self {
        Self {
            layer_idx: params.layer_idx,
            hidden_size: params.hidden_size,
            ffn_size: params.ffn_size,
            num_heads: params.num_heads,
            num_kv_heads: params.num_kv_heads,
            head_dim: params.head_dim,
            ple_width: params.ple_width,
        }
    }
}

struct InputRows {
    token_ids: Vec<u32>,
    slab: Slab,
}

impl InputRows {
    fn rows(&self) -> usize {
        self.slab.rows()
    }
}

struct QueryState {
    position: u32,
    local: u32,
    token_id: u32,
    q: Vec<i32>,
    residual: Vec<i32>,
}

struct LayerWeights {
    w_q: Matrix,
    w_k: Matrix,
    w_v: Matrix,
    w_o: Matrix,
    w_gate: Matrix,
    w_up: Matrix,
    w_down: Matrix,
    ple_input_gate: Matrix,
    ple_layer_projection: Matrix,
}

impl LayerWeights {
    fn from_layer(layer: &TransformerLayer, params: &LayerParams) -> Result<Self> {
        let hidden = params.hidden_size as usize;
        let ffn = params.ffn_size as usize;
        let q_len = params.num_heads as usize * params.head_dim as usize;
        let kv_len = params.num_kv_heads as usize * params.head_dim as usize;
        let ple_width = params.ple_width as usize;
        Ok(Self {
            w_q: Matrix::from_region("w_q", &layer.w_q, q_len, hidden)?,
            w_k: Matrix::from_region("w_k", &layer.w_k, kv_len, hidden)?,
            w_v: Matrix::from_region("w_v", &layer.w_v, kv_len, hidden)?,
            w_o: Matrix::from_region("w_o", &layer.w_o, hidden, q_len)?,
            w_gate: Matrix::from_region("w_gate", &layer.w_gate, ffn, hidden)?,
            w_up: Matrix::from_region("w_up", &layer.w_up, ffn, hidden)?,
            w_down: Matrix::from_region("w_down", &layer.w_down, hidden, ffn)?,
            ple_input_gate: Matrix::from_region(
                "ple_input_gate",
                &layer.ple_input_gate,
                ple_width,
                hidden,
            )?,
            ple_layer_projection: Matrix::from_region(
                "ple_layer_projection",
                &layer.ple_layer_projection,
                hidden,
                ple_width,
            )?,
        })
    }
}

fn validate_layer_params(params: &LayerParams) -> Result<LayerParams> {
    let hidden = params.hidden_size as usize;
    let ffn = params.ffn_size as usize;
    let heads = params.num_heads as usize;
    let kv_heads = params.num_kv_heads as usize;
    let head_dim = params.head_dim as usize;
    if hidden == 0 || ffn == 0 {
        bail!("transformer layer requires non-zero hidden_size and ffn_size");
    }
    if heads == 0 || kv_heads == 0 || head_dim == 0 {
        bail!("transformer layer requires non-zero num_heads, num_kv_heads and head_dim");
    }
    if heads % kv_heads != 0 {
        bail!(
            "layer {}: {heads} query heads do not group evenly over {kv_heads} kv heads",
            params.layer_idx
        );
    }
    let rotary_dim = params.rotary_dim as usize;
    if rotary_dim % 2 != 0 {
        bail!(
            "layer {}: rotary_dim {rotary_dim} must be even",
            params.layer_idx
        );
    }
    if rotary_dim > head_dim {
        bail!(
            "layer {}: rotary_dim {rotary_dim} exceeds head_dim {head_dim}",
            params.layer_idx
        );
    }
    let freq_base_dim = params.rope_freq_base_dim as usize;
    if rotary_dim > 0 && (freq_base_dim < 2 || freq_base_dim % 2 != 0) {
        bail!(
            "layer {}: rope_freq_base_dim {freq_base_dim} must be even and at least 2",
            params.layer_idx
        );
    }
    if rotary_dim > 0 && params.rope_base <= 0 {
        bail!("layer {}: rope_base must be positive", params.layer_idx);
    }
    if params.norm_eps < 0 {
        bail!("rms-norm epsilon must be non-negative");
    }
    if params.donor_a_layer == -1 || params.donor_b_layer == -1 {
        bail!("donor candidate -1 is reserved for a layer's own cache");
    }
    if params.kv_donor_layer >= 0
        && params.kv_donor_layer != params.donor_a_layer
        && params.kv_donor_layer != params.donor_b_layer
    {
        bail!(
            "layer {} borrows K/V from {}, outside committed candidates {} and {}",
            params.layer_idx,
            params.kv_donor_layer,
            params.donor_a_layer,
            params.donor_b_layer
        );
    }

    for (name, page, values) in [
        ("norm_input", &params.norm_input, hidden),
        ("norm_post_attn", &params.norm_post_attn, hidden),
        ("norm_pre_ffw", &params.norm_pre_ffw, hidden),
        ("norm_post_ffw", &params.norm_post_ffw, hidden),
        ("q_norm", &params.q_norm, head_dim),
        ("k_norm", &params.k_norm, head_dim),
    ] {
        let expected = values * 4;
        if page.len() != expected {
            bail!(
                "layer {} {name} has {} bytes, expected {expected} ({values} values)",
                params.layer_idx,
                page.len()
            );
        }
    }
    Ok(params.clone())
}

fn activation_rows(activations: &ActivationSequence, hidden: usize) -> Result<InputRows> {
    let mut token_ids = Vec::with_capacity(activations.rows.len());
    let mut rows = Vec::with_capacity(activations.rows.len());
    for (row_idx, row) in activations.rows.iter().enumerate() {
        let values = unpack_page_i32s(&row.values)?;
        if values.len() != hidden {
            bail!(
                "activation row {row_idx} has {} values, expected hidden_size {hidden}",
                values.len()
            );
        }
        token_ids.push(row.token_id);
        rows.push(values);
    }
    Ok(InputRows {
        token_ids,
        slab: Slab::from_rows(rows)?,
    })
}

fn project_tokens(
    rows: &InputRows,
    weights: &LayerWeights,
    params: &LayerParams,
    start_position: u32,
) -> Result<(Vec<QueryState>, Vec<KeyRow>)> {
    let norm_input = unpack_page_i32s(&params.norm_input)?;
    let q_norm = unpack_page_i32s(&params.q_norm)?;
    let k_norm = unpack_page_i32s(&params.k_norm)?;
    let hidden = params.hidden_size as usize;
    let normed = normed_input_slab(rows, &norm_input, params.norm_eps, hidden)?;
    let mut q = linear_slab(&normed, &weights.w_q)?;
    let mut k = linear_slab(&normed, &weights.w_k)?;
    let mut v = linear_slab(&normed, &weights.w_v)?;

    let head_dim = params.head_dim as usize;
    for row_idx in 0..rows.rows() {
        // RoPE follows the absolute Raster cursor, including during decode.
        finish_qkv_row(
            start_position + row_idx as u32,
            q.row_mut(row_idx),
            k.row_mut(row_idx),
            v.row_mut(row_idx),
            &q_norm,
            &k_norm,
            params,
        )?;
    }

    let mut queries = Vec::with_capacity(rows.rows());
    let mut keys = Vec::with_capacity(rows.rows());
    for row_idx in 0..rows.rows() {
        let local = row_idx as u32;
        let position = start_position + local;
        let q_row = q.row(row_idx).to_vec();
        let k_row = k.row(row_idx).to_vec();
        let v_row = v.row(row_idx).to_vec();
        let expected_q = params.num_heads as usize * head_dim;
        let expected_kv = params.num_kv_heads as usize * head_dim;
        if q_row.len() != expected_q || k_row.len() != expected_kv || v_row.len() != expected_kv {
            bail!(
                "Q/K/V width mismatch at position {position}: q {} k {} v {}",
                q_row.len(),
                k_row.len(),
                v_row.len()
            );
        }
        queries.push(QueryState {
            position,
            local,
            token_id: rows.token_ids[row_idx],
            q: q_row,
            residual: rows.slab.row(row_idx).to_vec(),
        });
        keys.push(KeyRow {
            position,
            k: pack_i32_page(&k_row),
            v: pack_i32_page(&v_row),
        });
    }
    Ok((queries, keys))
}

fn normed_input_slab(
    rows: &InputRows,
    norm_input: &[i32],
    norm_eps: i64,
    hidden: usize,
) -> Result<Slab> {
    let mut normed = Slab::zeroed(rows.rows(), hidden);
    if crate::tensor::use_parallel() {
        normed
            .as_flat_mut()
            .par_chunks_mut(hidden.max(1))
            .enumerate()
            .try_for_each(|(row_idx, out)| {
                out.copy_from_slice(rows.slab.row(row_idx));
                rms_norm(out, norm_input, norm_eps).map_err(anyhow::Error::msg)
            })?;
    } else {
        for row_idx in 0..rows.rows() {
            let out = normed.row_mut(row_idx);
            out.copy_from_slice(rows.slab.row(row_idx));
            rms_norm(out, norm_input, norm_eps).map_err(anyhow::Error::msg)?;
        }
    }
    Ok(normed)
}

fn finish_qkv_row(
    position: u32,
    q: &mut [i32],
    k: &mut [i32],
    v: &mut [i32],
    q_norm: &[i32],
    k_norm: &[i32],
    params: &LayerParams,
) -> Result<()> {
    let head_dim = params.head_dim as usize;
    for head in q.chunks_mut(head_dim) {
        rms_norm(head, q_norm, params.norm_eps).map_err(anyhow::Error::msg)?;
        apply_rope(head, params, position);
    }
    for head in k.chunks_mut(head_dim) {
        rms_norm(head, k_norm, params.norm_eps).map_err(anyhow::Error::msg)?;
        apply_rope(head, params, position);
    }
    for head in v.chunks_mut(head_dim) {
        value_rms_norm(head, params.norm_eps).map_err(anyhow::Error::msg)?;
    }
    Ok(())
}

fn apply_rope(head: &mut [i32], params: &LayerParams, position: u32) {
    let rotary_dim = params.rotary_dim as usize;
    if rotary_dim == 0 || position == 0 {
        return;
    }
    let mut lanes: Vec<Act> = head.iter().map(|bits| Act::from_bits(*bits)).collect();
    rope_rotate_pairs_in_place(
        &mut lanes,
        rotary_dim,
        params.rope_freq_base_dim as usize,
        Acc::from_bits(params.rope_base),
        position as usize,
    );
    for (lane, rotated) in head.iter_mut().zip(lanes.iter()) {
        *lane = rotated.to_bits();
    }
}

fn attend_tokens(
    queries: &[QueryState],
    prior_keys: &[KeyRow],
    own_keys: &[KeyRow],
    donor_a_keys: &[KeyRow],
    donor_b_keys: &[KeyRow],
    ple_rows: &[PleRow],
    weights: &LayerWeights,
    params: &LayerParams,
) -> Result<Vec<ActivationRow>> {
    if crate::tensor::use_parallel() {
        queries
            .par_iter()
            .map(|query| {
                attend_token(
                    query,
                    prior_keys,
                    own_keys,
                    donor_a_keys,
                    donor_b_keys,
                    ple_rows,
                    weights,
                    params,
                )
            })
            .collect()
    } else {
        queries
            .iter()
            .map(|query| {
                attend_token(
                    query,
                    prior_keys,
                    own_keys,
                    donor_a_keys,
                    donor_b_keys,
                    ple_rows,
                    weights,
                    params,
                )
            })
            .collect()
    }
}

fn attend_token(
    query: &QueryState,
    prior_keys: &[KeyRow],
    own_keys: &[KeyRow],
    donor_a_keys: &[KeyRow],
    donor_b_keys: &[KeyRow],
    ple_rows: &[PleRow],
    weights: &LayerWeights,
    params: &LayerParams,
) -> Result<ActivationRow> {
    let context = if params.kv_donor_layer == -1 {
        attention_context(query, &[prior_keys, own_keys], params)?
    } else if params.kv_donor_layer == params.donor_a_layer {
        attention_context(query, &[donor_a_keys], params)?
    } else if params.kv_donor_layer == params.donor_b_layer {
        attention_context(query, &[donor_b_keys], params)?
    } else {
        bail!(
            "layer {} borrows K/V from {}, outside committed candidates {} and {}",
            params.layer_idx,
            params.kv_donor_layer,
            params.donor_a_layer,
            params.donor_b_layer
        );
    };
    let attn_proj = weights.w_o.matvec(&context)?;
    let residual = attn_residual(query, attn_proj, params)?;
    let ff_in = pre_ff_norm(&residual, params)?;
    let gate = weights.w_gate.matvec(&ff_in)?;
    let up = weights.w_up.matvec(&ff_in)?;
    let gated = gelu_mul(gate, up)?;
    let ff = weights.w_down.matvec(&gated)?;
    let xs = finish_mlp(&residual, &ff_in, ff, params)?;
    let ple_row = ple_rows
        .get(query.local as usize)
        .ok_or_else(|| anyhow::anyhow!("missing PLE row at local position {}", query.local))?;
    let ple_gate = weights.ple_input_gate.matvec(&xs)?;
    let ple_gated = ple_gate_mul(ple_gate, ple_row)?;
    let projected = weights.ple_layer_projection.matvec(&ple_gated)?;
    let values = finish_layer(&xs, projected, params)?;
    Ok(ActivationRow {
        token_id: query.token_id,
        values: pack_i32_page(&values),
    })
}

fn attention_context(
    query: &QueryState,
    key_sources: &[&[KeyRow]],
    params: &LayerParams,
) -> Result<Vec<i32>> {
    let heads = params.num_heads as usize;
    let kv_heads = params.num_kv_heads as usize;
    let head_dim = params.head_dim as usize;
    let group = heads / kv_heads.max(1);
    let start = window_start(query.position, params.sliding_window);
    let window_len = query.position.saturating_sub(start) as usize + 1;
    let mut visible = Vec::new();
    let mut scores = vec![0_i32; window_len * heads];
    let mut filled = 0usize;
    for key in key_sources.iter().flat_map(|keys| keys.iter()) {
        if !is_visible(query.position, key.position, params.sliding_window) {
            continue;
        }
        let k = unpack_page_i32s(&key.k)?;
        if k.len() != kv_heads * head_dim {
            bail!(
                "attention key width mismatch at position {}: got {}, expected {}",
                key.position,
                k.len(),
                kv_heads * head_dim
            );
        }
        for head in 0..heads {
            let kv_head = head / group.max(1);
            let q_head = &query.q[head * head_dim..(head + 1) * head_dim];
            let k_head = &k[kv_head * head_dim..(kv_head + 1) * head_dim];
            let slot = key
                .position
                .checked_sub(start)
                .ok_or_else(|| anyhow::anyhow!("visible key precedes the attention window"))?
                as usize;
            if slot >= window_len {
                bail!("visible key falls outside the weight list");
            }
            scores[slot * heads + head] = dot_bits(q_head, k_head);
        }
        visible.push(key);
        filled += 1;
    }
    if filled != window_len {
        bail!("attention window has {filled} of {window_len} positions scored");
    }

    let mut weights = scores.clone();
    let mut logits = Vec::with_capacity(window_len);
    for head in 0..heads {
        logits.clear();
        for key in 0..window_len {
            logits.push(Act::from_bits(scores[key * heads + head]));
        }
        let head_weights = attention_softmax(&logits);
        for (key, weight) in head_weights.iter().enumerate() {
            weights[key * heads + head] = weight.to_bits();
        }
    }

    let mut acc = vec![0_i64; heads * head_dim];
    for key in &visible {
        let slot = key
            .position
            .checked_sub(start)
            .ok_or_else(|| anyhow::anyhow!("visible key precedes the attention window"))?;
        let slot = slot as usize;
        if slot >= window_len {
            bail!("visible key falls outside the weight list");
        }
        let v = unpack_page_i32s(&key.v)?;
        if v.len() != kv_heads * head_dim {
            bail!(
                "attention value width mismatch at position {}: got {}, expected {}",
                key.position,
                v.len(),
                kv_heads * head_dim
            );
        }
        for head in 0..heads {
            let kv_head = head / group.max(1);
            let weight = weights[slot * heads + head];
            let v_head = &v[kv_head * head_dim..(kv_head + 1) * head_dim];
            for (lane, value) in v_head.iter().enumerate() {
                mac_weighted_value(&mut acc[head * head_dim + lane], weight, *value);
            }
        }
    }
    Ok(acc.into_iter().map(requantize_acc).collect())
}

fn output_kv(
    prior_keys: &[KeyRow],
    own_keys: &[KeyRow],
    start_position: u32,
    sliding_window: u32,
) -> Vec<KeyRow> {
    let keep_from = window_start(start_position, sliding_window);
    prior_keys
        .iter()
        .filter(|key| key.position >= keep_from)
        .cloned()
        .chain(own_keys.iter().cloned())
        .collect()
}

fn window_start(query_position: u32, sliding_window: u32) -> u32 {
    if sliding_window == 0 {
        0
    } else {
        query_position
            .saturating_add(1)
            .saturating_sub(sliding_window)
    }
}

fn is_visible(query_position: u32, key_position: u32, sliding_window: u32) -> bool {
    key_position <= query_position
        && (sliding_window == 0 || query_position - key_position < sliding_window)
}

fn attn_residual(
    query: &QueryState,
    mut attn_proj: Vec<i32>,
    params: &LayerParams,
) -> Result<Vec<i32>> {
    rms_norm(
        &mut attn_proj,
        &unpack_page_i32s(&params.norm_post_attn)?,
        params.norm_eps,
    )
    .map_err(anyhow::Error::msg)?;
    add_row(&mut attn_proj, &query.residual).map_err(anyhow::Error::msg)?;
    Ok(attn_proj)
}

fn pre_ff_norm(residual: &[i32], params: &LayerParams) -> Result<Vec<i32>> {
    let mut normed = residual.to_vec();
    rms_norm(
        &mut normed,
        &unpack_page_i32s(&params.norm_pre_ffw)?,
        params.norm_eps,
    )
    .map_err(anyhow::Error::msg)?;
    Ok(normed)
}

fn gelu_mul(mut gate: Vec<i32>, up: Vec<i32>) -> Result<Vec<i32>> {
    if gate.len() != up.len() {
        bail!("gate/up width mismatch: {} vs {}", gate.len(), up.len());
    }
    for (value, up_value) in gate.iter_mut().zip(&up) {
        *value = ::prefill_range::input::mul(gelu(*value), *up_value);
    }
    Ok(gate)
}

fn finish_mlp(
    residual: &[i32],
    _ff_in: &[i32],
    mut ff: Vec<i32>,
    params: &LayerParams,
) -> Result<Vec<i32>> {
    rms_norm(
        &mut ff,
        &unpack_page_i32s(&params.norm_post_ffw)?,
        params.norm_eps,
    )
    .map_err(anyhow::Error::msg)?;
    add_row(&mut ff, residual).map_err(anyhow::Error::msg)?;
    Ok(ff)
}

fn ple_gate_mul(mut gate: Vec<i32>, per_layer_input: &PleRow) -> Result<Vec<i32>> {
    let ple = unpack_page_i32s(&per_layer_input.values)?;
    if gate.len() != ple.len() {
        bail!(
            "per-layer input has {} values, gate has {}",
            ple.len(),
            gate.len()
        );
    }
    for (slot, embedded) in gate.iter_mut().zip(ple.iter()) {
        *slot = ::prefill_range::input::mul(gelu(*slot), *embedded);
    }
    Ok(gate)
}

fn finish_layer(xs: &[i32], mut projected: Vec<i32>, params: &LayerParams) -> Result<Vec<i32>> {
    rms_norm(
        &mut projected,
        &unpack_page_i32s(&params.ple_post_norm)?,
        params.norm_eps,
    )
    .map_err(anyhow::Error::msg)?;
    add_row(&mut projected, xs).map_err(anyhow::Error::msg)?;
    if params.layer_scalar != 0 {
        scale_row(&mut projected, params.layer_scalar);
    }
    Ok(projected)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ::prefill_range::input::{pack_i32s, LayerParams, TransformerLayer};
    use raster::Bytes;
    use std::sync::Arc;

    fn tiny_params() -> LayerParams {
        LayerParams {
            layer_idx: 0,
            hidden_size: 2,
            ffn_size: 2,
            num_heads: 1,
            num_kv_heads: 1,
            head_dim: 2,
            sliding_window: 0,
            attn_scale: 0,
            layer_scalar: 0,
            norm_eps: 0,
            rope_base: 1_i64 << 32,
            rotary_dim: 0,
            rope_freq_base_dim: 2,
            kv_donor_layer: -1,
            donor_a_layer: -2,
            donor_b_layer: -3,
            norm_input: pack_i32s(&[1 << 16, 1 << 16]),
            norm_post_attn: pack_i32s(&[1 << 16, 1 << 16]),
            norm_pre_ffw: pack_i32s(&[1 << 16, 1 << 16]),
            norm_post_ffw: pack_i32s(&[1 << 16, 1 << 16]),
            q_norm: pack_i32s(&[1 << 16, 1 << 16]),
            k_norm: pack_i32s(&[1 << 16, 1 << 16]),
            ple_width: 2,
            ple_post_norm: pack_i32s(&[1 << 16, 1 << 16]),
        }
    }

    fn paged(values: &[i32]) -> Bytes<196_608> {
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        Bytes::<196_608>::paged(bytes).unwrap()
    }

    fn tiny_layer() -> TransformerLayer {
        let matrix = paged(&[1 << 16, 0, 0, 1 << 16]);
        TransformerLayer {
            params: tiny_params(),
            w_q: matrix.clone(),
            w_k: matrix.clone(),
            w_v: matrix.clone(),
            w_o: matrix.clone(),
            w_gate: matrix.clone(),
            w_up: matrix.clone(),
            w_down: matrix.clone(),
            ple_input_gate: matrix.clone(),
            ple_layer_projection: matrix,
        }
    }

    #[test]
    fn qkv_projection_uses_absolute_rotary_positions() {
        const ONE: i32 = 1 << 16;
        let mut layer = tiny_layer();
        layer.params.rotary_dim = 2;
        // Identity projections of [ONE, ONE] normalise exactly; these weights
        // leave the rotary pair [ONE, 0] in both Q and K.
        layer.params.q_norm = pack_i32s(&[ONE, 0]);
        layer.params.k_norm = pack_i32s(&[ONE, 0]);
        let params = &layer.params;
        let weights = LayerWeights::from_layer(&layer, params).unwrap();

        // Raster starts its cursor at start_position and advances once per row.
        for positions in [[0, 1], [1, 2], [2, 3], [7, 8]] {
            for row_count in [1, 2] {
                let rows = InputRows {
                    token_ids: vec![7; row_count],
                    slab: Slab::from_rows(vec![vec![ONE, ONE]; row_count]).unwrap(),
                };
                let (queries, keys) =
                    project_tokens(&rows, &weights, params, positions[0]).unwrap();
                for (local, (query, key)) in queries.iter().zip(&keys).enumerate() {
                    let position = positions[local];
                    let mut expected = [Act::from_bits(ONE), Act::from_bits(0)];
                    rope_rotate_pairs_in_place(
                        &mut expected,
                        2,
                        2,
                        Acc::from_bits(params.rope_base),
                        position as usize,
                    );
                    let expected = expected.map(|lane| lane.to_bits());
                    if position == 1 {
                        assert_eq!(expected, [35408, 55146]);
                    }
                    assert_eq!(query.position, position);
                    assert_eq!(query.local, local as u32);
                    assert_eq!(key.position, position);
                    assert_eq!(query.q, expected, "Q at position {position}");
                    assert_eq!(unpack_page_i32s(&key.k).unwrap(), expected);
                    assert_eq!(unpack_page_i32s(&key.v).unwrap(), [ONE, ONE]);
                }
            }
        }
    }

    #[test]
    fn rejects_invalid_head_grouping() {
        let mut params = tiny_params();
        params.num_heads = 3;
        params.num_kv_heads = 2;
        assert!(validate_layer_params(&params)
            .unwrap_err()
            .to_string()
            .contains("do not group evenly"));
    }

    #[test]
    fn sliding_visibility_matches_wip_predicate() {
        assert!(is_visible(3, 2, 2));
        assert!(!is_visible(3, 1, 2));
        assert!(!is_visible(3, 4, 0));
        assert!(is_visible(3, 0, 0));
    }

    #[test]
    fn weight_cache_reuses_prepared_weights_for_same_layer_key() {
        let cache = PrefillRangeWeightCache::with_capacity(1);
        let layer = tiny_layer();
        let params = validate_layer_params(&layer.params).unwrap();
        let key = MaterializationCacheKey {
            param: String::from("layer"),
            path: "layer.rastered".into(),
            index_path: "layer.rindex".into(),
            commitment: String::from("abc"),
            type_name: std::any::type_name::<TransformerLayer>(),
        };

        let first = cache.get_or_prepare(&layer, &params, &key).unwrap();
        let second = cache.get_or_prepare(&layer, &params, &key).unwrap();

        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!(cache.state.lock().unwrap().weights.len(), 1);
    }
}
