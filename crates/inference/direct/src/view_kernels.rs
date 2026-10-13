use anyhow::{bail, Result};
use det_num::ops::{attention_softmax, rope_rotate_pairs_in_place};
use det_num::{Acc, Act};
use inference_kernels::tensor::{
    dot_bits, linear_slab_from_source, mac_weighted_value, matvec_from_source, requantize_acc,
    MatrixSource, Slab,
};
use prefill_range::input::{
    add_row, gelu, rms_norm, scale_row, value_rms_norm, ActivationRow, ActivationSequence, KeyRow,
    LayerParams, PleLayerInputs, PleRow,
};
use raster::List;

use crate::model::{DirectFinalHead, DirectInferenceModel, DirectPleLayer, DirectTransformerLayer};

pub fn run_input_embedding_view(
    model: &DirectInferenceModel,
    prompt: &prompt_prepare::input::PromptTokenization,
) -> Result<ActivationSequence> {
    let mut rows = Vec::with_capacity(prompt.token_ids.len());
    for token_id in prompt.token_ids.iter().copied() {
        rows.push(ActivationRow {
            token_id,
            values: prefill_range::input::pack_i32s(&model.scaled_embedding_row(token_id)?),
        });
    }
    Ok(ActivationSequence {
        rows: List::from(rows),
        errors: List::new(),
        kv: List::new(),
        start_position: 0,
    })
}

pub fn run_decode_embed_view(
    model: &DirectInferenceModel,
    edge: &decode_select_token::input::DecodeEdge,
) -> Result<ActivationSequence> {
    if !edge.has_selected {
        bail!("decode-embed received the empty decode edge with no selected token");
    }
    Ok(ActivationSequence {
        rows: List::from(vec![ActivationRow {
            token_id: edge.token_id,
            values: prefill_range::input::pack_i32s(&model.scaled_embedding_row(edge.token_id)?),
        }]),
        errors: List::new(),
        kv: List::new(),
        start_position: edge.decode_position,
    })
}

pub fn run_ple_prepare_view(
    embedded: &ActivationSequence,
    layer: &DirectPleLayer<'_>,
) -> Result<PleLayerInputs> {
    let mut rows = Vec::with_capacity(embedded.rows.len());
    let norm_weights = unpack(&layer.params.norm_weights)?;
    for row in embedded.rows.iter() {
        let activation = unpack(&row.values)?;
        if activation.len() != layer.params.hidden_size as usize {
            bail!(
                "activation has {} values, expected in_width {}",
                activation.len(),
                layer.params.hidden_size
            );
        }
        let mut projected = matvec_from_source(&layer.projection, &activation)?;
        scale_row(&mut projected, layer.params.projection_scalar);
        prefill_prepare_aux::input::rms_norm(&mut projected, &norm_weights, layer.params.norm_eps)
            .map_err(anyhow::Error::msg)?;

        let embedding = layer.embeddings.row(row.token_id as usize)?;
        let mut embedded = embedding
            .subslice(layer.embedding_start, layer.params.ple_width as usize)?
            .values();
        scale_row(&mut embedded, layer.params.embedding_scale);
        for (value, projected_value) in embedded.iter_mut().zip(projected) {
            *value = prefill_prepare_aux::input::add_sat(*value, projected_value);
        }
        scale_row(&mut embedded, layer.params.input_scale);
        rows.push(PleRow {
            values: prefill_range::input::pack_i32s(&embedded),
        });
    }
    Ok(PleLayerInputs {
        layer_idx: layer.params.layer_idx,
        rows: List::from(rows),
        errors: List::new(),
    })
}

pub fn run_prefill_range_view(
    activations: &ActivationSequence,
    layer: &DirectTransformerLayer<'_>,
    prior_kv: &ActivationSequence,
    donor_a_kv: &ActivationSequence,
    donor_b_kv: &ActivationSequence,
    ple: &PleLayerInputs,
) -> Result<ActivationSequence> {
    let params = validate_layer_params(&layer.params)?;
    let input_rows = activation_rows(activations, params.hidden_size as usize)?;
    if input_rows.rows() == 0 {
        return Ok(ActivationSequence {
            rows: List::new(),
            errors: List::new(),
            kv: List::new(),
            start_position: activations.start_position,
        });
    }
    let (queries, own_keys) =
        project_tokens(&input_rows, layer, &params, activations.start_position)?;
    let output_rows = attend_tokens(
        &queries,
        prior_kv.kv.as_slice(),
        &own_keys,
        donor_a_kv.kv.as_slice(),
        donor_b_kv.kv.as_slice(),
        ple.rows.as_slice(),
        layer,
        &params,
    )?;
    let output_kv = output_kv(
        prior_kv.kv.as_slice(),
        &own_keys,
        activations.start_position,
        params.sliding_window,
    );
    Ok(ActivationSequence {
        rows: List::from(output_rows),
        errors: List::new(),
        kv: List::from(output_kv),
        start_position: activations.start_position,
    })
}

pub fn score_prefill_logits_view(
    activations: &ActivationSequence,
    head: &DirectFinalHead<'_>,
) -> Result<decode_select_token::input::PrefillLogits> {
    let position = activations.rows.iter().last().ok_or_else(|| {
        anyhow::anyhow!("prefill produced no activation rows; there is no final position to score")
    })?;
    let mut values = unpack(&position.values)?;
    if values.len() != head.params.hidden_size as usize {
        bail!(
            "final position has {} values, expected hidden_size {}",
            values.len(),
            head.params.hidden_size
        );
    }
    let norm = unpack(&head.params.norm_weights)?;
    prefill_finalize::input::rms_norm(&mut values, &norm, head.params.norm_eps)
        .map_err(anyhow::Error::msg)?;
    let mut logits = Vec::with_capacity(head.projection.rows());
    for token_id in 0..head.projection.rows() {
        let value = head.projection.row_dot(token_id, &values)?;
        let value = prefill_finalize::input::softcap(value, head.params.softcap);
        logits.push(decode_select_token::input::LogitEntry {
            token_id: token_id as u32,
            value,
        });
    }
    Ok(decode_select_token::input::PrefillLogits {
        decode_position: activations.start_position + activations.rows.len() as u32,
        logits: List::from(logits),
        errors: List::new(),
    })
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
    if params.rotary_dim as usize % 2 != 0 {
        bail!(
            "layer {}: rotary_dim {} must be even",
            params.layer_idx,
            params.rotary_dim
        );
    }
    if params.rotary_dim > params.head_dim {
        bail!(
            "layer {}: rotary_dim {} exceeds head_dim {}",
            params.layer_idx,
            params.rotary_dim,
            params.head_dim
        );
    }
    if params.norm_eps < 0 {
        bail!("rms-norm epsilon must be non-negative");
    }
    Ok(params.clone())
}

fn activation_rows(activations: &ActivationSequence, hidden: usize) -> Result<InputRows> {
    let mut token_ids = Vec::with_capacity(activations.rows.len());
    let mut rows = Vec::with_capacity(activations.rows.len());
    for (row_idx, row) in activations.rows.iter().enumerate() {
        let values = unpack(&row.values)?;
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
    weights: &DirectTransformerLayer<'_>,
    params: &LayerParams,
    start_position: u32,
) -> Result<(Vec<QueryState>, Vec<KeyRow>)> {
    let norm_input = unpack(&params.norm_input)?;
    let q_norm = unpack(&params.q_norm)?;
    let k_norm = unpack(&params.k_norm)?;
    let hidden = params.hidden_size as usize;
    let normed = normed_input_slab(rows, &norm_input, params.norm_eps, hidden)?;
    let mut q = linear_slab_from_source(&normed, &weights.w_q)?;
    let mut k = linear_slab_from_source(&normed, &weights.w_k)?;
    let mut v = linear_slab_from_source(&normed, &weights.w_v)?;

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
            k: prefill_range::input::pack_i32s(&k_row),
            v: prefill_range::input::pack_i32s(&v_row),
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
    for row_idx in 0..rows.rows() {
        let out = normed.row_mut(row_idx);
        out.copy_from_slice(rows.slab.row(row_idx));
        rms_norm(out, norm_input, norm_eps).map_err(anyhow::Error::msg)?;
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
    weights: &DirectTransformerLayer<'_>,
    params: &LayerParams,
) -> Result<Vec<ActivationRow>> {
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

fn attend_token(
    query: &QueryState,
    prior_keys: &[KeyRow],
    own_keys: &[KeyRow],
    donor_a_keys: &[KeyRow],
    donor_b_keys: &[KeyRow],
    ple_rows: &[PleRow],
    weights: &DirectTransformerLayer<'_>,
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
    let attn_proj = matvec_from_source(&weights.w_o, &context)?;
    let residual = attn_residual(query, attn_proj, params)?;
    let ff_in = pre_ff_norm(&residual, params)?;
    let gate = matvec_from_source(&weights.w_gate, &ff_in)?;
    let up = matvec_from_source(&weights.w_up, &ff_in)?;
    let gated = gelu_mul(gate, up)?;
    let ff = matvec_from_source(&weights.w_down, &gated)?;
    let xs = finish_mlp(&residual, ff, params)?;
    let ple_row = ple_rows
        .get(query.local as usize)
        .ok_or_else(|| anyhow::anyhow!("missing PLE row at local position {}", query.local))?;
    let ple_gate = matvec_from_source(&weights.ple_input_gate, &xs)?;
    let ple_gated = ple_gate_mul(ple_gate, ple_row)?;
    let projected = matvec_from_source(&weights.ple_layer_projection, &ple_gated)?;
    let values = finish_layer(&xs, projected, params)?;
    Ok(ActivationRow {
        token_id: query.token_id,
        values: prefill_range::input::pack_i32s(&values),
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
        let k = unpack(&key.k)?;
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
            .ok_or_else(|| anyhow::anyhow!("visible key precedes the attention window"))?
            as usize;
        if slot >= window_len {
            bail!("visible key falls outside the weight list");
        }
        let v = unpack(&key.v)?;
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
        &unpack(&params.norm_post_attn)?,
        params.norm_eps,
    )
    .map_err(anyhow::Error::msg)?;
    add_row(&mut attn_proj, &query.residual).map_err(anyhow::Error::msg)?;
    Ok(attn_proj)
}

fn pre_ff_norm(residual: &[i32], params: &LayerParams) -> Result<Vec<i32>> {
    let mut normed = residual.to_vec();
    rms_norm(&mut normed, &unpack(&params.norm_pre_ffw)?, params.norm_eps)
        .map_err(anyhow::Error::msg)?;
    Ok(normed)
}

fn gelu_mul(mut gate: Vec<i32>, up: Vec<i32>) -> Result<Vec<i32>> {
    if gate.len() != up.len() {
        bail!("gate/up width mismatch: {} vs {}", gate.len(), up.len());
    }
    for (value, up_value) in gate.iter_mut().zip(&up) {
        *value = prefill_range::input::mul(gelu(*value), *up_value);
    }
    Ok(gate)
}

fn finish_mlp(residual: &[i32], mut ff: Vec<i32>, params: &LayerParams) -> Result<Vec<i32>> {
    rms_norm(&mut ff, &unpack(&params.norm_post_ffw)?, params.norm_eps)
        .map_err(anyhow::Error::msg)?;
    add_row(&mut ff, residual).map_err(anyhow::Error::msg)?;
    Ok(ff)
}

fn ple_gate_mul(mut gate: Vec<i32>, per_layer_input: &PleRow) -> Result<Vec<i32>> {
    let ple = unpack(&per_layer_input.values)?;
    if gate.len() != ple.len() {
        bail!(
            "per-layer input has {} values, gate has {}",
            ple.len(),
            gate.len()
        );
    }
    for (slot, embedded) in gate.iter_mut().zip(ple.iter()) {
        *slot = prefill_range::input::mul(gelu(*slot), *embedded);
    }
    Ok(gate)
}

fn finish_layer(xs: &[i32], mut projected: Vec<i32>, params: &LayerParams) -> Result<Vec<i32>> {
    rms_norm(
        &mut projected,
        &unpack(&params.ple_post_norm)?,
        params.norm_eps,
    )
    .map_err(anyhow::Error::msg)?;
    add_row(&mut projected, xs).map_err(anyhow::Error::msg)?;
    if params.layer_scalar != 0 {
        scale_row(&mut projected, params.layer_scalar);
    }
    Ok(projected)
}

fn unpack(page: &raster::BytesPage) -> Result<Vec<i32>> {
    prefill_range::input::unpack_i32s(page).map_err(anyhow::Error::msg)
}
