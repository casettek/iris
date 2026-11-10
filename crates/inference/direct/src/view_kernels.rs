use anyhow::{bail, Result};
use inference_kernels::prefill_range::{layer_rows_and_keys, output_kv, LayerMatrices};
use inference_kernels::tensor::{matvec_from_source, MatrixSource};
use prefill_range::input::{
    scale_row, ActivationRow, ActivationSequence, LayerParams, PleLayerInputs, PleRow,
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
        model: model_identity::ModelContext::embedding(
            &bound_model_context(model),
            &Default::default(),
        ),
        rows: List::from(rows),
        errors: List::new(),
        kv: List::new(),
        start_position: 0,
    })
}

pub fn run_decode_embed_view(
    model: &DirectInferenceModel,
    edge: &sequence_check::input::DecodeEdge,
) -> Result<ActivationSequence> {
    if !edge.has_selected {
        bail!("decode-embed received the empty decode edge with no selected token");
    }
    Ok(ActivationSequence {
        model: model_identity::ModelContext::embedding(
            &bound_model_context(model),
            &Default::default(),
        ),
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

        let mut embedded = layer.embeddings.row_columns(
            row.token_id as usize,
            layer.embedding_start,
            layer.params.ple_width as usize,
        )?;
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
        model: embedded.model.auxiliary(layer.params.layer_idx),
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
    let model = activations.model.layer(
        params.layer_idx,
        &ple.model,
        &prior_kv.model,
        &donor_a_kv.model,
        &donor_b_kv.model,
        params.kv_donor_layer,
        params.donor_a_layer,
        params.donor_b_layer,
    );
    // The layer itself is the staged pass's kernel, reading the mapped weights.
    let computed = layer_rows_and_keys(
        activations,
        prior_kv.kv.as_slice(),
        donor_a_kv.kv.as_slice(),
        donor_b_kv.kv.as_slice(),
        ple.rows.as_slice(),
        &LayerMatrices {
            w_q: &layer.w_q,
            w_k: &layer.w_k,
            w_v: &layer.w_v,
            w_o: &layer.w_o,
            w_gate: &layer.w_gate,
            w_up: &layer.w_up,
            w_down: &layer.w_down,
            ple_input_gate: &layer.ple_input_gate,
            ple_layer_projection: &layer.ple_layer_projection,
        },
        &params,
    )?;
    let Some((output_rows, own_keys)) = computed else {
        return Ok(ActivationSequence {
            model,
            rows: List::new(),
            errors: List::new(),
            kv: List::new(),
            start_position: activations.start_position,
        });
    };
    let output_kv = output_kv(
        prior_kv.kv.as_slice(),
        &own_keys,
        activations.start_position,
        params.sliding_window,
    );
    Ok(ActivationSequence {
        model,
        rows: List::from(output_rows),
        errors: List::new(),
        kv: List::from(output_kv),
        start_position: activations.start_position,
    })
}

pub fn score_prefill_logits_view(
    activations: &ActivationSequence,
    head: &DirectFinalHead<'_>,
) -> Result<crate::generation::PrefillLogits> {
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
    sequence_check::input::rms_norm(&mut values, &norm, head.params.norm_eps)
        .map_err(anyhow::Error::msg)?;
    let mut logits = Vec::with_capacity(head.projection.rows());
    for token_id in 0..head.projection.rows() {
        let value = head.projection.row_dot(token_id, &values)?;
        let value = sequence_check::input::softcap(value, head.params.softcap);
        logits.push(crate::generation::LogitEntry {
            token_id: token_id as u32,
            value,
        });
    }
    Ok(crate::generation::PrefillLogits {
        decode_position: activations.start_position + activations.rows.len() as u32,
        logits: List::from(logits),
        errors: List::new(),
    })
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

fn unpack(page: &raster::BytesPage) -> Result<Vec<i32>> {
    prefill_range::input::unpack_i32s(page).map_err(anyhow::Error::msg)
}

pub(crate) fn bound_model_context(model: &DirectInferenceModel) -> model_identity::ModelContext {
    let bytes = model
        .manifest()
        .descriptor()
        .ok()
        .and_then(|d| d.encode().ok())
        .unwrap_or_default();
    model_identity::ModelContext::bind(&bytes, &model.manifest().model_id)
}
