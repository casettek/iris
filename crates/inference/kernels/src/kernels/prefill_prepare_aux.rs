use anyhow::{bail, Result};
use prefill_prepare_aux::input::{
    add_sat, pack_i32s, rms_norm, scale_row, unpack_i32s, unpack_i32s_at, ActivationSequence,
    PleLayer, PleLayerInputs, PleLayerParams, PleRow,
};
use raster::List;
use rayon::prelude::*;

use crate::tensor::{self, Matrix};

pub struct PrefillPrepareAuxDirectInputs<'a> {
    pub embedded: &'a ActivationSequence,
    pub layer: &'a PleLayer,
}

pub fn run_prefill_prepare_aux_direct(
    inputs: PrefillPrepareAuxDirectInputs<'_>,
) -> Result<PleLayerInputs> {
    let params = validate_params(&inputs.layer.params)?;
    let projection = Matrix::from_region(
        "projection",
        &inputs.layer.projection,
        params.ple_width as usize,
        params.hidden_size as usize,
    )?;
    let rows = if tensor::use_parallel() {
        inputs
            .embedded
            .rows
            .par_iter()
            .map(|row| prepare_row(row, inputs.layer, &params, &projection))
            .collect::<Result<Vec<_>>>()?
    } else {
        inputs
            .embedded
            .rows
            .iter()
            .map(|row| prepare_row(row, inputs.layer, &params, &projection))
            .collect::<Result<Vec<_>>>()?
    };

    Ok(PleLayerInputs {
        layer_idx: params.layer_idx,
        rows: List::from(rows),
        errors: List::new(),
    })
}

fn validate_params(params: &PleLayerParams) -> Result<PleLayerParams> {
    if params.hidden_size == 0 || params.ple_width == 0 {
        bail!("PLE layer requires non-zero hidden_size and ple_width");
    }
    let expected_norm = params.ple_width as usize * 4;
    if params.norm_weights.len() != expected_norm {
        bail!(
            "layer {} norm weights have {} bytes, expected {expected_norm}",
            params.layer_idx,
            params.norm_weights.len()
        );
    }
    if params.norm_eps < 0 {
        bail!("PLE rms-norm epsilon must be non-negative");
    }
    Ok(params.clone())
}

fn prepare_row(
    row: &prefill_prepare_aux::input::ActivationRow,
    layer: &PleLayer,
    params: &PleLayerParams,
    projection: &Matrix,
) -> Result<PleRow> {
    let activation = unpack_i32s(&row.values).map_err(anyhow::Error::msg)?;
    if activation.len() != params.hidden_size as usize {
        bail!(
            "activation has {} values, expected in_width {}",
            activation.len(),
            params.hidden_size
        );
    }

    let mut projected = projection.matvec(&activation)?;
    scale_row(&mut projected, params.projection_scalar);
    rms_norm(
        &mut projected,
        &unpack_i32s(&params.norm_weights).map_err(anyhow::Error::msg)?,
        params.norm_eps,
    )
    .map_err(anyhow::Error::msg)?;

    let mut embedded = ple_embedding(row.token_id, layer, params)?;
    scale_row(&mut embedded, params.embedding_scale);
    for (value, projected_value) in embedded.iter_mut().zip(projected) {
        *value = add_sat(*value, projected_value);
    }
    scale_row(&mut embedded, params.input_scale);

    Ok(PleRow {
        values: pack_i32s(&embedded),
    })
}

fn ple_embedding(token_id: u32, layer: &PleLayer, params: &PleLayerParams) -> Result<Vec<i32>> {
    let page_size = layer.embeddings.page_size();
    let byte_off = token_id as u64 * params.ple_width as u64 * 4;
    let page_idx = if page_size == 0 {
        0
    } else {
        byte_off / page_size
    };
    let page = layer
        .embeddings
        .pages()
        .get(page_idx as usize)
        .ok_or_else(|| {
            anyhow::anyhow!("PLE row width mismatch for token {token_id}: missing embedding page")
        })?;
    unpack_i32s_at(page, byte_off, params.ple_width)
        .map_err(anyhow::Error::msg)
        .map_err(|_| anyhow::anyhow!("PLE row width mismatch for token {token_id}: missing row"))
}
