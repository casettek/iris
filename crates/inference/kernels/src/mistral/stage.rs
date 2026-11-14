//! The `layer-mistral` stage, natively: what the stage program computes, from
//! the same committed inputs.
//!
//! The stage's activations are the pipeline's activation type, which every
//! stage defines field for field. `prefill-range`'s definition is used here,
//! as the native paths of the stages around this one use it.

use anyhow::{bail, Result};
use layer_mistral::input::{LayerParams as StageParams, TransformerLayer};
use prefill_range::input::ActivationSequence;
use raster::{Bytes, BytesPage, List};

use super::layer::{layer_rows_and_keys, LayerMatrices, LayerParams};
use crate::tensor::MatrixSource;

pub struct LayerStageInputs<'a> {
    pub activations: &'a ActivationSequence,
    pub layer: &'a TransformerLayer,
    pub prior_kv: &'a ActivationSequence,
}

/// The stage's committed parameters as the native layer takes them.
pub fn native_params(params: &StageParams) -> Result<LayerParams> {
    layer_mistral::check_layer_params(params).map_err(anyhow::Error::msg)?;
    let values = |page| layer_mistral::input::unpack_i32s(page).map_err(anyhow::Error::msg);
    Ok(LayerParams {
        layer_idx: params.layer_idx,
        hidden_size: params.hidden_size,
        ffn_size: params.ffn_size,
        num_heads: params.num_heads,
        num_kv_heads: params.num_kv_heads,
        head_dim: params.head_dim,
        norm_eps: params.norm_eps,
        attention_scale_q32: params.attention_scale_q32,
        rope_inverse_frequencies: layer_mistral::input::unpack_i64s(
            &params.rope_inverse_frequencies,
        )
        .map_err(anyhow::Error::msg)?,
        max_positions: params.max_positions,
        norm_attention: values(&params.norm_attention)?,
        norm_feed_forward: values(&params.norm_feed_forward)?,
    })
}

/// A committed `Q8_0` matrix read where it lies: its pages hold whole rows.
pub struct PagedQ8<'a> {
    pages: &'a [BytesPage],
    rows: usize,
    cols: usize,
    row_bytes: usize,
    rows_per_page: usize,
}

impl<'a> PagedQ8<'a> {
    pub fn new<const P: u64>(
        name: &str,
        region: &'a Bytes<P>,
        rows: usize,
        cols: usize,
    ) -> Result<Self> {
        let Some(row_bytes) = det_num::q8::row_bytes(cols) else {
            bail!("{name} rows of {cols} values are not whole Q8_0 blocks");
        };
        if region.page_size() != P || P as usize % row_bytes != 0 {
            bail!("{name} pages of {P} bytes do not hold whole rows of {row_bytes}");
        }
        if region.byte_len() != (rows * row_bytes) as u64 {
            bail!(
                "{name} has {} bytes, expected {} ({rows} x {cols} Q8_0)",
                region.byte_len(),
                rows * row_bytes
            );
        }
        let pages = region.pages().as_slice();
        let rows_per_page = P as usize / row_bytes;
        for (index, page) in pages.iter().enumerate() {
            let rows_here = rows_per_page.min(rows - (index * rows_per_page).min(rows));
            if page.len() != rows_here * row_bytes {
                bail!("{name} page {index} is {} bytes", page.len());
            }
        }
        if pages.len() != rows.div_ceil(rows_per_page) {
            bail!("{name} has {} pages for {rows} rows", pages.len());
        }
        Ok(Self {
            pages,
            rows,
            cols,
            row_bytes,
            rows_per_page,
        })
    }

    fn row(&self, row_idx: usize) -> &[u8] {
        let start = (row_idx % self.rows_per_page) * self.row_bytes;
        &self.pages[row_idx / self.rows_per_page].as_slice()[start..start + self.row_bytes]
    }
}

impl MatrixSource for PagedQ8<'_> {
    fn rows(&self) -> usize {
        self.rows
    }

    fn cols(&self) -> usize {
        self.cols
    }

    fn row_values_into(&self, _row_idx: usize, _out: &mut Vec<i32>) -> Result<()> {
        bail!("Q8_0 rows have no Q16.16 values; products use row_dot")
    }

    fn row_dot(&self, row_idx: usize, input: &[i32]) -> Result<i32> {
        if row_idx >= self.rows {
            bail!(
                "deterministic linear row index {row_idx} out of range for {} rows",
                self.rows
            );
        }
        det_num::q8::dot_row(input, self.row(row_idx)).map_err(anyhow::Error::msg)
    }
}

/// The stage's output for one range.
///
/// The context checks are the stage's own: after the layer below, with this
/// layer's cache from the range before, which must hold every earlier
/// position, over the rows the context names. A range that fails them passes
/// on no rows and carries the cache, as the stage does.
pub fn run_layer_stage(inputs: LayerStageInputs<'_>) -> Result<ActivationSequence> {
    let params = native_params(&inputs.layer.params)?;
    let start_position = inputs.activations.start_position;
    let mut model = inputs
        .activations
        .model
        .plain_layer(params.layer_idx, &inputs.prior_kv.model);
    let mut coverage = layer_mistral::cache_coverage(&inputs.prior_kv.model);
    for key in inputs.prior_kv.kv.iter() {
        layer_mistral::consume_cache_position(&mut coverage, key.position);
    }
    model = layer_mistral::validate_cache_coverage(model, coverage);
    model.validate_rows(start_position, inputs.activations.rows.len() as u64);

    let carried = inputs.prior_kv.kv.as_slice().to_vec();
    let refused = model.execution_fault != 0 && !model.execution.as_slice().is_empty();
    let computed = if refused {
        None
    } else {
        let hidden = params.hidden_size as usize;
        let ffn = params.ffn_size as usize;
        let q_len = (params.num_heads * params.head_dim) as usize;
        let kv_len = (params.num_kv_heads * params.head_dim) as usize;
        let layer = inputs.layer;
        let w_q = PagedQ8::new("w_q", &layer.w_q, q_len, hidden)?;
        let w_k = PagedQ8::new("w_k", &layer.w_k, kv_len, hidden)?;
        let w_v = PagedQ8::new("w_v", &layer.w_v, kv_len, hidden)?;
        let w_o = PagedQ8::new("w_o", &layer.w_o, hidden, q_len)?;
        let w_gate = PagedQ8::new("w_gate", &layer.w_gate, ffn, hidden)?;
        let w_up = PagedQ8::new("w_up", &layer.w_up, ffn, hidden)?;
        let w_down = PagedQ8::new("w_down", &layer.w_down, hidden, ffn)?;
        layer_rows_and_keys(
            inputs.activations,
            &carried,
            &LayerMatrices {
                w_q: &w_q,
                w_k: &w_k,
                w_v: &w_v,
                w_o: &w_o,
                w_gate: &w_gate,
                w_up: &w_up,
                w_down: &w_down,
            },
            &params,
        )?
    };
    let (rows, own_keys) = computed.unwrap_or_default();
    Ok(ActivationSequence {
        model,
        rows: List::from(rows),
        errors: List::new(),
        kv: List::from(carried.into_iter().chain(own_keys).collect::<Vec<_>>()),
        start_position,
    })
}
