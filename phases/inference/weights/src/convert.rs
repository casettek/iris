//! GGUF to detwgt v3: the one place model weights pass through floating point.
//!
//! `Q8_0` tensors are copied block for block, so their committed bytes are
//! the publisher's. Float tensors (norms, the small per-layer-embedding
//! matrices, scalars) become Q16.16 through `det_num`'s exact conversion. Any
//! tensor the mapping does not know fails the conversion rather than being
//! dropped.

use std::path::Path;

use anyhow::{bail, ensure, Context, Result};

use crate::gguf::{self, Gguf};
use crate::DetwgtWriter;

const PREFIX: &str = "model.language_model.";

/// Tensors whose products and lookups run on `Q8_0` blocks.
const BLOCK_TENSORS: &[&str] = &[
    "embed_tokens.weight",
    "embed_tokens_per_layer.weight",
    "lm_head.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
];

/// Whether a detwgt tensor of this name is stored as `Q8_0` blocks.
pub fn is_block_tensor(name: &str) -> bool {
    BLOCK_TENSORS.iter().any(|suffix| name.ends_with(suffix))
}

/// The detwgt name of a GGUF tensor; `None` for tensors conversion skips.
pub fn detwgt_name(gguf_name: &str) -> Result<Option<String>> {
    let global = match gguf_name {
        "token_embd.weight" => Some("embed_tokens.weight"),
        "per_layer_token_embd.weight" => Some("embed_tokens_per_layer.weight"),
        "per_layer_model_proj.weight" => Some("per_layer_model_projection.weight"),
        "per_layer_proj_norm.weight" => Some("per_layer_projection_norm.weight"),
        "output_norm.weight" => Some("norm.weight"),
        "output.weight" => Some("lm_head.weight"),
        // RoPE comes from config.json; llama.cpp's per-dimension factors are
        // its own encoding of the same partial rotation.
        "rope_freqs.weight" => return Ok(None),
        _ => None,
    };
    if let Some(name) = global {
        return Ok(Some(format!("{PREFIX}{name}")));
    }
    let (layer, field) = gguf_name
        .strip_prefix("blk.")
        .and_then(|rest| rest.split_once('.'))
        .with_context(|| format!("GGUF tensor '{gguf_name}' has no known mapping"))?;
    let field = match field {
        "attn_q.weight" => "self_attn.q_proj.weight",
        "attn_k.weight" => "self_attn.k_proj.weight",
        "attn_v.weight" => "self_attn.v_proj.weight",
        "attn_output.weight" => "self_attn.o_proj.weight",
        "attn_q_norm.weight" => "self_attn.q_norm.weight",
        "attn_k_norm.weight" => "self_attn.k_norm.weight",
        "attn_norm.weight" => "input_layernorm.weight",
        "post_attention_norm.weight" => "post_attention_layernorm.weight",
        "ffn_norm.weight" => "pre_feedforward_layernorm.weight",
        "post_ffw_norm.weight" => "post_feedforward_layernorm.weight",
        "ffn_gate.weight" => "mlp.gate_proj.weight",
        "ffn_up.weight" => "mlp.up_proj.weight",
        "ffn_down.weight" => "mlp.down_proj.weight",
        "inp_gate.weight" => "per_layer_input_gate.weight",
        "proj.weight" => "per_layer_projection.weight",
        "post_norm.weight" => "post_per_layer_input_norm.weight",
        "layer_output_scale.weight" => "layer_scalar",
        _ => bail!("GGUF tensor '{gguf_name}' has no known mapping"),
    };
    let layer: usize = layer
        .parse()
        .with_context(|| format!("GGUF tensor '{gguf_name}' has a malformed layer index"))?;
    Ok(Some(format!("{PREFIX}layers.{layer}.{field}")))
}

/// What conversion wrote.
pub struct Converted {
    pub block_tensors: usize,
    pub q16_tensors: usize,
}

/// Converts `gguf_path` into `out` as detwgt v3. With `dequantize`, block
/// tensors are written as Q16.16 values instead (each weight rounded once),
/// for running the weights through code that predates `Q8_0`.
pub fn convert(gguf_path: &Path, out: &Path, dequantize: bool) -> Result<Converted> {
    let gguf = Gguf::open(gguf_path)?;
    let mut plan = Vec::new();
    for (name, info) in &gguf.tensors {
        if let Some(target) = detwgt_name(name)? {
            plan.push((target, name, info));
        }
    }
    plan.sort_by(|a, b| a.0.cmp(&b.0));
    let mut writer = DetwgtWriter::create(out, plan.len() as u64)?;
    let mut converted = Converted {
        block_tensors: 0,
        q16_tensors: 0,
    };
    for (target, name, info) in plan {
        // GGUF lists the column dimension first; detwgt lists it last.
        let dims = info.dims.iter().rev().copied().collect::<Vec<_>>();
        let data = gguf.data(name)?;
        if is_block_tensor(&target) {
            ensure!(
                info.kind == gguf::Q8_0,
                "'{name}' must be Q8_0 to become '{target}', found GGUF type {}",
                info.kind
            );
            if dequantize {
                let cols = *dims.last().unwrap_or(&0) as usize;
                let row_bytes = det_num::q8::row_bytes(cols).context("Q8_0 row width")?;
                let mut values = vec![0; info.element_count() as usize];
                for (row, out) in data
                    .chunks_exact(row_bytes)
                    .zip(values.chunks_exact_mut(cols))
                {
                    det_num::q8::dequantize_row(row, out).map_err(anyhow::Error::msg)?;
                }
                writer.q16(&target, &dims, &values)?;
            } else {
                writer.q8_0(&target, &dims, data)?;
            }
            converted.block_tensors += 1;
        } else {
            let values = float_values(info.kind, data)
                .with_context(|| format!("'{name}' cannot become Q16.16 '{target}'"))?;
            writer.q16(&target, &dims, &values)?;
            converted.q16_tensors += 1;
        }
    }
    writer.finish()?;
    Ok(converted)
}

/// Exact Q16.16 values of a float tensor.
fn float_values(kind: u32, data: &[u8]) -> Result<Vec<i32>> {
    let to_q16 = |x: f32| -> Result<i32> {
        ensure!(x.is_finite(), "tensor holds a non-finite value");
        Ok(det_num::f32_to_wgt(x).to_bits())
    };
    match kind {
        gguf::F32 => data
            .chunks_exact(4)
            .map(|b| to_q16(f32::from_le_bytes(b.try_into().unwrap())))
            .collect(),
        gguf::BF16 => data
            .chunks_exact(2)
            .map(|b| {
                to_q16(f32::from_bits(
                    u32::from(u16::from_le_bytes([b[0], b[1]])) << 16,
                ))
            })
            .collect(),
        other => bail!("unsupported float type {other}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_map_to_the_hugging_face_layout() {
        assert_eq!(
            detwgt_name("blk.12.ffn_down.weight").unwrap().as_deref(),
            Some("model.language_model.layers.12.mlp.down_proj.weight")
        );
        assert_eq!(
            detwgt_name("blk.0.layer_output_scale.weight")
                .unwrap()
                .as_deref(),
            Some("model.language_model.layers.0.layer_scalar")
        );
        assert_eq!(
            detwgt_name("token_embd.weight").unwrap().as_deref(),
            Some("model.language_model.embed_tokens.weight")
        );
        assert_eq!(detwgt_name("rope_freqs.weight").unwrap(), None);
        assert!(detwgt_name("blk.0.mystery.weight").is_err());
        assert!(detwgt_name("vision.weight").is_err());
        assert!(is_block_tensor(
            "model.language_model.layers.3.self_attn.k_proj.weight"
        ));
        assert!(!is_block_tensor(
            "model.language_model.layers.3.self_attn.k_norm.weight"
        ));
        assert!(!is_block_tensor(
            "model.language_model.per_layer_model_projection.weight"
        ));
    }
}
