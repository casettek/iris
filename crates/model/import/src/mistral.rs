//! What is particular to importing a `mistral-dense` model: Ministral 3.
//!
//! Its shape, the tensors a bundle may hold, and its layer externals. The
//! tokenizer, the embedding table and the head are written by the shared code.

use std::collections::BTreeSet;
use std::error::Error;

use inference_artifacts::MistralShape;
use inference_kernels::mistral::num::{attention_scale_q32, yarn_inverse_frequencies};
use layer_mistral::input::{LayerParams, TransformerLayer};

use super::externals::page_of_i32s;
use super::{external_line, mapped_weights, write_external, ArtifactLayout, ChainLayers};

const LAYER_PROJECT: &str = "raster-stages/layer-mistral";
const LANGUAGE_MODEL: &str = "model.language_model.";

/// The tensors of the language model that the import reads, per model and per
/// layer.
const MODEL_TENSORS: &[&str] = &["embed_tokens.weight", "norm.weight"];
const LAYER_TENSORS: &[&str] = &[
    "input_layernorm.weight",
    "pre_feedforward_layernorm.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
];

/// Refuses a bundle with language-model weights the import does not read: it
/// would import cleanly as a model that computes something else.
pub(super) fn require_known_tensors<'a>(
    names: impl Iterator<Item = &'a str>,
    layers: u32,
) -> Result<(), Box<dyn Error>> {
    let known = |name: &str| {
        let Some(rest) = name.strip_prefix(LANGUAGE_MODEL) else {
            // Another tower's weights are not this chain's.
            return true;
        };
        if MODEL_TENSORS.contains(&rest) {
            return true;
        }
        rest.strip_prefix("layers.")
            .and_then(|rest| rest.split_once('.'))
            .is_some_and(|(index, suffix)| {
                index.parse::<u32>().is_ok_and(|index| index < layers)
                    && LAYER_TENSORS.contains(&suffix)
            })
    };
    let unknown: Vec<&str> = names.filter(|name| !known(name)).collect();
    if unknown.is_empty() {
        return Ok(());
    }
    Err(format!(
        "the bundle has {} language-model tensor(s) a mistral-dense import does not read, \
         first: {}",
        unknown.len(),
        unknown[0]
    )
    .into())
}

/// What the chain template needs of this model's layers: each one runs alone,
/// after the layer below, with its own cache.
pub(super) fn chain_layers(shape: &MistralShape) -> ChainLayers {
    ChainLayers {
        layers: shape.num_hidden_layers as usize,
        project: LAYER_PROJECT,
        stage: "layer_mistral",
        auxiliary: false,
        donor_inputs: false,
        num_kv_shared_layers: 0,
        donors: Vec::new(),
    }
}

/// Writes one external per layer: its parameters and its seven matrices.
pub(super) fn write_layers(
    weights: &detwgt::MmapDetwgt,
    shape: &MistralShape,
    layout: &ArtifactLayout,
    stages: &mut Vec<String>,
) -> Result<(), Box<dyn Error>> {
    let frequencies = raster::BytesPage::__from_parts(
        0,
        0,
        shape
            .rope_inverse_frequencies
            .iter()
            .flat_map(|frequency| frequency.to_le_bytes())
            .collect(),
    );
    let hidden = u64::from(shape.hidden_size);
    let ffn = u64::from(shape.intermediate_size);
    let q_width = u64::from(shape.num_attention_heads) * u64::from(shape.head_dim);
    let kv_width = u64::from(shape.num_key_value_heads) * u64::from(shape.head_dim);
    for layer_idx in 0..shape.num_hidden_layers {
        let at = |suffix: &str| format!("{LANGUAGE_MODEL}layers.{layer_idx}.{suffix}");
        // A matrix of another shape would stop the layer's stage, not the import.
        for (suffix, rows, cols) in [
            ("self_attn.q_proj.weight", q_width, hidden),
            ("self_attn.k_proj.weight", kv_width, hidden),
            ("self_attn.v_proj.weight", kv_width, hidden),
            ("self_attn.o_proj.weight", hidden, q_width),
            ("mlp.gate_proj.weight", ffn, hidden),
            ("mlp.up_proj.weight", ffn, hidden),
            ("mlp.down_proj.weight", hidden, ffn),
        ] {
            let dims = weights.tensor(&at(suffix))?.dims.clone();
            if dims.len() != 2 || dims[0] as u64 != rows || dims[1] as u64 != cols {
                return Err(format!(
                    "{} is {dims:?}, and this model's shape needs {rows} x {cols}",
                    at(suffix)
                )
                .into());
            }
        }
        let layer = TransformerLayer {
            params: LayerParams {
                layer_idx,
                hidden_size: shape.hidden_size,
                ffn_size: shape.intermediate_size,
                num_heads: shape.num_attention_heads,
                num_kv_heads: shape.num_key_value_heads,
                head_dim: shape.head_dim,
                norm_eps: shape.norm_eps,
                attention_scale_q32: shape.attention_scale_q32,
                max_positions: shape.max_positions,
                rope_inverse_frequencies: frequencies.clone(),
                norm_attention: page_of_i32s(&weights.values(&at("input_layernorm.weight"))?),
                norm_feed_forward: page_of_i32s(
                    &weights.values(&at("pre_feedforward_layernorm.weight"))?,
                ),
            },
            w_q: mapped_weights::q8_tensor(weights, &at("self_attn.q_proj.weight"))?,
            w_k: mapped_weights::q8_tensor(weights, &at("self_attn.k_proj.weight"))?,
            w_v: mapped_weights::q8_tensor(weights, &at("self_attn.v_proj.weight"))?,
            w_o: mapped_weights::q8_tensor(weights, &at("self_attn.o_proj.weight"))?,
            w_gate: mapped_weights::q8_tensor(weights, &at("mlp.gate_proj.weight"))?,
            w_up: mapped_weights::q8_tensor(weights, &at("mlp.up_proj.weight"))?,
            w_down: mapped_weights::q8_tensor(weights, &at("mlp.down_proj.weight"))?,
        };
        layer_mistral::check_layer_params(&layer.params)?;

        let name = format!("layer{layer_idx}");
        let commitment = write_external(&layer, layout, LAYER_PROJECT, &name)?;
        let upstream = match layer_idx.checked_sub(1) {
            Some(below) => format!("layer_mistral_l{below}"),
            None => String::from("input_embedding"),
        };
        stages.push(format!(
            "[[chain.stage]]\n\
             name = \"layer_mistral_l{layer_idx}\"\n\
             project = \"{LAYER_PROJECT}\"\n\
             inputs.activations = {{ from = \"{upstream}\" }}\n\
             inputs.prior_kv = {{ from = \"input_embedding\" }}\n\
             {}",
            external_line("layer", layout, LAYER_PROJECT, &name, &commitment)
        ));
    }
    Ok(())
}

/// The model types a `mistral-dense` bundle's `config.json` declares.
pub(super) fn is_mistral_dense(model_type: &str) -> bool {
    matches!(model_type, "ministral3" | "mistral3")
}

fn whole(text: &serde_json::Value, key: &str) -> Result<u64, Box<dyn Error>> {
    text.get(key)
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| format!("config.json text_config has no {key}").into())
}

fn real(section: &serde_json::Value, key: &str) -> Result<f64, Box<dyn Error>> {
    section
        .get(key)
        .and_then(serde_json::Value::as_f64)
        .ok_or_else(|| format!("config.json rope_parameters has no {key}").into())
}

/// YaRN's correction dimension for a number of rotations, as the reference
/// implementations compute it.
fn correction_dim(dims: f64, original_positions: f64, rotations: f64, base: f64) -> f64 {
    dims * (original_positions / (rotations * 2.0 * std::f64::consts::PI)).ln() / (2.0 * base.ln())
}

/// A bound that must not sit on a whole number, where rounding it down or up
/// would depend on the last bit of a logarithm.
fn settled(value: f64) -> Result<f64, Box<dyn Error>> {
    if (value - value.round()).abs() < 1e-6 {
        return Err(format!("a YaRN bound of {value} is too close to a whole number").into());
    }
    Ok(value)
}

pub(super) fn shape_from_config(text: &serde_json::Value) -> Result<MistralShape, Box<dyn Error>> {
    if text.get("hidden_act").and_then(serde_json::Value::as_str) != Some("silu") {
        return Err("a mistral-dense model's hidden_act must be silu".into());
    }
    if !text
        .get("sliding_window")
        .is_none_or(serde_json::Value::is_null)
    {
        return Err("a mistral-dense model has no sliding window".into());
    }
    let head_dim = whole(text, "head_dim")? as u32;
    let rope = text
        .get("rope_parameters")
        .ok_or("config.json text_config has no rope_parameters")?;
    if rope.get("rope_type").and_then(serde_json::Value::as_str) != Some("yarn") {
        return Err("a mistral-dense model's rope_type must be yarn".into());
    }
    // Both scale a rotation's length. At one they cancel, and nothing here
    // applies them.
    for key in ["mscale", "mscale_all_dim"] {
        if real(rope, key)? != 1.0 {
            return Err(format!("rope_parameters.{key} must be 1").into());
        }
    }
    let base = real(rope, "rope_theta")?;
    let factor = real(rope, "factor")?;
    let original = real(rope, "original_max_position_embeddings")?;
    if base.fract() != 0.0 || factor.fract() != 0.0 || original.fract() != 0.0 {
        return Err("rope_theta, factor and original_max_position_embeddings must be whole".into());
    }
    let dims = f64::from(head_dim);
    let low = settled(correction_dim(
        dims,
        original,
        real(rope, "beta_fast")?,
        base,
    ))?
    .floor()
    .max(0.0);
    let high = settled(correction_dim(
        dims,
        original,
        real(rope, "beta_slow")?,
        base,
    ))?
    .ceil()
    .min(dims - 1.0);
    let rope_inverse_frequencies = yarn_inverse_frequencies(
        base as u64,
        head_dim / 2,
        factor as u32,
        low as u32,
        high as u32,
    )
    .ok_or("the model's rotation needs head_dim / 2 to be a power of two")?;
    // The model multiplies a query by `1 + beta * ln(floor(position / original) + 1)`,
    // which is one for every position below `original`. Past it the term is
    // not built, so those positions are refused.
    let max_positions = match rope
        .get("llama_4_scaling_beta")
        .and_then(serde_json::Value::as_f64)
    {
        Some(beta) if beta != 0.0 => original as u32,
        _ => whole(text, "max_position_embeddings")? as u32,
    };
    Ok(MistralShape {
        hidden_size: whole(text, "hidden_size")? as u32,
        num_hidden_layers: whole(text, "num_hidden_layers")? as u32,
        num_attention_heads: whole(text, "num_attention_heads")? as u32,
        num_key_value_heads: whole(text, "num_key_value_heads")? as u32,
        head_dim,
        intermediate_size: whole(text, "intermediate_size")? as u32,
        vocab_size: whole(text, "vocab_size")? as u32,
        norm_eps: det_num::f32_to_acc(
            text.get("rms_norm_eps")
                .and_then(serde_json::Value::as_f64)
                .ok_or("config.json text_config has no rms_norm_eps")? as f32,
        )
        .to_bits(),
        attention_scale_q32: attention_scale_q32(head_dim),
        rope_inverse_frequencies,
        max_positions,
        tied_head: text
            .get("tie_word_embeddings")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false),
    })
}

/// The token that ends an answer. The config does not give its id, so it is
/// the tokenizer's `</s>`.
pub(super) fn eos_ids(tokenizer: &serde_json::Value) -> Result<BTreeSet<u32>, Box<dyn Error>> {
    let id = tokenizer
        .get("model")
        .and_then(|model| model.get("vocab"))
        .and_then(|vocab| vocab.get("</s>"))
        .and_then(serde_json::Value::as_u64)
        .ok_or("tokenizer.json has no </s> token")?;
    Ok(BTreeSet::from([id as u32]))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> serde_json::Value {
        serde_json::json!({
            "head_dim": 128, "hidden_act": "silu", "hidden_size": 3072,
            "intermediate_size": 9216, "max_position_embeddings": 262144,
            "model_type": "ministral3", "num_attention_heads": 32,
            "num_hidden_layers": 26, "num_key_value_heads": 8, "rms_norm_eps": 1e-05,
            "rope_parameters": {
                "beta_fast": 32.0, "beta_slow": 1.0, "factor": 16.0,
                "llama_4_scaling_beta": 0.1, "mscale": 1.0, "mscale_all_dim": 1.0,
                "original_max_position_embeddings": 16384, "rope_theta": 1000000.0,
                "rope_type": "yarn", "type": "yarn"
            },
            "sliding_window": null, "tie_word_embeddings": true, "vocab_size": 131072
        })
    }

    #[test]
    fn ministral_3b_shape() {
        let shape = shape_from_config(&config()).unwrap();
        assert_eq!(shape.rope_inverse_frequencies.len(), 64);
        assert_eq!(shape.rope_inverse_frequencies[0], 1 << 32);
        // Pairs below 20 keep their frequency; pairs from 37 up have a sixteenth.
        let plain = |pair: f64| 1_000_000_f64.powf(-pair / 64.0);
        let real = |pair: usize| shape.rope_inverse_frequencies[pair] as f64 / 4_294_967_296.0;
        assert!((real(19) - plain(19.0)).abs() < 1e-8);
        assert!((real(40) - plain(40.0) / 16.0).abs() < 1e-8);
        assert_eq!(shape.max_positions, 16_384);
        assert_eq!(shape.attention_scale_q32, 379_625_062);
        assert!(shape.tied_head);
    }

    #[test]
    fn refuses_what_the_layer_does_not_compute() {
        let mut gelu = config();
        gelu["hidden_act"] = serde_json::json!("gelu");
        assert!(shape_from_config(&gelu).is_err());
        let mut windowed = config();
        windowed["sliding_window"] = serde_json::json!(4096);
        assert!(shape_from_config(&windowed).is_err());
        let mut scaled = config();
        scaled["rope_parameters"]["mscale"] = serde_json::json!(2.0);
        assert!(shape_from_config(&scaled).is_err());
    }
}
