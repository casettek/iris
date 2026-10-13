use std::collections::BTreeSet;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

use inference_artifacts::{
    write_json, DirectInferBundle, DirectInferProvenance, DirectInferShape, ModelManifest,
    MODEL_MANIFEST_JSON,
};
use sha2::{Digest, Sha256};

use super::{det_act_bits, f32_to_q16, ImportConfig, Shape};

pub fn write_model_manifest(
    args: &ImportConfig,
    eos_ids: &BTreeSet<u32>,
    shape: &Shape,
    text_config: &serde_json::Value,
    raster_manifest: Option<(&Path, &str)>,
) -> Result<PathBuf, Box<dyn Error>> {
    let artifact_dir = args.artifact_dir();
    fs::create_dir_all(&artifact_dir)?;
    let manifest_path = artifact_dir.join(MODEL_MANIFEST_JSON);
    let model_detwgt = args.model_dir.join("model.detwgt");
    let config = args.model_dir.join("config.json");
    let tokenizer_path = args.model_dir.join("tokenizer.json");

    let direct = ModelManifest {
        version: 2,
        bundle: DirectInferBundle {
            model_detwgt_path: manifest_relative_path(&artifact_dir, &model_detwgt)?,
            model_detwgt_sha256: sha256_file(&model_detwgt)?,
            config_path: manifest_relative_path(&artifact_dir, &config)?,
            config_sha256: sha256_file(&config)?,
            tokenizer_path: manifest_relative_path(&artifact_dir, &tokenizer_path)?,
            tokenizer_sha256: sha256_file(&tokenizer_path)?,
        },
        shape: DirectInferShape {
            hidden_size: shape.hidden as u32,
            num_hidden_layers: shape.layers as u32,
            num_attention_heads: shape.heads as u32,
            num_key_value_heads: shape.kv_heads as u32,
            head_dim: shape.head_dim as u32,
            global_head_dim: shape.global_head_dim as u32,
            vocab_size: shape.vocab as u32,
            hidden_size_per_layer_input: shape.ple_width as u32,
            sliding_window: shape.sliding_window,
            layer_types: shape.layer_types.clone(),
            num_kv_shared_layers: shape.num_kv_shared_layers as u32,
            norm_eps: shape.norm_eps,
            rope_base_sliding: (shape.rope_base_sliding as i64) << 32,
            rope_base_full: (shape.rope_base_full as i64) << 32,
            full_partial_rotary_factor_q16: f32_to_q16(shape.full_partial_rotary_factor),
            embedding_scale: det_act_bits((shape.hidden as f32).sqrt()),
            ple_embedding_scale: det_act_bits((shape.ple_width as f32).sqrt()),
            ple_projection_scalar: det_act_bits((shape.hidden as f32).powf(-0.5)),
            ple_input_scale: det_act_bits(2f32.powf(-0.5)),
            final_logit_softcap: text_config
                .get("final_logit_softcapping")
                .and_then(serde_json::Value::as_f64)
                .map(f32_to_q16)
                .unwrap_or(0),
        },
        eos_token_ids: eos_ids.iter().copied().collect(),
        provenance: raster_manifest
            .map(|(path, text)| -> Result<_, Box<dyn Error>> {
                Ok(DirectInferProvenance {
                    raster_manifest_path: manifest_relative_path(&artifact_dir, path)?,
                    raster_manifest_sha256: format!("{:x}", Sha256::digest(text.as_bytes())),
                })
            })
            .transpose()?,
    };

    write_json(&manifest_path, &direct)
        .map_err(|error| -> Box<dyn Error> { error.to_string().into() })?;
    println!("wrote {}", manifest_path.display());
    Ok(manifest_path)
}

pub fn warn_partial_direct_manifest_not_refreshed(args: &ImportConfig) {
    eprintln!(
        "model manifest not refreshed; run a full `raster-inference model import ...` \
         to regenerate {}",
        args.artifact_dir().join(MODEL_MANIFEST_JSON).display()
    );
}

fn manifest_relative_path(artifact_dir: &Path, source: &Path) -> Result<PathBuf, Box<dyn Error>> {
    let dir = fs::canonicalize(artifact_dir)?;
    let source = fs::canonicalize(source)?;
    Ok(source.strip_prefix(dir).unwrap_or(&source).to_path_buf())
}

fn sha256_file(path: &Path) -> Result<String, Box<dyn Error>> {
    inference_artifacts::file_sha256(path).map_err(|error| error.to_string().into())
}
