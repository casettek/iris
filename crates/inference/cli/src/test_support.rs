use std::fs;
use std::path::{Path, PathBuf};

use inference_artifacts::{file_sha256, write_json, ModelManifest};
use serde_json::json;

/// Small bundle for exercising preparation without running any model stages.
pub(crate) fn model_fixture(base: &Path, name: &str) -> PathBuf {
    let dir = base.join("models").join(name);
    fs::create_dir_all(&dir).unwrap();
    fs::write(dir.join("model.detwgt"), name).unwrap();
    fs::write(dir.join("config.json"), "{}").unwrap();
    fs::write(
        dir.join("tokenizer.json"),
        r#"{"model":{"type":"BPE","vocab":{"h":0},"merges":[]}}"#,
    )
    .unwrap();
    fs::write(dir.join("Raster.toml"), format!(r#"[chain]
name = "{name}"
[[chain.stage]]
name = "prompt_merge_seed"
project = "raster-stages/prompt-merge"
inputs.tokenizer = {{ external = {{ path = "tokenizer.rastered", index_path = "tokenizer.rindex", commitment = "tok" }} }}
inputs.initial_pieces = {{ external = {{ path = "pieces.rastered", index_path = "pieces.rindex", commitment = "pieces" }} }}
[[chain.repeat]]
name = "tokenize"
index = "b"
count = 0
[[chain.stage]]
name = "prompt_prepare"
project = "raster-stages/prompt-prepare"
[[chain.repeat]]
name = "decode"
index = "t"
count = 0
"#)).unwrap();
    let manifest: ModelManifest = serde_json::from_value(json!({
        "version": 2,
        "bundle": {
            "model_detwgt_path": "model.detwgt", "model_detwgt_sha256": file_sha256(&dir.join("model.detwgt")).unwrap(),
            "config_path": "config.json", "config_sha256": file_sha256(&dir.join("config.json")).unwrap(),
            "tokenizer_path": "tokenizer.json", "tokenizer_sha256": file_sha256(&dir.join("tokenizer.json")).unwrap()
        },
        "shape": {
            "hidden_size": 2, "num_hidden_layers": 0, "num_attention_heads": 1,
            "num_key_value_heads": 1, "head_dim": 2, "global_head_dim": 2,
            "vocab_size": 1, "hidden_size_per_layer_input": 1, "sliding_window": 0,
            "layer_types": [], "num_kv_shared_layers": 0, "norm_eps": 0,
            "rope_base_sliding": 0, "rope_base_full": 0, "full_partial_rotary_factor_q16": 0,
            "embedding_scale": 0, "ple_embedding_scale": 0, "ple_projection_scalar": 0,
            "ple_input_scale": 0, "final_logit_softcap": 0
        },
        "eos_token_ids": [],
        "provenance": { "raster_manifest_path": "Raster.toml", "raster_manifest_sha256": file_sha256(&dir.join("Raster.toml")).unwrap() }
    })).unwrap();
    let path = dir.join("manifest.json");
    write_json(&path, &manifest).unwrap();
    path
}

pub(crate) fn run_spec(base: &Path, name: &str) -> PathBuf {
    let dir = base.join("specs");
    fs::create_dir_all(&dir).unwrap();
    let path = dir.join("inference.toml");
    fs::write(&path, format!("model_manifest = \"../models/{name}/manifest.json\"\nprompt = \"h\"\nraw_prompt = true\ntokens = 3\n")).unwrap();
    path
}
