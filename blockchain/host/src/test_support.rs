use std::fs;
use std::path::{Path, PathBuf};

use model_manifest::{file_sha256, ModelManifest};
use procedure_definition::write_json;
use serde_json::json;

/// Small bundle for exercising preparation without running any model stages.
pub fn model_fixture(base: &Path, name: &str) -> PathBuf {
    let dir = base.join("models").join(name);
    fs::create_dir_all(&dir).unwrap();
    fs::write(dir.join("model.detwgt"), name).unwrap();
    fs::write(dir.join("config.json"), "{}").unwrap();
    fs::write(
        dir.join("tokenizer.json"),
        r#"{"model":{"type":"BPE","vocab":{"h":0},"merges":[]}}"#,
    )
    .unwrap();
    fs::write(dir.join("Raster.toml"), format!(r#"[pipeline]
name = "{name}"

[pipeline.input.prompt_marks]
index = "j"
path = "target/iris/run-prompt-placeholder/marks_{{j}}.rastered"
index_path = "target/iris/run-prompt-placeholder/marks_{{j}}.rindex"
commitments = []

[pipeline.input.prompt_text]
index = "j"
path = "target/iris/run-prompt-placeholder/text_{{j}}.rastered"
index_path = "target/iris/run-prompt-placeholder/text_{{j}}.rindex"
commitments = []

[pipeline.input.claimed_ids]
index = "k"
path = "target/iris/run-prompt-placeholder/claimed_{{k}}.rastered"
index_path = "target/iris/run-prompt-placeholder/claimed_{{k}}.rindex"
commitments = []

[[pipeline.stage]]
name = "prompt_bind_p0"
project = "raster-stages/prompt-bind"
inputs.text = {{ input = "prompt_text_0" }}
inputs.config = {{ external = {{ path = "target/iris/run-prompt-placeholder/config.rastered", index_path = "target/iris/run-prompt-placeholder/config.rindex", commitment = "0" }} }}
inputs.schedule = {{ external = {{ path = "target/iris/run-prompt-placeholder/schedule.rastered", index_path = "target/iris/run-prompt-placeholder/schedule.rindex", commitment = "0" }} }}
inputs.text_digest = {{ external = {{ path = "target/iris/run-prompt-placeholder/text_digest.rastered", index_path = "target/iris/run-prompt-placeholder/text_digest.rindex", commitment = "0" }} }}
inputs.source = {{ external = {{ path = "model/prompt-bind/no_source.rastered", index_path = "model/prompt-bind/no_source.rindex", commitment = "n0" }} }}
inputs.previous = {{ external = {{ path = "empty_prompt.rastered", index_path = "empty_prompt.rindex", commitment = "0" }} }}

[[pipeline.stage]]
name = "prompt_check_p0"
project = "raster-stages/prompt-check"
inputs.bound = {{ from = "prompt_bind_p0" }}
inputs.previous = {{ external = {{ path = "empty_prompt.rastered", index_path = "empty_prompt.rindex", commitment = "0" }} }}
inputs.marks = {{ input = "prompt_marks_0" }}

[[pipeline.repeat]]
name = "prompt"
index = "j"
start = 1
count = 0

  [[pipeline.repeat.stage]]
name = "prompt_bind_p{{j}}"
project = "raster-stages/prompt-bind"
inputs.text = {{ input = "prompt_text_{{j}}" }}
inputs.config = {{ external = {{ path = "target/iris/run-prompt-placeholder/config.rastered", index_path = "target/iris/run-prompt-placeholder/config.rindex", commitment = "0" }} }}
inputs.schedule = {{ external = {{ path = "target/iris/run-prompt-placeholder/schedule.rastered", index_path = "target/iris/run-prompt-placeholder/schedule.rindex", commitment = "0" }} }}
inputs.text_digest = {{ external = {{ path = "target/iris/run-prompt-placeholder/text_digest.rastered", index_path = "target/iris/run-prompt-placeholder/text_digest.rindex", commitment = "0" }} }}
inputs.source = {{ external = {{ path = "model/prompt-bind/no_source.rastered", index_path = "model/prompt-bind/no_source.rindex", commitment = "n0" }} }}
inputs.previous = {{ from = "prompt_bind_p{{j-1}}", first = "prompt_bind_p0" }}

  [[pipeline.repeat.stage]]
name = "prompt_check_p{{j}}"
project = "raster-stages/prompt-check"
inputs.bound = {{ from = "prompt_bind_p{{j}}" }}
inputs.previous = {{ from = "prompt_check_p{{j-1}}", first = "prompt_check_p0" }}
inputs.marks = {{ input = "prompt_marks_{{j}}" }}

[[pipeline.repeat]]
name = "claimed_embed"
index = "k"
count = 0

[[pipeline.repeat]]
name = "layers"
index = "l"
start = 0
count = 1

  [[pipeline.repeat.stage]]
name = "range_p0_l{{l}}"
project = "raster-stages/prefill-range"

  [[pipeline.repeat.stage]]
name = "range_p{{j}}_l{{l}}"
index = "j"
start = 1
count = 0
project = "raster-stages/prefill-range"

  [[pipeline.repeat.stage]]
name = "range_c{{k}}_l{{l}}"
index = "k"
count = 0
project = "raster-stages/prefill-range"

[[pipeline.stage]]
name = "check_c0"
project = "raster-stages/sequence-check"

[[pipeline.repeat]]
name = "claimed"
index = "k"
start = 1
count = 0

"#)).unwrap();
    let commitments = [model_manifest::ModelCommitment {
        role: String::from("embedding"),
        commitment: name.to_string(),
    }];
    let manifest: ModelManifest = serde_json::from_value(json!({
        "version": model_manifest::MODEL_MANIFEST_VERSION,
        "model_id": model_manifest::model_id_of(&commitments),
        "model_commitments": commitments,
        "range_size": 4,
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
        "provenance": {
            "raster_manifest_path": "Raster.toml",
            "raster_manifest_sha256": file_sha256(&dir.join("Raster.toml")).unwrap()
        }
    })).unwrap();
    let path = dir.join("manifest.json");
    write_json(&path, &manifest).unwrap();
    path
}

/// Writes a specification of one model, with the case of a one-letter prompt
/// beside it, and returns the specification's path.
pub fn specification(base: &Path, name: &str) -> PathBuf {
    let dir = base.join("specs");
    fs::create_dir_all(&dir).unwrap();
    let path = dir.join(procedure_definition::SPECIFICATION_TOML);
    fs::write(
        &path,
        format!(
            "raw_prompt = true
max_new_tokens = 3

[[model]]
manifest = \"../models/{name}/manifest.json\"
"
        ),
    )
    .unwrap();
    fs::write(
        case_beside(&path),
        "prompt = \"h\"
",
    )
    .unwrap();
    path
}

/// The case file beside a specification [`specification`] wrote.
pub fn case_beside(specification: &Path) -> PathBuf {
    specification.with_file_name(procedure_definition::CASE_TOML)
}
