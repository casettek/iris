use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use model_manifest::{
    DirectInferBundle, DirectInferShape, ModelManifest, MODEL_ARTIFACTS_DIR, MODEL_MANIFEST_JSON,
};
use procedure_definition::{write_json, CASE_TOML, SPECIFICATION_TOML};
use procedure_direct::{DirectInferenceConfig, DirectInferenceExecutor};
use sha2::{Digest, Sha256};

const ONE: i32 = 1 << 16;
/// Hidden width of the fixtures: one `Q8_0` block, the narrowest int8 row.
const W: usize = 32;

struct TensorFixture {
    name: &'static str,
    dims: Vec<u64>,
    element_width: u32,
    values: Vec<i32>,
}

#[test]
fn direct_executor_reports_missing_direct_manifest_message() {
    let dir = std::env::temp_dir().join(format!(
        "direct-infer-missing-manifest-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time should move forward")
            .as_nanos()
    ));
    fs::create_dir_all(&dir).expect("temp dir should create");

    let error = DirectInferenceExecutor
        .run(DirectInferenceConfig {
            base_dir: dir.clone(),
            specification_path: dir.join(SPECIFICATION_TOML),
            case_path: dir.join(CASE_TOML),
            inference: 0,
        })
        .expect_err("missing manifest should fail");

    assert!(!format!("{error:#}").contains("scaffolded"));
    assert!(format!("{error:#}").contains("failed to load specification"));

    let _ = fs::remove_dir_all(dir);
}

#[test]
#[ignore = "requires local Gemma model artifacts and runs the full direct inference path"]
fn current_say_ciao_direct_first_token_matches_claim_reference() {
    let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("repo root should resolve");
    let result = DirectInferenceExecutor
        .run(DirectInferenceConfig {
            base_dir: repo.clone(),
            specification_path: repo.join(SPECIFICATION_TOML),
            case_path: repo.join(CASE_TOML),
            inference: 0,
        })
        .unwrap();

    assert_eq!(result.generated_token_ids.first().copied(), Some(150917));
    // The specification may ask for more than one token.
    assert!(result.generated_text.starts_with("Ciao"));
}

#[test]
fn direct_executor_runs_from_direct_manifest_without_raster_toml() {
    let dir = temp_dir("runtime");
    let artifact_dir = dir.join(MODEL_ARTIFACTS_DIR);
    fs::create_dir_all(&artifact_dir).unwrap();

    let detwgt = dir.join("model.detwgt");
    write_detwgt(
        &detwgt,
        &[
            TensorFixture {
                name: "model.language_model.embed_tokens.weight",
                dims: vec![3, W as u64],
                element_width: 8,
                values: one_hot_rows(&[None, Some(0), Some(1)]),
            },
            TensorFixture {
                name: "model.language_model.lm_head.weight",
                dims: vec![3, W as u64],
                element_width: 8,
                values: one_hot_rows(&[None, None, Some(0)]),
            },
            TensorFixture {
                name: "model.language_model.norm.weight",
                dims: vec![W as u64],
                element_width: 32,
                values: vec![ONE; W],
            },
        ],
    );
    let config = dir.join("config.json");
    fs::write(&config, r#"{"text_config":{"tie_word_embeddings":false}}"#).unwrap();
    let tokenizer = dir.join("tokenizer.json");
    fs::write(
        &tokenizer,
        r#"{"model":{"type":"BPE","vocab":{"<pad>":0,"h":1,"Hi":2},"merges":[]},"added_tokens":[]}"#,
    )
    .unwrap();

    let manifest_path = artifact_dir.join(MODEL_MANIFEST_JSON);
    write_json(
        &manifest_path,
        &ModelManifest {
            version: model_manifest::MODEL_MANIFEST_VERSION,
            family: model_manifest::ModelFamily::GemmaDense.name().to_string(),
            model_id: model_manifest::model_id_of(&[]),
            model_commitments: Vec::new(),
            range_size: 512,
            bundle: DirectInferBundle {
                model_detwgt_path: PathBuf::from("../../model.detwgt"),
                model_detwgt_sha256: sha256_file(&detwgt),
                config_path: PathBuf::from("../../config.json"),
                config_sha256: sha256_file(&config),
                tokenizer_path: PathBuf::from("../../tokenizer.json"),
                tokenizer_sha256: sha256_file(&tokenizer),
            },
            shape: DirectInferShape {
                hidden_size: W as u32,
                num_hidden_layers: 0,
                num_attention_heads: 1,
                num_key_value_heads: 1,
                head_dim: 2,
                global_head_dim: 2,
                vocab_size: 3,
                hidden_size_per_layer_input: 1,
                sliding_window: 0,
                layer_types: Vec::new(),
                num_kv_shared_layers: 0,
                norm_eps: 0,
                rope_base_sliding: 10_000_i64 << 32,
                rope_base_full: 1_000_000_i64 << 32,
                full_partial_rotary_factor_q16: ONE,
                embedding_scale: ONE,
                ple_embedding_scale: ONE,
                ple_projection_scalar: ONE,
                ple_input_scale: ONE,
                final_logit_softcap: 0,
            }
            .into(),
            eos_token_ids: Vec::new(),
            provenance: None,
        },
    )
    .unwrap();
    write_specification(&dir, &manifest_path, "h", 1);

    let result = DirectInferenceExecutor
        .run(DirectInferenceConfig {
            base_dir: dir.clone(),
            specification_path: dir.join(SPECIFICATION_TOML),
            case_path: dir.join(CASE_TOML),
            inference: 0,
        })
        .unwrap();

    assert_eq!(result.generated_token_ids, vec![2]);
    assert_eq!(
        result.generated_token_ids_sha256,
        format!("{:x}", Sha256::digest(b"[2]"))
    );
    assert_eq!(result.generated_text, "Hi");
    assert_eq!(result.stop_reason, "max_new_tokens");
    assert!(!dir.join("Raster.toml").exists());

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn direct_executor_stops_at_eos_and_reports_only_selected_tokens() {
    for eos_id in [2, 0] {
        let dir = temp_dir("eos");
        fs::create_dir_all(&dir).unwrap();
        write_direct_fixture(&dir);
        let manifest_path = dir.join(MODEL_ARTIFACTS_DIR).join(MODEL_MANIFEST_JSON);
        let mut manifest: ModelManifest = procedure_definition::read_json(&manifest_path).unwrap();
        manifest.eos_token_ids = vec![eos_id];
        write_json(&manifest_path, &manifest).unwrap();
        let specification_path = write_specification(&dir, &manifest_path, "h", 8);
        let report = DirectInferenceExecutor
            .run_with_diagnostics(DirectInferenceConfig {
                base_dir: dir.clone(),
                specification_path,
                case_path: dir.join(CASE_TOML),
                inference: 0,
            })
            .unwrap();
        // The tiny model selects 2 first, then 0 after embedding token 2.
        let expected = if eos_id == 2 { vec![2] } else { vec![2, 0] };
        assert_eq!(report.report.input_token_count, Some(1));
        assert_eq!(report.report.result.generated_token_ids, expected);
        assert_eq!(
            report.report.result.generated_token_count as usize,
            expected.len()
        );
        assert_eq!(report.report.result.stop_reason, "eos");
        assert_eq!(
            report.report.result.generated_token_ids_sha256,
            format!(
                "{:x}",
                Sha256::digest(serde_json::to_vec(&expected).unwrap())
            )
        );
        let last = expected.len() - 1;
        assert!(report
            .boundaries
            .iter()
            .any(|b| b.stage == format!("decode_select_t{last}")));
        assert!(!report
            .boundaries
            .iter()
            .any(|b| b.stage == format!("decode_embed_t{last}")));
        assert!(!report
            .boundaries
            .iter()
            .any(|b| b.stage == format!("decode_select_t{}", last + 1)));
        // Stopping at EOS short of the budget is within the run's rules.
        assert_eq!(report.report.result.output_fault, 0);
        fs::remove_dir_all(dir).unwrap();
    }
}

#[test]
fn direct_executor_keeps_a_maximum_budget_when_no_eos_is_selected() {
    let dir = temp_dir("token-limit");
    fs::create_dir_all(&dir).unwrap();
    write_direct_fixture(&dir);
    let manifest_path = dir.join(MODEL_ARTIFACTS_DIR).join(MODEL_MANIFEST_JSON);
    let specification_path = write_specification(&dir, &manifest_path, "hhhh", 3);
    let report = DirectInferenceExecutor
        .run_with_report(DirectInferenceConfig {
            base_dir: dir.clone(),
            specification_path,
            case_path: dir.join(CASE_TOML),
            inference: 0,
        })
        .unwrap();
    assert_eq!(report.input_token_count, Some(4));
    assert_eq!(report.result.generated_token_count, 3);
    assert_eq!(report.result.stop_reason, "max_new_tokens");
    // The record names the run, as the chain's final record does.
    assert_eq!(
        report.result.input_text_sha256,
        format!("{:x}", Sha256::digest(b"hhhh"))
    );
    assert_eq!(report.result.max_new_tokens, 3);
    assert_eq!(report.result.prompt_tokens, 4);
    assert!(report.result.prompt_complete);
    assert_eq!(report.result.output_fault, 0);
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn direct_executor_rejects_changed_bundle_before_inference() {
    let dir = temp_dir("changed-bundle");
    fs::create_dir_all(&dir).unwrap();
    let specification_path = write_direct_fixture(&dir);
    for (name, label) in [
        ("model.detwgt", "model weights"),
        ("config.json", "model config"),
        ("tokenizer.json", "model tokenizer"),
    ] {
        let path = dir.join(name);
        let original = fs::read(&path).unwrap();
        fs::write(&path, b"replaced").unwrap();
        let error = DirectInferenceExecutor
            .run(DirectInferenceConfig {
                base_dir: dir.clone(),
                specification_path: specification_path.clone(),
                case_path: dir.join(CASE_TOML),
                inference: 0,
            })
            .unwrap_err();
        assert!(
            format!("{error:#}").contains(&format!("{label} hash mismatch")),
            "{error:#}"
        );
        fs::write(path, original).unwrap();
    }
    fs::remove_dir_all(dir).unwrap();
}

fn write_detwgt(path: &Path, tensors: &[TensorFixture]) {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"DNWGTV0\0");
    bytes.extend_from_slice(&3_u32.to_le_bytes());
    bytes.extend_from_slice(&1_u32.to_le_bytes());
    bytes.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    for tensor in tensors {
        bytes.extend_from_slice(&(tensor.name.len() as u32).to_le_bytes());
        bytes.extend_from_slice(tensor.name.as_bytes());
        bytes.extend_from_slice(&(tensor.dims.len() as u32).to_le_bytes());
        for dim in &tensor.dims {
            bytes.extend_from_slice(&dim.to_le_bytes());
        }
        bytes.extend_from_slice(&(tensor.values.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&tensor.element_width.to_le_bytes());
        let payload_len = match tensor.element_width {
            8 => q8_exact(&tensor.values).len(),
            width => tensor.values.len() * (width as usize / 8),
        };
        bytes.extend_from_slice(&(payload_len as u64).to_le_bytes());
        bytes.extend_from_slice(&0_u64.to_le_bytes());
        while bytes.len() % 64 != 0 {
            bytes.push(0);
        }
        match tensor.element_width {
            8 => bytes.extend(q8_exact(&tensor.values)),
            16 => {
                for value in &tensor.values {
                    bytes.extend_from_slice(&(*value as i16).to_le_bytes());
                }
            }
            32 => {
                for value in &tensor.values {
                    bytes.extend_from_slice(&value.to_le_bytes());
                }
            }
            other => panic!("unsupported fixture width {other}"),
        }
    }
    fs::write(path, bytes).unwrap();
}

fn write_direct_fixture(dir: &Path) -> PathBuf {
    let artifact_dir = dir.join(MODEL_ARTIFACTS_DIR);
    fs::create_dir_all(&artifact_dir).unwrap();
    let detwgt = dir.join("model.detwgt");
    write_tiny_model_detwgt(&detwgt);
    let config = dir.join("config.json");
    fs::write(&config, r#"{"text_config":{"tie_word_embeddings":false}}"#).unwrap();
    let tokenizer = dir.join("tokenizer.json");
    fs::write(&tokenizer, tiny_tokenizer_json()).unwrap();
    let manifest_path = artifact_dir.join(MODEL_MANIFEST_JSON);
    write_json(
        &manifest_path,
        &tiny_direct_manifest(&detwgt, &config, &tokenizer),
    )
    .unwrap();
    write_specification(dir, &manifest_path, "h", 1)
}

fn write_tiny_model_detwgt(path: &Path) {
    write_detwgt(
        path,
        &[
            TensorFixture {
                name: "model.language_model.embed_tokens.weight",
                dims: vec![3, W as u64],
                element_width: 8,
                values: one_hot_rows(&[None, Some(0), Some(1)]),
            },
            TensorFixture {
                name: "model.language_model.lm_head.weight",
                dims: vec![3, W as u64],
                element_width: 8,
                values: one_hot_rows(&[None, None, Some(0)]),
            },
            TensorFixture {
                name: "model.language_model.norm.weight",
                dims: vec![W as u64],
                element_width: 32,
                values: vec![ONE; W],
            },
        ],
    );
}

fn tiny_direct_manifest(detwgt: &Path, config: &Path, tokenizer: &Path) -> ModelManifest {
    ModelManifest {
        version: model_manifest::MODEL_MANIFEST_VERSION,
        family: model_manifest::ModelFamily::GemmaDense.name().to_string(),
        model_id: model_manifest::model_id_of(&[]),
        model_commitments: Vec::new(),
        range_size: 512,
        bundle: DirectInferBundle {
            model_detwgt_path: PathBuf::from("../../model.detwgt"),
            model_detwgt_sha256: sha256_file(detwgt),
            config_path: PathBuf::from("../../config.json"),
            config_sha256: sha256_file(config),
            tokenizer_path: PathBuf::from("../../tokenizer.json"),
            tokenizer_sha256: sha256_file(tokenizer),
        },
        shape: DirectInferShape {
            hidden_size: W as u32,
            num_hidden_layers: 0,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            head_dim: 2,
            global_head_dim: 2,
            vocab_size: 3,
            hidden_size_per_layer_input: 1,
            sliding_window: 0,
            layer_types: Vec::new(),
            num_kv_shared_layers: 0,
            norm_eps: 0,
            rope_base_sliding: 10_000_i64 << 32,
            rope_base_full: 1_000_000_i64 << 32,
            full_partial_rotary_factor_q16: ONE,
            embedding_scale: ONE,
            ple_embedding_scale: ONE,
            ple_projection_scalar: ONE,
            ple_input_scale: ONE,
            final_logit_softcap: 0,
        }
        .into(),
        eos_token_ids: Vec::new(),
        provenance: None,
    }
}

/// Writes a specification and, beside it, the case of this prompt.
fn write_specification(
    dir: &Path,
    manifest_path: &Path,
    prompt: &str,
    max_new_tokens: u32,
) -> PathBuf {
    let path = dir.join(SPECIFICATION_TOML);
    fs::write(
        &path,
        format!(
            "raw_prompt = true
max_new_tokens = {max_new_tokens}

[[model]]
manifest = {:?}
",
            manifest_path.strip_prefix(dir).unwrap().to_string_lossy(),
        ),
    )
    .unwrap();
    fs::write(
        dir.join(CASE_TOML),
        format!(
            "prompt = {prompt:?}
"
        ),
    )
    .unwrap();
    path
}

fn tiny_tokenizer_json() -> &'static str {
    r#"{"model":{"type":"BPE","vocab":{"<pad>":0,"h":1,"Hi":2},"merges":[]},"added_tokens":[]}"#
}

/// Rows `W` wide: `Some(col)` is `ONE` at `col`, `None` is all zero.
fn one_hot_rows(hot: &[Option<usize>]) -> Vec<i32> {
    hot.iter()
        .flat_map(|col| (0..W).map(move |i| if Some(i) == *col { ONE } else { 0 }))
        .collect()
}

/// Q16.16 `values` as exact `Q8_0` blocks: each block takes the largest
/// power-of-two scale that represents all its values exactly.
fn q8_exact(values: &[i32]) -> Vec<u8> {
    let mut out = Vec::new();
    for block in values.chunks(32) {
        assert_eq!(block.len(), 32, "Q8_0 rows are whole blocks");
        let k = (0..=14)
            .find(|k| {
                let step = ONE >> k;
                block
                    .iter()
                    .all(|v| v % step == 0 && (v / step).abs() <= 127)
            })
            .expect("values are not exactly representable as Q8_0");
        out.extend_from_slice(&(((15 - k) as u16) << 10).to_le_bytes());
        out.extend(block.iter().map(|v| (v / (ONE >> k)) as i8 as u8));
    }
    out
}

fn sha256_file(path: &Path) -> String {
    format!("{:x}", Sha256::digest(fs::read(path).unwrap()))
}

fn temp_dir(label: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "direct-infer-{label}-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ))
}
