use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use direct_infer::detwgt::MmapDetwgt;
use direct_infer::model::{DirectFinalHead, DirectMatrixView, DirectTransformerLayer};
use direct_infer::view_kernels::{run_prefill_range_view, score_prefill_logits_view};
use direct_infer::{DirectInferenceConfig, DirectInferenceExecutor};
use inference_artifacts::{
    write_json, DirectInferBundle, DirectInferShape, InferenceRunSpec, ModelManifest,
    INFERENCE_RUN_SPEC_TOML, MODEL_ARTIFACTS_DIR, MODEL_MANIFEST_JSON,
};
use inference_kernels::kernels::decode_select_token::{
    run_decode_select_token_direct, DecodeSelectTokenDirectInputs,
};
use inference_kernels::tensor::{dot_bits, matvec_from_source, Matrix, MatrixSource};
use input_embedding::input::EmbeddingTable;
use output_finalize::input::{DecoderTable, DecoderToken};
use prefill_finalize::input::{FinalHead, FinalHeadParams};
use prefill_range::input::{
    pack_i32s, unpack_i32s, ActivationRow, ActivationSequence, LayerParams, PleLayerInputs, PleRow,
    TransformerLayer,
};
use prompt_prepare::input::{
    vocab_bucket_of, BpePieces, MergeBucket, PromptTokenizer, TokenEntry, VocabBucket,
};
use raster::{Bytes, List};
use sha2::{Digest, Sha256};
use staged_infer::{UncheckpointedInferenceConfig, UncheckpointedInferenceExecutor};

const ONE: i32 = 1 << 16;

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
            run_spec_path: dir.join(INFERENCE_RUN_SPEC_TOML),
        })
        .expect_err("missing manifest should fail");

    assert!(!format!("{error:#}").contains("scaffolded"));
    assert!(format!("{error:#}").contains("failed to load run spec"));

    let _ = fs::remove_dir_all(dir);
}

#[test]
fn mmap_detwgt_parses_directory_and_sign_extends_i16() {
    let dir = temp_dir("loader");
    fs::create_dir_all(&dir).unwrap();
    let detwgt = dir.join("model.detwgt");
    write_detwgt(
        &detwgt,
        &[TensorFixture {
            name: "tiny",
            dims: vec![2],
            element_width: 16,
            values: vec![-1, 2],
        }],
    );
    let model = MmapDetwgt::open(&detwgt).unwrap();
    assert_eq!(model.values("tiny").unwrap(), vec![-1, 2]);
    assert_eq!(model.slice("tiny").unwrap().value_at(0).unwrap(), -1);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn detwgt_matrix_view_matches_eager_matrix_matvec() {
    let dir = temp_dir("matrix-view");
    fs::create_dir_all(&dir).unwrap();
    let detwgt = dir.join("model.detwgt");
    let values = vec![ONE, 0, 0, 0, ONE, ONE];
    write_detwgt(
        &detwgt,
        &[TensorFixture {
            name: "matrix",
            dims: vec![2, 3],
            element_width: 32,
            values: values.clone(),
        }],
    );
    let model = MmapDetwgt::open(&detwgt).unwrap();
    let view = DirectMatrixView::new(model.matrix("matrix", 2, 3).unwrap());
    let eager = Matrix::from_region("matrix", &paged(&values), 2, 3).unwrap();
    let input = [ONE, 2 * ONE, 3 * ONE];

    assert_eq!(
        matvec_from_source(&view, &input).unwrap(),
        matvec_from_source(&eager, &input).unwrap()
    );

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn row_dot_matches_dot_bits_for_i16_and_i32() {
    let dir = temp_dir("row-dot");
    fs::create_dir_all(&dir).unwrap();
    let detwgt = dir.join("model.detwgt");
    write_detwgt(
        &detwgt,
        &[
            TensorFixture {
                name: "i16_matrix",
                dims: vec![2, 2],
                element_width: 16,
                values: vec![1, -2, 3, 4],
            },
            TensorFixture {
                name: "i32_matrix",
                dims: vec![2, 2],
                element_width: 32,
                values: vec![ONE, 0, 0, ONE],
            },
        ],
    );
    let model = MmapDetwgt::open(&detwgt).unwrap();

    let i16_view = DirectMatrixView::new(model.matrix("i16_matrix", 2, 2).unwrap());
    assert_eq!(
        i16_view.row_dot(0, &[5, 6]).unwrap(),
        dot_bits(&[5, 6], &[1, -2])
    );
    let i32_view = DirectMatrixView::new(model.matrix("i32_matrix", 2, 2).unwrap());
    assert_eq!(
        i32_view.row_dot(1, &[7, 8]).unwrap(),
        dot_bits(&[7, 8], &[0, ONE])
    );

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn final_head_logits_feed_shared_decode_select_tie_break() {
    let dir = temp_dir("stream-final-head");
    fs::create_dir_all(&dir).unwrap();
    let detwgt = dir.join("model.detwgt");
    let projection = vec![ONE, 0, ONE, 0, 0, 0];
    write_detwgt(
        &detwgt,
        &[TensorFixture {
            name: "projection",
            dims: vec![3, 2],
            element_width: 32,
            values: projection.clone(),
        }],
    );
    let model = MmapDetwgt::open(&detwgt).unwrap();
    let head = DirectFinalHead {
        params: FinalHeadParams {
            hidden_size: 2,
            norm_eps: 0,
            softcap: 0,
            norm_weights: pack_i32_page(&[ONE, ONE]),
        },
        projection: DirectMatrixView::new(model.matrix("projection", 3, 2).unwrap()),
    };
    let activations = prefill_range::input::ActivationSequence {
        rows: List::from(vec![prefill_range::input::ActivationRow {
            token_id: 1,
            values: pack_i32_page(&[ONE, 0]),
        }]),
        errors: List::new(),
        kv: List::new(),
        start_position: 0,
    };
    let logits = score_prefill_logits_view(&activations, &head).unwrap();
    let prior = inference_kernels::kernels::decode_init::run_decode_init_direct(
        inference_kernels::kernels::decode_init::DecodeInitDirectInputs,
    )
    .unwrap();
    let edge = run_decode_select_token_direct(DecodeSelectTokenDirectInputs {
        logits: &logits,
        prior: &prior,
    })
    .unwrap();

    assert_eq!(logits.decode_position, 1);
    assert_eq!(logits.logits.len(), 3);
    assert_eq!(edge.token_id, 0, "equal scores keep the earliest token");
    assert_eq!(
        edge.generated_token_ids.iter().copied().collect::<Vec<_>>(),
        vec![0]
    );

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_decode_rotary_outputs_match_prefill_across_chunks() {
    let dir = temp_dir("decode-rope");
    fs::create_dir_all(&dir).unwrap();
    let detwgt = dir.join("model.detwgt");
    let identity = [ONE, 0, 0, ONE];
    write_detwgt(
        &detwgt,
        &[TensorFixture {
            name: "identity",
            dims: vec![2, 2],
            element_width: 32,
            values: identity.to_vec(),
        }],
    );
    let model = MmapDetwgt::open(&detwgt).unwrap();
    let matrix = DirectMatrixView::new(model.matrix("identity", 2, 2).unwrap());
    let norm = pack_i32s(&[ONE, ONE]);
    let params = LayerParams {
        layer_idx: 0,
        hidden_size: 2,
        ffn_size: 2,
        num_heads: 1,
        num_kv_heads: 1,
        head_dim: 2,
        sliding_window: 0,
        attn_scale: ONE,
        layer_scalar: 0,
        norm_eps: 0,
        rope_base: 10_000_i64 << 32,
        rotary_dim: 2,
        rope_freq_base_dim: 2,
        kv_donor_layer: -1,
        donor_a_layer: -2,
        donor_b_layer: -3,
        norm_input: norm.clone(),
        norm_post_attn: norm.clone(),
        norm_pre_ffw: norm.clone(),
        norm_post_ffw: norm.clone(),
        q_norm: pack_i32s(&[ONE, 0]),
        k_norm: pack_i32s(&[ONE, 0]),
        ple_width: 2,
        ple_post_norm: norm,
    };
    let view = DirectTransformerLayer {
        params: params.clone(),
        w_q: matrix,
        w_k: matrix,
        w_v: matrix,
        w_o: matrix,
        w_gate: matrix,
        w_up: matrix,
        w_down: matrix,
        ple_input_gate: matrix,
        ple_layer_projection: matrix,
    };
    let layer = TransformerLayer {
        params,
        w_q: paged(&identity),
        w_k: paged(&identity),
        w_v: paged(&identity),
        w_o: paged(&identity),
        w_gate: paged(&identity),
        w_up: paged(&identity),
        w_down: paged(&identity),
        ple_input_gate: paged(&identity),
        ple_layer_projection: paged(&identity),
    };
    let empty = ActivationSequence {
        rows: List::new(),
        errors: List::new(),
        kv: List::new(),
        start_position: 0,
    };
    let activations = ActivationSequence {
        rows: List::from(
            (0..5)
                .map(|token_id| ActivationRow {
                    token_id,
                    // Distinct V rows make attention outputs sensitive to Q.
                    values: pack_i32s(&[ONE, if token_id % 2 == 0 { ONE } else { -ONE }]),
                })
                .collect::<Vec<_>>(),
        ),
        ..empty.clone()
    };
    let ple = PleLayerInputs {
        layer_idx: 0,
        rows: List::from(vec![
            PleRow {
                values: pack_i32s(&[ONE, ONE])
            };
            5
        ]),
        errors: List::new(),
    };
    let run_staged = |input: &ActivationSequence, prior: &ActivationSequence| {
        inference_kernels::run_prefill_range_direct(inference_kernels::PrefillRangeDirectInputs {
            activations: input,
            layer: &layer,
            layer_cache_key: None,
            prior_kv: prior,
            donor_a_kv: &empty,
            donor_b_kv: &empty,
            ple: &ple,
        })
        .unwrap()
    };
    let reference = run_staged(&activations, &empty);
    assert_eq!(unpack_i32s(&reference.kv[1].k).unwrap(), [35408, 55146]);

    let mut prior = empty.clone();
    // Prefill one token, decode twice, then process a two-token chunk.
    for (start, end) in [(0, 1), (1, 2), (2, 3), (3, 5)] {
        let chunk = ActivationSequence {
            rows: List::from(activations.rows.as_slice()[start..end].to_vec()),
            start_position: start as u32,
            ..empty.clone()
        };
        let direct = run_prefill_range_view(&chunk, &view, &prior, &empty, &empty, &ple).unwrap();
        let staged = run_staged(&chunk, &prior);
        for output in [&direct, &staged] {
            assert!(output.errors.is_empty());
            assert_eq!(output.start_position, start as u32);
            assert_eq!(
                serde_json::to_value(&output.rows).unwrap(),
                serde_json::to_value(&reference.rows.as_slice()[start..end]).unwrap(),
                "activation rows at start_position {start}"
            );
            assert_eq!(
                serde_json::to_value(&output.kv).unwrap(),
                serde_json::to_value(&reference.kv.as_slice()[..end]).unwrap(),
                "all cached K/V rows at start_position {start}"
            );
        }
        prior = direct;
    }

    fs::remove_dir_all(dir).unwrap();
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
            run_spec_path: repo.join(INFERENCE_RUN_SPEC_TOML),
        })
        .unwrap();

    assert_eq!(result.generated_token_ids.first().copied(), Some(150917));
    assert_eq!(result.generated_text, "Ciao");
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
                dims: vec![3, 2],
                element_width: 32,
                values: vec![0, 0, ONE, 0, 0, ONE],
            },
            TensorFixture {
                name: "model.language_model.lm_head.weight",
                dims: vec![3, 2],
                element_width: 32,
                values: vec![0, 0, 0, 0, ONE, 0],
            },
            TensorFixture {
                name: "model.language_model.norm.weight",
                dims: vec![2],
                element_width: 32,
                values: vec![ONE, ONE],
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
            version: 2,
            bundle: DirectInferBundle {
                model_detwgt_path: PathBuf::from("../../model.detwgt"),
                model_detwgt_sha256: sha256_file(&detwgt),
                config_path: PathBuf::from("../../config.json"),
                config_sha256: sha256_file(&config),
                tokenizer_path: PathBuf::from("../../tokenizer.json"),
                tokenizer_sha256: sha256_file(&tokenizer),
            },
            shape: DirectInferShape {
                hidden_size: 2,
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
            },
            eos_token_ids: Vec::new(),
            provenance: None,
        },
    )
    .unwrap();
    write_run_spec(&dir, &manifest_path, "h", 1);

    let result = DirectInferenceExecutor
        .run(DirectInferenceConfig {
            base_dir: dir.clone(),
            run_spec_path: dir.join(INFERENCE_RUN_SPEC_TOML),
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
fn direct_executor_matches_uncheckpointed_staged_result() {
    let dir = temp_dir("parity");
    fs::create_dir_all(&dir).unwrap();
    let run_spec_path = write_direct_fixture(&dir);
    write_staged_fixture(&dir);

    let direct = DirectInferenceExecutor
        .run(DirectInferenceConfig {
            base_dir: dir.clone(),
            run_spec_path,
        })
        .unwrap();
    let staged = UncheckpointedInferenceExecutor
        .run(UncheckpointedInferenceConfig {
            base_dir: dir.clone(),
            manifest_path: dir.join("Raster.toml"),
        })
        .unwrap();

    assert_eq!(direct, staged);
    assert_eq!(direct.generated_token_ids, vec![2]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn direct_executor_rejects_changed_bundle_before_inference() {
    let dir = temp_dir("changed-bundle");
    fs::create_dir_all(&dir).unwrap();
    let run_spec_path = write_direct_fixture(&dir);
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
                run_spec_path: run_spec_path.clone(),
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

struct TensorFixture {
    name: &'static str,
    dims: Vec<u64>,
    element_width: u32,
    values: Vec<i32>,
}

fn write_detwgt(path: &Path, tensors: &[TensorFixture]) {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"DNWGTV0\0");
    bytes.extend_from_slice(&2_u32.to_le_bytes());
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
        let payload_len = tensor.values.len() * (tensor.element_width as usize / 8);
        bytes.extend_from_slice(&(payload_len as u64).to_le_bytes());
        bytes.extend_from_slice(&0_u64.to_le_bytes());
        while bytes.len() % 64 != 0 {
            bytes.push(0);
        }
        match tensor.element_width {
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
    write_run_spec(dir, &manifest_path, "h", 1)
}

fn write_tiny_model_detwgt(path: &Path) {
    write_detwgt(
        path,
        &[
            TensorFixture {
                name: "model.language_model.embed_tokens.weight",
                dims: vec![3, 2],
                element_width: 32,
                values: vec![0, 0, ONE, 0, 0, ONE],
            },
            TensorFixture {
                name: "model.language_model.lm_head.weight",
                dims: vec![3, 2],
                element_width: 32,
                values: vec![0, 0, 0, 0, ONE, 0],
            },
            TensorFixture {
                name: "model.language_model.norm.weight",
                dims: vec![2],
                element_width: 32,
                values: vec![ONE, ONE],
            },
        ],
    );
}

fn tiny_direct_manifest(detwgt: &Path, config: &Path, tokenizer: &Path) -> ModelManifest {
    ModelManifest {
        version: 2,
        bundle: DirectInferBundle {
            model_detwgt_path: PathBuf::from("../../model.detwgt"),
            model_detwgt_sha256: sha256_file(detwgt),
            config_path: PathBuf::from("../../config.json"),
            config_sha256: sha256_file(config),
            tokenizer_path: PathBuf::from("../../tokenizer.json"),
            tokenizer_sha256: sha256_file(tokenizer),
        },
        shape: DirectInferShape {
            hidden_size: 2,
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
        },
        eos_token_ids: Vec::new(),
        provenance: None,
    }
}

fn write_run_spec(dir: &Path, manifest_path: &Path, prompt: &str, tokens: u32) -> PathBuf {
    let path = dir.join(INFERENCE_RUN_SPEC_TOML);
    let spec = InferenceRunSpec {
        model_manifest: manifest_path.strip_prefix(dir).unwrap().to_path_buf(),
        prompt: Some(prompt.to_string()),
        prompt_file: None,
        raw_prompt: true,
        tokens,
    };
    fs::write(
        &path,
        format!(
            "model_manifest = {:?}\nprompt = {:?}\nraw_prompt = true\ntokens = {}\n",
            spec.model_manifest.to_string_lossy(),
            prompt,
            spec.tokens
        ),
    )
    .unwrap();
    path
}

fn write_staged_fixture(dir: &Path) {
    let tokenizer = tiny_prompt_tokenizer();
    let pieces = BpePieces {
        pieces: List::from(vec![prompt_prepare::input::BpePiece {
            text: "h".into(),
            segment: 0,
        }]),
    };
    let embedding = EmbeddingTable {
        hidden_size: 2,
        embedding_scale: ONE,
        values: paged(&[0, 0, ONE, 0, 0, ONE]),
    };
    let head = FinalHead {
        params: FinalHeadParams {
            hidden_size: 2,
            norm_eps: 0,
            softcap: 0,
            norm_weights: pack_i32_page(&[ONE, ONE]),
        },
        projection: paged(&[0, 0, 0, 0, ONE, 0]),
    };
    let decoder = DecoderTable {
        tokens: List::from(vec![
            DecoderToken {
                token: String::from("<pad>"),
                special: false,
                terminal: false,
            },
            DecoderToken {
                token: String::from("hello"),
                special: false,
                terminal: false,
            },
            DecoderToken {
                token: String::from("Hi"),
                special: false,
                terminal: false,
            },
        ]),
    };

    let tokenizer_commitment =
        write_staged_external(dir, "raster-stages/prompt-prepare", "tokenizer", &tokenizer);
    let pieces_commitment = write_staged_external(
        dir,
        "raster-stages/prompt-prepare",
        "initial_pieces",
        &pieces,
    );
    let embedding_commitment = write_staged_external(
        dir,
        "raster-stages/input-embedding",
        "embedding",
        &embedding,
    );
    let head_commitment =
        write_staged_external(dir, "raster-stages/prefill-finalize", "head", &head);
    let decoder_commitment =
        write_staged_external(dir, "raster-stages/output-finalize", "decoder", &decoder);

    fs::write(
        dir.join("Raster.toml"),
        format!(
            r#"[chain]
name = "direct-parity"
version = "0.1.0"

[[chain.stage]]
name = "prompt_prepare"
project = "raster-stages/prompt-prepare"
inputs.tokenizer = {{ external = {{ path = "raster-stages/prompt-prepare/tokenizer.rastered", index_path = "raster-stages/prompt-prepare/tokenizer.rindex", commitment = "{tokenizer_commitment}" }} }}
inputs.merged_pieces = {{ external = {{ path = "raster-stages/prompt-prepare/initial_pieces.rastered", index_path = "raster-stages/prompt-prepare/initial_pieces.rindex", commitment = "{pieces_commitment}" }} }}

[[chain.stage]]
name = "input_embedding"
project = "raster-stages/input-embedding"
inputs.prompt = {{ from = "prompt_prepare" }}
inputs.embedding = {{ external = {{ path = "raster-stages/input-embedding/embedding.rastered", index_path = "raster-stages/input-embedding/embedding.rindex", commitment = "{embedding_commitment}" }} }}

[[chain.stage]]
name = "prefill_finalize"
project = "raster-stages/prefill-finalize"
inputs.activations = {{ from = "input_embedding" }}
inputs.head = {{ external = {{ path = "raster-stages/prefill-finalize/head.rastered", index_path = "raster-stages/prefill-finalize/head.rindex", commitment = "{head_commitment}" }} }}

[[chain.stage]]
name = "decode_init"
project = "raster-stages/decode-init"

[[chain.stage]]
name = "decode_select_token"
project = "raster-stages/decode-select-token"
inputs.logits = {{ from = "prefill_finalize" }}
inputs.prior = {{ from = "decode_init" }}

[[chain.stage]]
name = "output_finalize"
project = "raster-stages/output-finalize"
inputs.edge = {{ from = "decode_select_token" }}
inputs.decoder = {{ external = {{ path = "raster-stages/output-finalize/decoder.rastered", index_path = "raster-stages/output-finalize/decoder.rindex", commitment = "{decoder_commitment}" }} }}
"#
        ),
    )
    .unwrap();
}

fn write_staged_external<T: serde::Serialize>(
    base: &Path,
    stage_dir: &str,
    name: &str,
    value: &T,
) -> String {
    let dir = base.join(stage_dir);
    fs::create_dir_all(&dir).unwrap();
    raster::write_raster_files(
        value,
        &dir.join(format!("{name}.rastered")),
        &dir.join(format!("{name}.rindex")),
    )
    .unwrap()
}

fn tiny_prompt_tokenizer() -> PromptTokenizer {
    PromptTokenizer {
        vocab_bucket_count: 1,
        merge_bucket_count: 1,
        vocab_buckets: List::from(vec![VocabBucket {
            entries: List::from(
                [("<pad>", 0), ("h", 1), ("Hi", 2)]
                    .into_iter()
                    .map(|(token, id)| TokenEntry {
                        token: token.to_string(),
                        id,
                    })
                    .filter(|entry| vocab_bucket_of(&entry.token, 1) == 0)
                    .collect::<Vec<_>>(),
            ),
        }]),
        merge_buckets: List::from(vec![MergeBucket { rules: List::new() }]),
    }
}

fn tiny_tokenizer_json() -> &'static str {
    r#"{"model":{"type":"BPE","vocab":{"<pad>":0,"h":1,"Hi":2},"merges":[]},"added_tokens":[]}"#
}

fn paged(values: &[i32]) -> Bytes<196_608> {
    Bytes::<196_608>::paged(bytes_of_i32s(values)).unwrap()
}

fn pack_i32_page(values: &[i32]) -> raster::BytesPage {
    raster::BytesPage::__from_parts(0, 0, bytes_of_i32s(values))
}

fn bytes_of_i32s(values: &[i32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
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
