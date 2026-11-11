use prefill_range::input::{pack_i32s, LayerParams, TransformerLayer};
use raster::{Bytes, List};
use serde::Serialize;
use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};
const ONE: i32 = 1 << 16;
fn paged(values: &[i32]) -> Bytes<184_320> {
    Bytes::paged(
        values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    )
    .unwrap()
}
/// One `Q8_0` block wide: the narrowest an int8 matrix can be.
const N: usize = 32;
fn identity() -> Vec<i32> {
    (0..N * N)
        .map(|i| if i % (N + 1) == 0 { ONE } else { 0 })
        .collect()
}
/// `rows` one-hot `Q8_0` rows of width `N` (row `r` is 1 at column `r`),
/// each scaled by the half-precision `scale`. Exact: every value is 0 or `scale`.
fn q8_one_hot<const P: u64>(rows: usize, scale: u16) -> Bytes<P> {
    let mut bytes = Vec::with_capacity(rows * 34);
    for row in 0..rows {
        bytes.extend_from_slice(&scale.to_le_bytes());
        bytes.extend((0..N).map(|col| u8::from(col == row)));
    }
    Bytes::paged(bytes).unwrap()
}
fn q8_identity() -> Bytes<195_840> {
    q8_one_hot(N, 0x3c00)
}
fn tiny_layer(kv_donor_layer: i32) -> TransformerLayer {
    let norm = pack_i32s(&[ONE; N]);
    TransformerLayer {
        params: LayerParams {
            layer_idx: 0,
            hidden_size: N as u32,
            ffn_size: N as u32,
            num_heads: 1,
            num_kv_heads: 1,
            head_dim: N as u32,
            sliding_window: 0,
            attn_scale: 0,
            layer_scalar: 0,
            norm_eps: 0,
            rope_base: 1_i64 << 32,
            rotary_dim: 0,
            rope_freq_base_dim: 2,
            kv_donor_layer,
            donor_a_layer: if kv_donor_layer >= 0 {
                kv_donor_layer
            } else {
                -2
            },
            donor_b_layer: -3,
            norm_input: norm.clone(),
            norm_post_attn: norm.clone(),
            norm_pre_ffw: norm.clone(),
            norm_post_ffw: norm.clone(),
            q_norm: norm.clone(),
            k_norm: norm.clone(),
            ple_width: N as u32,
            ple_post_norm: norm,
        },
        w_q: q8_identity(),
        w_k: q8_identity(),
        w_v: q8_identity(),
        w_o: q8_identity(),
        w_gate: q8_identity(),
        w_up: q8_identity(),
        w_down: q8_identity(),
        ple_input_gate: paged(&identity()),
        ple_layer_projection: paged(&identity()),
    }
}

struct Fixture {
    base: PathBuf,
    stages: Vec<(String, String)>,
}
impl Fixture {
    fn new() -> Self {
        let base = std::env::temp_dir().join(format!(
            "iris-worker-parity-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&base).unwrap();
        Self {
            base,
            stages: Vec::new(),
        }
    }
    fn external<T: Serialize>(&self, name: &str, value: &T) -> String {
        raster::init();
        let (data, index, commitment) = raster::encode_raster_value(value).unwrap();
        let path = self.base.join(format!("{name}.rastered"));
        let index_path = self.base.join(format!("{name}.rindex"));
        fs::write(&path, data).unwrap();
        fs::write(&index_path, index).unwrap();
        format!(
            "{{ external = {{ path = {}, index_path = {}, commitment = {} }} }}",
            toml::Value::String(path.to_string_lossy().into_owned()),
            toml::Value::String(index_path.to_string_lossy().into_owned()),
            toml::Value::String(commitment)
        )
    }
    fn manifest(&mut self) {
        let decoder_table = output_finalize::input::DecoderTable {
            tokens: List::from(vec![
                output_finalize::input::DecoderToken {
                    token: "a".into(),
                    special: false,
                    terminal: false,
                },
                output_finalize::input::DecoderToken {
                    token: "b".into(),
                    special: false,
                    terminal: false,
                },
            ]),
        };
        let mut descriptor = model_identity::ModelDescriptor {
            version: 1,
            tokenizer: [1; 32],
            decoder: raster::core::tree::value_root(&decoder_table).unwrap(),
            embedding: [2; 32],
            head: [3; 32],
            auxiliary: vec![[4; 32]; 2],
            transformer: vec![[5; 32]; 2],
        };
        let no_prompt = self.external(
            "no_prompt",
            &input_embedding::input::PromptTokenization::default(),
        );
        let no_claim = self.external(
            "no_claim",
            &input_embedding::input::ClaimedOutput::default(),
        );
        let claimed = [1, 0, 1].map(|id: u32| {
            self.external(
                &format!("claimed_{id}"),
                &input_embedding::input::ClaimedOutput {
                    token_ids: List::from(vec![id]),
                },
            )
        });
        let embedding = self.external(
            "embedding",
            &input_embedding::input::EmbeddingTable {
                hidden_size: N as u32,
                embedding_scale: ONE,
                values: q8_one_hot(2, 0x3c00),
            },
        );
        let page_witness = self.external(
            "page_witness",
            &model_identity::sparse::witness(q8_one_hot::<195_840>(2, 0x3c00).pages.as_slice())
                .unwrap(),
        );
        let head = self.external(
            "head",
            &sequence_check::input::FinalHead {
                params: sequence_check::input::FinalHeadParams {
                    hidden_size: N as u32,
                    norm_eps: 1,
                    softcap: 0,
                    norm_weights: pack_i32s(&[ONE; N]),
                },
                projection: q8_one_hot(2, 0x3c00),
            },
        );
        let decoder = self.external("decoder", &decoder_table);
        let witness = self.external(
            "witness",
            &output_finalize::decoder_witness(&decoder_table).unwrap(),
        );
        let mut aux = Vec::new();
        let mut layers = Vec::new();
        for l in 0..2 {
            let mut layer = tiny_layer(-1);
            layer.params.layer_idx = l;
            layer.params.layer_scalar = ONE;
            layer.params.attn_scale = ONE;
            layer.params.norm_eps = 1;
            if l == 1 {
                layer.w_up = q8_one_hot(N, 0x3800); // 0.5 · identity
            }
            layers.push(self.external(&format!("layer{l}"), &layer));
            aux.push(self.external(
                &format!("aux{l}"),
                &prefill_prepare_aux::input::PleLayer {
                    params: prefill_prepare_aux::input::PleLayerParams {
                        layer_idx: l,
                        hidden_size: N as u32,
                        ple_width: N as u32,
                        embedding_scale: ONE,
                        projection_scalar: ONE,
                        input_scale: ONE,
                        norm_eps: 1,
                        norm_weights: pack_i32s(&[ONE; N]),
                    },
                    embeddings: q8_one_hot(2, 0x3c00),
                    projection: paged(&identity()),
                },
            ));
        }
        let root = |binding: &str| -> [u8; 32] {
            let parsed: toml::Value = format!("value = {binding}").parse().unwrap();
            hex::decode(parsed["value"]["external"]["commitment"].as_str().unwrap())
                .unwrap()
                .try_into()
                .unwrap()
        };
        descriptor.embedding = root(&embedding);
        descriptor.head = root(&head);
        descriptor.auxiliary = aux.iter().map(|b| root(b)).collect();
        descriptor.transformer = layers.iter().map(|b| root(b)).collect();
        let context = input_embedding::prompt::PromptContext {
            model: model_identity::ModelContext::bind(
                &descriptor.encode().unwrap(),
                &hex::encode(descriptor.model_id().unwrap()),
            ),
            model_id: hex::encode(descriptor.model_id().unwrap()),
            max_new_tokens: 3,
            range_size: 1,
            prompt_tokens: 2,
            complete: true,
            ..Default::default()
        };
        let prompt = self.external(
            "prompt",
            &input_embedding::input::PromptTokenization {
                token_ids: List::from(vec![0, 1]),
                context,
                ..Default::default()
            },
        );
        let descriptor = self.external(
            "descriptor",
            &raster::BytesPage::__from_parts(0, 0, descriptor.encode().unwrap()),
        );
        let mut text = String::from("[chain]\nname = \"worker-parity\"\nversion = \"0.1.0\"\n");
        let mut add = |name: String, project: &str, inputs: Vec<(&str, String)>| {
            text.push_str(&format!(
                "\n[[chain.stage]]\nname = {}\nproject = {}\n",
                toml::Value::String(name.clone()),
                toml::Value::String(project.into())
            ));
            for (param, binding) in inputs {
                text.push_str(&format!("inputs.{param} = {binding}\n"));
            }
            self.stages.push((name, project.into()));
        };
        let from = |s: &str| format!("{{ from = {} }}", toml::Value::String(s.into()));
        add("range_origin".into(), "range-origin", vec![]);
        add(
            "embed_p0".into(),
            "input-embedding",
            vec![
                ("prompt", prompt.clone()),
                ("claimed", no_claim),
                ("previous", from("range_origin")),
                ("embedding", embedding.clone()),
                ("witness", page_witness.clone()),
            ],
        );
        for l in 0..2 {
            add(
                format!("aux_p0_l{l}"),
                "prefill-prepare-aux",
                vec![
                    ("embedded", from("embed_p0")),
                    ("witness", page_witness.clone()),
                    ("layer", aux[l].clone()),
                ],
            );
        }
        for l in 0..2 {
            add(
                format!("range_p0_l{l}"),
                "prefill-range",
                vec![
                    (
                        "activations",
                        from(if l == 0 { "embed_p0" } else { "range_p0_l0" }),
                    ),
                    ("prior_kv", from("range_origin")),
                    ("donor_a_kv", from("range_origin")),
                    ("donor_b_kv", from("range_origin")),
                    ("ple", from(&format!("aux_p0_l{l}"))),
                    ("layer", layers[l].clone()),
                ],
            );
        }
        add("decode_init".into(), "decode-init", vec![]);
        for t in 0..3 {
            add(
                format!("embed_c{t}"),
                "input-embedding",
                vec![
                    ("prompt", no_prompt.clone()),
                    ("claimed", claimed[t].clone()),
                    (
                        "previous",
                        from(&if t == 0 {
                            "embed_p0".into()
                        } else {
                            format!("embed_c{}", t - 1)
                        }),
                    ),
                    ("embedding", embedding.clone()),
                    ("witness", page_witness.clone()),
                ],
            );
            for l in 0..2 {
                add(
                    format!("aux_c{t}_l{l}"),
                    "prefill-prepare-aux",
                    vec![
                        ("embedded", from(&format!("embed_c{t}"))),
                        ("witness", page_witness.clone()),
                        ("layer", aux[l].clone()),
                    ],
                );
            }
            for l in 0..2 {
                add(
                    format!("range_c{t}_l{l}"),
                    "prefill-range",
                    vec![
                        (
                            "activations",
                            from(&if l == 0 {
                                format!("embed_c{t}")
                            } else {
                                format!("range_c{t}_l0")
                            }),
                        ),
                        (
                            "prior_kv",
                            from(&if t == 0 {
                                format!("range_p0_l{l}")
                            } else {
                                format!("range_c{}_l{l}", t - 1)
                            }),
                        ),
                        ("donor_a_kv", from("range_origin")),
                        ("donor_b_kv", from("range_origin")),
                        ("ple", from(&format!("aux_c{t}_l{l}"))),
                        ("layer", layers[l].clone()),
                    ],
                );
            }
            add(
                format!("check_c{t}"),
                "sequence-check",
                vec![
                    ("activations", from(&format!("range_c{t}_l1"))),
                    (
                        "previous",
                        from(&if t == 0 {
                            "range_p0_l1".into()
                        } else {
                            format!("range_c{}_l1", t - 1)
                        }),
                    ),
                    ("claimed", claimed[t].clone()),
                    (
                        "prior",
                        from(&if t == 0 {
                            "decode_init".into()
                        } else {
                            format!("check_c{}", t - 1)
                        }),
                    ),
                    ("prompt", prompt.clone()),
                    ("head", head.clone()),
                ],
            );
        }
        add(
            "output_finalize".into(),
            "output-finalize",
            vec![
                ("edge", from("check_c2")),
                ("decoder", decoder),
                ("descriptor", descriptor),
                ("witness", witness),
                ("prompt", prompt),
            ],
        );
        fs::write(self.base.join("Raster.toml"), text).unwrap();
    }
    fn run(&self, budget: usize, workers: usize, subprocess: bool) -> PathBuf {
        self.run_mode(budget, workers, subprocess, "selective")
    }
    fn run_mode(&self, budget: usize, workers: usize, subprocess: bool, mode: &str) -> PathBuf {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_staged-infer"));
        cmd.current_dir(&self.base)
            .args(["chain", "run"])
            .env("STAGED_INFER_WEIGHT_CACHE_BYTES", budget.to_string())
            .env("STAGED_INFER_PLE_LOADING", mode)
            .env("STAGED_INFER_AUX_PARALLELISM", workers.to_string());
        if subprocess {
            cmd.arg("--staged-subprocess");
        }
        let output = cmd.output().unwrap();
        assert!(
            output.status.success(),
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        let root = self.base.join("target/staged-infer/runs");
        fs::read_dir(root)
            .unwrap()
            .map(|e| e.unwrap().path())
            .max()
            .unwrap()
    }
    fn compare(&self, a: &Path, b: &Path) {
        for (name, _) in &self.stages {
            for file in [
                "output.bin",
                "output.rindex",
                "output_manifest.json",
                "input_manifest.json",
            ] {
                assert_eq!(
                    fs::read(a.join(name).join(file)).unwrap(),
                    fs::read(b.join(name).join(file)).unwrap(),
                    "{name}/{file}"
                );
            }
            let ac =
                inference_artifacts::read_checkpoint_from_stage_dir(name, &a.join(name)).unwrap();
            let bc =
                inference_artifacts::read_checkpoint_from_stage_dir(name, &b.join(name)).unwrap();
            assert_eq!(
                inference_artifacts::checkpoint_hash(&ac).unwrap(),
                inference_artifacts::checkpoint_hash(&bc).unwrap(),
                "{name} checkpoint hash"
            );
        }
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        if !std::thread::panicking() {
            let _ = fs::remove_dir_all(&self.base);
        }
    }
}

#[test]
fn repeated_ranges_have_identical_checkpoints_with_worker_and_cache_reuse() {
    let mut f = Fixture::new();
    f.manifest();
    let baseline = f.run_mode(0, 1, true, "eager");
    f.compare(&baseline, &f.run(0, 1, true));
    for (budget, workers) in [
        (0, 1),
        (32 * 1024, 2),
        (64 * 1024 * 1024, 1),
        (64 * 1024 * 1024, 2),
    ] {
        let run = f.run(budget, workers, false);
        f.compare(&baseline, &run);
        let perf: serde_json::Value =
            serde_json::from_slice(&fs::read(run.join("staged-performance.json")).unwrap())
                .unwrap();
        let starts = perf["auxiliary_process_starts"].as_u64().unwrap();
        assert_eq!(perf["stages"][0]["name"], "range_origin");
        assert_eq!(perf["stages"][1]["name"], "embed_p0");
        // The initial embedding retains only small verified metadata now.
        let initial_retained = perf["stages"][1]["cache"]["retained_bytes"]
            .as_u64()
            .unwrap();
        assert!(initial_retained <= 16 * 1024);
        if budget == 0 {
            assert_eq!(initial_retained, 0);
        }
        assert!(starts >= 1 && starts <= workers as u64);
        assert!(perf["main_cache"]["retained_bytes"].as_u64().unwrap() <= budget as u64);
        if budget >= 64 * 1024 * 1024 {
            assert!(perf["main_cache"]["hits"].as_u64().unwrap() > 0);
            assert!(perf["stages"]
                .as_array()
                .unwrap()
                .iter()
                .any(|s| !s["worker"].is_null() && s["cache"]["hits"].as_u64().unwrap() > 0));
        }
    }
}

#[test]
fn workers_reject_bad_protocol_and_failed_stage_inputs() {
    use std::io::Write;
    use std::process::Stdio;
    for request in [
        "not-json\n".to_owned(),
        "{\"Shutdown\":{\"version\":999}}\n".to_owned(),
        format!(
            "{}\n",
            serde_json::json!({"Run":{"version":1,"job":{
                "id":7,"name":"aux_p0_l0","kind":{"PrefillPrepareAux":{"layer":0}},
                "input":"missing-input.json","input_manifest":"missing-manifest.json","output_dir":"unused-output"
            }}})
        ),
    ] {
        let mut child = Command::new(env!("CARGO_BIN_EXE_staged-infer"))
            .args(["--aux-worker", "1024"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        child
            .stdin
            .take()
            .unwrap()
            .write_all(request.as_bytes())
            .unwrap();
        let result = child.wait_with_output().unwrap();
        assert!(!result.status.success());
        if request.contains("missing-input") {
            let response: serde_json::Value = serde_json::from_slice(&result.stdout).unwrap();
            assert_eq!(response["id"], 7);
            assert!(response["error"].is_string());
            assert!(response["timings"].is_null());
        }
    }
    // A graceful shutdown is accepted by both executable frontends (CLI tested separately).
    let mut child = Command::new(env!("CARGO_BIN_EXE_staged-infer"))
        .args(["--aux-worker", "0"])
        .stdin(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(b"{\"Shutdown\":{\"version\":1}}\n")
        .unwrap();
    assert!(child.wait().unwrap().success());
}

#[test]
fn selective_input_failure_does_not_publish_and_unknown_mode_fails_early() {
    let mut f = Fixture::new();
    f.manifest();
    let bad_mode = Command::new(env!("CARGO_BIN_EXE_staged-infer"))
        .current_dir(&f.base)
        .args(["chain", "run"])
        .env("STAGED_INFER_PLE_LOADING", "typo")
        .output()
        .unwrap();
    assert!(!bad_mode.status.success());
    assert!(String::from_utf8_lossy(&bad_mode.stderr).contains("must be selective or eager"));
    assert!(!f.base.join("target/staged-infer/runs").exists());
    let run = f.run(0, 1, false);
    let source = run.join("aux_p0_l0");
    let dest = f.base.join("target/staged-infer/runs/rejected/aux_p0_l0");
    fs::create_dir_all(&dest).unwrap();
    for file in ["input.json", "input_manifest.json"] {
        fs::copy(source.join(file), dest.join(file)).unwrap();
    }
    // Change projection bytes while retaining the authorized commitment/index.
    let path = f.base.join("aux0.rastered");
    let mut data = fs::read(&path).unwrap();
    *data.last_mut().unwrap() ^= 0x40;
    fs::write(path, data).unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_staged-infer"))
        .current_dir(&f.base)
        .arg("--run-stage")
        .arg(&dest)
        .arg("--input")
        .arg(dest.join("input.json"))
        .arg("--input-manifest")
        .arg(dest.join("input_manifest.json"))
        .env("STAGED_INFER_PLE_LOADING", "selective")
        .env(raster_runtime::auth::AUTH_ENV, "0")
        .env(raster_runtime::OUTPUT_DIR_ENV, &dest)
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(
        String::from_utf8_lossy(&result.stderr)
            .contains("typed selection differs from verified payload"),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    for file in ["output.bin", "output.rindex", "output_manifest.json"] {
        assert!(!dest.join(file).exists());
    }
}

#[test]
fn worker_reuse_preserves_output_isolation_and_reports_failed_writes() {
    use std::io::Write;
    use std::process::Stdio;
    let mut f = Fixture::new();
    f.manifest();
    let run = f.run(1 << 20, 2, false);
    let source = run.join("aux_p0_l0");
    let good = f.base.join("worker-good");
    fs::create_dir_all(&good).unwrap();
    let blocked = f.base.join("worker-blocked");
    fs::write(&blocked, b"not a directory").unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_staged-infer"))
        .args(["--aux-worker", "1048576"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut input = child.stdin.take().unwrap();
    for (id, dest) in [(1, &good), (2, &blocked)] {
        let request = serde_json::json!({"Run":{"version":1,"job":{
            "id":id,"name":"aux_p0_l0","kind":{"PrefillPrepareAux":{"layer":0}},
            "input":source.join("input.json"),"input_manifest":source.join("input_manifest.json"),"output_dir":dest
        }}});
        writeln!(input, "{request}").unwrap();
    }
    drop(input);
    let result = child.wait_with_output().unwrap();
    assert!(!result.status.success());
    let responses: Vec<serde_json::Value> = String::from_utf8(result.stdout)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(responses.len(), 2);
    assert!(responses[0]["error"].is_null());
    assert!(responses[1]["error"].is_string());
    assert_eq!(responses[1]["cache"]["hits"], 1);
    assert_eq!(
        fs::read(good.join("output.bin")).unwrap(),
        fs::read(source.join("output.bin")).unwrap()
    );
    assert_eq!(fs::read(blocked).unwrap(), b"not a directory");
}

#[test]
fn early_checkpoint_divergence_stops_and_reaps_workers() {
    let mut f = Fixture::new();
    f.manifest();
    let run = f.run(1 << 20, 2, false);
    let mut hashes: Vec<_> = f
        .stages
        .iter()
        .map(|(name, _)| {
            let checkpoint =
                inference_artifacts::read_checkpoint_from_stage_dir(name, &run.join(name)).unwrap();
            inference_artifacts::checkpoint_hash(&checkpoint).unwrap()
        })
        .collect();
    let stop = f
        .stages
        .iter()
        .position(|(name, _)| name == "aux_c1_l0")
        .unwrap();
    hashes[stop] = "0".repeat(64);
    fs::write(
        f.base.join("claimed-hashes.json"),
        serde_json::to_vec(&hashes).unwrap(),
    )
    .unwrap();
    let result = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "challenge_cleanup_child",
            "--ignored",
            "--nocapture",
        ])
        .env("IRIS_WORKER_TEST_BASE", &f.base)
        .env("IRIS_WORKER_TEST_STOP", stop.to_string())
        .env("STAGED_INFER_WEIGHT_CACHE_BYTES", "1048576")
        .env("STAGED_INFER_AUX_PARALLELISM", "2")
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    );
}

#[test]
#[ignore = "subprocess helper for early-divergence test"]
fn challenge_cleanup_child() {
    // The parent invokes this in isolation because the executor scopes process cwd/env.
    let Some(base) = std::env::var_os("IRIS_WORKER_TEST_BASE") else {
        return;
    };
    let base = PathBuf::from(base);
    let stop: usize = std::env::var("IRIS_WORKER_TEST_STOP")
        .unwrap()
        .parse()
        .unwrap();
    let path = base.join("claimed-hashes.json");
    let hashes = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    let result = staged_infer::CheckpointedInferenceExecutor
        .run_until_hash_divergence(staged_infer::CheckpointHashChallengeConfig {
            base_dir: base.clone(),
            manifest_path: base.join("Raster.toml"),
            current_exe: env!("CARGO_BIN_EXE_staged-infer").into(),
            staged_backend: staged_infer::chain_runner::StagedExecutionBackend::InProcess,
            claimed_hashes_path: path,
            claimed_hashes: hashes,
        })
        .unwrap();
    assert_eq!(result.divergence.unwrap().checkpoint_index, stop);
    assert_eq!(result.recomputed_trace.checkpoints.len(), stop + 1);
    assert!(!result.chain_dir.join("range_c1_l0").exists());
    let mut system = sysinfo::System::new();
    system.refresh_processes(sysinfo::ProcessesToUpdate::All, true);
    let parent = sysinfo::Pid::from_u32(std::process::id());
    assert!(
        system
            .processes()
            .values()
            .all(|p| p.thread_kind().is_some() || p.parent() != Some(parent)),
        "auxiliary child survived early divergence"
    );
}

#[test]
#[ignore = "requires actual Raster stage binaries in STAGED_INFER_REPLAY_BIN_DIR"]
fn repeated_ranges_match_authenticated_raster_stage_execution() {
    let bin = PathBuf::from(
        std::env::var_os("STAGED_INFER_REPLAY_BIN_DIR").expect("build Raster stage binaries first"),
    );
    let mut f = Fixture::new();
    f.manifest();
    let run = f.run(64 * 1024 * 1024, 2, false);
    let replay = f.base.join("authenticated-replay");
    fs::create_dir_all(&replay).unwrap();
    for (name, project) in &f.stages {
        let source = run.join(name);
        let dest = replay.join(name);
        fs::create_dir_all(&dest).unwrap();
        let mut input: serde_json::Value =
            serde_json::from_slice(&fs::read(source.join("input.json")).unwrap()).unwrap();
        for entry in input.as_object_mut().unwrap().values_mut() {
            for field in ["path", "index_path"] {
                if let Some(path) = entry.get(field).and_then(|v| v.as_str()) {
                    if let Ok(relative) = Path::new(path).strip_prefix(&run) {
                        entry[field] = serde_json::Value::String(
                            replay.join(relative).to_string_lossy().into_owned(),
                        );
                    }
                }
            }
        }
        fs::write(dest.join("input.json"), serde_json::to_vec(&input).unwrap()).unwrap();
        fs::copy(
            source.join("input_manifest.json"),
            dest.join("input_manifest.json"),
        )
        .unwrap();
        let result = Command::new(bin.join(format!("{project}{}", std::env::consts::EXE_SUFFIX)))
            .args([
                "--input",
                dest.join("input.json").to_str().unwrap(),
                "--input-manifest",
                dest.join("input_manifest.json").to_str().unwrap(),
            ])
            .env(raster_runtime::auth::AUTH_ENV, "1")
            .env(raster_runtime::OUTPUT_DIR_ENV, &dest)
            .env(raster_runtime::TRACE_PATH_ENV, dest.join("trace.json"))
            .env(raster_runtime::TRACE_FORMAT_ENV, "json")
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{name}: {}\n{}",
            String::from_utf8_lossy(&result.stdout),
            String::from_utf8_lossy(&result.stderr)
        );
    }
    f.compare(&run, &replay);
}
