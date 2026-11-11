#![cfg(feature = "replay-profile")]
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{SystemTime, UNIX_EPOCH};

use inference_artifacts::{read_json, EXECUTION_TIMES_JSON};
use staged_infer::chain_runner::StagedExecutionBackend;
use staged_infer::{CheckpointedInferenceConfig, CheckpointedInferenceExecutor, ParityPolicy};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../..")
        .canonicalize()
        .unwrap()
}

fn unique_dir(parent: &Path) -> PathBuf {
    let path = parent.join(format!(
        "{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    fs::create_dir_all(&path).unwrap();
    path.canonicalize().unwrap()
}

#[cfg(unix)]
#[test]
fn native_prefix_stops_before_later_sequential_and_parallel_stages() {
    use std::os::unix::fs::PermissionsExt;

    let root = unique_dir(&std::env::temp_dir().join("replay-profile-prefix"));
    let manifest = root.join("Raster.toml");
    // The first stage is given empty values for everything it reads. It
    // records what is wrong with them and finishes, which is all the
    // scheduler needs of it here.
    let external = |name: &str, value: &dyn Fn() -> (Vec<u8>, Vec<u8>, String)| {
        let (data, index, commitment) = value();
        fs::write(root.join(format!("{name}.rastered")), data).unwrap();
        fs::write(root.join(format!("{name}.rindex")), index).unwrap();
        format!(
            "inputs.{name} = {{ external = {{ path = \"{name}.rastered\", index_path = \"{name}.rindex\", commitment = \"{commitment}\" }} }}\n"
        )
    };
    raster::init();
    let decoder = output_finalize::input::DecoderTable::default();
    let inputs = [
        external("edge", &|| {
            raster::encode_raster_value(&output_finalize::input::DecodeEdge::default()).unwrap()
        }),
        external("prompt", &|| {
            raster::encode_raster_value(&output_finalize::prompt::PromptState::default()).unwrap()
        }),
        external("decoder", &|| {
            raster::encode_raster_value(&decoder).unwrap()
        }),
        external("descriptor", &|| {
            raster::encode_raster_value(&raster::BytesPage::__from_parts(0, 0, Vec::new())).unwrap()
        }),
        external("witness", &|| {
            raster::encode_raster_value(&output_finalize::decoder_witness(&decoder).unwrap())
                .unwrap()
        }),
    ]
    .concat();
    // Later stages intentionally have no usable inputs: they must not execute.
    fs::write(
        &manifest,
        format!(
            r#"
[chain]
name = "prefix-test"
[[chain.stage]]
name = "output_finalize"
project = "raster-stages/output-finalize"
{inputs}
[[chain.stage]]
name = "prefill_prepare_aux_l0"
project = "raster-stages/prefill-prepare-aux"
[[chain.stage]]
name = "prefill_prepare_aux_l1"
project = "raster-stages/prefill-prepare-aux"
[[chain.stage]]
name = "prefill_prepare_aux_l2"
project = "raster-stages/prefill-prepare-aux"
[[chain.stage]]
name = "check_c0"
project = "raster-stages/sequence-check"
"#
        ),
    )
    .unwrap();
    let config = CheckpointedInferenceConfig {
        base_dir: root.clone(),
        manifest_path: manifest,
        current_exe: PathBuf::from("/not-an-executable"),
        staged_backend: StagedExecutionBackend::InProcess,
        parity_policy: ParityPolicy::Skip,
    };
    let sequential = CheckpointedInferenceExecutor
        .run_through_stage(config.clone(), "output_finalize")
        .unwrap();
    assert!(sequential.output_path.is_file());
    assert!(sequential.output_index_path.is_file());
    assert!(sequential.input_manifest_path.is_file());
    assert!(!sequential.chain_dir.join("prefill_prepare_aux_l0").exists());
    assert_eq!(
        sequential.project_dir,
        root.join("raster-stages/output-finalize")
    );

    // A stand-in aux worker publishes a valid artifact. This test exercises the
    // actual scheduler and subprocess boundary without a model/kernel workload.
    let worker = root.join("aux-worker");
    let source = sequential
        .stage_dir
        .to_string_lossy()
        .replace('\'', "'\\''");
    fs::write(
        &worker,
        format!(
            r#"#!/usr/bin/env python3
import sys,json,shutil,pathlib
for line in sys.stdin:
    request=json.loads(line)
    if "Shutdown" in request: break
    job=request["Run"]["job"]
    for name in ["output.bin","output.rindex","output_manifest.json"]:
        shutil.copyfile('{source}/'+name, pathlib.Path(job["output_dir"])/name)
    timings={{k:{{"secs":0,"nanos":1}} for k in ["input_load_duration","kernel_duration","encode_write_duration","direct_stage_duration"]}}
    cache={{k:0 for k in ["budget_bytes","retained_bytes","hits","misses","bypasses","preparation_duration_ns"]}}
    print(json.dumps({{"version":1,"id":job["id"],"name":job["name"],"timings":timings,"cache":cache,"error":None}}),flush=True)
"#
        ),
    )
    .unwrap();
    fs::set_permissions(&worker, fs::Permissions::from_mode(0o755)).unwrap();
    let mut parallel_config = config.clone();
    parallel_config.current_exe = worker;
    let parallel = CheckpointedInferenceExecutor
        .run_through_stage(parallel_config, "prefill_prepare_aux_l1")
        .unwrap();
    assert!(parallel
        .chain_dir
        .join("prefill_prepare_aux_l0/output.bin")
        .exists());
    assert!(parallel.output_path.is_file());
    assert!(!parallel.chain_dir.join("prefill_prepare_aux_l2").exists());
    assert!(!parallel.chain_dir.join("check_c0").exists());
    let times: serde_json::Value =
        read_json(&parallel.chain_dir.join(EXECUTION_TIMES_JSON)).unwrap();
    assert_eq!(times["stages"].as_array().unwrap().len(), 3);
    assert_eq!(times["aux_waves"][0]["stage_count"], 2);
    assert_eq!(
        times["aux_waves"][0]["last_stage"],
        "prefill_prepare_aux_l1"
    );
    let count = fs::read_dir(root.join("target/staged-infer/runs"))
        .unwrap()
        .count();
    assert!(CheckpointedInferenceExecutor
        .run_through_stage(config, "prefill_prepare_aux")
        .is_err());
    assert_eq!(
        fs::read_dir(root.join("target/staged-infer/runs"))
            .unwrap()
            .count(),
        count
    );
    fs::remove_dir_all(root).unwrap();
}

fn logged(mut command: Command, path: &Path) {
    let log = File::create(path).unwrap();
    let status = command
        .stdout(Stdio::from(log.try_clone().unwrap()))
        .stderr(Stdio::from(log))
        .status()
        .unwrap();
    assert!(
        status.success(),
        "command failed; inspect {}",
        path.display()
    );
}

#[test]
#[ignore = "requires matching cargo-raster, RISC Zero guest toolchain, and guest compilation"]
fn replay_profile_tokenizer_smoke() {
    let root = repo_root();
    let dir = unique_dir(&root.join("target/replay-profile-smoke"));
    println!("Replay profile smoke artifacts: {}", dir.display());
    let exe = env!("CARGO_BIN_EXE_iris");
    let mut import = Command::new(exe);
    import
        .current_dir(&root)
        .args(["model", "import", "--model"])
        .arg(root.join("model-bundles/parity-gemma"))
        .arg("--artifact-root")
        .arg(dir.join("model-artifacts"))
        .args(["--model-id", "parity-gemma", "--no-stage-fixtures"]);
    logged(import, &dir.join("import.log"));
    let run = dir.join("inference.toml");
    // The first prompt range's token check.
    fs::write(&run, "model_manifest = \"model-artifacts/parity-gemma/manifest.json\"\nprompt = \"abcdefghijklmn?!\"\nraw_prompt = true\ntokens = 1\n").unwrap();
    let mut profile = Command::new(exe);
    profile
        .current_dir(&root)
        .args(["replay-profile", "--run"])
        .arg(&run)
        .args(["--stage", "prompt_check_p0", "--top-n", "1"]);
    let log_path = dir.join("replay-profile.log");
    logged(profile, &log_path);
    let log = fs::read_to_string(&log_path).unwrap();
    let report_path = log
        .lines()
        .find_map(|line| line.strip_prefix("Replay profile report: "))
        .expect("missing summary path");
    let report: serde_json::Value = read_json(Path::new(report_path)).unwrap();
    assert_eq!(report["version"], 3);
    assert_eq!(report["complete"], true);
    assert_eq!(report["stage"], "prompt_check_p0");
    assert_eq!(report["top_calls"].as_array().unwrap().len(), 1);
    assert!(report["top_calls"][0]["user_cycles"].as_u64().unwrap() > 0);
    assert!(Path::new(report_path)
        .parent()
        .unwrap()
        .join("native-profile.json")
        .is_file());
    assert!(Path::new(report_path)
        .parent()
        .unwrap()
        .join("raster-replay-profile.json")
        .is_file());
    println!("Verified tokenizer replay profile: {report_path}");
}
