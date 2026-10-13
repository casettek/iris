use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{SystemTime, UNIX_EPOCH};

use inference_artifacts::replay_profile::{ReplayProfileStatus, StageReplayProfile};
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
    // Later stages intentionally have no usable inputs: they must not execute.
    fs::write(
        &manifest,
        r#"
[chain]
name = "prefix-test"
[[chain.stage]]
name = "decode_init"
project = "raster-stages/decode-init"
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
name = "output_finalize"
project = "raster-stages/output-finalize"
"#,
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
        .run_through_stage(config.clone(), "decode_init")
        .unwrap();
    assert!(sequential.output_path.is_file());
    assert!(sequential.output_index_path.is_file());
    assert!(sequential.input_manifest_path.is_file());
    assert!(!sequential.chain_dir.join("prefill_prepare_aux_l0").exists());
    assert_eq!(
        sequential.project_dir,
        root.join("raster-stages/decode-init")
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
            r#"#!/bin/sh
test "$1" = "--run-stage" || exit 10
for name in output.bin output.rindex output_manifest.json; do
  cp '{source}'/"$name" "$2/$name" || exit 11
done
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
    assert!(!parallel.chain_dir.join("output_finalize").exists());
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
    // The pinned near-window prompt exceeds the seed stage's eight-merge
    // batch, so the expanded chain includes a repeated merge stage.
    fs::write(&run, "model_manifest = \"model-artifacts/parity-gemma/manifest.json\"\nprompt = \"abcdefghijklmn?!\"\nraw_prompt = true\ntokens = 1\n").unwrap();
    let mut profile = Command::new(exe);
    profile
        .current_dir(&root)
        .args(["replay-profile", "--run"])
        .arg(&run)
        .args(["--stage", "prompt_merge_b0", "--top-n", "1"]);
    let log_path = dir.join("replay-profile.log");
    logged(profile, &log_path);
    let log = fs::read_to_string(&log_path).unwrap();
    let report_path = log
        .lines()
        .find_map(|line| line.strip_prefix("Replay profile report: "))
        .expect("missing summary path");
    let report: StageReplayProfile = read_json(Path::new(report_path)).unwrap();
    assert!(matches!(report.status, ReplayProfileStatus::Complete));
    assert_eq!(report.stage, "prompt_merge_b0");
    assert_eq!(report.selected_tile_ids.len(), 1);
    assert_eq!(report.native.output_matches_baseline, Some(true));
    assert_eq!(report.replay.output_matches_baseline, Some(true));
    assert!(report.replay_tiles[0].max_guest_cycles.unwrap() > 0);
    assert!(report.replay_tiles[0].total_guest_cycles > 0);
    assert!(
        report.replay_tiles[0]
            .transition_overhead
            .as_ref()
            .unwrap()
            .guest_cycles
            > 0
    );
    assert_eq!(report.version, 2);
    assert!(report.native_tiles.is_empty());
    assert!(report.native_ranking_path.as_ref().unwrap().is_file());
    let json: serde_json::Value = read_json(Path::new(report_path)).unwrap();
    assert!(json.get("native_tiles").is_none());
    assert!(!json.to_string().contains("_ns"));
    assert!(log.contains("heaviest replay"));
    assert!(log.contains("guest cycles across"));
    assert!(log.contains("Sample transition overhead (one invocation):"));
    assert!(
        !log.contains("_ns"),
        "native fields leaked into terminal: {log}"
    );
    assert!(
        !log.contains(" ns"),
        "native timings leaked into terminal: {log}"
    );
    assert!(report.native.profile_path.unwrap().is_file());
    assert!(report.replay.profile_path.unwrap().is_file());
    assert!(!report
        .stage_dir
        .parent()
        .unwrap()
        .join("prompt_prepare")
        .exists());
    println!("Verified tokenizer replay profile: {report_path}");
}
