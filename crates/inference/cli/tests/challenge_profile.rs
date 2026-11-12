// Real native -> replay -> targeted corruption -> fault proof. No model required.
use inference_artifacts::{file_sha256, read_json, write_json};
use serde_json::{json, Value};
use std::{
    fs,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    time::{SystemTime, UNIX_EPOCH},
};
fn logged(command: &mut Command, dir: &Path, name: &str) -> String {
    let path = dir.join(format!("{name}.log"));
    let file = fs::File::create(&path).unwrap();
    let status = command
        .stdout(file.try_clone().unwrap())
        .stderr(Stdio::from(file))
        .status()
        .unwrap();
    assert!(status.success(), "failed {name}; see {}", path.display());
    fs::read_to_string(path).unwrap()
}
#[test]
#[ignore = "requires current cargo-raster, RISC Zero guest toolchain, Docker and real proving"]
fn recursive_heavy_window_end_to_end() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../..")
        .canonicalize()
        .unwrap();
    let dir = root.join("target/heavy-window-e2e").join(format!(
        "{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let project = dir.join("project");
    fs::create_dir_all(project.join("src")).unwrap();
    println!("Heavy-window artifacts: {}", dir.display());
    for name in ["lib.rs", "main.rs"] {
        fs::copy(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures/heavy-window")
                .join(name),
            project.join("src").join(name),
        )
        .unwrap();
    }
    let raster = root.parent().unwrap().join("raster");
    fs::write(
        project.join("Cargo.toml"),
        format!(
            r#"
[package]
name = "heavy-window-fixture"
version="0.1.0"
edition="2021"
[workspace]
[features]
default=["std"]
std=["raster/std"]
profiling=["raster/profiling"]
[dependencies]
raster = {{ path = {:?}, default-features = false }}
serde={{version="1",default-features=false,features=["derive","alloc"]}}
[patch.crates-io]
risc0-zkvm-platform={{path={:?}}}
risc0-circuit-keccak={{path={:?}}}
risc0-zkvm={{path={:?}}}
"#,
            raster.join("crates/raster"),
            raster.join("vendor/risc0-zkvm-platform"),
            raster.join("vendor/risc0-circuit-keccak"),
            raster.join("vendor/risc0-zkvm")
        ),
    )
    .unwrap();
    let mut build = Command::new("cargo");
    build
        .current_dir(&project)
        .args(["raster", "build", "--backend", "risc0"]);
    logged(&mut build, &dir, "build");
    let input = dir.join("input.json");
    let inputs = dir.join("input_manifest.json");

    #[derive(serde::Serialize, serde::Deserialize, raster::Selectable)]
    struct Values {
        values: raster::prelude::List<u64>,
    }
    let data = dir.join("values.rastered");
    let index = dir.join("values.rindex");
    let root = raster::write_raster_files(
        &Values {
            values: (1..=8).collect::<Vec<_>>().into(),
        },
        &data,
        &index,
    )
    .unwrap();
    write_json(
        &input,
        &json!({"source":{"path":data,"index_path":index,"load_preference":"read"}}),
    )
    .unwrap();
    write_json(
        &inputs,
        &json!({"source":{"type":"sha256","encoding":"raster","commitment":root}}),
    )
    .unwrap();

    let original = dir.join("original.commit.bin");
    let mut run = Command::new("cargo");
    run.current_dir(&project)
        .args(["raster", "run", "--save-trace", "--commit"])
        .arg(&original)
        .args(["--fraud-proof-window-size", "2", "--input"])
        .arg(&input)
        .arg("--input-manifest")
        .arg(&inputs);
    let stdout = logged(&mut run, &dir, "commit");
    let stage_dir = PathBuf::from(
        stdout
            .lines()
            .find_map(|l| l.trim().strip_prefix("Run artifacts dir: "))
            .unwrap(),
    );
    fs::copy(&inputs, stage_dir.join("input_manifest.json")).unwrap();
    let stage = "heavy_fixture";
    let checkpoint =
        inference_artifacts::read_checkpoint_from_stage_dir(stage, &stage_dir).unwrap();
    let model = dir.join("model.json");
    fs::write(&model, "{}").unwrap();
    let manifest = dir.join("Raster.toml");
    fs::write(
        &manifest,
        format!(
            "[chain]\nname=\"fixture\"\n[[chain.stage]]\nname=\"{stage}\"\nproject={:?}\n",
            project
        ),
    )
    .unwrap();
    let prepared = dir.join("prepared.json");
    write_json(&prepared,&json!({"version":inference_artifacts::PREPARED_RUN_VERSION,"run_spec_path":dir.join("inference.toml"),"model_manifest_path":model,
        "model_id":"","raw_prompt":true,
        "model_manifest_sha256":file_sha256(&model).unwrap(),"prompt":{"resolved_prompt":"","rendered_prompt":"","initial_pieces":[],"eos_token_ids":[]},
        "tokens":1,"run_manifest_path":manifest,"run_manifest_sha256":file_sha256(&manifest).unwrap()})).unwrap();
    let checkpoints = dir.join("checkpoint_trace.json");
    write_json(&checkpoints, &json!([checkpoint])).unwrap();
    let claim = dir.join("claim.json");
    write_json(&claim,&json!({"version":1,"checkpoint_trace_path":checkpoints,"prepared_run_path":prepared,
        "input":{"stage":stage,"commitment":checkpoint.input_commitment},"output":{"stage":stage,"commitment":checkpoint.output_commitment}})).unwrap();
    let package = dir.join("replay_package.json");
    write_json(&package,&json!({"version":1,"stage":stage,"replay_run_dir":stage_dir.parent(),
        "stage_dir":stage_dir,"input_path":input,"input_manifest_path":inputs,"output_path":stage_dir.join("output.bin"),
        "output_index_path":stage_dir.join("output.rindex"),"output_manifest_path":stage_dir.join("output_manifest.json"),"commit_path":original})).unwrap();
    let challenge = dir.join("challenge.json");
    write_json(&challenge,&json!({"version":1,"stage":stage,"recomputed_trace_path":checkpoints,
        "divergence_path":dir.join("unused.json"),"challenge_trace_path":dir.join("unused.json"),"replay_package_path":package,"raster_commit_path":original})).unwrap();
    let iris = env!("CARGO_BIN_EXE_iris");
    let profile_dir = dir.join("profile");
    let mut profile = Command::new(iris);
    profile
        .args(["challenge", "profile", "--challenge"])
        .arg(&challenge)
        .arg("--claim-context")
        .arg(&claim)
        .arg("--output")
        .arg(&profile_dir);
    logged(&mut profile, &dir, "profile");
    let summary = profile_dir.join("challenge-profile.json");
    let measurement: Value = read_json(&summary).unwrap();
    assert_eq!(measurement["complete"], true);
    assert_eq!(measurement["version"], 2);
    assert!(measurement["calls_replayed"].as_u64().unwrap() <= 3);
    assert_eq!(measurement["selection_policy"], "native-runtime");
    for phase in ["native-timing", "selective-capture"] {
        let log = fs::read_to_string(profile_dir.join(phase).join("native.stdout.log")).unwrap();
        let pass = PathBuf::from(
            log.lines()
                .find_map(|l| l.trim().strip_prefix("Run artifacts dir: "))
                .unwrap(),
        );
        assert!(!pass.join("trace.ndjson").exists() && !pass.join("trace.bin").exists());
        assert!(!pass.join("profile.ndjson").exists() && !pass.join("profile.json").exists());
    }
    // Failed capture must never publish a replayable run or complete manifest.
    let selected: Value = read_json(&profile_dir.join("native-selection.json")).unwrap();
    for (name, budget, expected) in [
        ("budget", 1u64, "sample budget exceeded"),
        (
            "mapping",
            8589934592,
            "selected invocation mapping mismatch",
        ),
        ("missing", 8589934592, "capture coverage differs"),
    ] {
        let mut selection = selected.clone();
        if name == "mapping" {
            selection["calls"][0]["coordinates"] = json!([9999]);
        }
        if name == "missing" {
            selection["calls"][0]["exec_index"] = json!(1);
        }
        let path = dir.join(format!("negative-{name}.json"));
        write_json(&path, &selection).unwrap();
        let output = Command::new("cargo")
            .current_dir(&project)
            .args(["raster", "run", "--commit"])
            .arg(dir.join(format!("negative-{name}.commit.bin")))
            .args(["--fraud-proof-window-size", "2", "--capture-profile-calls"])
            .arg(path)
            .arg("--sample-max-bytes")
            .arg(budget.to_string())
            .arg("--input")
            .arg(&input)
            .arg("--input-manifest")
            .arg(&inputs)
            .output()
            .unwrap();
        let text = format!(
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        fs::write(dir.join(format!("negative-{name}.log")), &text).unwrap();
        assert!(
            !output.status.success() && text.contains(expected),
            "{name}: {text}"
        );
        let run = PathBuf::from(
            text.lines()
                .find_map(|l| l.trim().strip_prefix("Run artifacts dir: "))
                .unwrap(),
        );
        assert!(!run.join("run.json").exists() && !run.join("selected-capture.json").exists());
    }
    // A trace-backed oracle must give identical cycles for these same calls.
    let mut oracle = Command::new("cargo");
    oracle
        .current_dir(&project)
        .args(["raster", "replay", "profile"])
        .arg(&stage_dir);
    logged(&mut oracle, &dir, "trace-oracle");
    let legacy: Value = read_json(&stage_dir.join("replay-profile.json")).unwrap();
    for call in measurement["top_calls"].as_array().unwrap() {
        let measured = legacy["tiles"][call["tile_id"].as_str().unwrap()]["replayed"]
            .as_array()
            .unwrap()
            .iter()
            .find(|c| c["exec_index"] == call["exec_index"])
            .expect("selected native invocation in oracle sample");
        assert_eq!(measured["user_cycles"], call["user_cycles"]);
        assert!(call["p50_user_cycles"].is_null());
    }
    assert!(!measurement["target"].is_null());
    let proof_dir = dir.join("fault");
    let mut corrupt = Command::new(iris);
    corrupt
        .args(["challenge", "corrupt", "--challenge"])
        .arg(&challenge)
        .arg("--profile")
        .arg(&summary)
        .arg("--output")
        .arg(&proof_dir);
    logged(&mut corrupt, &dir, "corrupt");
    let before = file_sha256(&original).unwrap();
    let mut proof = Command::new(iris);
    proof
        .args(["fault", "prove", "--challenge"])
        .arg(&challenge)
        .arg("--claim-context")
        .arg(&claim)
        .arg("--commitment")
        .arg(proof_dir.join("corrupted_commit.bin"))
        .arg("--output")
        .arg(&proof_dir)
        .env("RISC0_DEV_MODE", "0");
    logged(&mut proof, &dir, "proof");
    let timing: Value = read_json(&proof_dir.join("fault_timings.json")).unwrap();
    assert_eq!(timing["selected_window"], measurement["target"]);
    assert_eq!(file_sha256(&original).unwrap(), before);
    let result: Value = read_json(&proof_dir.join("fault_bundle.json")).unwrap();
    assert_eq!(result["verified"], true);
}
