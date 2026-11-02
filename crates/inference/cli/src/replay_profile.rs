//! Standalone stage profiling, using the same Raster v4 report helpers as challenges.
use anyhow::{ensure, Result};
use inference_artifacts::write_json;
use raster_analysis::replay_profile::ReplayProfile;
use staged_infer::chain_runner::StagedExecutionBackend;
use staged_infer::{CheckpointedInferenceConfig, CheckpointedInferenceExecutor, ParityPolicy};
use std::{
    fs::{self, File},
    path::PathBuf,
    process::{Command, Stdio},
};
#[derive(Debug, Clone)]
pub struct ReplayProfileOptions {
    pub base_dir: PathBuf,
    pub run_spec_path: PathBuf,
    pub current_exe: PathBuf,
    pub stage: String,
    pub top_n: usize,
}
pub fn run_replay_profile(options: ReplayProfileOptions) -> Result<PathBuf> {
    ensure!(options.top_n > 0, "--top-n must be positive");
    let prepared = crate::claim::prepare_claim_run(&crate::claim::ClaimBuildOptions {
        base_dir: options.base_dir.clone(),
        run_spec_path: options.run_spec_path,
        current_exe: options.current_exe.clone(),
        staged_backend: StagedExecutionBackend::InProcess,
    })?;
    let prefix = CheckpointedInferenceExecutor.run_through_stage(
        CheckpointedInferenceConfig {
            base_dir: options.base_dir,
            manifest_path: prepared.manifest_path,
            current_exe: options.current_exe,
            staged_backend: StagedExecutionBackend::InProcess,
            parity_policy: ParityPolicy::Skip,
        },
        &options.stage,
    )?;
    let output = prefix.chain_dir.join("replay-profile").join(&prefix.stage);
    fs::create_dir_all(&output)?;
    let log = output.join("native.stdout.log");
    let status = Command::new("cargo")
        .current_dir(&prefix.project_dir)
        .args([
            "raster",
            "run",
            "--features",
            "profiling",
            "--trace-format",
            "json",
            "--input",
        ])
        .arg(&prefix.input_path)
        .arg("--input-manifest")
        .arg(&prefix.input_manifest_path)
        .stdout(File::create(&log)?)
        .stderr(File::create(output.join("native.stderr.log"))?)
        .stdin(Stdio::null())
        .status()?;
    ensure!(
        status.success(),
        "native profiling failed; see {}",
        output.display()
    );
    let run = crate::challenge_profile::run_directory(&log, &prefix.project_dir)?;
    crate::challenge_profile::verify_output(&prefix.stage, &prefix.stage_dir, &run)?;
    let status = Command::new("cargo")
        .current_dir(&prefix.project_dir)
        .args(["raster", "replay", "profile"])
        .arg(&run)
        .stdout(File::create(output.join("replay.stdout.log"))?)
        .stderr(File::create(output.join("replay.stderr.log"))?)
        .stdin(Stdio::null())
        .status()?;
    ensure!(
        status.success(),
        "guest replay failed; see {}",
        output.display()
    );
    let raw = run.join("replay-profile.json");
    let profile = ReplayProfile::from_json(&fs::read(&raw)?)?;
    let length = fs::read_to_string(run.join("trace.ndjson"))?
        .lines()
        .filter(|l| !l.trim().is_empty())
        .count();
    let mut summary = crate::challenge_profile::summarize(&profile, length, 128, options.top_n)?;
    summary["version"] = 3.into();
    summary["kind"] = "stage-replay-profile".into();
    summary["stage"] = prefix.stage.into();
    summary["complete"] = profile.complete.into();
    summary["run_dir"] = serde_json::to_value(&run)?;
    // Standalone measurements are not authorized targets for challenge corruption.
    summary.as_object_mut().unwrap().remove("target");
    summary.as_object_mut().unwrap().remove("target_reason");
    fs::copy(raw, output.join("raster-replay-profile.json"))?;
    fs::copy(run.join("profile.json"), output.join("native-profile.json"))?;
    let path = output.join("replay-profile.json");
    write_json(&path, &summary)?;
    println!("{}", profile.to_text());
    println!("Replay profile report: {}", path.display());
    Ok(path)
}
