//! Observability of a frozen challenge using Raster's existing run/replay interfaces.
use anyhow::{ensure, Context, Result};
use inference_artifacts::{
    file_sha256, read_json, write_json, ChallengeBundle, CheckpointTrace, ClaimBundle, PreparedRun,
    ReplayPackage,
};
use raster_analysis::replay_profile::ReplayProfile;
use raster_compiler::{CfsBuilder, Project};

use raster_prover::commitment_file::CommitmentFile;

use serde_json::{json, Value};
use std::{
    fs::{self, File},
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

#[derive(Debug, clap::Args)]
pub struct ProfileArgs {
    #[arg(long)]
    pub challenge: PathBuf,
    #[arg(long)]
    pub claim_context: PathBuf,
    #[arg(long)]
    pub output: PathBuf,
    #[arg(long, default_value_t = 3)]
    pub top_tiles: usize,
    #[arg(long, default_value_t = 1)]
    pub calls_per_tile: usize,
    #[arg(long, default_value_t = 8589934592u64)]
    pub sample_max_bytes: u64,
}
pub(crate) fn reference(bundle: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_owned()
    } else {
        bundle.parent().unwrap().join(path)
    }
}
fn logged(command: &mut Command, directory: &Path, name: &str) -> Result<PathBuf> {
    let stdout = directory.join(format!("{name}.stdout.log"));
    let status = command
        .stdin(Stdio::null())
        .stdout(File::create(&stdout)?)
        .stderr(File::create(directory.join(format!("{name}.stderr.log")))?)
        .status()?;
    ensure!(
        status.success(),
        "{name} failed; see logs in {}",
        directory.display()
    );
    Ok(stdout)
}
pub(crate) fn run_directory(log: &Path, project: &Path) -> Result<PathBuf> {
    let mut paths = std::collections::BTreeSet::new();
    for line in BufReader::new(File::open(log)?).lines() {
        let line = line?;
        if let Some(path) = line.trim().strip_prefix("Run artifacts dir: ") {
            paths.insert(fs::canonicalize(project.join(path))?);
        }
    }
    ensure!(
        paths.len() == 1,
        "Raster did not report exactly one run directory"
    );
    let path = paths.into_iter().next().unwrap();
    ensure!(
        path.starts_with(fs::canonicalize(project.join("target/raster/runs"))?),
        "unexpected Raster run directory"
    );
    Ok(path)
}
pub(crate) fn summarize(
    report: &ReplayProfile,
    length: usize,
    window: usize,
    top_n: usize,
) -> Result<Value> {
    ensure!(
        report.complete && report.failure.is_none(),
        "replay profile is incomplete"
    );
    ensure!(window >= 2, "invalid commitment window");
    ensure!(
        report.version == 5 || report.calls_per_tile == 10,
        "unsupported replay sampling policy"
    );
    if report.version == 5 {
        ensure!(
            report
                .selection
                .as_ref()
                .is_some_and(|s| s["policy"] == "native-runtime"),
            "missing native selection policy"
        );
    }
    if report.version == 5 {
        let selected = report
            .selection
            .as_ref()
            .and_then(|s| s["calls"].as_array())
            .context("missing selected calls")?;
        ensure!(
            selected.len()
                == report
                    .tiles
                    .values()
                    .map(|t| t.replayed.len())
                    .sum::<usize>(),
            "incomplete selected coverage"
        );
    }
    let mut calls = Vec::new();
    let mut recorded = 0u64;
    let mut medians = serde_json::Map::new();
    for (tile_id, tile) in &report.tiles {
        ensure!(
            tile.replayed.len() as u64
                == if report.version == 5 {
                    report
                        .selection
                        .as_ref()
                        .and_then(|s| s["calls"].as_array())
                        .context("missing selected calls")?
                        .iter()
                        .filter(|c| c["tile_id"] == *tile_id)
                        .count() as u64
                } else {
                    tile.calls_recorded.min(report.calls_per_tile as u64)
                },
            "incomplete replay sample for tile {tile_id}"
        );
        recorded = recorded
            .checked_add(tile.calls_recorded)
            .context("call count overflow")?;
        let mut cycles: Vec<_> = tile.replayed.iter().map(|c| c.user_cycles).collect();
        cycles.sort_unstable();
        if cycles.is_empty() {
            continue;
        }
        let p50 = if report.version == 5 && cycles.len() < 2 {
            None
        } else {
            Some(cycles[(cycles.len() - 1) / 2])
        };
        medians.insert(tile_id.clone(), json!({"p50_user_cycles":p50,"calls_recorded":tile.calls_recorded,"calls_replayed":cycles.len()}));
        for call in &tile.replayed {
            ensure!(
                call.exec_index > 0 && call.exec_index <= length as u64,
                "replay index outside trace"
            );
            calls.push(json!({"tile_id":tile_id,"exec_index":call.exec_index,
                "coordinates":call.coordinates,"user_cycles":call.user_cycles,"padded_cycles":call.padded_cycles,
                "segments":call.segments,"executor_duration_ns":call.exec_ns,"native_duration_ns":null,
                "p50_user_cycles":p50}));
        }
    }
    calls.sort_by(|a, b| {
        b["user_cycles"]
            .as_u64()
            .cmp(&a["user_cycles"].as_u64())
            .then(a["exec_index"].as_u64().cmp(&b["exec_index"].as_u64()))
    });
    let mut seen = std::collections::BTreeSet::new();
    ensure!(
        calls
            .iter()
            .all(|c| seen.insert(c["exec_index"].as_u64().unwrap())),
        "duplicate replay execution index"
    );
    let target = calls.iter().enumerate().find_map(|(index, call)| {
        let h = usize::try_from(call["exec_index"].as_u64()?)
            .ok()?
            .checked_sub(1)?;
        let j = h.checked_add(window - 1)?;
        if j < window || j >= length.checked_sub(window)? {
            return None;
        }
        Some(
            json!({"rank":index+1,"tile_id":call["tile_id"],"exec_index":call["exec_index"],
            "coordinates":call["coordinates"],"user_cycles":call["user_cycles"],
            "window_start":h,"window_end":j,"window_size":window}),
        )
    });
    let reason = match target.as_ref().and_then(|t| t["rank"].as_u64()) {
        Some(1) => "Heaviest measured invocation starts the window.",
        Some(_) => "Higher-ranked invocations cannot start a full interior window; using the highest-ranked eligible invocation.",
        None => "No measured invocation can start a full interior window.",
    };
    Ok(
        json!({"calls_recorded":recorded,"calls_replayed":calls.len(),"calls_per_tile":report.calls_per_tile,
        "top_calls":calls.iter().take(top_n).collect::<Vec<_>>(),"tiles":medians,"target":target,"target_reason":reason}),
    )
}

pub(crate) fn verify_output(stage: &str, baseline_dir: &Path, run: &Path) -> Result<()> {
    let baseline = inference_artifacts::read_checkpoint_from_stage_dir(stage, baseline_dir)?;
    let output: Value = read_json(&run.join("output_manifest.json"))?;
    ensure!(
        output["output"]["commitment"] == baseline.output_commitment
            && file_sha256(&run.join("output.bin"))? == baseline.output_sha256,
        "profiling output differs from baseline"
    );
    Ok(())
}
pub fn run(args: ProfileArgs) -> Result<PathBuf> {
    let challenge_path = fs::canonicalize(args.challenge)?;
    let claim_path = fs::canonicalize(args.claim_context)?;
    let bundle: ChallengeBundle = read_json(&challenge_path)?;
    let claim: ClaimBundle = read_json(&claim_path)?;
    ensure!(
        bundle.version == 1 && claim.version == 1,
        "unsupported bundle version"
    );
    let prepared: PreparedRun = read_json(&reference(
        &claim_path,
        claim
            .prepared_run_path
            .as_ref()
            .context("missing frozen run")?,
    ))?;
    ensure!(prepared.version == 2, "unsupported frozen run version");
    let manifest = prepared
        .run_manifest_path
        .as_ref()
        .context("missing run manifest")?;
    inference_artifacts::verify_file_sha256(
        manifest,
        prepared
            .run_manifest_sha256
            .as_ref()
            .context("missing run hash")?,
        "frozen run",
    )?;
    inference_artifacts::verify_file_sha256(
        &prepared.model_manifest_path,
        &prepared.model_manifest_sha256,
        "frozen model",
    )?;
    let package: ReplayPackage =
        read_json(&reference(&challenge_path, &bundle.replay_package_path))?;
    ensure!(
        package.version == 1 && package.stage == bundle.stage,
        "replay package stage mismatch"
    );
    let baseline =
        inference_artifacts::read_checkpoint_from_stage_dir(&bundle.stage, &package.stage_dir)?;
    let trace: CheckpointTrace = read_json(&reference(&claim_path, &claim.checkpoint_trace_path))?;
    ensure!(
        trace
            .checkpoints
            .iter()
            .filter(|c| c.stage == bundle.stage)
            .count()
            == 1
            && trace.checkpoints.iter().any(|c| c == &baseline),
        "challenge differs from claim checkpoint"
    );
    let original_path = fs::canonicalize(reference(&challenge_path, &bundle.raster_commit_path))?;
    let original = CommitmentFile::read_memory(&original_path)?;
    original.validate()?;
    let original_hash = file_sha256(&original_path)?;
    let project = staged_infer::chain_runner::stage_project(manifest, &bundle.stage)?;
    fs::create_dir_all(&args.output)?;
    let output = fs::canonicalize(args.output)?;
    let summary_path = output.join("challenge-profile.json");
    ensure!(
        !summary_path.exists() && !output.join("profile.commit.bin").exists(),
        "profile output already used"
    );
    let input = fs::canonicalize(&package.input_path)?;
    let inputs = fs::canonicalize(&package.input_manifest_path)?;
    let input_hash = file_sha256(&input)?;
    let inputs_hash = file_sha256(&inputs)?;
    let mut verify = Command::new("cargo");
    verify
        .current_dir(&project)
        .args(["raster", "program", "--verify"]);
    logged(&mut verify, &output, "identity")?;
    let cfs = CfsBuilder::new(&Project::new(project.clone())?).build()?;
    use sha2::Digest;
    let cfs_hash = hex::encode(sha2::Sha256::digest(serde_json::to_vec(&cfs)?));
    let lock_path = project.join("Raster.lock");
    let lock_hash = file_sha256(&lock_path)?;
    ensure!(
        args.top_tiles > 0 && args.calls_per_tile > 0 && args.sample_max_bytes > 0,
        "profile limits must be positive"
    );
    let mut durations = serde_json::Map::new();
    let mut selection_path = None;
    let mut capture_run = None;
    let mut native_timings = Value::Null;
    for phase in ["native-timing", "selective-capture"] {
        profile_progress(phase)?;
        let started = std::time::Instant::now();
        let pass = output.join(phase);
        fs::create_dir_all(&pass)?;
        let commit = pass.join("profile.commit.bin");
        let mut native = Command::new("cargo");
        native
            .current_dir(&project)
            .args(["raster", "run", "--fraud-proof-window-size"])
            .arg(original.window_size().to_string())
            .arg("--commit")
            .arg(&commit)
            .arg("--input")
            .arg(&input)
            .arg("--input-manifest")
            .arg(&inputs);
        if phase == "native-timing" {
            native
                .arg("--profile-native-top")
                .arg("--profile-top-tiles")
                .arg(args.top_tiles.to_string())
                .arg("--profile-calls-per-tile")
                .arg(args.calls_per_tile.to_string());
        } else {
            native
                .arg("--capture-profile-calls")
                .arg(
                    selection_path
                        .as_ref()
                        .context("missing native selection")?,
                )
                .arg("--sample-max-bytes")
                .arg(args.sample_max_bytes.to_string());
        }
        write_json(
            &pass.join("native-invocation.json"),
            &json!({"version":1,"project_root":fs::canonicalize(&project)?,"input":input,"input_manifest":inputs}),
        )?;
        let log = logged(&mut native, &pass, "native")?;
        let run_dir = run_directory(&log, &project)?;
        let run: Value = read_json(&run_dir.join("run.json"))?;
        ensure!(
            run["version"] == 2 && run["cfs_digest"] == cfs_hash,
            "selective run identity mismatch"
        );
        ensure!(
            run["input"].as_str() == input.to_str()
                && run["input_manifest"].as_str() == inputs.to_str(),
            "run inputs mismatch"
        );
        let rerun = CommitmentFile::read_memory(&commit)?;
        rerun.validate()?;
        ensure!(
            postcard::to_allocvec(&original)? == postcard::to_allocvec(&rerun)?,
            "profiling rerun commitment differs from original"
        );
        verify_output(&bundle.stage, &package.stage_dir, &run_dir)?;
        if phase == "native-timing" {
            let timing_report: Value = read_json(&run_dir.join("native-timings.json"))?;
            ensure!(timing_report["version"] == 1 && timing_report["complete"] == true,
                "incomplete native timing history");
            native_timings = timing_report["calls"].clone();
            ensure!(native_timings.is_array(), "missing native timing history");
            let path = run_dir.join("native-selection.json");
            fs::copy(&path, output.join("native-selection.json"))?;
            selection_path = Some(path);
        } else {
            capture_run = Some(run_dir);
        }
        durations.insert(
            phase.into(),
            json!(started.elapsed().as_secs_f64() * 1000.0),
        );
    }
    let run_dir = capture_run.context("missing capture run")?;
    profile_progress("guest-measurement")?;
    let started = std::time::Instant::now();
    let mut replay = Command::new("cargo");
    replay
        .current_dir(&project)
        .args(["raster", "replay", "profile"])
        .arg(&run_dir);
    logged(&mut replay, &output, "replay")?;
    durations.insert(
        "guest-measurement".into(),
        json!(started.elapsed().as_secs_f64() * 1000.0),
    );
    let report = ReplayProfile::from_json(&fs::read(run_dir.join("replay-profile.json"))?)?;
    let run: Value = read_json(&run_dir.join("run.json"))?;
    ensure!(
        report.version == 5 && run["run_id"].as_str() == Some(report.run_id.as_str()),
        "selective replay report mismatch"
    );
    let capture: raster_runtime::selective_profile::Manifest =
        read_json(&run_dir.join("selected-capture.json"))?;
    capture.validate()?;
    ensure!(
        capture.complete
            && capture.step_count == original.len() as u64
            && capture.calls.len()
                == report
                    .tiles
                    .values()
                    .map(|t| t.replayed.len())
                    .sum::<usize>(),
        "invalid capture coverage"
    );
    for (id, tile) in &report.tiles {
        if !tile.replayed.is_empty() {
            ensure!(
                tile.image_id.is_some() && tile.image_id == tile.lock_image_id,
                "guest identity mismatch"
            );
        }
        for call in &tile.replayed {
            ensure!(
                capture.calls.iter().any(|c| c.tile_id == *id
                    && c.exec_index == call.exec_index
                    && c.coordinates == call.coordinates),
                "guest invocation differs from capture"
            );
        }
    }
    let mut summary = summarize(&report, original.len(), original.window_size(), 3)?;
    for call in summary["top_calls"].as_array_mut().unwrap() {
        let native = capture
            .calls
            .iter()
            .find(|c| Some(c.exec_index) == call["exec_index"].as_u64())
            .context("missing selected native measurement")?;
        call["native_duration_ns"] = json!(native.native_duration_ns);
    }
    ensure!(
        file_sha256(&input)? == input_hash
            && file_sha256(&inputs)? == inputs_hash
            && file_sha256(&lock_path)? == lock_hash
            && file_sha256(&original_path)? == original_hash,
        "frozen artifacts changed during profiling"
    );
    fs::copy(
        run_dir.join("replay-profile.json"),
        output.join("replay-profile.json"),
    )?;
    fs::copy(
        run_dir.join("selected-capture.json"),
        output.join("selected-capture.json"),
    )?;
    summary.as_object_mut().unwrap().extend(json!({"version":2,"kind":"challenge-profile","complete":true,"stage":bundle.stage,
        "original_commitment_sha256":original_hash,"commitment_length":original.len(),"window_size":original.window_size(),
        "input_sha256":input_hash,"input_manifest_sha256":inputs_hash,"cfs_digest":cfs_hash,"lock_sha256":lock_hash,
        "run_dir":run_dir,"commitment_matches":true,"output_matches":true,"mapping_verified":true,
        "selection_policy":"native-runtime","native_candidates":capture.calls,"native_timings":native_timings,"top_tiles":args.top_tiles,"selected_tile_count":report.tiles.values().filter(|t|!t.replayed.is_empty()).count(),
        "phase_durations_ms":durations,"sample_payload_bytes":capture.payload_bytes,"sample_max_bytes":args.sample_max_bytes}).as_object().unwrap().clone());
    write_json(&summary_path, &summary)?;
    println!("Challenge profile report: {}", summary_path.display());
    Ok(summary_path)
}

fn profile_progress(phase: &str) -> Result<()> {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_millis();
    println!(
        "IRIS_PROFILE_PROGRESS {}",
        json!({"version":1,"phase":phase,"started_at_ms":now})
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use raster_analysis::replay_profile::{ReplayedCall, TileReplay};
    fn report() -> ReplayProfile {
        let mut p = ReplayProfile::new("test", ["recursive".into()], 10);
        p.complete = true;
        p.tiles.insert(
            "recursive".into(),
            TileReplay {
                calls_recorded: 4,
                replayed: [(95, 900), (11, 800), (12, 700), (13, 600)]
                    .into_iter()
                    .map(|(exec_index, user_cycles)| ReplayedCall {
                        exec_index,
                        user_cycles,
                        coordinates: Default::default(),
                        input_bytes: 1,
                        output_bytes: 1,
                        byte_rank: 1,
                        segments: 1,
                        padded_cycles: 1024,
                        exec_ns: 10,
                    })
                    .collect(),
                ..Default::default()
            },
        );
        p
    }
    #[test]
    fn ranked_invocations_and_fallback() {
        let s = summarize(&report(), 100, 8, 3).unwrap();
        assert_eq!(s["top_calls"].as_array().unwrap().len(), 3);
        assert_eq!(s["top_calls"][0]["exec_index"], 95);
        assert_eq!(s["top_calls"][0]["p50_user_cycles"], 700);
        assert_eq!(s["target"]["rank"], 2);
        assert_eq!(s["target"]["window_start"], 10);
        assert_eq!(s["target"]["window_end"], 17);
    }
    #[test]
    fn no_eligible_and_partial() {
        let mut p = report();
        assert!(summarize(&p, 100, 64, 3).unwrap()["target"].is_null());
        p.complete = false;
        assert!(summarize(&p, 100, 8, 3).is_err());
    }
    #[test]
    fn report_version_and_sample_coverage_are_checked() {
        let p = report();
        assert_eq!(ReplayProfile::from_json(&p.to_json().unwrap()).unwrap(), p);
        let mut value = serde_json::to_value(&p).unwrap();
        value["version"] = 3.into();
        assert!(ReplayProfile::from_json(&serde_json::to_vec(&value).unwrap()).is_err());
        let mut partial = p.clone();
        partial.tiles.get_mut("recursive").unwrap().replayed.pop();
        assert!(summarize(&partial, 100, 8, 3).is_err());
    }
    #[test]
    fn ties_use_execution_order_and_indices_are_one_based() {
        let mut p = report();
        let calls = &mut p.tiles.get_mut("recursive").unwrap().replayed;
        for call in calls.iter_mut() {
            call.user_cycles = 10;
        }
        assert_eq!(
            summarize(&p, 100, 8, 3).unwrap()["top_calls"][0]["exec_index"],
            11
        );
        p.tiles.get_mut("recursive").unwrap().replayed[0].exec_index = 0;
        assert!(summarize(&p, 100, 8, 3).is_err());
    }
    #[test]
    fn native_selection_omits_single_sample_median_and_checks_coverage() {
        let mut p = report();
        p.version = 5;
        p.calls_per_tile = 1;
        p.tiles.get_mut("recursive").unwrap().replayed.truncate(1);
        p.selection = Some(json!({"policy":"native-runtime","calls":[{"tile_id":"recursive"}]}));
        assert_eq!(ReplayProfile::from_json(&p.to_json().unwrap()).unwrap(), p);
        assert!(summarize(&p, 100, 8, 3).unwrap()["top_calls"][0]["p50_user_cycles"].is_null());
        p.selection.as_mut().unwrap()["calls"]
            .as_array_mut()
            .unwrap()
            .push(json!({"tile_id":"missing"}));
        assert!(summarize(&p, 100, 8, 3).is_err());
    }
}
