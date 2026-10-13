use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use anyhow::{bail, ensure, Context, Result};
use inference_artifacts::replay_profile::{
    NativeInvocation, NativeTileProfile, ProfilePass, ReplayProfileStatus, ReplayTileMeasurement,
    ReplayTransitionOverhead, StageReplayProfile, NATIVE_RANKING_JSON, REPLAY_PROFILE_JSON,
};
use inference_artifacts::{
    file_sha256, read_checkpoint_from_stage_dir, read_json, verify_file_sha256, write_json,
    Checkpoint, PreparedRun,
};
use raster_analysis::replay_profile::ReplayProfile;
use raster_runtime::profiling::{ExecutionProfile, ProfileRecord, TileProfileRecord};
use staged_infer::chain_runner::StagedExecutionBackend;
use staged_infer::{
    CheckpointedInferenceConfig, CheckpointedInferenceExecutor, ParityPolicy, StagePrefixRun,
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
    ensure_raster_cli(&options.base_dir)?;
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
    // Exact stage validation has happened before this stage name becomes a path.
    let report_dir = prefix.chain_dir.join("replay-profile").join(&prefix.stage);
    fs::create_dir_all(&report_dir)?;
    let mut report = new_report(prefix, prepared.prepared_run, options.top_n, &report_dir)?;
    execute_profile(&mut report, &report_dir, run_profile_command)
}

fn new_report(
    prefix: StagePrefixRun,
    prepared_run: PreparedRun,
    top_n: usize,
    report_dir: &Path,
) -> Result<StageReplayProfile> {
    Ok(StageReplayProfile {
        version: 2,
        kind: "stage-replay-profile".into(),
        status: ReplayProfileStatus::Running,
        error: None,
        prepared_run,
        stage: prefix.stage.clone(),
        project_dir: prefix.project_dir,
        baseline: read_checkpoint_from_stage_dir(&prefix.stage, &prefix.stage_dir)?,
        stage_dir: prefix.stage_dir,
        input_sha256: file_sha256(&prefix.input_path)?,
        input_manifest_sha256: file_sha256(&prefix.input_manifest_path)?,
        native: profile_pass(
            "native",
            report_dir,
            &prefix.input_path,
            &prefix.input_manifest_path,
        ),
        replay: profile_pass(
            "replay",
            report_dir,
            &prefix.input_path,
            &prefix.input_manifest_path,
        ),
        input_path: prefix.input_path,
        input_manifest_path: prefix.input_manifest_path,
        top_n,
        selection_rule:
            "maximum native elapsed time per tile ID descending; tile ID ascending on ties".into(),
        selected_tile_ids: Vec::new(),
        native_ranking_path: None,
        native_tiles: Vec::new(),
        replay_complete: None,
        replay_invocation_limit: None,
        replay_tiles: Vec::new(),
    })
}

fn ensure_raster_cli(base_dir: &Path) -> Result<()> {
    let help = Command::new("cargo")
        .current_dir(base_dir)
        .args(["raster", "run", "--help"])
        .output()
        .context(
            "failed to inspect Raster CLI; install raster-cli from the matching Raster checkout",
        )?;
    let text = String::from_utf8_lossy(&help.stdout);
    ensure!(help.status.success() && text.contains("--profile") && text.contains("replay"),
        "Raster CLI lacks replay profiling; install raster-cli from the same Raster checkout as this repo's dependencies (cargo install --path ../raster/crates/raster-cli --locked --force)");
    Ok(())
}

fn profile_pass(mode: &str, dir: &Path, input: &Path, manifest: &Path) -> ProfilePass {
    ProfilePass {
        command: vec![
            "cargo".into(),
            "raster".into(),
            "run".into(),
            "--profile".into(),
            mode.into(),
            "--input".into(),
            input.to_string_lossy().into_owned(),
            "--input-manifest".into(),
            manifest.to_string_lossy().into_owned(),
        ],
        stdout_path: dir.join(format!("{mode}.stdout.log")),
        stderr_path: dir.join(format!("{mode}.stderr.log")),
        run_dir: None,
        profile_path: None,
        command_succeeded: None,
        output_matches_baseline: None,
    }
}

// Inject only process execution: tests still exercise report loading, selection,
// parity, preservation and error handling through the production workflow.
fn execute_profile(
    report: &mut StageReplayProfile,
    dir: &Path,
    mut run: impl FnMut(&Path, &mut ProfilePass, ProfileOutput) -> Result<()>,
) -> Result<PathBuf> {
    let path = dir.join(REPLAY_PROFILE_JSON);
    write_json(&path, report)?;
    println!("Replay profile report: {}", path.display());
    let outcome = profile_stage(report, dir, &path, &mut run);
    if let Err(error) = &outcome {
        report.status = ReplayProfileStatus::Failed;
        report.error = Some(format!("{error:#}"));
    }
    write_json(&path, report)?;
    print_report(report, &path);
    outcome.with_context(|| {
        format!(
            "replay profiling failed; report and logs: {}",
            path.display()
        )
    })?;
    Ok(path)
}

fn profile_stage(
    report: &mut StageReplayProfile,
    dir: &Path,
    report_path: &Path,
    run: &mut impl FnMut(&Path, &mut ProfilePass, ProfileOutput) -> Result<()>,
) -> Result<()> {
    verify_inputs(report)?;
    println!("Profiling native Raster stage {}", report.stage);
    run(
        &report.project_dir,
        &mut report.native,
        ProfileOutput::LogOnly,
    )?;
    let native_path = preserve_profile(
        &mut report.native,
        "profile.json",
        &dir.join("native-profile.json"),
    )?;
    let native: ExecutionProfile = read_json(&native_path)?;
    ensure!(
        native.version == 3,
        "unsupported native profile version {} (expected 3)",
        native.version
    );
    report.native_tiles = rank_native_tiles(&native)?;
    let ranking_path = dir.join(NATIVE_RANKING_JSON);
    write_json(&ranking_path, &report.native_tiles)?;
    report.native_ranking_path = Some(ranking_path);
    // Large stages can have millions of records. Release discovery data before
    // the second Raster process records its own full authenticated execution.
    drop(native);
    verify_inputs(report)?;
    check_pass_output(&report.baseline, &mut report.native)?;
    ensure!(
        report.native.command_succeeded == Some(true),
        "native profiling command failed; see {}",
        report.native.stderr_path.display()
    );
    report.selected_tile_ids = report
        .native_tiles
        .iter()
        .take(report.top_n)
        .map(|t| t.tile_id.clone())
        .collect();
    if report.selected_tile_ids.is_empty() {
        report.status = ReplayProfileStatus::NoTiles;
        return Ok(());
    }
    for tile in &report.selected_tile_ids {
        report
            .replay
            .command
            .extend(["--tile".into(), tile.clone()]);
    }
    // Save the native result and exact selection before the expensive replay pass.
    write_json(report_path, report)?;
    println!("Selected tile IDs: {}", report.selected_tile_ids.join(", "));
    verify_inputs(report)?;
    run(
        &report.project_dir,
        &mut report.replay,
        ProfileOutput::Stream,
    )?;
    let replay_path = preserve_profile(
        &mut report.replay,
        "replay-profile.json",
        &dir.join("raster-replay-profile.json"),
    )?;
    let replay: ReplayProfile = read_json(&replay_path)?;
    replay
        .validate_format()
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    report.replay_complete = Some(replay.complete);
    report.replay_invocation_limit = replay.invocation_limit;
    report.replay_tiles = replay
        .tiles
        .iter()
        .map(|(id, tile)| ReplayTileMeasurement {
            tile_id: id.clone(),
            recorded_invocations: tile.invocations,
            profiled_invocations: tile.profiled_invocations,
            image_id: tile.image_id.clone(),
            max_guest_cycles: tile.max_guest_cycles,
            total_guest_cycles: tile.total_guest_cycles,
            heaviest_exec_index: tile.heaviest_invocation.as_ref().map(|i| i.exec_index),
            heaviest_coordinates: tile
                .heaviest_invocation
                .as_ref()
                .map(|i| i.coordinates.0.clone()),
            transition_overhead: tile.transition_overhead.as_ref().map(|overhead| {
                ReplayTransitionOverhead {
                    guest_cycles: overhead.guest_cycles,
                    sampled_exec_index: overhead.invocation.exec_index,
                    sampled_coordinates: overhead.invocation.coordinates.0.clone(),
                    transition_image_id: overhead.transition_image_id.clone(),
                    context: overhead.context.clone(),
                    context_version: overhead.context_version,
                }
            }),
        })
        .collect();
    report.replay_tiles.sort_by(|a, b| {
        b.max_guest_cycles
            .cmp(&a.max_guest_cycles)
            .then_with(|| a.tile_id.cmp(&b.tile_id))
    });
    verify_inputs(report)?;
    check_pass_output(&report.baseline, &mut report.replay)?;
    ensure!(
        replay.tiles.keys().collect::<BTreeSet<_>>()
            == report.selected_tile_ids.iter().collect::<BTreeSet<_>>(),
        "replay report tile IDs differ from selected tile IDs"
    );
    if let Some(failure) = &replay.failure {
        bail!(
            "partial replay profile: {} at {:?}: {} (phase: {:?})",
            failure.tile,
            failure.invocation.coordinates.0,
            failure.message,
            failure.phase,
        );
    }
    ensure!(
        report.replay.command_succeeded == Some(true),
        "replay profiling command failed; see {}",
        report.replay.stderr_path.display()
    );
    ensure!(
        replay.complete,
        "Raster returned an incomplete replay profile"
    );
    for tile in &report.replay_tiles {
        let native = report
            .native_tiles
            .iter()
            .find(|n| n.tile_id == tile.tile_id)
            .unwrap();
        ensure!(
            tile.recorded_invocations == native.invocations,
            "tile {} invocation count changed between native and replay runs",
            tile.tile_id
        );
        let expected = replay
            .invocation_limit
            .map_or(tile.recorded_invocations, |limit| {
                limit.min(tile.recorded_invocations)
            });
        ensure!(
            tile.profiled_invocations == expected && tile.max_guest_cycles.is_some(),
            "incomplete measurements for tile {}",
            tile.tile_id
        );
    }
    report.status = ReplayProfileStatus::Complete;
    Ok(())
}

fn verify_inputs(report: &StageReplayProfile) -> Result<()> {
    verify_file_sha256(
        &report.input_path,
        &report.input_sha256,
        "selected stage inputs",
    )?;
    verify_file_sha256(
        &report.input_manifest_path,
        &report.input_manifest_sha256,
        "selected stage input manifest",
    )
}

fn preserve_profile(pass: &mut ProfilePass, filename: &str, destination: &Path) -> Result<PathBuf> {
    let source = pass
        .run_dir
        .as_ref()
        .context("Raster did not report its run artifact directory")?
        .join(filename);
    fs::copy(&source, destination).map_err(|error| {
        let context = if pass.command_succeeded == Some(false)
            && error.kind() == std::io::ErrorKind::NotFound
            && !source.exists()
        {
            // A failed native run never starts replay or writes a replay report.
            // Raster currently prints child-process stderr inside its stdout,
            // so both logs are needed to reveal the failure that came first.
            format!(
                "Raster profiling command failed before producing {filename}; no measurements were loaded.\nRecent stdout ({}):\n{}\nRecent stderr ({}):\n{}",
                pass.stdout_path.display(), diagnostic_log_tail(&pass.stdout_path),
                pass.stderr_path.display(), diagnostic_log_tail(&pass.stderr_path),
            )
        } else {
            format!(
                "missing or unreadable Raster profile {} (command succeeded: {:?}); see {} and {}",
                source.display(), pass.command_succeeded,
                pass.stdout_path.display(), pass.stderr_path.display(),
            )
        };
        anyhow::Error::new(error).context(context)
    })?;
    pass.profile_path = Some(destination.to_path_buf());
    Ok(destination.to_path_buf())
}

fn diagnostic_log_tail(path: &Path) -> String {
    let result = (|| -> std::io::Result<String> {
        let mut file = File::open(path)?;
        let start = file.metadata()?.len().saturating_sub(4096);
        file.seek(SeekFrom::Start(start))?;
        let mut bytes = Vec::new();
        file.take(4096).read_to_end(&mut bytes)?;
        Ok(String::from_utf8_lossy(&bytes).trim().to_string())
    })();
    result.unwrap_or_else(|error| format!("Could not read log: {error}"))
}

fn check_pass_output(baseline: &Checkpoint, pass: &mut ProfilePass) -> Result<()> {
    let dir = pass
        .run_dir
        .as_ref()
        .context("missing Raster run directory")?;
    let manifest: serde_json::Value = read_json(&dir.join("output_manifest.json"))?;
    let commitment = manifest["output"]["commitment"]
        .as_str()
        .context("Raster output manifest has no output.commitment")?;
    let sha256 = file_sha256(&dir.join("output.bin"))?;
    let matched = commitment == baseline.output_commitment && sha256 == baseline.output_sha256;
    pass.output_matches_baseline = Some(matched);
    ensure!(matched, "native/Raster parity mismatch for {}: output commitment expected {}, got {}; output SHA-256 expected {}, got {}", baseline.stage, baseline.output_commitment, commitment, baseline.output_sha256, sha256);
    Ok(())
}

fn native_invocation(tile: &TileProfileRecord) -> NativeInvocation {
    NativeInvocation {
        invocation_index: tile.invocation_index,
        coordinates: tile.coordinates.0.clone(),
        total_duration_ns: tile.total_duration_ns,
        user_duration_ns: tile.user_duration_ns,
        raster_overhead_ns: tile.raster_overhead_ns,
        input_bytes: tile.input_bytes,
        output_bytes: tile.output_bytes,
    }
}

fn rank_native_tiles(profile: &ExecutionProfile) -> Result<Vec<NativeTileProfile>> {
    let mut tiles = BTreeMap::<String, NativeTileProfile>::new();
    for record in &profile.records {
        let ProfileRecord::Tile(tile) = record else {
            continue;
        };
        let aggregate = tiles
            .entry(tile.tile_id.clone())
            .or_insert_with(|| NativeTileProfile {
                tile_id: tile.tile_id.clone(),
                invocations: 0,
                total_duration_ns: 0,
                total_user_duration_ns: 0,
                total_raster_overhead_ns: 0,
                max_input_bytes: 0,
                max_output_bytes: 0,
                slowest_invocation: native_invocation(tile),
            });
        aggregate.invocations += 1;
        aggregate.total_duration_ns = aggregate
            .total_duration_ns
            .checked_add(tile.total_duration_ns)
            .context("native duration total overflow")?;
        aggregate.total_user_duration_ns = aggregate
            .total_user_duration_ns
            .checked_add(tile.user_duration_ns)
            .context("native body duration total overflow")?;
        aggregate.total_raster_overhead_ns = aggregate
            .total_raster_overhead_ns
            .checked_add(tile.raster_overhead_ns)
            .context("native overhead duration total overflow")?;
        aggregate.max_input_bytes = aggregate.max_input_bytes.max(tile.input_bytes);
        aggregate.max_output_bytes = aggregate.max_output_bytes.max(tile.output_bytes);
        if tile.total_duration_ns > aggregate.slowest_invocation.total_duration_ns {
            aggregate.slowest_invocation = native_invocation(tile);
        }
    }
    let mut tiles: Vec<_> = tiles.into_values().collect();
    tiles.sort_by(|a, b| {
        b.slowest_invocation
            .total_duration_ns
            .cmp(&a.slowest_invocation.total_duration_ns)
            .then_with(|| a.tile_id.cmp(&b.tile_id))
    });
    Ok(tiles)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProfileOutput {
    LogOnly,
    Stream,
}

fn run_profile_command(
    project: &Path,
    pass: &mut ProfilePass,
    output: ProfileOutput,
) -> Result<()> {
    run_profile_command_with_output(project, pass, output, &mut std::io::stdout().lock())
}

fn run_profile_command_with_output(
    project: &Path,
    pass: &mut ProfilePass,
    output: ProfileOutput,
    terminal: &mut impl Write,
) -> Result<()> {
    let stderr = File::create(&pass.stderr_path)?;
    let mut stdout_log = File::create(&pass.stdout_path)?;
    let mut child = Command::new(&pass.command[0])
        .args(&pass.command[1..])
        .current_dir(project)
        .stdout(Stdio::piped())
        .stderr(Stdio::from(stderr))
        .spawn()
        .context("failed to start Raster profiling command")?;
    let stdout = child.stdout.take().context("missing Raster stdout pipe")?;
    // The run directory is printed in Raster's header, before program output.
    // Read only that header so a program's own output cannot redirect artifacts.
    let mut in_header = true;
    let result = (|| -> Result<()> {
        for line in BufReader::new(stdout).lines() {
            let line = line?;
            writeln!(stdout_log, "{line}")?;
            if in_header {
                observe_run_header(&line, project, &mut pass.run_dir)?;
                if line.starts_with("Mode:") {
                    in_header = false;
                }
            }
            // Keep discovery timings available in the native log without
            // presenting them as replay measurements in the terminal.
            if output == ProfileOutput::Stream {
                writeln!(terminal, "{line}")?;
                terminal.flush()?;
            }
        }
        Ok(())
    })();
    if result.is_err() {
        let _ = child.kill();
    }
    let status = child.wait()?;
    pass.command_succeeded = Some(status.success());
    result?;
    ensure!(
        pass.run_dir.is_some(),
        "Raster did not print a run artifact directory; see {}",
        pass.stderr_path.display()
    );
    Ok(())
}

fn observe_run_header(line: &str, project: &Path, run_dir: &mut Option<PathBuf>) -> Result<()> {
    if let Some(path) = line.strip_prefix("  Run artifacts dir: ") {
        ensure!(
            !path.trim().is_empty(),
            "Raster reported an empty run directory"
        );
        ensure!(
            run_dir.is_none(),
            "Raster reported multiple run artifact directories"
        );
        let path = Path::new(path);
        *run_dir = Some(if path.is_absolute() {
            path.to_path_buf()
        } else {
            project.join(path)
        });
    }
    Ok(())
}

fn print_report(report: &StageReplayProfile, path: &Path) {
    println!("\n{}", report_text(report, path));
}

fn report_text(report: &StageReplayProfile, path: &Path) -> String {
    let status = match report.status {
        ReplayProfileStatus::Running => "running",
        ReplayProfileStatus::Complete => "complete",
        ReplayProfileStatus::NoTiles => "no tiles",
        ReplayProfileStatus::Failed => "failed",
    };
    let mut lines = vec![
        format!("Replay profile: {} ({status})", report.stage),
        "All measurements below are zkVM guest cycles. Replay totals exclude the separately sampled transition overhead.".into(),
        "Guest execution only; no proofs or budget verdict. Native selection timings are saved separately in native-ranking.json and native-profile.json.".into(),
    ];
    if let Some(limit) = report.replay_invocation_limit {
        lines.push(format!("Raster measures at most the first {limit} invocations per tile; later invocations may be heavier."));
    }
    for tile in &report.replay_tiles {
        let maximum = tile
            .max_guest_cycles
            .map_or_else(|| "unavailable".into(), |v| v.to_string());
        let coordinate = tile
            .heaviest_coordinates
            .as_ref()
            .map_or_else(|| "unavailable".into(), |v| format!("{v:?}"));
        let total = if tile.profiled_invocations == 0 {
            "not measured".into()
        } else {
            format!("{} guest cycles", tile.total_guest_cycles)
        };
        lines.push(format!(
            "  {}: heaviest replay {maximum} guest cycles at {coordinate}; replay total {total} across {} measured invocations ({} recorded)",
            tile.tile_id,
            tile.profiled_invocations,
            tile.recorded_invocations,
        ));
        lines.push(match &tile.transition_overhead {
            Some(overhead) => format!(
                "    Sample transition overhead (one invocation): {} guest cycles at {:?} (representative continuation)",
                overhead.guest_cycles, overhead.sampled_coordinates,
            ),
            None => "    Sample transition overhead (one invocation): not measured".into(),
        });
    }
    if matches!(report.status, ReplayProfileStatus::NoTiles) {
        lines.push("No tile invocations recorded; replay profiling was skipped.".into());
    }
    if let Some(error) = &report.error {
        lines.push(format!("Incomplete: {error}"));
    }
    lines.push(format!("Report: {}", path.display()));
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;
    use raster_analysis::replay_profile::{
        InvocationLocation, ProfileFailurePhase, ReplayProfileFailure, TransitionOverhead,
    };
    use serde_json::json;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn record(id: &str, index: u64, duration: u64) -> ProfileRecord {
        ProfileRecord::Tile(
            serde_json::from_value(json!({
                "tile_id": id, "invocation_index": index, "coordinates": [index as u32],
                "depth": 1, "total_duration_ns": duration, "user_duration_ns": duration / 2,
                "raster_overhead_ns": duration - duration / 2,
                "input_bytes": index * 10, "output_bytes": index * 2
            }))
            .unwrap(),
        )
    }

    struct Fixture {
        root: PathBuf,
        dir: PathBuf,
        report: StageReplayProfile,
    }

    impl Fixture {
        fn new() -> Self {
            static NEXT: AtomicUsize = AtomicUsize::new(0);
            let root = std::env::temp_dir().join(format!(
                "replay-profile-test-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            let model = crate::test_support::model_fixture(&root, "model");
            let run = crate::test_support::run_spec(&root, "model");
            assert!(model.exists());
            let prepared = crate::claim::prepare_claim_run(&crate::claim::ClaimBuildOptions {
                base_dir: root.clone(),
                run_spec_path: run,
                current_exe: PathBuf::from("unused"),
                staged_backend: StagedExecutionBackend::InProcess,
            })
            .unwrap();
            let stage_dir = root.join("chain/prompt_merge_b1");
            fs::create_dir_all(&stage_dir).unwrap();
            fs::write(stage_dir.join("input.json"), b"{}").unwrap();
            fs::write(stage_dir.join("input_manifest.json"), b"{}").unwrap();
            fs::write(stage_dir.join("output.bin"), b"baseline").unwrap();
            write_json(
                &stage_dir.join("output_manifest.json"),
                &json!({"output": {"commitment": "expected-root"}}),
            )
            .unwrap();
            let prefix = StagePrefixRun {
                chain_dir: root.join("chain"),
                stage: "prompt_merge_b1".into(),
                project_dir: root.join("project with spaces"),
                input_path: stage_dir.join("input.json"),
                input_manifest_path: stage_dir.join("input_manifest.json"),
                output_path: stage_dir.join("output.bin"),
                output_index_path: stage_dir.join("output.rindex"),
                output_manifest_path: stage_dir.join("output_manifest.json"),
                stage_dir,
            };
            let dir = root.join("report");
            fs::create_dir_all(&dir).unwrap();
            let report = new_report(prefix, prepared.prepared_run, 3, &dir).unwrap();
            Self { root, dir, report }
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.root);
        }
    }

    fn fake_pass(pass: &mut ProfilePass, source: &Path, baseline: &Path) {
        fs::create_dir_all(source).unwrap();
        for name in ["output.bin", "output_manifest.json"] {
            fs::copy(baseline.join(name), source.join(name)).unwrap();
        }
        fs::write(&pass.stdout_path, "captured stdout").unwrap();
        fs::write(&pass.stderr_path, "captured stderr").unwrap();
        pass.run_dir = Some(source.to_path_buf());
        pass.command_succeeded = Some(true);
    }

    fn replay(invocations: u64, complete: bool) -> ReplayProfile {
        let mut profile = ReplayProfile::new("replay-run".into(), ["heavy".into()]);
        profile.tiles.get_mut("heavy").unwrap().invocations = invocations;
        let measured = if complete { invocations.min(128) } else { 1 };
        for index in 0..measured {
            let location: InvocationLocation =
                serde_json::from_value(json!({"exec_index": index, "coordinates": [index as u32]}))
                    .unwrap();
            profile.record("heavy", location, index + 100).unwrap();
        }
        if measured > 0 {
            profile.tiles.get_mut("heavy").unwrap().transition_overhead =
                Some(TransitionOverhead {
                    guest_cycles: 1234,
                    invocation: serde_json::from_value(
                        json!({"exec_index": 0, "coordinates": [0]}),
                    )
                    .unwrap(),
                    transition_image_id: "transition-image".into(),
                    context: "representative-continuation".into(),
                    context_version: 1,
                });
        }
        profile.complete = complete;
        if !complete {
            profile.failure = Some(ReplayProfileFailure {
                phase: ProfileFailurePhase::Replay,
                tile: "heavy".into(),
                invocation: serde_json::from_value(json!({"exec_index": 1, "coordinates": [1]}))
                    .unwrap(),
                message: "guest output mismatch".into(),
            });
        }
        profile
    }

    #[test]
    fn ranking_uses_per_invocation_maximum_with_distinct_ids_and_stable_ties() {
        let profile = ExecutionProfile::new(
            vec![
                record("frequent", 1, 90),
                record("frequent", 2, 90),
                record("z_heavy", 3, 100),
                record("a_heavy", 4, 100),
            ],
            None,
            None,
        );
        let ranked = rank_native_tiles(&profile).unwrap();
        assert_eq!(
            ranked
                .iter()
                .map(|t| t.tile_id.as_str())
                .collect::<Vec<_>>(),
            ["a_heavy", "z_heavy", "frequent"]
        );
        assert_eq!(ranked[2].total_duration_ns, 180);
        assert_eq!(ranked[2].invocations, 2);
        assert_eq!(ranked[2].max_input_bytes, 20);
        assert_eq!(ranked[2].slowest_invocation.invocation_index, 1);
        assert_eq!(ranked[0].slowest_invocation.user_duration_ns, 50);
    }

    #[test]
    fn successful_workflow_preserves_inputs_selection_reports_and_capped_coverage() {
        let mut fixture = Fixture::new();
        let mut calls = Vec::new();
        let baseline = fixture.report.stage_dir.clone();
        let root = fixture.root.clone();
        execute_profile(&mut fixture.report, &fixture.dir, |_, pass, output| {
            assert_eq!(
                output,
                if calls.is_empty() {
                    ProfileOutput::LogOnly
                } else {
                    ProfileOutput::Stream
                }
            );
            calls.push(pass.command.clone());
            let mode = &pass.command[4];
            let source = root.join(mode);
            fake_pass(pass, &source, &baseline);
            if calls.len() == 1 {
                write_json(
                    &source.join("profile.json"),
                    &ExecutionProfile::new(
                        (0..150).map(|i| record("heavy", i, i + 1)).collect(),
                        None,
                        None,
                    ),
                )?;
            } else {
                write_json(&source.join("replay-profile.json"), &replay(150, true))?;
            }
            Ok(())
        })
        .unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(&calls[0][5..9], &calls[1][5..9]);
        assert_eq!(&calls[1][9..], ["--tile", "heavy"]);
        assert!(matches!(
            fixture.report.status,
            ReplayProfileStatus::Complete
        ));
        assert_eq!(fixture.report.selected_tile_ids, ["heavy"]);
        assert_eq!(fixture.report.native.output_matches_baseline, Some(true));
        assert_eq!(fixture.report.replay.output_matches_baseline, Some(true));
        assert_eq!(fixture.report.replay_tiles[0].profiled_invocations, 128);
        assert_eq!(fixture.report.replay_tiles[0].recorded_invocations, 150);
        assert_eq!(fixture.report.replay_tiles[0].max_guest_cycles, Some(227));
        assert!(fixture.dir.join("native-profile.json").is_file());
        assert!(fixture.dir.join("raster-replay-profile.json").is_file());
        let saved: StageReplayProfile = read_json(&fixture.dir.join(REPLAY_PROFILE_JSON)).unwrap();
        assert_eq!(saved.version, 2);
        assert!(saved.native_tiles.is_empty());
        let json: serde_json::Value = read_json(&fixture.dir.join(REPLAY_PROFILE_JSON)).unwrap();
        assert!(json.get("native_tiles").is_none());
        assert!(!json.to_string().contains("_ns"));
        let ranked: Vec<NativeTileProfile> =
            read_json(saved.native_ranking_path.as_ref().unwrap()).unwrap();
        assert_eq!(ranked.len(), 1);
        assert_eq!(ranked[0].total_duration_ns, 11325);
        assert_eq!(ranked[0].slowest_invocation.total_duration_ns, 150);
        assert_eq!(saved.replay_invocation_limit, Some(128));
        let measured = &saved.replay_tiles[0];
        assert_eq!(measured.max_guest_cycles, Some(227));
        assert_eq!(measured.total_guest_cycles, 20928); // sum(100..=227), excluding overhead
        let overhead = measured.transition_overhead.as_ref().unwrap();
        assert_eq!(overhead.guest_cycles, 1234);
        assert_eq!(overhead.sampled_exec_index, 0);
        assert_eq!(overhead.sampled_coordinates, [0]);
        assert_eq!(overhead.transition_image_id, "transition-image");
        assert_eq!(overhead.context, "representative-continuation");
        assert_eq!(overhead.context_version, 1);
        let text = report_text(&saved, Path::new("report.json"));
        assert!(text.contains("heaviest replay 227 guest cycles at [127]"));
        assert!(text.contains("replay total 20928 guest cycles across 128 measured invocations"));
        assert!(
            text.contains("Sample transition overhead (one invocation): 1234 guest cycles at [0]")
        );
        assert!(!text.contains("native max:"));
        assert!(!text.contains(" ns"));
        // Old stage summaries and old upstream reports did not carry overhead.
        let mut legacy = serde_json::to_value(&saved).unwrap();
        legacy["version"] = json!(1);
        legacy["native_tiles"] = serde_json::to_value(&ranked).unwrap();
        legacy
            .as_object_mut()
            .unwrap()
            .remove("native_ranking_path");
        legacy["replay_tiles"][0]
            .as_object_mut()
            .unwrap()
            .remove("transition_overhead");
        let legacy: StageReplayProfile = serde_json::from_value(legacy).unwrap();
        assert_eq!(legacy.native_tiles[0].total_duration_ns, 11325);
        assert!(legacy.native_ranking_path.is_none());
        assert!(legacy.replay_tiles[0].transition_overhead.is_none());
        assert!(report_text(&legacy, Path::new("legacy.json"))
            .contains("Sample transition overhead (one invocation): not measured"));
        assert!(saved.native.run_dir.unwrap().ends_with("native"));
    }

    #[test]
    fn empty_tile_set_skips_replay() {
        let mut fixture = Fixture::new();
        let baseline = fixture.report.stage_dir.clone();
        let source = fixture.root.join("native");
        let mut calls = 0;
        execute_profile(&mut fixture.report, &fixture.dir, |_, pass, _| {
            calls += 1;
            fake_pass(pass, &source, &baseline);
            write_json(
                &source.join("profile.json"),
                &ExecutionProfile::new(vec![], None, None),
            )
        })
        .unwrap();
        assert_eq!(calls, 1);
        assert!(matches!(
            fixture.report.status,
            ReplayProfileStatus::NoTiles
        ));
        assert!(fixture.report.replay.command_succeeded.is_none());
    }

    #[test]
    fn native_failures_preserve_artifacts_and_prevent_replay() {
        for failure in [
            "output",
            "commitment",
            "missing-profile",
            "malformed-profile",
            "input",
            "exit",
        ] {
            let mut fixture = Fixture::new();
            let baseline = fixture.report.stage_dir.clone();
            let input = fixture.report.input_path.clone();
            let source = fixture.root.join("native");
            let mut calls = 0;
            let result = execute_profile(&mut fixture.report, &fixture.dir, |_, pass, _| {
                calls += 1;
                fake_pass(pass, &source, &baseline);
                if failure != "missing-profile" {
                    write_json(
                        &source.join("profile.json"),
                        &ExecutionProfile::new(vec![record("heavy", 1, 100)], None, None),
                    )?;
                }
                match failure {
                    "output" => fs::write(source.join("output.bin"), b"wrong")?,
                    "commitment" => write_json(
                        &source.join("output_manifest.json"),
                        &json!({"output": {"commitment": "wrong"}}),
                    )?,
                    "input" => fs::write(&input, b"changed")?,
                    "exit" => pass.command_succeeded = Some(false),
                    "malformed-profile" => fs::write(source.join("profile.json"), b"not-json")?,
                    _ => (),
                }
                Ok(())
            });
            assert!(result.is_err(), "{failure}");
            assert_eq!(calls, 1);
            assert!(matches!(fixture.report.status, ReplayProfileStatus::Failed));
            assert!(fixture.report.error.is_some());
            assert!(fixture.dir.join(REPLAY_PROFILE_JSON).is_file());
            assert!(fixture.report.native.stdout_path.is_file());
            assert!(source.join("output.bin").is_file());
            if failure != "missing-profile" {
                assert!(fixture.dir.join("native-profile.json").is_file());
            }
        }
    }

    #[test]
    fn partial_replay_retains_successful_measurements_and_failure_coordinate() {
        let mut fixture = Fixture::new();
        let baseline = fixture.report.stage_dir.clone();
        let root = fixture.root.clone();
        let mut calls = 0;
        let result = execute_profile(&mut fixture.report, &fixture.dir, |_, pass, _| {
            calls += 1;
            let source = root.join(&pass.command[4]);
            fake_pass(pass, &source, &baseline);
            if calls == 1 {
                write_json(
                    &source.join("profile.json"),
                    &ExecutionProfile::new(
                        vec![record("heavy", 0, 10), record("heavy", 1, 20)],
                        None,
                        None,
                    ),
                )?;
            } else {
                pass.command_succeeded = Some(false);
                write_json(&source.join("replay-profile.json"), &replay(2, false))?;
            }
            Ok(())
        });
        assert!(result.is_err());
        assert_eq!(fixture.report.replay_complete, Some(false));
        assert_eq!(fixture.report.replay_tiles[0].profiled_invocations, 1);
        assert_eq!(fixture.report.replay_tiles[0].total_guest_cycles, 100);
        assert_eq!(
            fixture.report.replay_tiles[0]
                .transition_overhead
                .as_ref()
                .unwrap()
                .guest_cycles,
            1234
        );
        assert!(fixture
            .report
            .error
            .as_ref()
            .unwrap()
            .contains("heavy at [1]: guest output mismatch"));
        assert!(fixture.dir.join("raster-replay-profile.json").is_file());
    }

    #[test]
    fn failed_overhead_keeps_replay_cycles_without_substituting_zero() {
        let mut fixture = Fixture::new();
        let baseline = fixture.report.stage_dir.clone();
        let root = fixture.root.clone();
        let mut calls = 0;
        let result = execute_profile(&mut fixture.report, &fixture.dir, |_, pass, _| {
            calls += 1;
            let source = root.join(&pass.command[4]);
            fake_pass(pass, &source, &baseline);
            if calls == 1 {
                write_json(
                    &source.join("profile.json"),
                    &ExecutionProfile::new(vec![record("heavy", 0, 10)], None, None),
                )?;
            } else {
                pass.command_succeeded = Some(false);
                let mut profile = replay(1, false);
                profile.tiles.get_mut("heavy").unwrap().transition_overhead = None;
                let failure = profile.failure.as_mut().unwrap();
                failure.phase = ProfileFailurePhase::TransitionOverhead;
                failure.invocation =
                    serde_json::from_value(json!({"exec_index": 0, "coordinates": [0]}))?;
                failure.message = "storage commitment mismatch".into();
                write_json(&source.join("replay-profile.json"), &profile)?;
            }
            Ok(())
        });
        assert!(result.is_err());
        let saved: StageReplayProfile = read_json(&fixture.dir.join(REPLAY_PROFILE_JSON)).unwrap();
        assert!(matches!(saved.status, ReplayProfileStatus::Failed));
        assert_eq!(saved.replay_complete, Some(false));
        assert_eq!(saved.replay_tiles[0].profiled_invocations, 1);
        assert_eq!(saved.replay_tiles[0].max_guest_cycles, Some(100));
        assert_eq!(saved.replay_tiles[0].total_guest_cycles, 100);
        assert!(saved.replay_tiles[0].transition_overhead.is_none());
        let text = report_text(&saved, Path::new("partial.json"));
        assert!(text.contains("replay total 100 guest cycles across 1 measured invocations"));
        assert!(text.contains("Sample transition overhead (one invocation): not measured"));
        assert!(
            text.contains("heavy at [0]: storage commitment mismatch (phase: TransitionOverhead)")
        );
    }

    #[test]
    fn missing_replay_report_surfaces_native_failure_from_both_logs() {
        let mut fixture = Fixture::new();
        let baseline = fixture.report.stage_dir.clone();
        let root = fixture.root.clone();
        let mut calls = 0;
        let result = execute_profile(&mut fixture.report, &fixture.dir, |_, pass, _| {
            calls += 1;
            let source = root.join(&pass.command[4]);
            fake_pass(pass, &source, &baseline);
            if calls == 1 {
                write_json(
                    &source.join("profile.json"),
                    &ExecutionProfile::new(vec![record("heavy", 0, 10)], None, None),
                )?;
            } else {
                pass.command_succeeded = Some(false);
                fs::write(&pass.stdout_path, format!(
                    "old output must be truncated\n{}\nProcess exited with: exit status: 101\nFailed to write binary trace: simulated disk full\n", "x".repeat(16384),
                ))?;
                fs::write(
                    &pass.stderr_path,
                    "Native execution failed; replay profiling was not started\n",
                )?;
            }
            Ok(())
        });
        assert!(result.is_err());
        let saved: StageReplayProfile = read_json(&fixture.dir.join(REPLAY_PROFILE_JSON)).unwrap();
        assert!(matches!(saved.status, ReplayProfileStatus::Failed));
        assert!(saved.replay_tiles.is_empty());
        let error = saved.error.as_ref().unwrap();
        assert!(
            error.contains("Raster profiling command failed before producing replay-profile.json")
        );
        assert!(error.contains("Failed to write binary trace: simulated disk full"));
        assert!(error.contains("Native execution failed; replay profiling was not started"));
        assert!(!error.contains("old output must be truncated"));
        assert!(error.len() < 10000);
        assert!(!fixture.dir.join("raster-replay-profile.json").exists());
    }

    #[test]
    fn artifact_discovery_uses_the_reported_directory_and_rejects_ambiguity() {
        let mut dir = None;
        observe_run_header(
            "  Run artifacts dir: /tmp/old run",
            Path::new("/project"),
            &mut dir,
        )
        .unwrap();
        assert_eq!(dir, Some(PathBuf::from("/tmp/old run")));
        assert!(observe_run_header(
            "  Run artifacts dir: /tmp/newer",
            Path::new("/project"),
            &mut dir
        )
        .is_err());
        let mut relative = None;
        observe_run_header(
            "  Run artifacts dir: target/raster/runs/exact",
            Path::new("/project"),
            &mut relative,
        )
        .unwrap();
        assert_eq!(
            relative,
            Some(PathBuf::from("/project/target/raster/runs/exact"))
        );
    }

    #[cfg(unix)]
    #[test]
    fn command_transport_records_exit_and_stderr_and_ignores_program_headers() {
        let mut fixture = Fixture::new();
        let source = fixture.root.join("reported run");
        fixture.report.native.command = vec![
            "sh".into(), "-c".into(),
            "printf '  Run artifacts dir: %s\\nMode: authenticated\\n  Run artifacts dir: /wrong\\n' \"$1\"; printf 'guest failed\\n' >&2; exit 7".into(),
            "sh".into(), source.to_string_lossy().into_owned(),
        ];
        let mut terminal = Vec::new();
        run_profile_command_with_output(
            &fixture.root,
            &mut fixture.report.native,
            ProfileOutput::LogOnly,
            &mut terminal,
        )
        .unwrap();
        assert!(terminal.is_empty());
        assert_eq!(fixture.report.native.run_dir, Some(source));
        assert_eq!(fixture.report.native.command_succeeded, Some(false));
        assert!(fs::read_to_string(&fixture.report.native.stdout_path)
            .unwrap()
            .contains("/wrong"));
        assert_eq!(
            fs::read_to_string(&fixture.report.native.stderr_path).unwrap(),
            "guest failed\n"
        );
    }

    #[cfg(unix)]
    #[test]
    fn command_transport_logs_native_timings_and_streams_replay_cycles() {
        let mut fixture = Fixture::new();
        let mut terminal = Vec::new();
        for (pass, output, measurement) in [
            (
                &mut fixture.report.native,
                ProfileOutput::LogOnly,
                "native tile: 100 ns",
            ),
            (
                &mut fixture.report.replay,
                ProfileOutput::Stream,
                "Replay progress: 1/128; 123 guest cycles",
            ),
        ] {
            pass.command = vec![
                "sh".into(),
                "-c".into(),
                "printf '  Run artifacts dir: target/run\nMode: authenticated\n%s\n' \"$1\"".into(),
                "sh".into(),
                measurement.into(),
            ];
            run_profile_command_with_output(&fixture.root, pass, output, &mut terminal).unwrap();
            assert_eq!(pass.command_succeeded, Some(true));
            assert!(fs::read_to_string(&pass.stdout_path)
                .unwrap()
                .contains(measurement));
        }
        let terminal = String::from_utf8(terminal).unwrap();
        assert!(terminal.contains("Replay progress: 1/128; 123 guest cycles"));
        assert!(!terminal.contains("native tile"));
        assert!(!terminal.contains(" ns"));
    }

    #[test]
    fn cli_requires_exact_stage_and_positive_top_n() {
        for args in [
            vec!["raster-inference", "replay-profile"],
            vec![
                "raster-inference",
                "replay-profile",
                "--stage",
                "prompt_merge_b1",
                "--top-n",
                "0",
            ],
        ] {
            assert!(crate::parse_for_test(args).is_err());
        }
        let cli = crate::Cli::try_parse_from([
            "raster-inference",
            "replay-profile",
            "--stage",
            "prompt_merge_b1",
        ])
        .unwrap();
        let Some(crate::Commands::ReplayProfile(args)) = cli.command else {
            panic!("wrong command")
        };
        assert_eq!(args.top_n.get(), 3);
        assert_eq!(args.run, PathBuf::from("inference.toml"));
    }
}
