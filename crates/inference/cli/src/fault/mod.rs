//! Fault workflow orchestration. Raster remains the execution/proof implementation.

use anyhow::{ensure, Context, Result};
use inference_artifacts::{
    file_sha256, read_json, write_json, ChallengeBundle, CheckpointTrace, ClaimBundle, PreparedRun,
    ReplayPackage,
};
use raster_core::transition::{TransitionJournal, TransitionState};
use raster_prover::commitment_file::CommitmentFile;
use serde_json::json;
use std::{
    fs,
    io::{BufRead, BufReader, Write},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    time::{Instant, SystemTime, UNIX_EPOCH},
};

#[derive(Debug, clap::Args)]
pub struct CorruptArgs {
    /// Frozen challenge profile selecting an interior window endpoint.
    #[arg(long)]
    pub profile: Option<PathBuf>,
    #[arg(long)]
    pub challenge: PathBuf,
    #[arg(long)]
    pub output: PathBuf,
}

#[derive(Debug, clap::Args)]
pub struct ProveArgs {
    /// Execute the fault window and measure guest cycles without creating a proof.
    #[arg(long)]
    pub estimate: bool,
    #[arg(long)]
    pub challenge: PathBuf,
    #[arg(long)]
    pub claim_context: PathBuf,
    #[arg(long)]
    pub commitment: PathBuf,
    #[arg(long)]
    pub output: PathBuf,
}

fn reference(bundle: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_owned()
    } else {
        bundle.parent().unwrap().join(path)
    }
}

fn commitment(path: &Path) -> Result<CommitmentFile> {
    let value = CommitmentFile::read_memory(path)?;
    value.validate()?;
    Ok(value)
}

fn flip_middle(value: &mut CommitmentFile) -> Result<(usize, usize, u64, u64)> {
    value.validate()?;
    let window = value.window_size();
    ensure!(
        value.len() > window * 2,
        "challenge has no interior fingerprint entry to corrupt"
    );
    let entry = window + (value.len() - 2 * window) / 2;
    flip_entry(value, entry)
}

fn flip_entry(value: &mut CommitmentFile, entry: usize) -> Result<(usize, usize, u64, u64)> {
    value.validate()?;
    ensure!(
        entry >= value.window_size() && entry < value.len().saturating_sub(value.window_size()),
        "target is not an interior fingerprint entry"
    );
    let bit = entry
        .checked_mul(value.fingerprint.bits_per_item())
        .context("fingerprint bit overflow")?;
    let word = bit / 64;
    let before = value.fingerprint.bits.get(word);
    value
        .fingerprint
        .bits
        .set(word, before ^ (1u64 << (bit % 64)));
    value.validate()?;
    Ok((entry, bit, before, value.fingerprint.bits.get(word)))
}

pub fn corrupt(args: CorruptArgs) -> Result<()> {
    let challenge_path = fs::canonicalize(args.challenge)?;
    let bundle: ChallengeBundle = read_json(&challenge_path)?;
    ensure!(bundle.version == 1, "unsupported challenge version");
    let source = reference(&challenge_path, &bundle.raster_commit_path);
    let mut value = commitment(&source)?;

    let profile: Option<serde_json::Value> = args
        .profile
        .as_ref()
        .map(|path| read_json(path))
        .transpose()?;
    let (entry, bit, before, after) = if let Some(profile) = &profile {
        ensure!(
            matches!(profile["version"].as_u64(), Some(1 | 2))
                && profile["kind"] == "challenge-profile"
                && profile["complete"] == true
                && profile["stage"] == bundle.stage
                && profile["original_commitment_sha256"] == file_sha256(&source)?
                && profile["commitment_matches"] == true
                && (if profile["version"] == 2 {
                    profile["mapping_verified"] == true
                } else {
                    profile["trace_matches"] == true
                })
                && profile["commitment_length"] == value.len()
                && profile["window_size"] == value.window_size(),
            "profile is not bound to this challenge commitment"
        );
        let target = &profile["target"];
        let start = target["window_start"]
            .as_u64()
            .context("profile has no eligible window")?;
        let end = target["window_end"]
            .as_u64()
            .context("missing window endpoint")?;
        ensure!(
            target["exec_index"].as_u64().and_then(|n| n.checked_sub(1)) == Some(start)
                && start.checked_add(value.window_size() as u64 - 1) == Some(end)
                && target["window_size"] == value.window_size(),
            "invalid profile window"
        );
        flip_entry(&mut value, usize::try_from(end)?)?
    } else {
        flip_middle(&mut value)?
    };

    fs::create_dir_all(&args.output)?;
    let output = fs::canonicalize(args.output)?;
    let target = output.join("corrupted_commit.bin");
    ensure!(!target.exists(), "corrupted commitment already exists");
    value.write(&target)?;
    write_json(
        &output.join("corruption.json"),
        &json!({
            "version": 1, "stage": bundle.stage, "entry_index": entry,
            "target": profile.as_ref().map(|p| &p["target"]),
            "selection_reason": profile.as_ref().map(|p| &p["target_reason"]), "fingerprint_bit": bit,
            "before_word": format!("{before:016x}"), "after_word": format!("{after:016x}"),
            "original_sha256": file_sha256(&source)?, "corrupted_sha256": file_sha256(&target)?,
            "commitment_path": target
        }),
    )?;
    Ok(())
}

fn progress(phase: &str, replay: Option<f64>, proving: Option<f64>) -> Result<()> {
    emit_progress(phase, replay, proving, None)
}

fn emit_progress(
    phase: &str,
    replay: Option<f64>,
    proving: Option<f64>,
    estimation: Option<f64>,
) -> Result<()> {
    let mut stdout = std::io::stdout().lock();
    writeln!(
        stdout,
        "IRIS_FAULT_PROGRESS {}",
        json!({"version": 1, "phase": phase,
        "started_at_ms": SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis(),
        "stage_replay_duration_ms": replay, "proof_duration_ms": proving, "estimation_duration_ms": estimation})
    )?;
    stdout.flush()?;
    Ok(())
}

pub fn prove(args: ProveArgs) -> Result<()> {
    let total = Instant::now();
    check_runtime_for(args.estimate)?;
    progress("preparing", None, None)?;
    let challenge_path = fs::canonicalize(args.challenge)?;
    let claim_path = fs::canonicalize(args.claim_context)?;
    let disputed_path = fs::canonicalize(args.commitment)?;
    let bundle: ChallengeBundle = read_json(&challenge_path)?;
    let claim: ClaimBundle = read_json(&claim_path)?;
    ensure!(
        bundle.version == 1 && claim.version == 1,
        "unsupported bundle version"
    );
    let prepared_path = reference(
        &claim_path,
        claim
            .prepared_run_path
            .as_ref()
            .context("claim has no frozen run")?,
    );
    let prepared: PreparedRun = read_json(&prepared_path)?;
    let manifest = prepared
        .run_manifest_path
        .as_ref()
        .context("missing run manifest")?;
    inference_artifacts::verify_file_sha256(
        manifest,
        prepared
            .run_manifest_sha256
            .as_ref()
            .context("missing manifest hash")?,
        "frozen run",
    )?;
    inference_artifacts::verify_file_sha256(
        &prepared.model_manifest_path,
        &prepared.model_manifest_sha256,
        "frozen model",
    )?;
    let claim_trace: CheckpointTrace =
        read_json(&reference(&claim_path, &claim.checkpoint_trace_path))?;
    ensure!(
        claim_trace.checkpoints.iter().all(|c| !c.stage.is_empty()
            && c.stage
                .chars()
                .all(|ch| ch.is_ascii_alphanumeric() || ch == '_' || ch == '-')),
        "unsafe checkpoint stage name"
    );
    let index = claim_trace
        .checkpoints
        .iter()
        .position(|c| c.stage == bundle.stage)
        .context("challenge stage absent from claim")?;
    let package: ReplayPackage =
        read_json(&reference(&challenge_path, &bundle.replay_package_path))?;
    ensure!(package.stage == bundle.stage, "challenge stage mismatch");
    let baseline =
        inference_artifacts::read_checkpoint_from_stage_dir(&bundle.stage, &package.stage_dir)?;
    ensure!(
        baseline == claim_trace.checkpoints[index],
        "challenge does not match original claimer checkpoint"
    );
    let disputed = commitment(&disputed_path)?;
    let original = commitment(&reference(&challenge_path, &bundle.raster_commit_path))?;
    ensure!(
        disputed.len() == original.len() && disputed.window_size() == original.window_size(),
        "disputed commitment shape differs from challenge"
    );
    fs::create_dir_all(&args.output)?;
    let output = fs::canonicalize(args.output)?;
    ensure!(
        !output.join("fault_bundle.json").exists()
            && !output.join("fault_estimate_bundle.json").exists()
            && !output.join("fault_estimate.json").exists()
            && !output.join("stage-replay").exists(),
        "fault output already used"
    );
    // Guest dependencies can change without changing the stage's tile source.
    // Check the locked identity before the expensive replay, without rewriting it.
    let project = staged_infer::chain_runner::stage_project(manifest, &bundle.stage)?;
    let status = Command::new("cargo")
        .args(["raster", "program", "--verify"])
        .current_dir(&project)
        .status()
        .context("check Raster program identity before fault replay")?;
    ensure!(status.success(), "Raster program identity preflight failed ({status}); rebuild the stage with cargo raster build --backend risc0");
    let preparation_ms = total.elapsed().as_secs_f64() * 1000.0;
    progress("stage-replay", None, None)?;
    // Reuse the selected stage's frozen inputs directly. The fault prover
    // stops at divergence, so it never needs a completed stage checkpoint.
    let input = fs::canonicalize(&package.input_path).context("fault audit input missing")?;
    let input_manifest =
        fs::canonicalize(&package.input_manifest_path).context("fault audit manifest missing")?;
    let mut replay_ms = 0.0;
    let proof_path = output.join("fault.receipt");
    let report_path = output.join(if args.estimate {
        "fault_estimate.json"
    } else {
        "raster_audit.json"
    });
    let execution_started = Instant::now();
    let mut command = Command::new("cargo");
    command
        .current_dir(&project)
        .args(["raster", "run"])
        .arg("--input")
        .arg(&input)
        .arg("--input-manifest")
        .arg(&input_manifest)
        .arg("--audit")
        .arg(&disputed_path);
    if args.estimate {
        command.arg("--estimate");
    } else {
        command.arg("--proof-output").arg(&proof_path);
    }
    let mut child = command
        .arg("--report")
        .arg(&report_path)
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .context("start Raster live audit; install the updated cargo-raster CLI")?;
    for line in BufReader::new(child.stdout.take().context("Raster stdout unavailable")?).lines() {
        let line = line?;
        if let Some(event) = line.strip_prefix("RASTER_AUDIT_PROGRESS ") {
            let event: serde_json::Value = serde_json::from_str(event)?;
            if event["phase"] == "proving" || event["phase"] == "estimating" {
                replay_ms = event["audit_duration_ms"]
                    .as_f64()
                    .context("Raster audit duration missing")?;
                if args.estimate {
                    emit_progress("estimating", Some(replay_ms), None, None)?;
                } else {
                    progress("proving", Some(replay_ms), None)?;
                }
            }
        }
        println!("{line}");
    }
    let status = child.wait()?;
    ensure!(status.success(), "Raster live audit failed ({status})");
    let report: serde_json::Value = read_json(&report_path)?;
    ensure!(report["version"] == 1, "unsupported Raster audit report");
    if report["outcome"] == "no-fault" {
        write_json(
            &output.join(if args.estimate {
                "fault_estimate_bundle.json"
            } else {
                "fault_bundle.json"
            }),
            &json!({"version":1,"outcome":"no-fault","stage":bundle.stage}),
        )?;
        anyhow::bail!("no fault found in the supplied challenge");
    }

    let corruption_path = disputed_path.parent().unwrap().join("corruption.json");
    let mut selected_window = serde_json::Value::Null;
    if corruption_path.is_file() {
        let corruption: serde_json::Value = read_json(&corruption_path)?;
        if !corruption["target"].is_null() {
            ensure!(
                corruption["corrupted_sha256"] == file_sha256(&disputed_path)?,
                "corruption manifest hash mismatch"
            );
            let target = &corruption["target"];
            let end = target["window_end"]
                .as_u64()
                .context("missing selected endpoint")?;
            ensure!(
                report["window_start"] == target["window_start"]
                    && (if args.estimate {
                        report["window_start"]
                            .as_u64()
                            .zip(report["actual_step_count"].as_u64())
                            .and_then(|(s, n)| s.checked_add(n))
                    } else {
                        report["window_end"].as_u64()
                    }) == end.checked_add(1)
                    && (args.estimate || report["detected_step"] == target["window_end"]),
                "proved window differs from profile target"
            );
            selected_window = target.clone();
        }
    }
    if args.estimate {
        validate_estimate_report(&report, &disputed)?;
        let estimation_ms = (execution_started.elapsed().as_secs_f64() * 1000.0
            - report["audit_duration_ms"].as_f64().unwrap_or(0.0))
        .max(0.0);
        replay_ms = report["audit_duration_ms"]
            .as_f64()
            .context("missing estimate audit duration")?;
        emit_progress("validating", Some(replay_ms), None, Some(estimation_ms))?;
        write_json(
            &output.join("fault_timings.json"),
            &json!({"version":1,"selected_window":selected_window,
            "preparation_duration_ms":preparation_ms,"stage_replay_duration_ms":replay_ms,
            "estimation_duration_ms":estimation_ms,"duration_ms":total.elapsed().as_secs_f64()*1000.0}),
        )?;
        write_json(
            &output.join("fault_estimate_bundle.json"),
            &json!({"version":1,"outcome":"estimated","verified":false,
            "stage":bundle.stage,"commitment_sha256":file_sha256(&disputed_path)?,
            "report_sha256":file_sha256(&report_path)?}),
        )?;
        return Ok(());
    }
    ensure!(
        report["outcome"] == "proved" && report["verified"] == true,
        "Raster did not produce a verified fault proof"
    );
    let validating_started = Instant::now();
    let receipt: risc0_zkvm::Receipt = postcard::from_bytes(&fs::read(&proof_path)?)?;
    receipt.verify(raster_prover::TRANSITION_GUEST_ID)?;
    let journal: TransitionJournal = receipt.journal.decode()?;
    ensure!(
        matches!(journal.current_state, TransitionState::Finished),
        "proof has not reached a fraud verdict"
    );
    verify_refuted_commitment(&journal.refuted_trace_commitment, &disputed)?;
    if !selected_window.is_null() {
        ensure!(
            Some(journal.init_state.init_frontier.position)
                == selected_window["window_start"].as_u64()
                && Some(journal.init_state.fingerprint.len() as u64)
                    == selected_window["window_size"].as_u64(),
            "receipt journal window differs from selected window"
        );
    }
    let proof_ms = report["proof_duration_ms"]
        .as_f64()
        .context("Raster proof duration missing")?
        + validating_started.elapsed().as_secs_f64() * 1000.0;
    progress("validating", Some(replay_ms), Some(proof_ms))?;
    let timings = json!({"version":1,"preparation_duration_ms":preparation_ms,
        "selected_window":selected_window,"stage_replay_duration_ms":replay_ms,"proof_duration_ms":proof_ms,"duration_ms":total.elapsed().as_secs_f64()*1000.0});
    write_json(&output.join("fault_timings.json"), &timings)?;
    write_json(
        &output.join("fault_bundle.json"),
        &json!({"version":1,"outcome":"proved","verified":true,
        "stage":bundle.stage,"commitment_sha256":file_sha256(&disputed_path)?,
        "proof_path":proof_path,"proof_sha256":file_sha256(&proof_path)?,"proof_bytes":fs::metadata(&proof_path)?.len(),"proof_seal_bytes":receipt.seal_size()}),
    )?;
    Ok(())
}

fn validate_estimate_report(report: &serde_json::Value, disputed: &CommitmentFile) -> Result<()> {
    ensure!(
        report["version"] == 1
            && report["kind"] == "fault-window-estimate"
            && report["outcome"] == "estimated"
            && report["complete"] == true
            && report["error"].is_null(),
        "Raster did not produce a complete fault estimate"
    );
    let count = report["actual_step_count"]
        .as_u64()
        .context("missing estimate count")?;
    ensure!(
        count > 0
            && report["measured_step_count"].as_u64() == Some(count)
            && report["configured_window_size"].as_u64() == Some(disputed.window_size() as u64),
        "incomplete estimate coverage"
    );
    let journal: TransitionJournal = serde_json::from_value(report["final_journal"].clone())?;
    ensure!(
        matches!(journal.current_state, TransitionState::Finished),
        "estimate did not reach fault verdict"
    );
    verify_refuted_commitment(&journal.refuted_trace_commitment, disputed)?;
    let digest: String = disputed
        .header()
        .digest()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    ensure!(
        report["commitment_digest"] == digest
            && report["window_start"].as_u64() == Some(journal.init_state.init_frontier.position)
            && count == journal.init_state.fingerprint.len() as u64,
        "estimate window binding mismatch"
    );
    Ok(())
}

// The transition guest binds to Raster's canonical header digest, not the
// transport file's SHA-256. Keep file hashes for artifact integrity only.
fn verify_refuted_commitment(refuted: &[u8], disputed: &CommitmentFile) -> Result<()> {
    ensure!(
        refuted == disputed.header().digest().as_slice(),
        "proof refutes a different challenge"
    );
    Ok(())
}

/// Read-only preflight for both automatic challenge policies.
pub fn check_challenge_programs(manifest: &Path) -> Result<()> {
    let manifest = fs::canonicalize(manifest).context("locate Raster manifest")?;
    let value: toml::Value = toml::from_str(&fs::read_to_string(&manifest)?)?;
    let projects = challenge_projects(&value, manifest.parent().unwrap())?;
    for project in projects {
        let status = Command::new("cargo")
            .args(["raster", "program", "--verify"])
            .current_dir(&project)
            .status()
            .with_context(|| format!("verify program in {}", project.display()))?;
        ensure!(
            status.success(),
            "Raster identity preflight failed for {}. Run cargo raster build --backend risc0 in that directory, then retry. No inference was started.",
            project.display()
        );
    }
    println!("Challenge program identities verified");
    Ok(())
}

fn challenge_projects(
    value: &toml::Value,
    base: &Path,
) -> Result<std::collections::BTreeSet<PathBuf>> {
    let chain = value.get("chain").context("Raster manifest has no chain")?;
    let stages_of = |table: &toml::Value| {
        table
            .get("stage")
            .and_then(toml::Value::as_array)
            .cloned()
            .unwrap_or_default()
    };
    // Most of the chain's stages are inside its repeats.
    let repeats = chain
        .get("repeat")
        .and_then(toml::Value::as_array)
        .cloned()
        .unwrap_or_default();
    let stages = stages_of(chain)
        .into_iter()
        .chain(repeats.iter().flat_map(stages_of));
    let mut projects = std::collections::BTreeSet::new();
    for stage in stages {
        let project = stage
            .get("project")
            .and_then(toml::Value::as_str)
            .context("challenge stage has no project")?;
        projects.insert(base.join(project));
    }
    ensure!(
        !projects.is_empty(),
        "Raster manifest has no challenge projects"
    );
    Ok(projects)
}

pub fn check_runtime() -> Result<()> {
    check_runtime_for(false)
}

fn check_runtime_for(estimate: bool) -> Result<()> {
    // A dev receipt is never a successful application proof.
    ensure!(
        !risc0_zkvm::is_dev_mode(),
        "fault proving requires RISC0_DEV_MODE disabled"
    );
    ensure!(
        !raster_prover::TRANSITION_GUEST_ELF.is_empty(),
        "Iris was built without proving guests; rebuild without RISC0_SKIP_BUILD"
    );
    let help = Command::new("cargo")
        .args(["raster", "run", "--help"])
        .output()
        .context("find Raster CLI")?;
    let help_text = String::from_utf8_lossy(&help.stdout);
    ensure!(
        help.status.success()
            && (!estimate || help_text.contains("--estimate"))
            && ["--audit", "--proof-output", "--report"]
                .iter()
                .all(|flag| help_text.contains(flag)),
        "install the updated Raster CLI with live audit-and-prove support"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use raster_core::trace::{ExecStep, ExecTarget, StepKind, StepRecord, StorageRoots};
    use raster_core::{merkle::EMPTY_LEAF, trace::Trace};
    #[test]
    fn challenge_preflight_resolves_and_deduplicates_every_stage_project() {
        let manifest: toml::Value = toml::from_str(
            r#"
[[chain.stage]]
name = "prompt_bind_p0"
project = "bind"
[[chain.repeat]]
name = "layers"
index = "l"
count = 2
[[chain.repeat.stage]]
name = "range_p{j}_l{l}"
project = "range"
[[chain.repeat.stage]]
name = "range_c{k}_l{l}"
project = "range"
[[chain.repeat]]
name = "claimed"
index = "k"
count = 1
[[chain.repeat.stage]]
name = "check_c{k}"
project = "/opt/check"
[[chain.stage]]
name = "output_finalize"
project = "finalize"
"#,
        )
        .unwrap();
        let projects = challenge_projects(&manifest, Path::new("/model")).unwrap();
        assert_eq!(
            projects,
            [
                PathBuf::from("/model/bind"),
                PathBuf::from("/model/range"),
                PathBuf::from("/opt/check"),
                PathBuf::from("/model/finalize")
            ]
            .into()
        );
        let only_top_level: toml::Value = toml::from_str(
            r#"
[[chain.stage]]
name = "output_finalize"
project = "finalize"
"#,
        )
        .unwrap();
        assert_eq!(
            challenge_projects(&only_top_level, Path::new("/model"))
                .unwrap()
                .len(),
            1
        );
        let empty: toml::Value = toml::from_str("[chain]\nname = \"empty\"\n").unwrap();
        assert!(challenge_projects(&empty, Path::new("/model")).is_err());
    }

    fn fixture(len: usize) -> CommitmentFile {
        let trace = Trace(
            (0..len)
                .map(|i| StepRecord {
                    exec_index: i as u64,
                    sequence_id: "main".into(),
                    coordinates: Default::default(),
                    kind: StepKind::Exec(ExecStep {
                        target: ExecTarget::Tile("tile".into()),
                        intra_sequence_index: i as i32,
                        input_commitment: vec![1; 32],
                        input_source_commitment: vec![],
                        output_commitment: vec![2; 32],
                        storage: StorageRoots {
                            root_before: vec![],
                            root_after: vec![],
                            index_root_before: vec![],
                            index_root_after: vec![],
                        },
                    }),
                    recur_progress_commitment: Default::default(),
                    recur_state: None,
                })
                .collect(),
        );
        CommitmentFile::build(
            &trace,
            &EMPTY_LEAF,
            raster_prover::trace::FraudProofConfig::from_window_size(2).unwrap(),
        )
        .unwrap()
    }
    #[test]
    fn proof_binding_uses_header_digest_and_rejects_a_different_commitment() {
        use sha2::{Digest, Sha256};

        let original = fixture(20);
        let mut disputed = original.clone();
        flip_middle(&mut disputed).unwrap();
        let header_digest = disputed.header().digest();
        let file_digest = Sha256::digest(postcard::to_allocvec(&disputed).unwrap());

        verify_refuted_commitment(&header_digest, &disputed).unwrap();
        assert!(verify_refuted_commitment(&original.header().digest(), &disputed).is_err());
        assert!(verify_refuted_commitment(file_digest.as_slice(), &disputed).is_err());
    }

    #[test]
    fn estimates_require_complete_coverage_and_a_bound_fault_journal() {
        let disputed = fixture(20);
        let frontier = json!({"position":0,"leaf":[],"ommers":[]});
        let digest = disputed.header().digest();
        let digest_hex: String = digest.iter().map(|b| format!("{b:02x}")).collect();
        let mut report = json!({"version":1,"kind":"fault-window-estimate",
        "outcome":"estimated","complete":true,"error":null,
        "actual_step_count":2,"measured_step_count":2,"configured_window_size":2,
        "window_start":0,"commitment_digest":digest_hex,
        "final_journal":{
            "init_state":{"init_frontier":frontier,"init_storage_frontier":frontier,
                "init_storage_root":[],"init_storage_index_root":[],"fingerprint":disputed.fingerprint},
            "current_state":"Finished","transition_image_id":[],"authorization_image_id":[],
            "input_manifest_commitment":[],"program_commitment":[],"refuted_trace_commitment":digest,
            "entrypoint_authorization":"NotRequired","output_authorization":"NotRequired",
            "window_is_terminal":false,"final_committed_root":null
        }});
        report["final_journal"]["init_state"]["fingerprint"]["len"] = json!(2);
        validate_estimate_report(&report, &disputed).unwrap();
        report["measured_step_count"] = json!(1);
        assert!(validate_estimate_report(&report, &disputed).is_err());
        report["measured_step_count"] = json!(2);
        report["final_journal"]["refuted_trace_commitment"] = json!([0]);
        assert!(validate_estimate_report(&report, &disputed).is_err());
    }

    #[test]
    fn corruption_changes_exactly_one_interior_bit_and_is_reversible() {
        let original = fixture(20);
        let mut corrupted = original.clone();
        let (entry, _, _, _) = flip_middle(&mut corrupted).unwrap();
        assert!(entry >= original.window_size() && entry < original.len() - original.window_size());
        assert_eq!(
            original
                .fingerprint
                .bits
                .iter()
                .zip(&corrupted.fingerprint.bits)
                .map(|(a, b)| (a ^ b).count_ones())
                .sum::<u32>(),
            1
        );
        corrupted.validate().unwrap();
        flip_middle(&mut corrupted).unwrap();
        assert_eq!(
            postcard::to_allocvec(&original).unwrap(),
            postcard::to_allocvec(&corrupted).unwrap()
        );
    }
    #[test]
    fn targeted_corruption_uses_exact_endpoint_and_preserves_original() {
        let original = fixture(20);
        let bytes = postcard::to_allocvec(&original).unwrap();
        let mut copy = original.clone();
        let h = 5;
        let j = h + original.window_size() - 1;
        assert_eq!(flip_entry(&mut copy, j).unwrap().0, j);
        assert_ne!(copy.header().digest(), original.header().digest());
        assert_eq!(postcard::to_allocvec(&original).unwrap(), bytes);
        flip_entry(&mut copy, j).unwrap();
        assert_eq!(postcard::to_allocvec(&copy).unwrap(), bytes);
        assert!(flip_entry(&mut copy, 0).is_err());
        assert!(flip_entry(&mut copy, 19).is_err());
    }
    #[test]
    fn short_commitment_cannot_corrupt_revealed_windows() {
        assert!(flip_middle(&mut fixture(4)).is_err());
    }
}
