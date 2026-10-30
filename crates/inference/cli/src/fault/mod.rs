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
    #[arg(long)]
    pub challenge: PathBuf,
    #[arg(long)]
    pub output: PathBuf,
}

#[derive(Debug, clap::Args)]
pub struct ProveArgs {
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
    let bit = entry * value.fingerprint.bits_per_item();
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
    let (entry, bit, before, after) = flip_middle(&mut value)?;
    fs::create_dir_all(&args.output)?;
    let output = fs::canonicalize(args.output)?;
    let target = output.join("corrupted_commit.bin");
    ensure!(!target.exists(), "corrupted commitment already exists");
    value.write(&target)?;
    write_json(
        &output.join("corruption.json"),
        &json!({
            "version": 1, "stage": bundle.stage, "entry_index": entry, "fingerprint_bit": bit,
            "before_word": format!("{before:016x}"), "after_word": format!("{after:016x}"),
            "original_sha256": file_sha256(&source)?, "corrupted_sha256": file_sha256(&target)?,
            "commitment_path": target
        }),
    )?;
    Ok(())
}

fn progress(phase: &str, replay: Option<f64>, proving: Option<f64>) -> Result<()> {
    let mut stdout = std::io::stdout().lock();
    writeln!(
        stdout,
        "IRIS_FAULT_PROGRESS {}",
        json!({"version": 1, "phase": phase,
        "started_at_ms": SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis(),
        "stage_replay_duration_ms": replay, "proof_duration_ms": proving})
    )?;
    stdout.flush()?;
    Ok(())
}

pub fn prove(args: ProveArgs) -> Result<()> {
    let total = Instant::now();
    check_runtime()?;
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
        !output.join("fault_bundle.json").exists() && !output.join("stage-replay").exists(),
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
    let report_path = output.join("raster_audit.json");
    let mut child = Command::new("cargo")
        .current_dir(&project)
        .args(["raster", "run"])
        .arg("--input")
        .arg(&input)
        .arg("--input-manifest")
        .arg(&input_manifest)
        .arg("--audit")
        .arg(&disputed_path)
        .arg("--proof-output")
        .arg(&proof_path)
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
            if event["phase"] == "proving" {
                replay_ms = event["audit_duration_ms"]
                    .as_f64()
                    .context("Raster audit duration missing")?;
                progress("proving", Some(replay_ms), None)?;
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
            &output.join("fault_bundle.json"),
            &json!({"version":1,"outcome":"no-fault","stage":bundle.stage}),
        )?;
        anyhow::bail!("no fault found in the supplied challenge");
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
    let proof_ms = report["proof_duration_ms"]
        .as_f64()
        .context("Raster proof duration missing")?
        + validating_started.elapsed().as_secs_f64() * 1000.0;
    progress("validating", Some(replay_ms), Some(proof_ms))?;
    let timings = json!({"version":1,"preparation_duration_ms":preparation_ms,
        "stage_replay_duration_ms":replay_ms,"proof_duration_ms":proof_ms,"duration_ms":total.elapsed().as_secs_f64()*1000.0});
    write_json(&output.join("fault_timings.json"), &timings)?;
    write_json(
        &output.join("fault_bundle.json"),
        &json!({"version":1,"outcome":"proved","verified":true,
        "stage":bundle.stage,"commitment_sha256":file_sha256(&disputed_path)?,
        "proof_path":proof_path,"proof_sha256":file_sha256(&proof_path)?,"proof_bytes":fs::metadata(&proof_path)?.len(),"proof_seal_bytes":receipt.seal_size()}),
    )?;
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
    let stages = value
        .get("chain")
        .and_then(|v| v.get("stage"))
        .and_then(toml::Value::as_array)
        .context("Raster manifest has no chain stages")?;
    let mut projects = std::collections::BTreeSet::new();
    let mut finalize = false;
    let mut range = false;
    for stage in stages {
        let name = stage
            .get("name")
            .and_then(toml::Value::as_str)
            .unwrap_or("");
        if name == "prefill_finalize" || name.starts_with("prefill_range_") {
            finalize |= name == "prefill_finalize";
            range |= name.starts_with("prefill_range_");
            let project = stage
                .get("project")
                .and_then(toml::Value::as_str)
                .context("challenge stage has no project")?;
            projects.insert(base.join(project));
        }
    }
    ensure!(
        finalize && range,
        "Raster manifest must contain prefill_finalize and prefill_range stages"
    );
    Ok(projects)
}

pub fn check_runtime() -> Result<()> {
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
    fn challenge_preflight_resolves_and_deduplicates_both_policies() {
        let manifest: toml::Value = toml::from_str(
            r#"
[[chain.stage]]
name = "prefill_range_l0"
project = "range"
[[chain.stage]]
name = "prefill_range_l1"
project = "range"
[[chain.stage]]
name = "prefill_finalize"
project = "/opt/finalize"
[[chain.stage]]
name = "output_finalize"
project = "unused"
"#,
        )
        .unwrap();
        let projects = challenge_projects(&manifest, Path::new("/model")).unwrap();
        assert_eq!(
            projects,
            [
                PathBuf::from("/model/range"),
                PathBuf::from("/opt/finalize")
            ]
            .into()
        );
        let missing: toml::Value = toml::from_str(
            r#"
[[chain.stage]]
name = "prefill_finalize"
project = "finalize"
"#,
        )
        .unwrap();
        assert!(challenge_projects(&missing, Path::new("/model")).is_err());
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
    fn short_commitment_cannot_corrupt_revealed_windows() {
        assert!(flip_middle(&mut fixture(4)).is_err());
    }
}
