//! Fault workflow orchestration. Raster remains the execution/proof implementation.
mod program;
mod proof;

use anyhow::{ensure, Context, Result};
use inference_artifacts::{
    file_sha256, read_json, write_json, ChallengeBundle, CheckpointTrace, ClaimBundle, PreparedRun,
    ReplayPackage,
};
use raster_core::{
    merkle::EMPTY_LEAF,
    trace::{Trace, TraceEvent},
    transition::{TransitionJournal, TransitionState},
};
use raster_prover::trace::{
    TraceCommitment, TraceCommitmentExt, TraceVerifier, VerificationResult,
};
use serde_json::json;
use std::{
    fs,
    io::{Read, Write},
    path::{Path, PathBuf},
    process::Command,
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

fn commitment(path: &Path) -> Result<TraceCommitment> {
    let value: TraceCommitment = postcard::from_bytes(&fs::read(path)?)?;
    value.validate()?;
    Ok(value)
}

fn flip_middle(value: &mut TraceCommitment) -> Result<(usize, usize, u64, u64)> {
    value.validate()?;
    let window = value.window_size();
    ensure!(
        value.len() > window * 2,
        "challenge has no interior fingerprint entry to corrupt"
    );
    let entry = window + (value.len() - 2 * window) / 2;
    let bit = entry * value.fingerprint.bits_per_item();
    let word = bit / 64;
    let before = value.fingerprint.bits[word];
    value.fingerprint.bits[word] ^= 1u64 << (bit % 64);
    value.validate()?;
    Ok((entry, bit, before, value.fingerprint.bits[word]))
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
    fs::write(&target, postcard::to_allocvec(&value)?)?;
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
    let replay_dir = output.join("stage-replay");
    super::challenge::seed_replay_directory(
        claim_path.parent().unwrap(),
        &claim_trace,
        index,
        &replay_dir,
    )?;
    let preparation_ms = total.elapsed().as_secs_f64() * 1000.0;
    progress("stage-replay", None, None)?;
    let replay_started = Instant::now();
    let status = Command::new("cargo")
        .args(["raster", "chain", "run"])
        .arg(manifest)
        .arg("--run")
        .arg(&replay_dir)
        .arg("--stage")
        .arg(&bundle.stage)
        .status()?;
    ensure!(status.success(), "Raster stage replay failed ({status})");
    let stage_dir = replay_dir.join(&bundle.stage);
    ensure!(
        inference_artifacts::read_checkpoint_from_stage_dir(&bundle.stage, &stage_dir)? == baseline,
        "fault prover replay differs from original claimer checkpoint"
    );
    let project = raster_compiler::Project::new(staged_infer::chain_runner::stage_project(
        manifest,
        &bundle.stage,
    )?)?;
    let cfs = raster_compiler::CfsBuilder::new(&project).build()?;
    let input = stage_dir.join("input.json");
    let input_manifest = stage_dir.join("input_manifest.json");
    let (trace, recorder) =
        load_trace(&stage_dir.join("trace.bin"), &cfs, &input, &input_manifest)?;
    ensure!(
        trace.len() == disputed.len(),
        "replay trace length differs from challenge"
    );
    let mut verifier = TraceVerifier::new(disputed.clone(), &EMPTY_LEAF, &cfs)?;
    let evidence = verifier.verify(&trace);
    let replay_ms = replay_started.elapsed().as_secs_f64() * 1000.0;
    let VerificationResult::Fraud(evidence) = evidence else {
        write_json(
            &output.join("fault_bundle.json"),
            &json!({"version":1,"outcome":"no-fault","stage":bundle.stage}),
        )?;
        anyhow::bail!("no fault found in the supplied challenge");
    };
    progress("proving", Some(replay_ms), None)?;
    let proving_started = Instant::now();
    let backend = raster_backend_risc0::Risc0Backend::new(project.output_dir.clone())
        .with_user_crate(project.root_dir.clone());
    let replayer = raster_prover::replay::Replayer::new(&backend, &project);
    let receipt = proof::prove(
        evidence,
        &trace,
        &cfs,
        &recorder,
        &replayer,
        Some(input_manifest.to_str().context("non-UTF8 manifest path")?),
        &disputed,
    );
    receipt.verify(raster_prover::TRANSITION_GUEST_ID)?;
    let journal: TransitionJournal = receipt.journal.decode()?;
    ensure!(
        matches!(journal.current_state, TransitionState::Finished),
        "proof has not reached a fraud verdict"
    );
    ensure!(
        hex::encode(&journal.refuted_trace_commitment) == file_sha256(&disputed_path)?,
        "proof refutes a different challenge"
    );
    let proof_path = output.join("fault.receipt");
    fs::write(&proof_path, postcard::to_allocvec(&receipt)?)?;
    let proof_ms = proving_started.elapsed().as_secs_f64() * 1000.0;
    progress("validating", Some(replay_ms), Some(proof_ms))?;
    let timings = json!({"version":1,"preparation_duration_ms":preparation_ms,
        "stage_replay_duration_ms":replay_ms,"proof_duration_ms":proof_ms,"duration_ms":total.elapsed().as_secs_f64()*1000.0});
    write_json(&output.join("fault_timings.json"), &timings)?;
    write_json(
        &output.join("fault_bundle.json"),
        &json!({"version":1,"outcome":"proved","verified":true,
        "stage":bundle.stage,"commitment_sha256":file_sha256(&disputed_path)?,
        "proof_path":proof_path,"proof_sha256":file_sha256(&proof_path)?,"proof_bytes":fs::metadata(&proof_path)?.len()}),
    )?;
    Ok(())
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
    Ok(())
}

fn load_trace(
    path: &Path,
    cfs: &raster_core::cfs::ControlFlowSchema,
    input: &Path,
    manifest: &Path,
) -> Result<(Trace, raster_runtime::TraceRecorder)> {
    let mut file = fs::File::open(path)?;
    let mut trace = Trace::new();
    let mut recorder = raster_runtime::TraceRecorder::new(cfs.clone());
    recorder.set_external_input(input.to_str(), manifest.to_str())?;
    loop {
        let mut len = [0u8; 4];
        if file.read(&mut len[..1])? == 0 {
            break;
        }
        file.read_exact(&mut len[1..])?;
        let mut frame = vec![0; u32::from_le_bytes(len) as usize];
        file.read_exact(&mut frame)?;
        let event: TraceEvent = postcard::from_bytes(&frame)?;
        trace.push(recorder.record(event));
    }
    Ok((trace, recorder))
}

#[cfg(test)]
mod tests {
    use super::*;
    use raster_core::trace::{ExecStep, ExecTarget, StepKind, StepRecord, StorageRoots};
    fn fixture(len: usize) -> TraceCommitment {
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
        TraceCommitment::try_build(
            &trace,
            &EMPTY_LEAF,
            raster_prover::trace::FraudProofConfig::from_window_size(2).unwrap(),
        )
        .unwrap()
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
