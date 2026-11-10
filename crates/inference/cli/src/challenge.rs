use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{bail, Context, Result};
use staged_infer::chain_runner::StagedExecutionBackend;
use staged_infer::{
    CheckpointHashChallengeConfig, CheckpointedInferenceConfig, CheckpointedInferenceExecutor,
    ParityPolicy,
};

use inference_artifacts::{
    build_checkpoint_trace, checkpoint_hashes, read_challenge_bundle, read_checkpoint_hashes,
    read_checkpoint_trace, read_claim_bundle, verify_file_sha256, write_challenge_artifacts,
    write_checkpoint_trace_artifact, ChallengeBundle, ChallengeSourcePath, Checkpoint,
    CheckpointTrace, Divergence, DivergenceReason, InferenceRunSpec, PreparedRun, ReplayPackage,
};
use raster_runtime::{read_raster_artifact_from_bytes, ReadLimits};

use crate::parity::compare::value_json;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChallengeInput {
    Claim(PathBuf),
    Trace(PathBuf),
    CheckpointHashes {
        claim_context_path: PathBuf,
        checkpoint_hashes_path: PathBuf,
    },
}

#[derive(Debug, Clone)]
pub struct ChallengeBuildOptions {
    pub base_dir: PathBuf,
    pub run_spec_path: PathBuf,
    pub current_exe: PathBuf,
    pub staged_backend: StagedExecutionBackend,
    pub input: ChallengeInput,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChallengeBuildResult {
    pub outcome: ChallengeBuildOutcome,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChallengeBuildOutcome {
    NoDivergence {
        claimed_reference_path: PathBuf,
        recomputed_trace_path: PathBuf,
        recomputed_chain_dir: PathBuf,
    },
    /// Every checkpoint matched, but the claim's own final record refutes it.
    ClaimedOutputRejected {
        reason: String,
        claimed_reference_path: PathBuf,
        recomputed_trace_path: PathBuf,
        recomputed_chain_dir: PathBuf,
    },
    DivergenceBuilt {
        divergence: Divergence,
        divergence_path: PathBuf,
        replay_package: ReplayPackage,
        replay_package_path: PathBuf,
        challenge_trace_path: PathBuf,
        challenge_bundle_path: PathBuf,
        challenge_bundle: ChallengeBundle,
    },
}

pub fn build_challenge(options: ChallengeBuildOptions) -> Result<ChallengeBuildResult> {
    let checkpoint_trace_started = Instant::now();
    challenge_progress("checkpoint-trace", None, None)?;
    let claimed_input = load_claimed_input(&options.input)?;
    let (manifest_path, prepared_run) = challenge_run(&options, &claimed_input)?;
    let executor = CheckpointedInferenceExecutor;
    let (recomputed_chain_dir, recomputed_trace, divergence, final_result) = match &claimed_input
        .checkpoints
    {
        ClaimedCheckpoints::Hashes { path, hashes } => {
            let recomputed = executor.run_until_hash_divergence(CheckpointHashChallengeConfig {
                base_dir: options.base_dir.clone(),
                manifest_path: manifest_path.clone(),
                current_exe: options.current_exe.clone(),
                staged_backend: options.staged_backend,
                claimed_hashes_path: path.clone(),
                claimed_hashes: hashes.clone(),
            })?;
            (
                recomputed.chain_dir,
                recomputed.recomputed_trace,
                recomputed.divergence,
                recomputed.final_result,
            )
        }
        ClaimedCheckpoints::Trace { .. } => {
            let recomputed = executor.run(CheckpointedInferenceConfig {
                base_dir: options.base_dir.clone(),
                manifest_path: manifest_path.clone(),
                current_exe: options.current_exe.clone(),
                staged_backend: options.staged_backend,
                parity_policy: ParityPolicy::Skip,
            })?;
            let recomputed_trace = build_checkpoint_trace(&recomputed.chain_dir, &manifest_path)?;
            let recomputed_trace_path = recomputed
                .chain_dir
                .join(inference_artifacts::CHECKPOINT_TRACE_JSON);
            let divergence =
                claimed_input.locate_divergence(&recomputed_trace_path, &recomputed_trace)?;
            (
                recomputed.chain_dir,
                recomputed_trace,
                divergence,
                recomputed.final_result,
            )
        }
    };
    let recomputed_trace_path =
        write_checkpoint_trace_artifact(&recomputed_chain_dir, &recomputed_trace)?;
    let checkpoint_trace_duration = checkpoint_trace_started.elapsed();

    let Some(divergence) = divergence else {
        // Every checkpoint agrees, so there is no stage to dispute. A claim can
        // still be refuted by its own final record: honest execution over
        // non-greedy ids commits a non-zero verdict. A bare trace claims nothing.
        if prepared_run.is_some() {
            if let Some(reason) = claimed_output_rejection(final_result.as_ref())? {
                return Ok(ChallengeBuildResult {
                    outcome: ChallengeBuildOutcome::ClaimedOutputRejected {
                        reason,
                        claimed_reference_path: claimed_input.reference_path().to_path_buf(),
                        recomputed_trace_path,
                        recomputed_chain_dir,
                    },
                });
            }
        }
        return Ok(ChallengeBuildResult {
            outcome: ChallengeBuildOutcome::NoDivergence {
                claimed_reference_path: claimed_input.reference_path().to_path_buf(),
                recomputed_trace_path,
                recomputed_chain_dir,
            },
        });
    };

    let recomputed_checkpoint = divergence.recomputed.as_ref().ok_or_else(|| {
        anyhow::anyhow!(
            "cannot build a Raster challenge for `{}` because the recomputed trace has no matching checkpoint",
            divergence.stage
        )
    })?;
    if recomputed_checkpoint.stage != divergence.stage {
        bail!(
            "cannot build a Raster challenge for stage-name divergence: claimed `{}`, recomputed `{}`",
            divergence.stage,
            recomputed_checkpoint.stage
        );
    }

    let challenge_dir = recomputed_chain_dir
        .join("challenge")
        .join(sanitize_stage_name(&divergence.stage));
    let replay_run_dir = challenge_dir.join("raster-replay");
    challenge_progress("replay-preparing", Some(checkpoint_trace_duration), None)?;
    seed_replay_directory(
        &recomputed_chain_dir,
        &recomputed_trace,
        divergence.checkpoint_index,
        &replay_run_dir,
    )?;
    let raster_replay_started = Instant::now();
    challenge_progress("raster-replay", Some(checkpoint_trace_duration), None)?;
    let replay_package = run_raster_replay(
        &options.base_dir,
        &manifest_path,
        &replay_run_dir,
        &divergence.stage,
    )?;
    let raster_replay_duration = raster_replay_started.elapsed();
    challenge_progress(
        "validating",
        Some(checkpoint_trace_duration),
        Some(raster_replay_duration),
    )?;
    let (divergence_path, replay_package_path, challenge_trace_path, challenge_bundle_path) =
        write_challenge_artifacts(
            &challenge_dir,
            claimed_input.source_path(),
            &recomputed_trace_path,
            &divergence,
            &replay_package,
        )?;
    inference_artifacts::write_json(
        &challenge_dir.join("challenge_timings.json"),
        &serde_json::json!({
            "version": 1,
            "stage": divergence.stage,
            "checkpoint_index": divergence.checkpoint_index,
            "checkpoint_trace_duration_ms": checkpoint_trace_duration.as_secs_f64() * 1000.0,
            "raster_replay_duration_ms": raster_replay_duration.as_secs_f64() * 1000.0,
        }),
    )?;
    let challenge_bundle = read_challenge_bundle(&challenge_bundle_path)?;
    // The bundle retains the commitment and replay inputs/outputs. Fault proving
    // records its own execution; this successful challenge's raw trace is unused.

    Ok(ChallengeBuildResult {
        outcome: ChallengeBuildOutcome::DivergenceBuilt {
            divergence,
            divergence_path,
            replay_package,
            replay_package_path,
            challenge_trace_path,
            challenge_bundle_path,
            challenge_bundle,
        },
    })
}

fn challenge_progress(
    phase: &str,
    checkpoint_trace_duration: Option<Duration>,
    raster_replay_duration: Option<Duration>,
) -> Result<()> {
    let progress = serde_json::json!({
        "version": 1,
        "phase": phase,
        "started_at_ms": SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis(),
        "checkpoint_trace_duration_ms": checkpoint_trace_duration.map(|d| d.as_secs_f64() * 1000.0),
        "raster_replay_duration_ms": raster_replay_duration.map(|d| d.as_secs_f64() * 1000.0),
    });
    let mut stdout = std::io::stdout().lock();
    writeln!(stdout, "IRIS_CHALLENGE_PROGRESS {progress}")?;
    stdout.flush()?;
    Ok(())
}

/// The manifest a challenger recomputes, and the run a clean recomputation
/// judges the claim by.
fn challenge_run(
    options: &ChallengeBuildOptions,
    claimed: &ClaimedInput,
) -> Result<(PathBuf, Option<PreparedRun>)> {
    let run_spec_path = resolve_bundle_path(&options.base_dir, &options.run_spec_path);
    if let Some(prepared) = &claimed.prepared_run {
        let rebuilt = rebuild_teacher_forced_run(options, prepared)?;
        return Ok((rebuilt.manifest_path, Some(rebuilt.prepared_run)));
    }

    // A bare trace has no frozen metadata. Prepare it from the selected run
    // spec, just as claim build does, rather than selecting a root template.
    let prepared = crate::claim::prepare_claim_run(&crate::claim::ClaimBuildOptions {
        base_dir: options.base_dir.clone(),
        run_spec_path,
        current_exe: options.current_exe.clone(),
        staged_backend: options.staged_backend,
        claimed_token_ids: None,
    })?;
    Ok((prepared.manifest_path, None))
}

/// Prepares the claim's prompt, token budget and claimed ids from the model the
/// challenger's `--run` selects, using the challenger's own template.
///
/// Recomputing the claimer's manifest would rerun the claimer's model and agree
/// with it. Preparing this run instead means a claim that used other weights, a
/// different cut or a different configuration differs at a stage's committed
/// inputs, which is a divergence to replay. Nothing from the claim beyond those
/// three things reaches the run, and none of them can stop the challenge.
fn rebuild_teacher_forced_run(
    options: &ChallengeBuildOptions,
    frozen: &PreparedRun,
) -> Result<crate::claim::PreparedClaimRun> {
    let run_spec_path = resolve_bundle_path(&options.base_dir, &options.run_spec_path);
    let selected = inference_artifacts::read_run_spec(&run_spec_path)?;
    let frozen_manifest = frozen
        .run_manifest_path
        .as_ref()
        .context("prepared run has no run_manifest_path")?;
    let run_spec = InferenceRunSpec {
        model_manifest: selected.model_manifest,
        prompt: Some(frozen.prompt.resolved_prompt.clone()),
        prompt_file: None,
        raw_prompt: frozen.raw_prompt,
        tokens: frozen.tokens,
    };
    let rebuilt = crate::claim::prepare_challenge_run(
        &crate::claim::ClaimBuildOptions {
            base_dir: options.base_dir.clone(),
            run_spec_path,
            current_exe: options.current_exe.clone(),
            staged_backend: options.staged_backend,
            claimed_token_ids: Some(frozen_claimed_ids(frozen_manifest)?),
        },
        &run_spec,
    )?;
    if rebuilt.prepared_run.model_id != frozen.model_id {
        eprintln!(
            "warning: the claim names model {}, but --run selects model {}; a divergence here means the claim did not run the selected model",
            frozen.model_id, rebuilt.prepared_run.model_id
        );
    }
    Ok(rebuilt)
}

/// The claimed ids a frozen run manifest binds, checked against its commitments.
fn frozen_claimed_ids(manifest_path: &Path) -> Result<Vec<u32>> {
    let manifest: toml::Value = toml::from_str(
        &fs::read_to_string(manifest_path)
            .with_context(|| format!("failed to read {}", manifest_path.display()))?,
    )
    .with_context(|| format!("failed to parse {}", manifest_path.display()))?;
    let input = &manifest["chain"]["input"]["claimed_ids"];
    let path_of = |key: &str, index: usize| -> Result<PathBuf> {
        let pattern = input
            .get(key)
            .and_then(toml::Value::as_str)
            .with_context(|| format!("frozen run manifest has no claimed_ids {key}"))?;
        Ok(PathBuf::from(pattern.replace("{k}", &index.to_string())))
    };
    let mut ids = Vec::new();
    for (index, commitment) in bound_commitments(manifest_path, "claimed_ids")?
        .iter()
        .enumerate()
    {
        let path = path_of("path", index)?;
        let bytes =
            fs::read(&path).with_context(|| format!("failed to read {}", path.display()))?;
        let index_path = path_of("index_path", index)?;
        let raster_index = fs::read(&index_path)
            .with_context(|| format!("failed to read {}", index_path.display()))?;
        let artifact =
            read_raster_artifact_from_bytes(&bytes, &raster_index, &ReadLimits::unbounded())?;
        if !artifact.roots_agree() || &artifact.structural_root != commitment {
            bail!(
                "{} is not the claimed range the frozen run manifest commits to",
                path.display()
            );
        }
        let range = value_json(&artifact.value)?;
        let range_ids = range["token_ids"]
            .as_array()
            .context("claimed range has no token ids")?;
        for id in range_ids {
            let id = id.as_u64().context("claimed id is not an integer")?;
            ids.push(u32::try_from(id).context("claimed id exceeds u32")?);
        }
    }
    Ok(ids)
}

pub fn locate_divergence(
    claimed_trace_path: &Path,
    claimed: &CheckpointTrace,
    recomputed_trace_path: &Path,
    recomputed: &CheckpointTrace,
) -> Option<Divergence> {
    let checkpoint_count = claimed.checkpoints.len().max(recomputed.checkpoints.len());
    for index in 0..checkpoint_count {
        let claimed_checkpoint = claimed.checkpoints.get(index);
        let recomputed_checkpoint = recomputed.checkpoints.get(index);
        let Some(reason) = divergence_reason(claimed_checkpoint, recomputed_checkpoint) else {
            continue;
        };
        let stage = claimed_checkpoint
            .or(recomputed_checkpoint)
            .map(|checkpoint| checkpoint.stage.clone())
            .unwrap_or_else(|| String::from("<unknown>"));
        return Some(Divergence {
            version: 1,
            checkpoint_index: index,
            stage,
            reason,
            claimed_trace_path: Some(claimed_trace_path.to_path_buf()),
            claimed_checkpoint_hashes_path: None,
            recomputed_trace_path: recomputed_trace_path.to_path_buf(),
            claimed_hash: None,
            recomputed_hash: None,
            claimed: claimed_checkpoint.cloned(),
            recomputed: recomputed_checkpoint.cloned(),
        });
    }
    None
}

pub fn locate_hash_divergence(
    claimed_hashes_path: &Path,
    claimed_hashes: &[String],
    recomputed_trace_path: &Path,
    recomputed: &CheckpointTrace,
) -> Result<Option<Divergence>> {
    let recomputed_hashes = checkpoint_hashes(recomputed)?;
    let checkpoint_count = claimed_hashes.len().max(recomputed_hashes.len());
    for index in 0..checkpoint_count {
        let claimed_hash = claimed_hashes.get(index);
        let recomputed_hash = recomputed_hashes.get(index);
        let Some(reason) = hash_divergence_reason(claimed_hash, recomputed_hash) else {
            continue;
        };
        let recomputed_checkpoint = recomputed.checkpoints.get(index);
        let stage = recomputed_checkpoint
            .map(|checkpoint| checkpoint.stage.clone())
            .unwrap_or_else(|| format!("<checkpoint-{index}>"));
        return Ok(Some(Divergence {
            version: 1,
            checkpoint_index: index,
            stage,
            reason,
            claimed_trace_path: None,
            claimed_checkpoint_hashes_path: Some(claimed_hashes_path.to_path_buf()),
            recomputed_trace_path: recomputed_trace_path.to_path_buf(),
            claimed_hash: claimed_hash.cloned(),
            recomputed_hash: recomputed_hash.cloned(),
            claimed: None,
            recomputed: recomputed_checkpoint.cloned(),
        }));
    }
    Ok(None)
}

fn divergence_reason(
    claimed: Option<&Checkpoint>,
    recomputed: Option<&Checkpoint>,
) -> Option<DivergenceReason> {
    match (claimed, recomputed) {
        (None, None) => None,
        (None, Some(_)) => Some(DivergenceReason::MissingClaimedCheckpoint),
        (Some(_), None) => Some(DivergenceReason::MissingRecomputedCheckpoint),
        (Some(claimed), Some(recomputed)) if claimed.stage != recomputed.stage => {
            Some(DivergenceReason::StageName)
        }
        (Some(claimed), Some(recomputed))
            if claimed.input_commitment != recomputed.input_commitment =>
        {
            Some(DivergenceReason::InputCommitment)
        }
        (Some(claimed), Some(recomputed))
            if claimed.output_commitment != recomputed.output_commitment =>
        {
            Some(DivergenceReason::OutputCommitment)
        }
        (Some(claimed), Some(recomputed)) if claimed.output_sha256 != recomputed.output_sha256 => {
            Some(DivergenceReason::OutputSha256)
        }
        _ => None,
    }
}

fn hash_divergence_reason(
    claimed: Option<&String>,
    recomputed: Option<&String>,
) -> Option<DivergenceReason> {
    match (claimed, recomputed) {
        (None, None) => None,
        (None, Some(_)) => Some(DivergenceReason::MissingClaimedHash),
        (Some(_), None) => Some(DivergenceReason::MissingRecomputedHash),
        (Some(claimed), Some(recomputed)) if claimed != recomputed => {
            Some(DivergenceReason::CheckpointHash)
        }
        _ => None,
    }
}

struct ClaimedInput {
    checkpoints: ClaimedCheckpoints,
    prepared_run: Option<PreparedRun>,
}

enum ClaimedCheckpoints {
    Trace {
        path: PathBuf,
        trace: CheckpointTrace,
    },
    Hashes {
        path: PathBuf,
        hashes: Vec<String>,
    },
}

impl ClaimedInput {
    fn reference_path(&self) -> &Path {
        self.checkpoints.path()
    }

    fn source_path(&self) -> ChallengeSourcePath<'_> {
        match &self.checkpoints {
            ClaimedCheckpoints::Trace { path, .. } => ChallengeSourcePath::Trace(path),
            ClaimedCheckpoints::Hashes { path, .. } => ChallengeSourcePath::CheckpointHashes(path),
        }
    }

    fn locate_divergence(
        &self,
        recomputed_trace_path: &Path,
        recomputed_trace: &CheckpointTrace,
    ) -> Result<Option<Divergence>> {
        match &self.checkpoints {
            ClaimedCheckpoints::Trace { path, trace } => Ok(locate_divergence(
                path,
                trace,
                recomputed_trace_path,
                recomputed_trace,
            )),
            ClaimedCheckpoints::Hashes { path, hashes } => {
                locate_hash_divergence(path, hashes, recomputed_trace_path, recomputed_trace)
            }
        }
    }
}

impl ClaimedCheckpoints {
    fn path(&self) -> &Path {
        match self {
            ClaimedCheckpoints::Trace { path, .. } => path,
            ClaimedCheckpoints::Hashes { path, .. } => path,
        }
    }
}

fn load_claimed_input(input: &ChallengeInput) -> Result<ClaimedInput> {
    match input {
        ChallengeInput::Claim(path) => {
            let bundle = read_claim_bundle(path)?;
            let bundle_dir = path
                .parent()
                .ok_or_else(|| anyhow::anyhow!("claim bundle has no parent directory"))?;
            let trace_path = resolve_bundle_path(bundle_dir, &bundle.checkpoint_trace_path);
            let trace = read_checkpoint_trace(&trace_path)?;
            let prepared_run = bundle
                .prepared_run_path
                .as_ref()
                .map(|path| read_prepared_run(&resolve_bundle_path(bundle_dir, path)))
                .transpose()?
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "claim bundle has no prepared_run_path; rebuild the claim with --run"
                    )
                })?;
            verify_prepared_run(&prepared_run)?;
            Ok(ClaimedInput {
                checkpoints: ClaimedCheckpoints::Trace {
                    path: trace_path,
                    trace,
                },
                prepared_run: Some(prepared_run),
            })
        }
        ChallengeInput::Trace(path) => {
            let trace = read_checkpoint_trace(path)?;
            Ok(ClaimedInput {
                checkpoints: ClaimedCheckpoints::Trace {
                    path: path.clone(),
                    trace,
                },
                prepared_run: None,
            })
        }
        ChallengeInput::CheckpointHashes {
            claim_context_path,
            checkpoint_hashes_path,
        } => {
            let bundle = read_claim_bundle(claim_context_path)?;
            let bundle_dir = claim_context_path
                .parent()
                .ok_or_else(|| anyhow::anyhow!("claim context bundle has no parent directory"))?;
            let prepared_run = bundle
                .prepared_run_path
                .as_ref()
                .map(|path| read_prepared_run(&resolve_bundle_path(bundle_dir, path)))
                .transpose()?
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "claim context bundle has no prepared_run_path; rebuild the claim with --run"
                    )
                })?;
            verify_prepared_run(&prepared_run)?;
            let hashes = read_checkpoint_hashes(checkpoint_hashes_path)?;
            Ok(ClaimedInput {
                checkpoints: ClaimedCheckpoints::Hashes {
                    path: checkpoint_hashes_path.clone(),
                    hashes,
                },
                prepared_run: Some(prepared_run),
            })
        }
    }
}

fn read_prepared_run(path: &Path) -> Result<PreparedRun> {
    let value: serde_json::Value = inference_artifacts::read_json(path)?;
    if value["version"].as_u64() != Some(inference_artifacts::PREPARED_RUN_VERSION as u64) {
        bail!("incompatible frozen prepared-run version: expected ranked-BPE version {}; preserve this claim and use its original implementation for replay", inference_artifacts::PREPARED_RUN_VERSION);
    }
    serde_json::from_value(value).context("incompatible frozen ranked-BPE pieces")
}

/// A claim is recomputed from the challenger's own model, so the claimer's model
/// files are neither needed nor trusted here.
fn verify_prepared_run(prepared: &PreparedRun) -> Result<()> {
    if prepared.version != inference_artifacts::PREPARED_RUN_VERSION {
        bail!(
            "incompatible frozen prepared-run version {}",
            prepared.version
        );
    }
    let manifest_path = prepared
        .run_manifest_path
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("prepared run has no run_manifest_path"))?;
    let manifest_sha256 = prepared
        .run_manifest_sha256
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("prepared run has no run_manifest_sha256"))?;
    verify_file_sha256(manifest_path, manifest_sha256, "frozen run manifest")
}

/// The commitments an indexed `[chain.input.<name>]` table binds, in order.
fn bound_commitments(manifest_path: &Path, name: &str) -> Result<Vec<String>> {
    let manifest: toml::Value = toml::from_str(
        &fs::read_to_string(manifest_path)
            .with_context(|| format!("failed to read {}", manifest_path.display()))?,
    )
    .with_context(|| format!("failed to parse {}", manifest_path.display()))?;
    manifest["chain"]["input"][name]["commitments"]
        .as_array()
        .with_context(|| format!("frozen run manifest has no {name} commitments"))?
        .iter()
        .map(|value| value.as_str().map(str::to_string))
        .collect::<Option<Vec<_>>>()
        .with_context(|| format!("{name} commitments are not strings"))
}

/// Why a claim whose every checkpoint matched is still rejected, if it is.
///
/// With no divergent stage to dispute, a claim stands or falls on its own
/// final record, in the order the chain established it: the prompt's
/// preparation, then the checks, the verdict, and the output's budget and EOS
/// rules. Every rule here reads only that record: a contract reading the same
/// committed output reaches the same answer.
///
/// Honest execution over non-greedy claimed ids publishes a non-zero verdict; a
/// claimer who wanted a clean one instead had to doctor a stage, which diverges
/// and is replayed rather than reaching this point.
pub(crate) fn claimed_output_rejection(
    final_result: Option<&inference_artifacts::InferenceResult>,
) -> Result<Option<String>> {
    let result = final_result
        .context("recomputation produced no final output to check the claim against")?;
    Ok(final_record_rejection(result))
}

fn final_record_rejection(result: &inference_artifacts::InferenceResult) -> Option<String> {
    if result.model_fault != 0 {
        return Some(format!(
            "consumed model data failed authentication (fault {})",
            result.model_fault
        ));
    }
    if result.prompt_fault != 0 {
        return Some(format!(
            "the prompt was not prepared correctly: {} (fault {}, at position {})",
            run_prep::fault_description(result.prompt_fault),
            result.prompt_fault,
            result.prompt_fault_position
        ));
    }
    if !result.prompt_complete {
        return Some(String::from(
            "the prompt was not prepared correctly: its ranges do not cover the whole prompt",
        ));
    }
    if result.check_fault != 0 {
        return Some(format!(
            "the claimed output was not checked: {} (fault {}, at sequence position {})",
            run_prep::check_fault_description(result.check_fault),
            result.check_fault,
            result.check_fault_position
        ));
    }
    if result.claim_mismatches > 0 {
        return Some(format!(
            "claimed output is not the greedy continuation: the committed verdict counts {} claimed token(s) that differ from the greedy selection, the first at sequence position {}",
            result.claim_mismatches, result.first_claim_mismatch_position
        ));
    }
    if result.output_fault != 0 {
        return Some(format!(
            "the claimed output breaks the run's rules: {} (fault {})",
            run_prep::output_fault_description(result.output_fault),
            result.output_fault
        ));
    }
    None
}

fn resolve_bundle_path(base: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        base.join(path)
    }
}

fn run_raster_replay(
    base_dir: &Path,
    manifest_path: &Path,
    replay_run_dir: &Path,
    stage: &str,
) -> Result<ReplayPackage> {
    fs::create_dir_all(replay_run_dir)
        .with_context(|| format!("failed to create {}", replay_run_dir.display()))?;
    let status = Command::new("cargo")
        .current_dir(base_dir)
        .args(["raster", "chain", "run", "--fraud-proof-window-size", "128"])
        .arg(manifest_path)
        .arg("--run")
        .arg(replay_run_dir)
        .arg("--stage")
        .arg(stage)
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()
        .context("failed to start authenticated Raster stage replay")?;
    if !status.success() {
        bail!("authenticated Raster stage replay for `{stage}` failed ({status})");
    }
    replay_package(stage, replay_run_dir)
}

fn replay_package(stage: &str, replay_run_dir: &Path) -> Result<ReplayPackage> {
    let stage_dir = replay_run_dir.join(stage);
    let package = ReplayPackage {
        version: 1,
        stage: stage.to_string(),
        replay_run_dir: replay_run_dir.to_path_buf(),
        stage_dir: stage_dir.clone(),
        input_path: stage_dir.join("input.json"),
        input_manifest_path: stage_dir.join("input_manifest.json"),
        output_path: stage_dir.join("output.bin"),
        output_index_path: stage_dir.join("output.rindex"),
        output_manifest_path: stage_dir.join("output_manifest.json"),
        commit_path: stage_dir.join("commit.bin"),
    };
    for path in [
        &package.input_path,
        &package.input_manifest_path,
        &package.output_path,
        &package.output_index_path,
        &package.output_manifest_path,
        &package.commit_path,
    ] {
        if !path.is_file() {
            bail!(
                "authenticated Raster replay did not write {}",
                path.display()
            );
        }
    }
    Ok(package)
}

pub(crate) fn seed_replay_directory(
    source_chain_dir: &Path,
    trace: &CheckpointTrace,
    checkpoint_index: usize,
    replay_run_dir: &Path,
) -> Result<()> {
    if replay_run_dir.exists() {
        fs::remove_dir_all(replay_run_dir)
            .with_context(|| format!("failed to reset {}", replay_run_dir.display()))?;
    }
    fs::create_dir_all(replay_run_dir)
        .with_context(|| format!("failed to create {}", replay_run_dir.display()))?;
    for checkpoint in trace.checkpoints.iter().take(checkpoint_index) {
        let source_stage_dir = source_chain_dir.join(&checkpoint.stage);
        copy_dir_recursive(&source_stage_dir, &replay_run_dir.join(&checkpoint.stage))
            .with_context(|| format!("failed to seed producer `{}`", checkpoint.stage))?;
    }
    Ok(())
}

fn copy_dir_recursive(source: &Path, destination: &Path) -> Result<()> {
    fs::create_dir_all(destination)
        .with_context(|| format!("failed to create {}", destination.display()))?;
    for entry in
        fs::read_dir(source).with_context(|| format!("failed to read {}", source.display()))?
    {
        let entry =
            entry.with_context(|| format!("failed to read entry in {}", source.display()))?;
        let source_path = entry.path();
        let destination_path = destination.join(entry.file_name());
        let file_type = entry
            .file_type()
            .with_context(|| format!("failed to inspect {}", source_path.display()))?;
        if file_type.is_dir() {
            copy_dir_recursive(&source_path, &destination_path)?;
        } else if file_type.is_file() {
            fs::copy(&source_path, &destination_path).with_context(|| {
                format!(
                    "failed to copy {} to {}",
                    source_path.display(),
                    destination_path.display()
                )
            })?;
        }
    }
    Ok(())
}

fn sanitize_stage_name(stage: &str) -> String {
    stage
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' {
                ch
            } else {
                '_'
            }
        })
        .collect()
}

impl ChallengeBuildOptions {
    pub fn from_current_dir(
        current_exe: PathBuf,
        staged_backend: StagedExecutionBackend,
        input: ChallengeInput,
        run_spec_path: PathBuf,
    ) -> Result<Self> {
        let base_dir = std::env::current_dir().context("failed to read current directory")?;
        Ok(Self {
            run_spec_path,
            base_dir,
            current_exe,
            staged_backend,
            input,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::Digest;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn commitment_only_challenge_has_complete_replay_artifacts() {
        let base = temp_dir("challenge-trace-cleanup");
        let stage = base.join("stage_a");
        fs::create_dir_all(&stage).unwrap();
        for name in [
            "input.json",
            "input_manifest.json",
            "output.bin",
            "output.rindex",
            "output_manifest.json",
            "commit.bin",
        ] {
            fs::write(stage.join(name), name.as_bytes()).unwrap();
        }
        let package = replay_package("stage_a", &base).unwrap();
        assert!(!stage.join("trace.bin").exists());
        // The same inputs/outputs and commitment remain sufficient for a replay package.
        replay_package("stage_a", &base).unwrap();
        assert_eq!(fs::read(&package.commit_path).unwrap(), b"commit.bin");
        fs::remove_dir_all(base).unwrap();
    }

    #[test]
    fn bare_trace_prepares_the_run_spec_model() {
        let base = temp_dir("trace-model");
        crate::test_support::model_fixture(&base, "selected");
        fs::write(base.join("Raster.toml"), "stale root template").unwrap();
        let options = ChallengeBuildOptions {
            base_dir: base.clone(),
            run_spec_path: crate::test_support::run_spec(&base, "selected"),
            current_exe: PathBuf::from("unused"),
            staged_backend: StagedExecutionBackend::InProcess,
            input: ChallengeInput::Trace(base.join("trace.json")),
        };
        let claimed = ClaimedInput {
            checkpoints: ClaimedCheckpoints::Trace {
                path: base.join("trace.json"),
                trace: trace(Vec::new()),
            },
            prepared_run: None,
        };
        // A bare trace has no claimed ids, so it discovers them from the run
        // spec's model; this fixture has no weights to discover them with.
        let error = challenge_run(&options, &claimed).err().unwrap();
        assert!(
            format!("{error:#}").contains("direct inference"),
            "{error:#}"
        );
        fs::remove_dir_all(base).unwrap();
    }

    #[test]
    fn identical_traces_have_no_divergence() {
        let trace = trace(vec![checkpoint("stage_a", "aaa", "111")]);

        assert_eq!(
            locate_divergence(
                Path::new("claimed.json"),
                &trace,
                Path::new("recomputed.json"),
                &trace,
            ),
            None
        );
    }

    #[test]
    fn commitment_mismatch_is_first_divergence() {
        let claimed = trace(vec![
            checkpoint("stage_a", "aaa", "111"),
            checkpoint("stage_b", "bbb", "222"),
        ]);
        let recomputed = trace(vec![
            checkpoint("stage_a", "aaa", "111"),
            checkpoint("stage_b", "ccc", "222"),
        ]);

        let divergence = locate_divergence(
            Path::new("claimed.json"),
            &claimed,
            Path::new("recomputed.json"),
            &recomputed,
        )
        .unwrap();

        assert_eq!(divergence.checkpoint_index, 1);
        assert_eq!(divergence.stage, "stage_b");
        assert_eq!(divergence.reason, DivergenceReason::OutputCommitment);
    }

    #[test]
    fn missing_checkpoint_is_divergence() {
        let claimed = trace(vec![checkpoint("stage_a", "aaa", "111")]);
        let recomputed = trace(vec![
            checkpoint("stage_a", "aaa", "111"),
            checkpoint("stage_b", "bbb", "222"),
        ]);

        let divergence = locate_divergence(
            Path::new("claimed.json"),
            &claimed,
            Path::new("recomputed.json"),
            &recomputed,
        )
        .unwrap();

        assert_eq!(divergence.checkpoint_index, 1);
        assert_eq!(
            divergence.reason,
            DivergenceReason::MissingClaimedCheckpoint
        );
    }

    #[test]
    fn hash_mismatch_is_first_divergence() {
        let recomputed = trace(vec![
            checkpoint("stage_a", "aaa", "111"),
            checkpoint("stage_b", "bbb", "222"),
        ]);
        let mut claimed_hashes = checkpoint_hashes(&recomputed).unwrap();
        let replacement = if claimed_hashes[1].starts_with('0') {
            "1"
        } else {
            "0"
        };
        claimed_hashes[1].replace_range(0..1, replacement);

        let divergence = locate_hash_divergence(
            Path::new("checkpoints.txt"),
            &claimed_hashes,
            Path::new("recomputed.json"),
            &recomputed,
        )
        .unwrap()
        .unwrap();

        assert_eq!(divergence.checkpoint_index, 1);
        assert_eq!(divergence.stage, "stage_b");
        assert_eq!(divergence.reason, DivergenceReason::CheckpointHash);
        assert_eq!(
            divergence.claimed_checkpoint_hashes_path,
            Some(PathBuf::from("checkpoints.txt"))
        );
        assert!(divergence.claimed_trace_path.is_none());
        assert_eq!(
            divergence.recomputed,
            Some(checkpoint("stage_b", "bbb", "222"))
        );
    }

    #[test]
    fn missing_claimed_hash_is_divergence_at_recomputed_stage() {
        let recomputed = trace(vec![
            checkpoint("stage_a", "aaa", "111"),
            checkpoint("stage_b", "bbb", "222"),
        ]);
        let claimed_hashes =
            checkpoint_hashes(&trace(vec![checkpoint("stage_a", "aaa", "111")])).unwrap();

        let divergence = locate_hash_divergence(
            Path::new("checkpoints.txt"),
            &claimed_hashes,
            Path::new("recomputed.json"),
            &recomputed,
        )
        .unwrap()
        .unwrap();

        assert_eq!(divergence.checkpoint_index, 1);
        assert_eq!(divergence.stage, "stage_b");
        assert_eq!(divergence.reason, DivergenceReason::MissingClaimedHash);
        assert_eq!(divergence.claimed_hash, None);
        assert!(divergence.recomputed_hash.is_some());
    }

    #[test]
    fn replay_seed_copies_prior_stages_without_mutating_source() {
        let base = temp_dir("seed");
        let source_run = base.join("source");
        let replay_run = base.join("replay");
        fs::create_dir_all(&source_run).unwrap();
        write_stage_dir(&source_run, "stage_a");
        write_stage_dir(&source_run, "stage_b");
        let trace = CheckpointTrace {
            checkpoints: vec![
                checkpoint("stage_a", "aaa", "111"),
                checkpoint("stage_b", "bbb", "222"),
            ],
        };

        seed_replay_directory(&source_run, &trace, 1, &replay_run).unwrap();

        assert!(replay_run.join("stage_a").join("output.bin").is_file());
        assert!(!replay_run.join("stage_b").exists());
        assert!(source_run.join("stage_b").join("output.bin").is_file());

        fs::remove_dir_all(base).unwrap();
    }

    #[test]
    fn legacy_frozen_runs_are_rejected_without_rewriting_them() {
        let base = temp_dir("legacy-prepared");
        fs::create_dir_all(&base).unwrap();
        let path = base.join("prepared_run.json");
        let legacy = br#"{"version":1,"prompt":{"initial_pieces":["h","</w>"]}}"#;
        fs::write(&path, legacy).unwrap();
        let error = read_prepared_run(&path).unwrap_err();
        assert!(error
            .to_string()
            .contains("incompatible frozen prepared-run version"));
        assert_eq!(fs::read(&path).unwrap(), legacy);
        fs::remove_dir_all(base).unwrap();
    }

    #[test]
    fn prepared_run_hash_mismatch_is_rejected() {
        let base = temp_dir("prepared-hash");
        fs::create_dir_all(&base).unwrap();
        let model_manifest = base.join("model-artifacts").join("manifest.json");
        fs::create_dir_all(model_manifest.parent().unwrap()).unwrap();
        fs::write(&model_manifest, b"model").unwrap();
        let run_manifest = base.join("runs").join("Raster.toml");
        fs::create_dir_all(run_manifest.parent().unwrap()).unwrap();
        fs::write(&run_manifest, b"[chain]\nname = \"test\"\n").unwrap();

        let prepared = PreparedRun {
            version: inference_artifacts::PREPARED_RUN_VERSION,
            run_spec_path: base.join("inference.toml"),
            model_manifest_path: model_manifest.clone(),
            model_manifest_sha256: format!("{:x}", sha2::Sha256::digest(b"model")),
            model_id: String::from("model-id"),
            raw_prompt: false,
            prompt: inference_artifacts::PreparedPrompt {
                resolved_prompt: String::from("hello"),
                rendered_prompt: String::from("hello"),
                initial_pieces: vec![inference_artifacts::PreparedPiece {
                    text: String::from("hello"),
                    segment: 0,
                }],
                eos_token_ids: Vec::new(),
            },
            tokens: 1,
            run_manifest_path: Some(run_manifest),
            run_manifest_sha256: Some(String::from("wrong")),
        };

        let error = verify_prepared_run(&prepared).unwrap_err();
        assert!(error.to_string().contains("run manifest hash mismatch"));

        fs::remove_dir_all(base).unwrap();
    }

    #[test]
    fn a_claim_is_judged_by_its_final_record() {
        let clean = inference_artifacts::InferenceResult {
            prompt_complete: true,
            ..Default::default()
        };
        let rejection = |result: inference_artifacts::InferenceResult| {
            claimed_output_rejection(Some(&result)).unwrap()
        };

        assert_eq!(rejection(clean.clone()), None);

        for (result, expected) in [
            (
                inference_artifacts::InferenceResult {
                    model_fault: 1,
                    ..clean.clone()
                },
                "model data failed authentication",
            ),
            (
                inference_artifacts::InferenceResult {
                    prompt_fault: run_prep::FAULT_NOT_THE_MERGE,
                    prompt_fault_position: 7,
                    ..clean.clone()
                },
                "not what the merge produces",
            ),
            (
                inference_artifacts::InferenceResult {
                    prompt_complete: false,
                    ..clean.clone()
                },
                "do not cover",
            ),
            (
                inference_artifacts::InferenceResult {
                    check_fault: 1,
                    check_fault_position: 9,
                    ..clean.clone()
                },
                "could not be scored",
            ),
            (
                inference_artifacts::InferenceResult {
                    claim_mismatches: 2,
                    first_claim_mismatch_position: 9,
                    ..clean.clone()
                },
                "counts 2 claimed token(s)",
            ),
            (
                inference_artifacts::InferenceResult {
                    output_fault: 3,
                    ..clean.clone()
                },
                "without reaching EOS",
            ),
        ] {
            let reason = rejection(result).expect("the record is not clean");
            assert!(reason.contains(expected), "{reason}");
        }

        // A claim must have a final record to be judged by.
        assert!(claimed_output_rejection(None).is_err());
    }

    fn teacher_forced_claim(base: &Path, model: &str, ids: Vec<u32>) -> PreparedRun {
        let prepared = crate::claim::prepare_claim_run(&crate::claim::ClaimBuildOptions {
            base_dir: base.to_path_buf(),
            run_spec_path: crate::test_support::run_spec(base, model),
            current_exe: PathBuf::from("unused"),
            staged_backend: StagedExecutionBackend::InProcess,
            claimed_token_ids: Some(ids),
        })
        .unwrap();
        prepared.prepared_run
    }

    fn bound_external(manifest: &Path, stage: &str, parameter: &str) -> String {
        let manifest: toml::Value = toml::from_str(&fs::read_to_string(manifest).unwrap()).unwrap();
        manifest["chain"]["stage"]
            .as_array()
            .unwrap()
            .iter()
            .find(|entry| entry["name"].as_str() == Some(stage))
            .and_then(|entry| {
                entry["inputs"][parameter]["external"]["commitment"]
                    .as_str()
                    .map(str::to_string)
            })
            .unwrap()
    }

    #[test]
    fn a_challenger_rebuilds_a_claim_from_its_own_model_and_finds_the_input_that_differs() {
        use crate::test_support::{model_fixture, run_spec};
        let base = temp_dir("rebuild-other-model");
        model_fixture(&base, "claimers-model");
        let selected = model_fixture(&base, "challengers-model");
        let frozen = teacher_forced_claim(&base, "claimers-model", vec![0, 0, 0]);
        let frozen_manifest = frozen.run_manifest_path.clone().unwrap();
        verify_prepared_run(&frozen).unwrap();

        let run_spec_path = run_spec(&base, "challengers-model");
        let claimed = ClaimedInput {
            checkpoints: ClaimedCheckpoints::Trace {
                path: base.join("trace.json"),
                trace: trace(Vec::new()),
            },
            prepared_run: Some(frozen.clone()),
        };
        let options = ChallengeBuildOptions {
            base_dir: base.clone(),
            run_spec_path,
            current_exe: PathBuf::from("unused"),
            staged_backend: StagedExecutionBackend::InProcess,
            input: ChallengeInput::Trace(base.join("trace.json")),
        };
        // The claimer's model files are not needed to challenge the claim.
        fs::remove_dir_all(base.join("models/claimers-model")).unwrap();
        verify_prepared_run(&frozen).unwrap();

        let (manifest, rebuilt) = challenge_run(&options, &claimed).unwrap();
        let rebuilt = rebuilt.unwrap();
        assert_ne!(manifest, frozen_manifest);
        assert!(fs::read_to_string(&manifest)
            .unwrap()
            .contains("name = \"challengers-model\""));
        assert_eq!(
            fs::canonicalize(&rebuilt.model_manifest_path).unwrap(),
            fs::canonicalize(&selected).unwrap()
        );
        assert_ne!(rebuilt.model_id, frozen.model_id);

        // Same text, so the first stage's text agrees; its configuration names
        // a different model, so `prompt_bind`'s committed inputs differ.
        assert_eq!(
            bound_external(&manifest, "prompt_bind", "text"),
            bound_external(&frozen_manifest, "prompt_bind", "text")
        );
        assert_ne!(
            bound_external(&manifest, "prompt_bind", "config"),
            bound_external(&frozen_manifest, "prompt_bind", "config")
        );
        // The claim's ids and its prompt carried over.
        assert_eq!(
            rebuilt.prompt.resolved_prompt,
            frozen.prompt.resolved_prompt
        );
        assert_eq!(rebuilt.tokens, frozen.tokens);
        assert_eq!(
            bound_commitments(&manifest, "claimed_ids").unwrap(),
            bound_commitments(&frozen_manifest, "claimed_ids").unwrap()
        );
        fs::remove_dir_all(base).unwrap();
    }

    #[test]
    fn a_challenger_on_the_same_model_rebuilds_the_same_committed_inputs() {
        let base = temp_dir("rebuild-same-model");
        crate::test_support::model_fixture(&base, "model");
        let frozen = teacher_forced_claim(&base, "model", vec![0, 0, 0]);
        let frozen_manifest = frozen.run_manifest_path.clone().unwrap();
        let options = ChallengeBuildOptions {
            base_dir: base.clone(),
            run_spec_path: crate::test_support::run_spec(&base, "model"),
            current_exe: PathBuf::from("unused"),
            staged_backend: StagedExecutionBackend::InProcess,
            input: ChallengeInput::Trace(base.join("trace.json")),
        };
        let claimed = ClaimedInput {
            checkpoints: ClaimedCheckpoints::Trace {
                path: base.join("trace.json"),
                trace: trace(Vec::new()),
            },
            prepared_run: Some(frozen),
        };
        // The run spec's own prompt plays no part.
        let spec = fs::read_to_string(&options.run_spec_path)
            .unwrap()
            .replace("prompt = \"h\"", "prompt = \"hh\"");
        fs::write(&options.run_spec_path, spec).unwrap();
        let (manifest, _) = challenge_run(&options, &claimed).unwrap();
        for parameter in ["text", "config"] {
            assert_eq!(
                bound_external(&manifest, "prompt_bind", parameter),
                bound_external(&frozen_manifest, "prompt_bind", parameter),
            );
        }
        for input in ["prompt_marks", "claimed_ids"] {
            assert_eq!(
                bound_commitments(&manifest, input).unwrap(),
                bound_commitments(&frozen_manifest, input).unwrap(),
            );
        }
        fs::remove_dir_all(base).unwrap();
    }

    #[test]
    fn a_tampered_claimed_range_is_not_read_as_the_claim() {
        let base = temp_dir("tampered-claimed");
        crate::test_support::model_fixture(&base, "model");
        let frozen = teacher_forced_claim(&base, "model", vec![0, 0, 0]);
        let manifest = frozen.run_manifest_path.unwrap();
        assert_eq!(frozen_claimed_ids(&manifest).unwrap(), [0, 0, 0]);
        let claimed = manifest.parent().unwrap().join("prompt/claimed_0.rastered");
        let other = manifest.parent().unwrap().join("other");
        let file = run_prep::write_claimed_ranges(&[0, 0, 1], 4, &other)
            .unwrap()
            .remove(0);
        fs::copy(&file.path, &claimed).unwrap();
        fs::copy(&file.index_path, claimed.with_extension("rindex")).unwrap();
        assert!(frozen_claimed_ids(&manifest)
            .unwrap_err()
            .to_string()
            .contains("is not the claimed range"));
        fs::remove_dir_all(base).unwrap();
    }

    fn trace(checkpoints: Vec<Checkpoint>) -> CheckpointTrace {
        CheckpointTrace { checkpoints }
    }

    fn checkpoint(stage: &str, commitment: &str, sha: &str) -> Checkpoint {
        Checkpoint {
            stage: stage.to_string(),
            input_commitment: String::from("input"),
            output_commitment: commitment.to_string(),
            output_sha256: sha.to_string(),
        }
    }

    fn write_stage_dir(base: &Path, stage: &str) {
        let stage_dir = base.join(stage);
        fs::create_dir_all(&stage_dir).unwrap();
        fs::write(stage_dir.join("input.json"), b"{}").unwrap();
        fs::write(stage_dir.join("input_manifest.json"), b"{}").unwrap();
        fs::write(stage_dir.join("output.bin"), b"output").unwrap();
        fs::write(stage_dir.join("output.rindex"), b"index").unwrap();
        fs::write(stage_dir.join("output_manifest.json"), b"manifest").unwrap();
    }

    fn temp_dir(label: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "iris-cli-challenge-{label}-{}-{nanos}",
            std::process::id()
        ))
    }
}
