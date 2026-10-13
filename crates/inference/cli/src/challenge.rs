use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

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
    CheckpointTrace, Divergence, DivergenceReason, PreparedRun, ReplayPackage,
};

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
    let claimed_input = load_claimed_input(&options.input)?;
    let manifest_path = challenge_manifest_path(&options, &claimed_input)?;
    let executor = CheckpointedInferenceExecutor;
    let (recomputed_chain_dir, recomputed_trace, divergence) = match &claimed_input.checkpoints {
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
            (recomputed.chain_dir, recomputed_trace, divergence)
        }
    };
    let recomputed_trace_path =
        write_checkpoint_trace_artifact(&recomputed_chain_dir, &recomputed_trace)?;

    let Some(divergence) = divergence else {
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
    seed_replay_directory(
        &recomputed_chain_dir,
        &recomputed_trace,
        divergence.checkpoint_index,
        &replay_run_dir,
    )?;
    let replay_package = run_raster_replay(
        &options.base_dir,
        &manifest_path,
        &replay_run_dir,
        &divergence.stage,
    )?;
    let (divergence_path, replay_package_path, challenge_trace_path, challenge_bundle_path) =
        write_challenge_artifacts(
            &challenge_dir,
            claimed_input.source_path(),
            &recomputed_trace_path,
            &divergence,
            &replay_package,
        )?;
    let challenge_bundle = read_challenge_bundle(&challenge_bundle_path)?;

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

fn challenge_manifest_path(
    options: &ChallengeBuildOptions,
    claimed: &ClaimedInput,
) -> Result<PathBuf> {
    let run_spec_path = resolve_bundle_path(&options.base_dir, &options.run_spec_path);
    if let Some(prepared) = &claimed.prepared_run {
        let run_spec = inference_artifacts::read_run_spec(&run_spec_path)?;
        let run_spec_dir = run_spec_path
            .parent()
            .context("run spec has no parent directory")?;
        let selected_model_path = resolve_bundle_path(run_spec_dir, &run_spec.model_manifest);
        // The run spec chooses the model; the claim retains its frozen prompt
        // and token count. Compare identities before touching the large bundle.
        inference_artifacts::verify_file_sha256(
            &selected_model_path,
            &prepared.model_manifest_sha256,
            "model selected by --run does not match the claim: model manifest",
        )?;
        let model =
            inference_artifacts::ModelManifest::load_verified(&prepared.model_manifest_path)?;
        model.verified_raster_template(&prepared.model_manifest_path)?;
        // load_claimed_input already checked the frozen run manifest's hash.
        return prepared
            .run_manifest_path
            .clone()
            .context("prepared run has no run_manifest_path");
    }

    // A bare trace has no frozen metadata. Prepare it from the selected run
    // spec, just as claim build does, rather than selecting a root template.
    let prepared = crate::claim::prepare_claim_run(&crate::claim::ClaimBuildOptions {
        base_dir: options.base_dir.clone(),
        run_spec_path,
        current_exe: options.current_exe.clone(),
        staged_backend: options.staged_backend,
    })?;
    Ok(prepared.manifest_path)
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

fn verify_prepared_run(prepared: &PreparedRun) -> Result<()> {
    if prepared.version != inference_artifacts::PREPARED_RUN_VERSION {
        bail!(
            "incompatible frozen prepared-run version {}",
            prepared.version
        );
    }
    verify_file_sha256(
        &prepared.model_manifest_path,
        &prepared.model_manifest_sha256,
        "frozen model manifest",
    )?;
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
        .args(["raster", "chain", "run"])
        .arg(manifest_path)
        .args(["--quiet-timings", "--run"])
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

fn seed_replay_directory(
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
    fn challenge_checks_selected_and_frozen_model_before_execution() {
        use crate::test_support::{model_fixture, run_spec};
        use inference_artifacts::{file_sha256, read_json};
        let base = temp_dir("model-selection");
        let model_path = model_fixture(&base, "model-a");
        model_fixture(&base, "model-b");
        let run_spec_path = run_spec(&base, "model-a");
        let prepared = crate::claim::prepare_claim_run(&crate::claim::ClaimBuildOptions {
            base_dir: base.clone(),
            run_spec_path: run_spec_path.clone(),
            current_exe: PathBuf::from("unused"),
            staged_backend: StagedExecutionBackend::InProcess,
        })
        .unwrap();
        let context_path = prepared
            .manifest_path
            .parent()
            .unwrap()
            .join(inference_artifacts::PREPARED_RUN_JSON);
        let frozen: PreparedRun = read_json(&context_path).unwrap();
        let claimed = ClaimedInput {
            checkpoints: ClaimedCheckpoints::Trace {
                path: base.join("trace.json"),
                trace: trace(Vec::new()),
            },
            prepared_run: Some(frozen.clone()),
        };
        let options = ChallengeBuildOptions {
            base_dir: base.clone(),
            run_spec_path: run_spec_path.clone(),
            current_exe: PathBuf::from("unused"),
            staged_backend: StagedExecutionBackend::InProcess,
            input: ChallengeInput::Trace(base.join("trace.json")),
        };
        verify_prepared_run(&frozen).unwrap();
        assert_eq!(
            challenge_manifest_path(&options, &claimed).unwrap(),
            prepared.manifest_path
        );

        // Current prompt/token edits must not change the claim's frozen run.
        let spec = fs::read_to_string(&run_spec_path)
            .unwrap()
            .replace("tokens = 3", "tokens = 8")
            .replace("prompt = \"h\"", "prompt = \"changed\"");
        fs::write(&run_spec_path, spec).unwrap();
        assert_eq!(
            challenge_manifest_path(&options, &claimed).unwrap(),
            prepared.manifest_path
        );
        assert_eq!(
            file_sha256(&prepared.manifest_path).unwrap(),
            frozen.run_manifest_sha256.unwrap()
        );

        run_spec(&base, "model-b");
        assert!(challenge_manifest_path(&options, &claimed)
            .unwrap_err()
            .to_string()
            .contains("does not match the claim"));
        run_spec(&base, "model-a");
        for (file, label) in [
            ("model.detwgt", "model weights"),
            ("config.json", "model config"),
            ("tokenizer.json", "model tokenizer"),
            ("Raster.toml", "Raster template"),
        ] {
            let path = model_path.parent().unwrap().join(file);
            let original = fs::read(&path).unwrap();
            fs::write(&path, "replaced").unwrap();
            assert!(challenge_manifest_path(&options, &claimed)
                .unwrap_err()
                .to_string()
                .contains(&format!("{label} hash mismatch")));
            fs::write(path, original).unwrap();
        }
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
        let manifest = challenge_manifest_path(&options, &claimed).unwrap();
        assert!(fs::read_to_string(manifest)
            .unwrap()
            .contains("name = \"selected\""));
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
            "raster-inference-cli-challenge-{label}-{}-{nanos}",
            std::process::id()
        ))
    }
}
