use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use procedure_definition::{CheckpointList, ConsensusOutcome, Divergence, InferenceResult};

use crate::cache::CachedStageValue;
use crate::chain_runner::{self, CheckpointHashVerifier, StagedExecutionBackend};
use crate::uncheckpointed::inference_result_from_generated;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParityPolicy {
    Skip,
    ReferenceStage(String),
}

#[derive(Debug, Clone)]
pub struct CheckpointedInferenceConfig {
    pub base_dir: PathBuf,
    pub manifest_path: PathBuf,
    pub current_exe: PathBuf,
    pub staged_backend: StagedExecutionBackend,
    pub parity_policy: ParityPolicy,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckpointedInferenceResult {
    pub chain_dir: PathBuf,
    pub selected_stage_dir: Option<PathBuf>,
    /// The first inference's result: the whole result of a procedure kind of one.
    pub final_result: Option<InferenceResult>,
    /// What a consensus chain states besides: its second inference's result,
    /// and whether the two agree.
    pub consensus: Option<ConsensusOutcome>,
}

#[derive(Debug, Clone)]
pub struct CheckpointHashChallengeConfig {
    pub base_dir: PathBuf,
    pub manifest_path: PathBuf,
    pub current_exe: PathBuf,
    pub staged_backend: StagedExecutionBackend,
    pub claimed_hashes_path: PathBuf,
    pub claimed_hashes: Vec<String>,
    /// The producers each of the claim's checkpoints points at, for a run of
    /// the chain as the claim arranges it (see
    /// [`CheckpointHashVerifier::claimed_sources`]).
    pub claimed_sources: Option<Vec<std::collections::BTreeMap<String, u32>>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckpointHashChallengeResult {
    pub chain_dir: PathBuf,
    pub recomputed_list: CheckpointList,
    pub divergence: Option<Divergence>,
    /// The output record of the last check recomputation reached of the
    /// first inference. It is the claim's once the last claimed range is
    /// reached.
    pub final_result: Option<InferenceResult>,
    /// What a consensus chain states besides, once recomputation reaches its
    /// result stage.
    pub consensus: Option<ConsensusOutcome>,
}

#[derive(Debug, Default)]
pub struct CheckpointedInferenceExecutor;

impl CheckpointedInferenceExecutor {
    /// Execute a native prefix and retain the selected stage's replay inputs.
    pub fn run_through_stage(
        &self,
        config: CheckpointedInferenceConfig,
        stage: &str,
    ) -> Result<chain_runner::StagePrefixRun> {
        if config.parity_policy != ParityPolicy::Skip {
            bail!("run_through_stage requires native execution (ParityPolicy::Skip)");
        }
        ensure_supported_manifest_path(&config.base_dir, &config.manifest_path)?;
        let _cwd = CurrentDirGuard::enter(&config.base_dir)?;
        chain_runner::run_through_stage(
            &config.manifest_path,
            stage,
            &config.current_exe,
            config.staged_backend,
        )
    }

    pub fn run(&self, config: CheckpointedInferenceConfig) -> Result<CheckpointedInferenceResult> {
        ensure_supported_manifest_path(&config.base_dir, &config.manifest_path)?;
        let _cwd = CurrentDirGuard::enter(&config.base_dir)?;
        let raster_stage = match &config.parity_policy {
            ParityPolicy::Skip => None,
            ParityPolicy::ReferenceStage(stage) => Some(stage.as_str()),
        };
        let run = chain_runner::run(
            &config.manifest_path,
            raster_stage,
            &config.current_exe,
            config.staged_backend,
        )?;
        let (final_result, consensus) = final_result(run.final_checks, run.final_output);
        Ok(CheckpointedInferenceResult {
            chain_dir: run.chain_dir,
            selected_stage_dir: run.selected_stage_dir,
            final_result,
            consensus,
        })
    }

    pub fn run_until_hash_divergence(
        &self,
        config: CheckpointHashChallengeConfig,
    ) -> Result<CheckpointHashChallengeResult> {
        ensure_supported_manifest_path(&config.base_dir, &config.manifest_path)?;
        let _cwd = CurrentDirGuard::enter(&config.base_dir)?;
        let verified = chain_runner::run_until_hash_divergence(
            &config.manifest_path,
            &config.current_exe,
            config.staged_backend,
            CheckpointHashVerifier {
                claimed_hashes_path: config.claimed_hashes_path,
                claimed_hashes: config.claimed_hashes,
                claimed_sources: config.claimed_sources,
            },
        )?;
        let (final_result, consensus) =
            final_result(verified.run.final_checks, verified.run.final_output);
        Ok(CheckpointHashChallengeResult {
            final_result,
            consensus,
            chain_dir: verified.run.chain_dir,
            recomputed_list: verified.recomputed_list,
            divergence: verified.divergence,
        })
    }
}

/// A run's result: each inference's last check's record, with what the
/// result stage made of it when the run reached that stage.
fn final_result(
    checks: Vec<CachedStageValue>,
    last: Option<CachedStageValue>,
) -> (Option<InferenceResult>, Option<ConsensusOutcome>) {
    let mut results = checks.into_iter().filter_map(|check| match check {
        CachedStageValue::DecodeEdge(edge) => Some(inference_result_from_generated(
            sequence_check::input::generated_output(&edge),
        )),
        _ => None,
    });
    let Some(mut first) = results.next() else {
        return (None, None);
    };
    match (last, results.next()) {
        (Some(CachedStageValue::ResultRecord(record)), _) => {
            first.result_code = record.code;
            first.result_forms_sha256 = record.forms_sha256;
            (Some(first), None)
        }
        (Some(CachedStageValue::ConsensusRecord(record)), Some(mut second)) => {
            first.result_code = record.first.code;
            first.result_forms_sha256 = record.first.forms_sha256;
            second.result_code = record.second.code;
            second.result_forms_sha256 = record.second.forms_sha256;
            let consensus = ConsensusOutcome {
                second,
                same_input: record.same_input,
                agree: record.agree,
                code: record.code,
            };
            (Some(first), Some(consensus))
        }
        _ => (Some(first), None),
    }
}

fn ensure_supported_manifest_path(base_dir: &Path, manifest_path: &Path) -> Result<()> {
    if !manifest_path.is_file() {
        bail!(
            "checkpointed inference manifest does not exist: {}",
            manifest_path.display()
        );
    }
    let canonical_base = base_dir
        .canonicalize()
        .with_context(|| format!("failed to canonicalize {}", base_dir.display()))?;
    let canonical_manifest = manifest_path
        .canonicalize()
        .with_context(|| format!("failed to canonicalize {}", manifest_path.display()))?;
    if !canonical_manifest.starts_with(&canonical_base) {
        bail!(
            "checkpointed inference manifest {} is outside {}",
            manifest_path.display(),
            base_dir.display()
        );
    }
    Ok(())
}

struct CurrentDirGuard {
    saved: PathBuf,
}

impl CurrentDirGuard {
    fn enter(base_dir: &Path) -> Result<Self> {
        let saved = std::env::current_dir().context("failed to read current directory")?;
        std::env::set_current_dir(base_dir)
            .with_context(|| format!("failed to enter {}", base_dir.display()))?;
        Ok(Self { saved })
    }
}

impl Drop for CurrentDirGuard {
    fn drop(&mut self) {
        let _ = std::env::set_current_dir(&self.saved);
    }
}

impl CheckpointedInferenceConfig {
    pub fn from_current_dir(
        current_exe: PathBuf,
        staged_backend: StagedExecutionBackend,
    ) -> Result<Self> {
        let base_dir = std::env::current_dir().context("failed to read current directory")?;
        Ok(Self {
            manifest_path: base_dir.join("Raster.toml"),
            base_dir,
            current_exe,
            staged_backend,
            parity_policy: ParityPolicy::Skip,
        })
    }

    pub fn reference_stage(mut self, stage: impl Into<String>) -> Self {
        self.parity_policy = ParityPolicy::ReferenceStage(stage.into());
        self
    }

    pub fn with_current_exe_from_env(staged_backend: StagedExecutionBackend) -> Result<Self> {
        Self::from_current_dir(
            std::env::current_exe().context("failed to locate current executable")?,
            staged_backend,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_from_current_dir_points_at_root_manifest() {
        let cwd = std::env::current_dir().unwrap();
        let config = CheckpointedInferenceConfig::from_current_dir(
            PathBuf::from("bin"),
            StagedExecutionBackend::InProcess,
        )
        .unwrap();

        assert_eq!(config.base_dir, cwd);
        assert_eq!(config.manifest_path, cwd.join("Raster.toml"));
        assert_eq!(config.parity_policy, ParityPolicy::Skip);
    }

    #[test]
    fn manifest_path_must_exist_inside_base_dir() {
        let base =
            std::env::temp_dir().join(format!("staged-infer-manifest-path-{}", std::process::id()));
        std::fs::create_dir_all(base.join("target/run")).unwrap();
        let manifest = base.join("target/run/Raster.toml");
        std::fs::write(&manifest, "[chain]\nname = \"test\"\n").unwrap();

        ensure_supported_manifest_path(&base, &manifest).unwrap();

        std::fs::remove_dir_all(base).unwrap();
    }
}
