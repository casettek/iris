use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use inference_artifacts::{CheckpointTrace, Divergence, InferenceResult};

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
    pub final_result: Option<InferenceResult>,
}

#[derive(Debug, Clone)]
pub struct CheckpointHashChallengeConfig {
    pub base_dir: PathBuf,
    pub manifest_path: PathBuf,
    pub current_exe: PathBuf,
    pub staged_backend: StagedExecutionBackend,
    pub claimed_hashes_path: PathBuf,
    pub claimed_hashes: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckpointHashChallengeResult {
    pub chain_dir: PathBuf,
    pub recomputed_trace: CheckpointTrace,
    pub divergence: Option<Divergence>,
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
        let final_result = match run.final_output {
            Some(CachedStageValue::GeneratedOutput(output)) => {
                Some(inference_result_from_generated(output))
            }
            _ => None,
        };
        Ok(CheckpointedInferenceResult {
            chain_dir: run.chain_dir,
            selected_stage_dir: run.selected_stage_dir,
            final_result,
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
            },
        )?;
        Ok(CheckpointHashChallengeResult {
            chain_dir: verified.run.chain_dir,
            recomputed_trace: verified.recomputed_trace,
            divergence: verified.divergence,
        })
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
