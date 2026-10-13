use std::path::PathBuf;

use anyhow::{Context, Result};
use direct_infer::{DirectInferenceConfig, DirectInferenceExecutor};
use staged_infer::InferenceRunReport;

pub fn run_infer(run_spec_path: PathBuf) -> Result<InferenceRunReport> {
    let mut config =
        DirectInferenceConfig::from_current_dir().context("failed to build infer configuration")?;
    config.run_spec_path = run_spec_path;
    DirectInferenceExecutor.run_with_report(config)
}
