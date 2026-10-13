use std::path::PathBuf;

use inference_artifacts::INFERENCE_RUN_SPEC_TOML;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirectInferenceConfig {
    pub base_dir: PathBuf,
    pub run_spec_path: PathBuf,
}

impl DirectInferenceConfig {
    pub fn from_current_dir() -> anyhow::Result<Self> {
        let base_dir = std::env::current_dir()?;
        Ok(Self {
            run_spec_path: base_dir.join(INFERENCE_RUN_SPEC_TOML),
            base_dir,
        })
    }
}
