use std::path::PathBuf;

use inference_artifacts::{CASE_TOML, SPECIFICATION_TOML};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirectInferenceConfig {
    pub base_dir: PathBuf,
    pub specification_path: PathBuf,
    /// The case the specification's procedure is run on.
    pub case_path: PathBuf,
    /// Which of the run's inferences to make: the index of its model among
    /// those the specification names. A procedure kind of one inference has only 0.
    pub inference: usize,
}

impl DirectInferenceConfig {
    pub fn from_current_dir() -> anyhow::Result<Self> {
        let base_dir = std::env::current_dir()?;
        Ok(Self {
            specification_path: base_dir.join(SPECIFICATION_TOML),
            case_path: base_dir.join(CASE_TOML),
            base_dir,
            inference: 0,
        })
    }
}
