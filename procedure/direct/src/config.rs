use std::path::PathBuf;

use procedure_definition::{CASE_TOML, SPECIFICATION_TOML};

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direct_infer_uses_direct_manifest_by_default() {
        let config = DirectInferenceConfig::from_current_dir().expect("config should build");
        assert_eq!(
            config.specification_path,
            config.base_dir.join(SPECIFICATION_TOML)
        );
        assert_eq!(config.case_path, config.base_dir.join(CASE_TOML));
    }
}
