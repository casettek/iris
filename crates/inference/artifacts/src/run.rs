use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

pub const INFERENCE_RUN_SPEC_TOML: &str = "inference.toml";
pub const PREPARED_RUN_JSON: &str = "prepared_run.json";
pub const PREPARED_RUN_VERSION: u32 = 2;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct InferenceRunSpec {
    pub model_manifest: PathBuf,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_file: Option<PathBuf>,
    #[serde(default)]
    pub raw_prompt: bool,
    #[serde(default = "default_tokens")]
    pub tokens: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PreparedRun {
    pub version: u32,
    pub run_spec_path: PathBuf,
    pub model_manifest_path: PathBuf,
    pub model_manifest_sha256: String,
    pub prompt: PreparedPrompt,
    pub tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub run_manifest_path: Option<PathBuf>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub run_manifest_sha256: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PreparedPrompt {
    pub resolved_prompt: String,
    pub rendered_prompt: String,
    pub initial_pieces: Vec<PreparedPiece>,
    pub eos_token_ids: Vec<u32>,
}

/// Frozen normalization/segmentation output. Old sentinel-based inputs cannot
/// deserialize as this shape and must be prepared again under the new identity.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PreparedPiece {
    pub text: String,
    pub segment: u32,
}

pub fn read_run_spec(path: &Path) -> Result<InferenceRunSpec> {
    let text = fs::read_to_string(path)
        .with_context(|| format!("failed to read run spec {}", path.display()))?;
    let spec: InferenceRunSpec = toml::from_str(&text)
        .with_context(|| format!("failed to parse run spec {}", path.display()))?;
    spec.validate()?;
    Ok(spec)
}

impl InferenceRunSpec {
    pub fn validate(&self) -> Result<()> {
        match (self.prompt.as_ref(), self.prompt_file.as_ref()) {
            (Some(_), Some(_)) => {
                anyhow::bail!("run spec must set either `prompt` or `prompt_file`, not both")
            }
            (None, None) => anyhow::bail!("run spec must set `prompt` or `prompt_file`"),
            _ => Ok(()),
        }
    }
}

fn default_tokens() -> u32 {
    1
}
