use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::pipeline::Pipeline;

pub const INFERENCE_RUN_SPEC_TOML: &str = "inference.toml";
pub const PREPARED_RUN_JSON: &str = "prepared_run.json";
pub const PREPARED_RUN_VERSION: u32 = 20;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct InferenceRunSpec {
    /// Which pipeline the run is of. `single` when the spec does not say.
    #[serde(default, skip_serializing_if = "Pipeline::is_default")]
    pub pipeline: Pipeline,
    /// The model of a pipeline that makes one inference.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_manifest: Option<PathBuf>,
    /// The models of a pipeline that makes several, in the order it runs
    /// them: one `[[model]]` table each.
    #[serde(default, rename = "model", skip_serializing_if = "Vec::is_empty")]
    pub models: Vec<ModelSpec>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_file: Option<PathBuf>,
    /// A prompt built from a document a signer vouches for, in place of
    /// `prompt` and `prompt_file`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<SourceSpec>,
    #[serde(default)]
    pub raw_prompt: bool,
    #[serde(default = "default_tokens")]
    pub tokens: u32,
}

/// One model of a run that names several. The path is relative to the run
/// spec.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ModelSpec {
    pub manifest: PathBuf,
}

/// The files of a run with a source. Paths are relative to the run spec.
/// See `docs/proposals/authenticated-prompt.md`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SourceSpec {
    /// How the prompt is made from the documents, and the forms of its output.
    pub recipe: PathBuf,
    /// The public key every attestation must be signed by.
    pub signer: PathBuf,
    /// The documents, in the order the recipe names them.
    pub documents: Vec<SourceDocumentSpec>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SourceDocumentSpec {
    /// The document, as the bytes its attestation is over.
    pub document: PathBuf,
    pub attestation: PathBuf,
}

/// One inference of a prepared run: its model, and the prompt as that model's
/// tokenizer prepares it.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PreparedInference {
    pub model_manifest_path: PathBuf,
    pub model_manifest_sha256: String,
    /// The identifier of the model the inference names.
    pub model_id: String,
    pub prompt: PreparedPrompt,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PreparedRun {
    pub version: u32,
    /// The pipeline the claim is of. A challenger's run spec must name it.
    pub pipeline: Pipeline,
    pub run_spec_path: PathBuf,
    /// The pipeline's inferences, in the order it runs them. Each holds the
    /// run's text, source record and forms, which are the same in all.
    pub inferences: Vec<PreparedInference>,
    pub raw_prompt: bool,
    pub tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub run_manifest_path: Option<PathBuf>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub run_manifest_sha256: Option<String>,
    /// The source stage's inputs, for a run that has a source. A challenger
    /// reruns the stage on these.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_inputs: Option<PreparedSourceInputs>,
}

/// One committed run input: its files and its commitment.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RunFileRef {
    pub path: PathBuf,
    pub index_path: PathBuf,
    pub commitment: String,
}

/// The three inputs of the `prompt-source` stage.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PreparedSourceInputs {
    pub signer: RunFileRef,
    pub recipe: RunFileRef,
    pub documents: RunFileRef,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PreparedPrompt {
    pub resolved_prompt: String,
    pub rendered_prompt: String,
    pub initial_pieces: Vec<PreparedPiece>,
    pub eos_token_ids: Vec<u32>,
    /// What the source step established, for a run whose prompt was built from
    /// signed documents. The first bind is given it, and the statement names
    /// its digest.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<source_core::SourceRecord>,
    /// The texts the result stage compares the output with. Empty for a run
    /// that names none.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub forms: Vec<source_core::Form>,
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
    reject_removed_chain(&text, path)?;
    spec.validate()?;
    Ok(spec)
}

/// Specs may still say `chain = "teacher-forced"`, the only staged chain left.
/// Asking for any other must not silently run this one.
fn reject_removed_chain(text: &str, path: &Path) -> Result<()> {
    let table: toml::Table = toml::from_str(text)
        .with_context(|| format!("failed to parse run spec {}", path.display()))?;
    match table.get("chain").map(toml::Value::as_str) {
        None | Some(Some("teacher-forced")) => Ok(()),
        Some(chain) => anyhow::bail!(
            "run spec {} asks for chain {}, but teacher-forced is the only staged chain; remove `chain`, or use `iris infer` for direct generation",
            path.display(),
            chain.unwrap_or("<not a string>")
        ),
    }
}

impl PreparedRun {
    /// The first inference: the only one of a pipeline that makes one, and
    /// where the run's text, source record and forms are read from.
    pub fn first(&self) -> Result<&PreparedInference> {
        self.inferences
            .first()
            .context("the prepared run holds no inference")
    }
}

impl InferenceRunSpec {
    /// The model manifest of each inference, in the order the pipeline runs
    /// them, as the spec writes the paths.
    pub fn model_manifests(&self) -> Vec<&Path> {
        self.model_manifest
            .iter()
            .map(PathBuf::as_path)
            .chain(self.models.iter().map(|model| model.manifest.as_path()))
            .collect()
    }

    /// The model manifest of one inference.
    pub fn model_manifest_of(&self, inference: usize) -> Result<&Path> {
        self.model_manifests().get(inference).copied().with_context(|| {
            format!(
                "the `{}` pipeline has no inference {inference}",
                self.pipeline
            )
        })
    }

    pub fn validate(&self) -> Result<()> {
        let inferences = self.pipeline.inferences();
        if inferences == 1 {
            anyhow::ensure!(
                self.model_manifest.is_some() && self.models.is_empty(),
                "a run of the `{}` pipeline names its model as `model_manifest`",
                self.pipeline
            );
        } else {
            anyhow::ensure!(
                self.model_manifest.is_none() && self.models.len() == inferences,
                "a run of the `{}` pipeline names its {inferences} models as `[[model]]` \
                 tables, each with a `manifest`, and no `model_manifest`; found {}",
                self.pipeline,
                self.models.len()
            );
        }
        if self.source.is_some() {
            anyhow::ensure!(
                self.prompt.is_none() && self.prompt_file.is_none(),
                "run spec with a `[source]` builds its own prompt; remove `prompt` and `prompt_file`"
            );
            return Ok(());
        }
        match (self.prompt.as_ref(), self.prompt_file.as_ref()) {
            (Some(_), Some(_)) => {
                anyhow::bail!("run spec must set either `prompt` or `prompt_file`, not both")
            }
            (None, None) => {
                anyhow::bail!("run spec must set `prompt`, `prompt_file` or a `[source]`")
            }
            _ => Ok(()),
        }
    }
}

fn default_tokens() -> u32 {
    1
}
