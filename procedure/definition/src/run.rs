use std::path::PathBuf;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::procedure_kind::ProcedureKind;

pub const PREPARED_RUN_JSON: &str = "prepared_run.json";
pub const PREPARED_RUN_VERSION: u32 = 22;

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
    /// The procedure kind the claim is of. A challenger's specification must name it.
    pub procedure_kind: ProcedureKind,
    pub specification_path: PathBuf,
    /// The case file the run was prepared from. Absent in the run a
    /// challenger rebuilds from a claimer's prepared run, whose case is the
    /// prompt that run holds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub case_path: Option<PathBuf>,
    /// The procedure kind's inferences, in the order it runs them. Each holds its
    /// prompt's text and the record of what that text was built from: the
    /// run's own, or for an inference a relay gives its prompt, the relay's.
    /// The forms are the same in all.
    pub inferences: Vec<PreparedInference>,
    pub raw_prompt: bool,
    pub max_new_tokens: u32,
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
    pub source: Option<source_phase::SourceRecord>,
    /// The texts the result stage compares the output with. Empty for a run
    /// that names none.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub forms: Vec<result_phase::Form>,
}

pub use inference_prompt::PreparedPiece;

impl PreparedRun {
    /// The first inference: the only one of a procedure kind that makes one, and
    /// where the run's text, source record and forms are read from.
    pub fn first(&self) -> Result<&PreparedInference> {
        self.inferences
            .first()
            .context("the prepared run holds no inference")
    }
}
