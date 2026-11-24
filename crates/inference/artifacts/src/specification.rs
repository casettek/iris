//! The two files a run is described by.
//!
//! A specification defines a procedure: the procedure kind, the models, the
//! parameters, and for a prompt built from signed documents the source and
//! the forms the output is read by. A case holds one case's own inputs: the
//! prompt taken as given, or the documents the specification's source names.
//!
//! Every stage input is built from the values these hold, and none from a
//! file's bytes.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use source_core::{DocumentName, Field, Form, Recipe, SignerKey};

use crate::procedure_kind::ProcedureKind;

pub const SPECIFICATION_TOML: &str = "specification.toml";
pub const CASE_TOML: &str = "case.toml";

/// `specification.toml`. Paths in it are relative to the file.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Specification {
    /// Which procedure kind the procedure is of. `single` when not given.
    #[serde(default, skip_serializing_if = "ProcedureKind::is_default")]
    pub procedure_kind: ProcedureKind,
    #[serde(default)]
    pub raw_prompt: bool,
    #[serde(default = "default_max_new_tokens")]
    pub max_new_tokens: u32,
    /// The models, in the order the procedure kind runs them: one `[[model]]`
    /// table each.
    #[serde(default, rename = "model")]
    pub models: Vec<SpecificationModel>,
    /// A prompt built from documents a signer vouches for, in place of a
    /// prompt taken as given.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<SpecificationSource>,
    /// The forms the result step compares the output with: one `[[form]]`
    /// table each.
    #[serde(default, rename = "form", skip_serializing_if = "Vec::is_empty")]
    pub forms: Vec<SpecificationForm>,
}

/// One model of a procedure, named by the manifest of its import.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SpecificationModel {
    pub manifest: PathBuf,
}

/// The `[source]` table: the recipe the prompt is built by, with its base
/// prompt as one string, and the public key every attestation must be signed
/// by. See `docs/proposals/authenticated-prompt.md`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SpecificationSource {
    pub base: String,
    #[serde(default)]
    pub tail: String,
    pub signer: SignerKey,
    /// The documents a case must give, in order.
    #[serde(rename = "document")]
    pub documents: Vec<SpecificationDocument>,
    #[serde(rename = "field")]
    pub fields: Vec<SpecificationField>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SpecificationDocument {
    pub server: String,
    pub target: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SpecificationField {
    pub lead: String,
    /// Which of the source's documents the value is in. The first, when not
    /// given.
    #[serde(default)]
    pub document: u32,
    /// The steps from the document's root: a key, or an array index.
    pub path: Vec<PathStep>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(untagged)]
pub enum PathStep {
    Key(String),
    Index(u32),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SpecificationForm {
    pub text: String,
    pub code: u32,
}

impl SpecificationField {
    /// The path as the source step reads it.
    pub fn path(&self) -> String {
        self.path
            .iter()
            .map(|step| match step {
                PathStep::Key(key) => key.clone(),
                PathStep::Index(index) => index.to_string(),
            })
            .collect::<Vec<_>>()
            .join(&source_core::PATH_SEPARATOR.to_string())
    }
}

impl SpecificationSource {
    /// The recipe as the source stage takes it.
    pub fn recipe(&self) -> Recipe {
        Recipe {
            base: source_core::text_chunks(&self.base).into(),
            documents: self
                .documents
                .iter()
                .map(|document| DocumentName {
                    server: document.server.clone(),
                    target: document.target.clone(),
                })
                .collect::<Vec<_>>()
                .into(),
            fields: self
                .fields
                .iter()
                .map(|field| Field {
                    lead: field.lead.clone(),
                    document: field.document,
                    path: field.path(),
                })
                .collect::<Vec<_>>()
                .into(),
            tail: self.tail.clone(),
        }
    }
}

/// `case.toml`: a prompt taken as given, in the file or beside it, or the
/// documents of the specification's source with their attestations. Paths in
/// it are relative to the file.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Case {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_file: Option<PathBuf>,
    /// The documents, in the order the specification's source names them:
    /// one `[[document]]` table each.
    #[serde(default, rename = "document", skip_serializing_if = "Vec::is_empty")]
    pub documents: Vec<CaseDocument>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CaseDocument {
    /// The document, as the bytes its attestation is over.
    pub file: PathBuf,
    pub attestation: PathBuf,
}

/// What replaced each key of the run spec, recipe file and signer file that
/// came before these two files.
const REPLACED_KEYS: [(&str, &str); 6] = [
    (
        "model_manifest",
        "a `[[model]]` table with a `manifest`, for every procedure kind",
    ),
    ("tokens", "`max_new_tokens`"),
    ("prompt", "`prompt` in the case file"),
    ("prompt_file", "`prompt_file` in the case file"),
    ("pipeline", "`procedure_kind`"),
    (
        "chain",
        "nothing: teacher-forced is the only staged chain, and it is not named",
    ),
];

const REPLACED_SOURCE_KEYS: [(&str, &str); 2] = [
    (
        "recipe",
        "the recipe's own `base`, `tail`, `[[source.document]]` and `[[source.field]]` in the \
         `[source]` table, with its forms as `[[form]]` tables",
    ),
    (
        "documents",
        "`[[document]]` tables in the case file, each with a `file` and an `attestation`",
    ),
];

/// A run spec from before the specification is refused by name: the general
/// refusal of an unknown key would not say what to write in its place.
fn reject_run_spec(text: &str, path: &Path) -> Result<()> {
    let Ok(table) = toml::from_str::<toml::Table>(text) else {
        // Not TOML at all: the reader's own error says so.
        return Ok(());
    };
    let refuse = |key: &str, replacement: &str| -> Result<()> {
        anyhow::bail!(
            "{} is a run spec from before the specification and case files: `{key}` is now \
             {replacement}",
            path.display()
        )
    };
    for (key, replacement) in REPLACED_KEYS {
        if table.contains_key(key) {
            refuse(key, replacement)?;
        }
    }
    if let Some(source) = table.get("source").and_then(toml::Value::as_table) {
        for (key, replacement) in REPLACED_SOURCE_KEYS {
            if source.contains_key(key) {
                refuse(key, replacement)?;
            }
        }
        if source.get("signer").is_some_and(toml::Value::is_str) {
            refuse(
                "signer",
                "the `[source.signer]` table, with the key's `scheme` and `key`",
            )?;
        }
    }
    Ok(())
}

pub fn read_specification(path: &Path) -> Result<Specification> {
    let text = fs::read_to_string(path)
        .with_context(|| format!("failed to read specification {}", path.display()))?;
    reject_run_spec(&text, path)?;
    let specification: Specification = toml::from_str(&text)
        .with_context(|| format!("failed to parse specification {}", path.display()))?;
    specification.validate()?;
    Ok(specification)
}

pub fn read_case(path: &Path) -> Result<Case> {
    let text = fs::read_to_string(path)
        .with_context(|| format!("failed to read case {}", path.display()))?;
    let case: Case = toml::from_str(&text)
        .with_context(|| format!("failed to parse case {}", path.display()))?;
    case.validate()?;
    Ok(case)
}

impl Specification {
    /// The model manifest of each inference, in the order the procedure kind
    /// runs them, as the specification writes the paths.
    pub fn model_manifests(&self) -> Vec<&Path> {
        self.models
            .iter()
            .map(|model| model.manifest.as_path())
            .collect()
    }

    /// The model manifest of one inference.
    pub fn model_manifest_of(&self, inference: usize) -> Result<&Path> {
        self.model_manifests()
            .get(inference)
            .copied()
            .with_context(|| {
                format!(
                    "the `{}` procedure kind has no inference {inference}",
                    self.procedure_kind
                )
            })
    }

    /// The forms as the result stage takes them.
    pub fn forms(&self) -> Vec<Form> {
        self.forms
            .iter()
            .map(|form| Form {
                text: form.text.clone(),
                code: form.code,
            })
            .collect()
    }

    pub fn validate(&self) -> Result<()> {
        let inferences = self.procedure_kind.inferences();
        anyhow::ensure!(
            self.models.len() == inferences,
            "a specification of the `{}` procedure kind names its {inferences} model(s) as \
             `[[model]]` tables, each with a `manifest`; found {}",
            self.procedure_kind,
            self.models.len()
        );
        // Forms are read with a source's prompt today. Taking them alone
        // would be a procedure nothing registers yet.
        anyhow::ensure!(
            self.forms.is_empty() || self.source.is_some(),
            "a specification lists `[[form]]` tables only with a `[source]`"
        );
        Ok(())
    }
}

impl Case {
    /// The case of a prompt taken as given.
    pub fn of_prompt(prompt: String) -> Self {
        Self {
            prompt: Some(prompt),
            ..Self::default()
        }
    }

    /// The case alone: it does not give a prompt two ways.
    pub fn validate(&self) -> Result<()> {
        let given = self.prompt.is_some() || self.prompt_file.is_some();
        anyhow::ensure!(
            self.prompt.is_none() || self.prompt_file.is_none(),
            "a case sets either `prompt` or `prompt_file`, not both"
        );
        anyhow::ensure!(
            !given || self.documents.is_empty(),
            "a case with `[[document]]` tables has its prompt built from them; remove `prompt` \
             and `prompt_file`"
        );
        Ok(())
    }

    /// The case as one of this specification's procedure: documents for a
    /// procedure with a source, and a prompt for one without.
    ///
    /// How many documents, and whether they are the ones the source names, is
    /// not held here. The source step holds it, and states a fault a claim
    /// can be made of.
    pub fn validate_for(&self, specification: &Specification) -> Result<()> {
        self.validate()?;
        match &specification.source {
            Some(_) => anyhow::ensure!(
                self.prompt.is_none() && self.prompt_file.is_none(),
                "the specification has a `[source]`, which builds the prompt: the case gives \
                 its documents as `[[document]]` tables, and no `prompt` or `prompt_file`"
            ),
            None => {
                anyhow::ensure!(
                    self.documents.is_empty(),
                    "the case gives documents, and the specification has no `[source]` to read \
                     them by"
                );
                anyhow::ensure!(
                    self.prompt.is_some() || self.prompt_file.is_some(),
                    "a case of a specification with no `[source]` sets `prompt` or `prompt_file`"
                );
            }
        }
        Ok(())
    }
}

fn default_max_new_tokens() -> u32 {
    1
}
