//! A run's source, as the files a run spec names.
//!
//! Reading the files is all that happens here. What they mean is in
//! `source-core`, which the stages will run too.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{ensure, Context, Result};
use inference_artifacts::SourceSpec;
use serde::{Deserialize, Serialize};
pub use source_core::{
    fault_description, run_result, Attestation, DocumentName, Field, Form, Recipe, ResultRecord,
    SignerKey, SourceDocument, SourceRecord, SourceRun, FAULT_NONE,
};

/// `recipe.json`: the recipe with its base prompt as one string, and the
/// forms the result step compares the output with.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RecipeFile {
    pub base: String,
    /// The documents the run must give, in order.
    pub documents: Vec<DocumentFile>,
    pub fields: Vec<FieldFile>,
    #[serde(default)]
    pub tail: String,
    #[serde(default)]
    pub forms: Vec<FormFile>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct DocumentFile {
    pub server: String,
    pub target: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct FieldFile {
    pub lead: String,
    /// Which of `documents` the value is in. The first, when not given.
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

impl FieldFile {
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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct FormFile {
    pub text: String,
    pub code: u32,
}

impl RecipeFile {
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

    pub fn forms(&self) -> Vec<Form> {
        self.forms
            .iter()
            .map(|form| Form {
                text: form.text.clone(),
                code: form.code,
            })
            .collect()
    }
}

/// One document a run gives: its bytes and the attestation over them.
#[derive(Debug, Clone)]
pub struct LoadedDocument {
    pub attestation: Attestation,
    pub body: Vec<u8>,
}

/// Everything a run's `[source]` table names, read from disk.
#[derive(Debug, Clone)]
pub struct LoadedSource {
    pub recipe: RecipeFile,
    pub signer: SignerKey,
    pub documents: Vec<LoadedDocument>,
}

fn beside(spec_dir: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        spec_dir.join(path)
    }
}

pub fn read_json<T: serde::de::DeserializeOwned>(path: &Path, what: &str) -> Result<T> {
    let text = fs::read_to_string(path)
        .with_context(|| format!("failed to read {what} {}", path.display()))?;
    serde_json::from_str(&text)
        .with_context(|| format!("failed to parse {what} {}", path.display()))
}

pub fn load_source(spec_dir: &Path, spec: &SourceSpec) -> Result<LoadedSource> {
    let documents = spec
        .documents
        .iter()
        .map(|document| {
            let body = beside(spec_dir, &document.document);
            Ok(LoadedDocument {
                attestation: read_json(&beside(spec_dir, &document.attestation), "attestation")?,
                body: fs::read(&body)
                    .with_context(|| format!("failed to read document {}", body.display()))?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(LoadedSource {
        recipe: read_json(&beside(spec_dir, &spec.recipe), "recipe")?,
        signer: read_json(&beside(spec_dir, &spec.signer), "signer key")?,
        documents,
    })
}

impl LoadedSource {
    /// The documents as the source stage takes them.
    pub fn documents(&self) -> source_core::SourceDocuments {
        let sources: Vec<SourceDocument> = self
            .documents
            .iter()
            .map(|document| SourceDocument {
                attestation: &document.attestation,
                body: &document.body,
            })
            .collect();
        source_core::source_documents(&sources)
    }

    /// The source step over these files: its record, with any fault in it,
    /// and the prompt text.
    pub fn run(&self) -> SourceRun {
        source_core::run_source_steps(&self.signer, &self.recipe.recipe(), &self.documents())
    }

    /// The result step over an output text.
    pub fn result(&self, output: &str) -> ResultRecord {
        run_result(&self.recipe.forms(), output)
    }
}

/// The prompt of a run with a source. A source with a fault gives no prompt:
/// direct inference is the gate, and no claim is built on such a run.
pub fn sourced_prompt(spec_dir: &Path, spec: &SourceSpec) -> Result<String> {
    Ok(sourced_run(spec_dir, spec)?.text)
}

/// The same, with the record the first bind is given.
pub fn sourced_run(spec_dir: &Path, spec: &SourceSpec) -> Result<SourceRun> {
    let run = load_source(spec_dir, spec)?.run();
    ensure!(
        run.record.fault == FAULT_NONE,
        "the run's source has fault {} at {}: {}",
        run.record.fault,
        run.record.fault_position,
        fault_description(run.record.fault)
    );
    Ok(run)
}
