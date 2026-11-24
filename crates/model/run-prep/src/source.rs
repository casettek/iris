//! A run's source, as the specification and the case give it.
//!
//! Reading the case's files is all that happens here. What they mean is in
//! `source-core`, which the stages will run too.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{ensure, Context, Result};
use inference_artifacts::{Case, Specification, SpecificationForm, SpecificationSource};
pub use source_core::{
    fault_description, forms_id, run_forms, Attestation, DocumentName, Field, Form, Recipe,
    SignerKey, SourceDocument, SourceRecord, SourceRun, FAULT_NONE,
};

/// One document a run gives: its bytes and the attestation over them.
#[derive(Debug, Clone)]
pub struct LoadedDocument {
    pub attestation: Attestation,
    pub body: Vec<u8>,
}

/// A specification's source and forms, with a case's documents read from
/// disk.
#[derive(Debug, Clone)]
pub struct LoadedSource {
    /// The specification's `[source]` table: the recipe and the signer's key.
    pub specification: SpecificationSource,
    /// The specification's forms.
    pub forms: Vec<SpecificationForm>,
    pub documents: Vec<LoadedDocument>,
}

fn beside(case_dir: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        case_dir.join(path)
    }
}

pub fn read_json<T: serde::de::DeserializeOwned>(path: &Path, what: &str) -> Result<T> {
    let text = fs::read_to_string(path)
        .with_context(|| format!("failed to read {what} {}", path.display()))?;
    serde_json::from_str(&text)
        .with_context(|| format!("failed to parse {what} {}", path.display()))
}

/// The source of a specification that has one, over a case's documents.
/// `case_dir` is the directory the case's paths are relative to.
pub fn load_source(
    specification: &Specification,
    case_dir: &Path,
    case: &Case,
) -> Result<Option<LoadedSource>> {
    case.validate_for(specification)?;
    let Some(source) = &specification.source else {
        return Ok(None);
    };
    let documents = case
        .documents
        .iter()
        .map(|document| {
            let body = beside(case_dir, &document.file);
            Ok(LoadedDocument {
                attestation: read_json(&beside(case_dir, &document.attestation), "attestation")?,
                body: fs::read(&body)
                    .with_context(|| format!("failed to read document {}", body.display()))?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Some(LoadedSource {
        specification: source.clone(),
        forms: specification.forms.clone(),
        documents,
    }))
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
        source_core::run_source_steps(
            &self.specification.signer,
            &self.specification.recipe(),
            &self.documents(),
        )
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

    /// The result step over an output text.
    pub fn result_code(&self, output: &str) -> u32 {
        source_core::run_forms(&self.forms(), output).code
    }
}

/// What a specification's source fixes before any document is fetched: who
/// must have signed the documents, the recipe the prompt is built by, and the
/// forms the output is read by.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceTerms {
    /// As in [`source_core::SourceRecord::signer`].
    pub signer: String,
    /// As in [`source_core::SourceRecord::recipe_sha256`].
    pub recipe_sha256: String,
    /// The forms the output is read by, in order.
    pub forms: Vec<Form>,
}

/// The terms of a specification with a source. None for one without.
pub fn source_terms(specification: &Specification) -> Option<SourceTerms> {
    let source = specification.source.as_ref()?;
    // The recipe's hash is over the recipe alone, so the source step over no
    // documents states it.
    let record = source_core::run_source_steps(
        &source.signer,
        &source.recipe(),
        &source_core::source_documents(&[]),
    )
    .record;
    Some(SourceTerms {
        signer: source.signer.key.clone(),
        recipe_sha256: record.recipe_sha256,
        forms: specification.forms(),
    })
}

/// The forms hash of a run that names no forms.
pub fn no_forms_id() -> String {
    forms_id(&run_forms(&[], ""))
}

/// The prompt of a run with a source. A source with a fault gives no prompt:
/// direct inference is the gate, and no claim is built on such a run.
pub fn sourced_prompt(source: &LoadedSource) -> Result<String> {
    Ok(sourced_run(source)?.text)
}

/// The same, with the record the first bind is given.
pub fn sourced_run(source: &LoadedSource) -> Result<SourceRun> {
    let run = source.run();
    ensure!(
        run.record.fault == FAULT_NONE,
        "the run's source has fault {} at {}: {}",
        run.record.fault,
        run.record.fault_position,
        fault_description(run.record.fault)
    );
    Ok(run)
}
