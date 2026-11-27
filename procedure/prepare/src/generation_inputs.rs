//! The inputs of a generation, kept in a record and read back from one.
//!
//! A generation's inputs are its case's: the prompt taken as given, or the
//! documents with their attestations and the question they are read by. The
//! question is the specification's `[source]` table. It is an input of the
//! run and not part of the procedure's id, so a record carries it, and
//! whoever makes the run again makes it on the record's question.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{ensure, Context, Result};
use procedure_definition::run_record::{GenerationInputs, GENERATION_INPUTS_JSON};
use procedure_definition::{
    read_json, read_specification, write_json, Case, CaseDocument, Specification, CASE_TOML,
    SPECIFICATION_TOML,
};

fn absolute(base: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        base.join(path)
    }
}

fn copy(from: &Path, to: &Path) -> Result<()> {
    fs::copy(from, to)
        .map(|_| ())
        .with_context(|| format!("failed to copy {} to {}", from.display(), to.display()))
}

/// Writes the inputs of a generation of `specification` on `case` into `dir`:
/// `inputs.json`, and the case's documents with their attestations beside it.
pub fn save(
    specification: &Specification,
    case_path: &Path,
    case: &Case,
    dir: &Path,
) -> Result<()> {
    let case_dir = case_path.parent().context("case has no parent directory")?;
    fs::create_dir_all(dir).with_context(|| format!("failed to create {}", dir.display()))?;
    let inputs = match &specification.source {
        Some(source) => {
            let mut documents = Vec::new();
            for (index, document) in case.documents.iter().enumerate() {
                let file = PathBuf::from(format!("document-{index}.bin"));
                let attestation = PathBuf::from(format!("attestation-{index}.json"));
                copy(&absolute(case_dir, &document.file), &dir.join(&file))?;
                copy(
                    &absolute(case_dir, &document.attestation),
                    &dir.join(&attestation),
                )?;
                documents.push(CaseDocument { file, attestation });
            }
            GenerationInputs {
                prompt: None,
                source: Some(source.clone()),
                documents,
            }
        }
        None => GenerationInputs {
            prompt: Some(crate::resolve_prompt(specification, case_dir, case)?.0),
            source: None,
            documents: Vec::new(),
        },
    };
    write_json(&dir.join(GENERATION_INPUTS_JSON), &inputs)
}

/// Writes, into `work_dir`, the specification and the case a generation is
/// made again from: the checker's own specification, naming its own imports
/// of the models, with the question of the record in `record_dir`, and the
/// record's case. Returns the two files' paths.
///
/// Both directories are given as absolute paths.
pub fn restore(
    record_dir: &Path,
    specification_path: &Path,
    work_dir: &Path,
) -> Result<(PathBuf, PathBuf)> {
    let inputs_path = record_dir.join(GENERATION_INPUTS_JSON);
    let inputs: GenerationInputs = read_json(&inputs_path)
        .with_context(|| format!("{} is not a generation's inputs", inputs_path.display()))?;
    let mut specification = read_specification(specification_path)?;
    let specification_dir = specification_path
        .parent()
        .context("specification has no parent directory")?;
    for model in &mut specification.models {
        model.manifest = absolute(specification_dir, &model.manifest);
    }
    match (&specification.source, &inputs.source) {
        (None, None) => {}
        // The signer is the procedure's. The rest of the table is the
        // question, which is the record's.
        (Some(own), Some(given)) => ensure!(
            own.signer == given.signer,
            "the record's evidence is signed for another signer than {} names",
            specification_path.display()
        ),
        (own, _) => anyhow::bail!(
            "the record's case {} evidence, and {} {} a source",
            if inputs.source.is_some() {
                "has"
            } else {
                "has no"
            },
            specification_path.display(),
            if own.is_some() { "has" } else { "has no" }
        ),
    }
    specification.source = inputs.source.clone();
    let case = match inputs.prompt {
        Some(prompt) => Case::of_prompt(prompt),
        None => Case {
            documents: inputs
                .documents
                .iter()
                .map(|document| CaseDocument {
                    file: record_dir.join(&document.file),
                    attestation: record_dir.join(&document.attestation),
                })
                .collect(),
            ..Case::default()
        },
    };
    fs::create_dir_all(work_dir)
        .with_context(|| format!("failed to create {}", work_dir.display()))?;
    let specification_out = work_dir.join(SPECIFICATION_TOML);
    fs::write(
        &specification_out,
        toml::to_string(&specification).context("the specification does not encode")?,
    )
    .with_context(|| format!("failed to write {}", specification_out.display()))?;
    let case_out = work_dir.join(CASE_TOML);
    fs::write(
        &case_out,
        toml::to_string(&case).context("the case does not encode")?,
    )
    .with_context(|| format!("failed to write {}", case_out.display()))?;
    Ok((specification_out, case_out))
}
