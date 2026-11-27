//! A run's inputs as values: what a claimer posts with its claim, and what a
//! challenger runs the claim's stages on.
//!
//! A stage is given three kinds of input: an earlier stage's output, fixed
//! data the procedure pins, and the run's own. The last are these. A
//! claimer writes them as it prepares its run and posts them with the claim.
//! A challenger reads them back and writes the same files from them, so its
//! run of every stage is on the inputs the claimer's was.

use std::fs;
use std::path::{Path, PathBuf};

use crate::RunFile;
use anyhow::{Context, Result};
use procedure_definition::run_inputs::{PostedInference, PostedRun, PostedSourceInputs, RunInput};
use procedure_definition::{
    write_json, Case, CaseDocument, PreparedSourceInputs, RunFileRef, CASE_TOML,
};

/// A run's inputs, beside its frozen manifest and beside a fetched claim.
pub const POSTED_RUN_BIN: &str = "posted_run.bin";

/// The commitment a stage's input list names each of the run's inputs by, in
/// the order of [`PostedRun::values`].
pub fn commitments(run: &PostedRun) -> Result<Vec<String>> {
    run.values()
        .into_iter()
        .map(crate::run_input_commitment)
        .collect()
}

pub fn save(run: &PostedRun, dir: &Path) -> Result<PathBuf> {
    fs::create_dir_all(dir).with_context(|| format!("failed to create {}", dir.display()))?;
    let path = dir.join(POSTED_RUN_BIN);
    fs::write(
        &path,
        postcard::to_allocvec(run).context("the run's inputs do not encode")?,
    )
    .with_context(|| format!("failed to write {}", path.display()))?;
    Ok(path)
}

pub fn load(dir: &Path) -> Result<PostedRun> {
    let path = dir.join(POSTED_RUN_BIN);
    let bytes = fs::read(&path).with_context(|| format!("failed to read {}", path.display()))?;
    postcard::from_bytes(&bytes)
        .with_context(|| format!("{} does not hold a run's inputs", path.display()))
}

/// A value of one crate's type as the same type of another crate. The prompt
/// types are compiled into the stage programs and into the host from one
/// source file, so they are two types with one encoding.
pub fn recast<A: serde::Serialize, B: serde::de::DeserializeOwned>(value: &A) -> Result<B> {
    postcard::from_bytes(&postcard::to_allocvec(value)?).context("a run value does not recast")
}

/// The files of one inference's inputs.
pub struct InferenceRunFiles {
    pub text: Vec<RunFile>,
    pub config: RunFile,
    pub schedule: RunFile,
    pub text_digest: RunFile,
    pub marks: Vec<RunFile>,
    pub claimed: Vec<RunFile>,
}

/// Writes one inference's inputs into `dir`, under the names the pipeline
/// template gives them.
pub fn write_inference(inference: &PostedInference, dir: &Path) -> Result<InferenceRunFiles> {
    let indexed = |values: Vec<RunInput<'_>>, name: &str| -> Result<Vec<RunFile>> {
        values
            .into_iter()
            .enumerate()
            .map(|(index, value)| crate::write_run_input(value, dir, &format!("{name}_{index}")))
            .collect()
    };
    Ok(InferenceRunFiles {
        text: indexed(inference.text.iter().map(RunInput::Text).collect(), "text")?,
        config: crate::write_run_input(RunInput::Config(&inference.config), dir, "config")?,
        schedule: crate::write_run_input(RunInput::Schedule(&inference.schedule), dir, "schedule")?,
        text_digest: crate::write_run_input(
            RunInput::TextDigest(&inference.text_digest),
            dir,
            "text_digest",
        )?,
        marks: indexed(
            inference
                .marks
                .iter()
                .map(|marks| RunInput::Marks(marks))
                .collect(),
            "marks",
        )?,
        claimed: indexed(
            inference
                .claimed
                .iter()
                .map(|ids| RunInput::Claimed(ids))
                .collect(),
            "claimed",
        )?,
    })
}

/// Writes the source stage's inputs into `dir`.
pub fn write_source(source: &PostedSourceInputs, dir: &Path) -> Result<PreparedSourceInputs> {
    let file = |value: RunInput<'_>, name: &str| -> Result<RunFileRef> {
        let file = crate::write_run_input(value, dir, name)?;
        Ok(RunFileRef {
            path: file.path,
            index_path: file.index_path,
            commitment: file.commitment,
        })
    };
    Ok(PreparedSourceInputs {
        signer: file(RunInput::Signer(&source.signer), "source_signer")?,
        recipe: file(RunInput::Recipe(&source.recipe), "source_recipe")?,
        documents: file(RunInput::Documents(&source.documents), "source_documents")?,
    })
}

/// The prompt text a run's slices concatenate to.
pub fn text_of(inference: &PostedInference) -> String {
    inference
        .text
        .iter()
        .flat_map(|slice| slice.chunks.iter().map(String::as_str))
        .collect()
}

/// A posted source's documents as files: each document's bytes, with its
/// attestation.
pub fn documents_of(source: &PostedSourceInputs) -> Vec<(Vec<u8>, &source_phase::Attestation)> {
    source
        .documents
        .attestations
        .iter()
        .enumerate()
        .map(|(index, attestation)| {
            let body = source
                .documents
                .chunks
                .iter()
                .filter(|chunk| chunk.document as usize == index)
                .flat_map(|chunk| chunk.bytes.as_slice().iter().copied())
                .collect();
            (body, attestation)
        })
        .collect()
}

/// A run's inputs as the case a specification of its procedure reads,
/// written into `dir`: the prompt text the slices concatenate to, in the case
/// file itself, or the source's documents with their attestations as files
/// beside it. Returns the case file's path.
///
/// Whether a run has a source is its procedure's, so the reader's
/// specification of that procedure reads this case as the run's own was read.
pub fn write_case(run: &PostedRun, dir: &Path) -> Result<PathBuf> {
    fs::create_dir_all(dir).with_context(|| format!("failed to create {}", dir.display()))?;
    let case = match &run.source {
        Some(posted) => {
            let mut documents = Vec::new();
            for (index, (document, attested)) in documents_of(posted).into_iter().enumerate() {
                let file = PathBuf::from(format!("posted-document-{index}.bin"));
                let body = dir.join(&file);
                fs::write(&body, &document)
                    .with_context(|| format!("failed to write {}", body.display()))?;
                let attestation = PathBuf::from(format!("posted-attestation-{index}.json"));
                write_json(&dir.join(&attestation), attested)?;
                documents.push(CaseDocument { file, attestation });
            }
            Case {
                documents,
                ..Case::default()
            }
        }
        None => Case::of_prompt(text_of(
            run.inferences
                .first()
                .context("the run's inputs are those of no inference")?,
        )),
    };
    let path = dir.join(CASE_TOML);
    fs::write(
        &path,
        toml::to_string(&case).context("the case does not encode")?,
    )
    .with_context(|| format!("failed to write {}", path.display()))?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The values a claim posts give the commitments run preparation's own
    /// writers bind: what a stage's input list names each input by.
    #[test]
    fn posted_values_have_the_commitments_run_preparation_binds() {
        let dir = std::env::temp_dir().join(format!(
            "iwebs-run-inputs-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = fs::remove_dir_all(&dir);
        let marks = vec![vec![1u32, 0, 0, 1], vec![1]];
        let ids = [5u32, 6, 7];
        let text = inference_prompt::prompt_text("hello there");
        let config = inference_prompt::RunConfig {
            model_id: "ab".repeat(32),
            raw_prompt: true,
            max_new_tokens: 3,
            range_size: 2,
        };
        let own_marks = crate::write_marks(&marks, &dir).unwrap();
        let own_claimed = crate::write_claimed_ranges(&ids, 2, &dir).unwrap();
        let own_text = crate::write_text_slices(std::slice::from_ref(&text), &dir).unwrap();
        let own_config = crate::write_run_config(&config, &dir).unwrap();
        let own_schedule = crate::write_run_schedule(2, 2, &dir).unwrap();

        let posted = PostedInference {
            config: recast(&config).unwrap(),
            schedule: inference_tokenizer::RunSchedule {
                prompt_ranges: 2,
                claimed_ranges: 2,
            },
            text_digest: Default::default(),
            text: vec![recast(&text).unwrap()],
            marks,
            claimed: ids.chunks(2).map(<[u32]>::to_vec).collect(),
        };
        assert_eq!(text_of(&posted), "hello there");
        let other = dir.join("posted");
        let files = write_inference(&posted, &other).unwrap();
        let commitments = |files: &[RunFile]| -> Vec<String> {
            files.iter().map(|file| file.commitment.clone()).collect()
        };
        assert_eq!(commitments(&files.marks), commitments(&own_marks));
        assert_eq!(commitments(&files.claimed), commitments(&own_claimed));
        assert_eq!(commitments(&files.text), commitments(&own_text));
        assert_eq!(files.config.commitment, own_config.commitment);
        assert_eq!(files.schedule.commitment, own_schedule.commitment);
        // And the same bytes: a challenger's files are the claimer's.
        assert_eq!(
            fs::read(&files.marks[0].path).unwrap(),
            fs::read(&own_marks[0].path).unwrap()
        );

        // A run's inputs read back as posted, and every one has a commitment.
        let run = PostedRun {
            source: None,
            forms: Default::default(),
            inferences: vec![posted],
            relay: None,
        };
        save(&run, &dir).unwrap();
        assert_eq!(load(&dir).unwrap(), run);
        assert_eq!(super::commitments(&run).unwrap().len(), run.values().len());
        // And as a case: the text its slices concatenate to.
        let case = write_case(&run, &dir.join("case")).unwrap();
        assert_eq!(
            procedure_definition::read_case(&case).unwrap(),
            Case::of_prompt(String::from("hello there"))
        );
        let _ = fs::remove_dir_all(&dir);
    }
}
