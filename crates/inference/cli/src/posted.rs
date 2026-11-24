//! A claim and a challenge as the other parties meet them: read back from
//! the data-availability layer, with nothing taken from the directory they
//! were built in.
//!
//! A challenger has the procedure and the models. [`fetch_claim`] gives
//! it the rest: the claim's checkpoint hashes, checkpoints and input
//! manifests, and the inputs the claim's run was given, with the case they
//! hold. A defender has its own run of the disputed stage.
//! [`fetch_challenge`] gives it the trace the challenge names.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{bail, ensure, Context, Result};
use inference_artifacts::availability::{
    check_holds, Blob, BlobContents, BlobEntry, CheckpointPackage, RequiredPosting,
};
use inference_artifacts::checks::{ChallengeSubmission, ClaimSubmission, CLAIM_SUBMISSION_JSON};
use inference_artifacts::{
    read_json, write_json, Case, CaseDocument, CheckpointList, CASE_TOML, CHECKPOINTS_TXT,
    CHECKPOINT_LIST_JSON,
};
use sha2::{Digest, Sha256};

use crate::posting::{contents_of, open_layer, trace_header_digest};

/// A stage's input list, in the stage's directory.
pub(crate) const INPUT_MANIFEST_JSON: &str = "input_manifest.json";

pub(crate) fn parent(path: &Path) -> Result<&Path> {
    path.parent()
        .with_context(|| format!("{} has no parent directory", path.display()))
}

/// A directory to write into that holds nothing yet, so that what is read
/// from the layer is never mixed with files of another origin.
pub(crate) fn fresh_dir(dir: &Path) -> Result<()> {
    if dir.exists() {
        ensure!(
            fs::read_dir(dir)
                .with_context(|| format!("failed to read {}", dir.display()))?
                .next()
                .is_none(),
            "{} is not empty: give a new directory",
            dir.display()
        );
    }
    fs::create_dir_all(dir).with_context(|| format!("failed to create {}", dir.display()))
}

#[derive(Debug, Clone)]
pub struct FetchClaimOptions {
    /// The claim's `claim_submission.json`, as it was submitted: with where
    /// its data is.
    pub submission_path: PathBuf,
    /// The reader's own specification of the procedure the claim names. The
    /// claim's procedure is held to it before anything is read.
    pub specification_path: PathBuf,
    pub pins_path: Option<PathBuf>,
    pub da_dir: Option<PathBuf>,
    pub out_dir: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FetchedClaim {
    pub submission_path: PathBuf,
    /// The claim's case, as the reader's specification runs it.
    pub case_path: PathBuf,
    pub blobs: usize,
    pub bytes: u64,
    pub checkpoints: usize,
    pub input_manifests: usize,
}

/// A stage name as a directory name. The names in a posted claim are the
/// claimer's, and the directory is the reader's.
pub(crate) fn stage_dir_name(stage: &str) -> Result<&str> {
    ensure!(
        !stage.is_empty()
            && stage
                .chars()
                .all(|ch| ch.is_ascii_alphanumeric() || ch == '_' || ch == '-'),
        "the claim names a stage `{stage}`, which is no stage name"
    );
    Ok(stage)
}

/// Read a submitted claim's data back from the layer into a new directory:
/// everything a challenger needs that its procedure and its models do
/// not fix.
///
/// The directory holds the submission, the claim's `checkpoints.txt` and
/// `checkpoint_list.json`, every stage's `input_manifest.json`, the inputs
/// the claim's run was given (`posted_run.bin`), and `case.toml`: the prompt
/// text, or the source's documents, that those inputs hold. The reader runs
/// it under its own specification. Nothing is written unless the blobs are
/// the claimer's and hold the checkpoint list the claim names and the inputs
/// of one run.
pub fn fetch_claim(options: FetchClaimOptions) -> Result<FetchedClaim> {
    let submission: ClaimSubmission = read_json(&options.submission_path)?;
    ensure!(
        !submission.availability.is_empty(),
        "{} names no posted data: the claim was not posted",
        options.submission_path.display()
    );
    let procedure = crate::claim_check::spec_procedure(
        &options.specification_path,
        options.pins_path.as_deref(),
    )?;
    ensure!(
        procedure.id() == submission.procedure,
        "the claim is of procedure {}, and {} describes {}",
        submission.procedure,
        options.specification_path.display(),
        procedure.id()
    );

    let layer = open_layer(options.da_dir.as_deref())?;
    let mut held = BlobContents::default();
    let mut entries = Vec::new();
    let mut bytes_read = 0u64;
    for (index, posted) in submission.availability.iter().enumerate() {
        let bytes = layer
            .read(&posted.pointer)
            .with_context(|| format!("blob {index} could not be read from the layer"))?;
        ensure!(
            bytes.len() as u64 == posted.length,
            "blob {index} is {} bytes, and the claim states {}",
            bytes.len(),
            posted.length
        );
        // Read with no outputs: a challenger makes its own, and needs none
        // of the claimer's to know what the blob holds.
        let contents = contents_of(&bytes, &[])
            .with_context(|| format!("blob {index} is not a blob of a claim"))?;
        ensure!(
            contents.submitter == submission.claimer,
            "blob {index} was posted for {}, and the claim is {}'s",
            contents.submitter,
            submission.claimer
        );
        held.submitter = contents.submitter.clone();
        held.extend(&contents);
        entries.extend(
            Blob::decode(&bytes)
                .map_err(|fault| anyhow::anyhow!("blob {index}: {fault}"))?
                .entries,
        );
        bytes_read += posted.length;
    }
    check_holds(&held, &RequiredPosting::of(&submission)).map_err(|rejection| {
        anyhow::anyhow!("what the claim posted is not its data: {rejection}")
    })?;

    let package = entries
        .iter()
        .find_map(|entry| match entry {
            BlobEntry::Checkpoints(package)
                if package.check().ok().as_deref() == Some(&submission.checkpoints_root) =>
            {
                Some(package)
            }
            _ => None,
        })
        .context("the claim's checkpoint list is not among what it posted")?;

    // The inputs the claim's run was given. A claim posts those of one run.
    let mut runs = entries.iter().filter_map(|entry| match entry {
        BlobEntry::Run(run) => Some(run),
        _ => None,
    });
    let run = runs
        .next()
        .context("the claim posted no inputs of its run: there is nothing to run its stages on")?;
    ensure!(
        runs.next().is_none(),
        "the claim posted the inputs of more than one run"
    );
    let first = run
        .inferences
        .first()
        .context("the claim posted the inputs of a run of no inference")?;

    fresh_dir(&options.out_dir)?;
    let out = fs::canonicalize(&options.out_dir)
        .with_context(|| format!("failed to resolve {}", options.out_dir.display()))?;
    crate::run_inputs::save(run, &out)?;
    // The case: what the claim posted, as the files a specification of its
    // procedure reads. Whether the claim has a source is the procedure's,
    // which was held to the reader's specification above.
    let case = match &run.source {
        Some(posted) => {
            let mut documents = Vec::new();
            for (index, (document, attested)) in crate::run_inputs::documents_of(posted)
                .into_iter()
                .enumerate()
            {
                let file = PathBuf::from(format!("posted-document-{index}.bin"));
                let body = out.join(&file);
                fs::write(&body, &document)
                    .with_context(|| format!("failed to write {}", body.display()))?;
                let attestation = PathBuf::from(format!("posted-attestation-{index}.json"));
                write_json(&out.join(&attestation), attested)?;
                documents.push(CaseDocument { file, attestation });
            }
            Case {
                documents,
                ..Case::default()
            }
        }
        // In the case itself: the text the posted slices concatenate to.
        None => Case::of_prompt(crate::run_inputs::text_of(first)),
    };

    write_package(&out, package)?;
    let submission_path = out.join(CLAIM_SUBMISSION_JSON);
    write_json(&submission_path, &submission)?;
    let case_path = out.join(CASE_TOML);
    fs::write(
        &case_path,
        toml::to_string(&case).context("the case does not encode")?,
    )
    .with_context(|| format!("failed to write {}", case_path.display()))?;
    let list: CheckpointList = serde_json::from_slice(&package.list)?;
    Ok(FetchedClaim {
        submission_path,
        case_path,
        blobs: submission.availability.len(),
        bytes: bytes_read,
        checkpoints: list.checkpoints.len(),
        input_manifests: package.input_manifests.len(),
    })
}

/// A checkpoint package as the files of a claim's directory: the hashes, the
/// checkpoints, and each stage's input manifest in the stage's directory.
fn write_package(dir: &Path, package: &CheckpointPackage) -> Result<()> {
    let write = |path: PathBuf, bytes: &[u8]| {
        fs::write(&path, bytes).with_context(|| format!("failed to write {}", path.display()))
    };
    write(dir.join(CHECKPOINTS_TXT), &package.hashes)?;
    write(dir.join(CHECKPOINT_LIST_JSON), &package.list)?;
    let manifests: BTreeMap<String, &Vec<u8>> = package
        .input_manifests
        .iter()
        .map(|manifest| (hex::encode(Sha256::digest(manifest)), manifest))
        .collect();
    let list: CheckpointList = serde_json::from_slice(&package.list)?;
    for checkpoint in &list.checkpoints {
        let manifest = manifests
            .get(&checkpoint.input_commitment)
            .with_context(|| {
                format!(
                    "the claim posted no input manifest for `{}`",
                    checkpoint.stage
                )
            })?;
        let stage_dir = dir.join(stage_dir_name(&checkpoint.stage)?);
        fs::create_dir_all(&stage_dir)
            .with_context(|| format!("failed to create {}", stage_dir.display()))?;
        write(stage_dir.join("input_manifest.json"), manifest)?;
    }
    Ok(())
}

#[derive(Debug, Clone)]
pub struct FetchChallengeOptions {
    /// The challenge's `challenge_submission.json`, as it was submitted.
    pub challenge_path: PathBuf,
    pub da_dir: Option<PathBuf>,
    /// Where to write the trace commitment (`commit.bin`).
    pub out_path: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FetchedChallenge {
    pub commitment_path: PathBuf,
    pub bytes: u64,
    pub header_digest: String,
}

/// Read the trace a submitted challenge names back from the layer: the
/// commitment a defender audits its own run of the stage against. It is
/// written only when it is the challenger's and its header is the one the
/// challenge names.
pub fn fetch_challenge(options: FetchChallengeOptions) -> Result<FetchedChallenge> {
    let submission: ChallengeSubmission = read_json(&options.challenge_path)?;
    ensure!(
        !submission.availability.is_empty(),
        "{} names no posted data: the challenge was not posted",
        options.challenge_path.display()
    );
    let wanted = submission.trace_header_digest();
    let layer = open_layer(options.da_dir.as_deref())?;
    for (index, posted) in submission.availability.iter().enumerate() {
        let bytes = layer
            .read(&posted.pointer)
            .with_context(|| format!("blob {index} could not be read from the layer"))?;
        let blob =
            Blob::decode(&bytes).map_err(|fault| anyhow::anyhow!("blob {index}: {fault}"))?;
        ensure!(
            blob.submitter == submission.challenger,
            "blob {index} was posted for {}, and the challenge is {}'s",
            blob.submitter,
            submission.challenger
        );
        for entry in &blob.entries {
            let BlobEntry::TraceBody { bytes: body } = entry else {
                continue;
            };
            if trace_header_digest(body).as_deref() != Some(wanted.as_str()) {
                continue;
            }
            if let Some(dir) = options.out_path.parent() {
                fs::create_dir_all(dir)
                    .with_context(|| format!("failed to create {}", dir.display()))?;
            }
            ensure!(
                !options.out_path.exists(),
                "{} already exists",
                options.out_path.display()
            );
            fs::write(&options.out_path, body)
                .with_context(|| format!("failed to write {}", options.out_path.display()))?;
            // As Raster reads a commitment file, which is this encoding.
            raster_prover::commitment_file::CommitmentFile::read_memory(&options.out_path)
                .with_context(|| {
                    format!(
                        "{} is not a Raster trace commitment",
                        options.out_path.display()
                    )
                })?;
            return Ok(FetchedChallenge {
                commitment_path: options.out_path,
                bytes: body.len() as u64,
                header_digest: wanted,
            });
        }
    }
    bail!("no blob the challenge posted holds the trace it names ({wanted})")
}
