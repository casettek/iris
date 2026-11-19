//! A claim and a challenge as the other parties meet them: read back from
//! the data-availability layer, with nothing taken from the directory they
//! were built in.
//!
//! A challenger has the configuration and the models. `fetch_claim` gives it
//! the rest: the claim's checkpoint list, checkpoints and input manifests,
//! and a run spec over the new information the claim posted. A defender has
//! its own run of the disputed stage. `fetch_challenge` gives it the trace
//! the challenge names.
//!
//! `corrupt_claim` is for tests of that path: it makes the claim a lying
//! claimer would submit, which states another output for one checkpoint and
//! posts its own list.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{bail, ensure, Context, Result};
use inference_artifacts::availability::{
    check_holds, Blob, BlobContents, BlobEntry, CheckpointPackage, RequiredPosting,
};
use inference_artifacts::contract::{
    is_address, source_record_id, ChallengeSubmission, ClaimSubmission, CLAIM_SUBMISSION_JSON,
};
use inference_artifacts::{
    checkpoint_hashes, checkpoints_root, read_checkpoint_hashes, read_checkpoint_trace, read_json,
    read_run_spec, write_checkpoint_hashes, write_json, CheckpointTrace, SourceDocumentSpec,
    CHECKPOINTS_TXT, CHECKPOINT_TRACE_JSON, INFERENCE_RUN_SPEC_TOML,
};
use sha2::{Digest, Sha256};

use crate::availability::{contents_of, open_layer, trace_header_digest, PreparedPosting};
use crate::claim::{corrupt_hash, select_checkpoint_index, CorruptCheckpointSelection};

/// The file `corrupt_claim` writes beside the lying claim: which checkpoint
/// it changed, in the form `claim corrupt --checkpoints` writes.
pub const CORRUPTION_JSON: &str = "checkpoints.corruption.json";

fn absolute(base: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        base.join(path)
    }
}

fn parent(path: &Path) -> Result<&Path> {
    path.parent()
        .with_context(|| format!("{} has no parent directory", path.display()))
}

/// A directory to write into that holds nothing yet, so that what is read
/// from the layer is never mixed with files of another origin.
fn fresh_dir(dir: &Path) -> Result<()> {
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
pub struct CorruptClaimOptions {
    /// The honest claim's `claim_submission.json`, in its directory.
    pub submission_path: PathBuf,
    pub selection: CorruptCheckpointSelection,
    /// The address the lying claim is attributed to. With none, the honest
    /// claim's.
    pub claimer: Option<String>,
    pub out_dir: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CorruptedClaim {
    pub submission_path: PathBuf,
    pub hashes_path: PathBuf,
    pub manifest_path: PathBuf,
    pub checkpoint_index: usize,
    pub stage: String,
    pub original_hash: String,
    pub corrupted_hash: String,
}

/// The claim a lying claimer submits in place of an honest one: the same run
/// and the same results, with another output stated for one checkpoint.
///
/// A checkpoint list is posted with the checkpoints it names, so a changed
/// hash alone is no claim: the lie is a checkpoint that states another
/// output, and the list and its root follow from it. The directory written
/// is ready for `claim post`: the submission, the list, the checkpoints, and
/// the honest claim's blobs with the lying list and checkpoints in place of
/// the honest ones.
pub fn corrupt_claim(options: CorruptClaimOptions) -> Result<CorruptedClaim> {
    let honest: ClaimSubmission = read_json(&options.submission_path)?;
    let dir = parent(&options.submission_path)?;
    let honest_hashes_path = dir.join(CHECKPOINTS_TXT);
    let honest_hashes = read_checkpoint_hashes(&honest_hashes_path)?;
    let mut trace = read_checkpoint_trace(&dir.join(CHECKPOINT_TRACE_JSON))?;
    ensure!(
        honest_hashes.len() == trace.checkpoints.len(),
        "{} lists {} checkpoints, and {CHECKPOINT_TRACE_JSON} holds {}",
        honest_hashes_path.display(),
        honest_hashes.len(),
        trace.checkpoints.len()
    );
    let index = select_checkpoint_index(&options.selection, &honest_hashes, Some(&trace))?;
    let stage = trace.checkpoints[index].stage.clone();
    trace.checkpoints[index].output_sha256 = corrupt_hash(&trace.checkpoints[index].output_sha256);
    let hashes = checkpoint_hashes(&trace)?;

    let claimer = options.claimer.unwrap_or_else(|| honest.claimer.clone());
    ensure!(
        is_address(&claimer),
        "`{claimer}` is not an address: give `0x` and 40 lowercase hex digits"
    );
    let mut claim = honest;
    claim.claimer = claimer.clone();
    claim.checkpoint_count = hashes.len() as u32;
    claim.checkpoints_root = checkpoints_root(&hashes)?;
    claim.availability.clear();

    fresh_dir(&options.out_dir)?;
    let hashes_path = options.out_dir.join(CHECKPOINTS_TXT);
    write_checkpoint_hashes(&hashes_path, &hashes)?;
    let trace_path = options.out_dir.join(CHECKPOINT_TRACE_JSON);
    write_json(&trace_path, &trace)?;
    let posting = PreparedPosting::read(dir)?.with_checkpoints(
        &claimer,
        &fs::read(&hashes_path)?,
        &fs::read(&trace_path)?,
    )?;
    // The proof of a claim's checkpoints reads its records out of the last
    // stage's output and the checks', each held to its checkpoint's hash. A
    // lie about one of those has no output behind it, so no proof states it
    // and no contract takes the claim.
    let contents = posting.contents().with_context(|| {
        format!(
            "checkpoint {index} (`{stage}`) is one the claim's records are read from: a claim \
             that states another output for it is refused when it arrives, so there is nothing \
             to challenge"
        )
    })?;
    check_holds(&contents, &RequiredPosting::of(&claim)).map_err(|rejection| {
        anyhow::anyhow!("the lying claim's data is not its own: {rejection}")
    })?;
    posting.write(&options.out_dir)?;
    let submission_path = options.out_dir.join(CLAIM_SUBMISSION_JSON);
    write_json(&submission_path, &claim)?;

    let manifest_path = options.out_dir.join(CORRUPTION_JSON);
    write_json(
        &manifest_path,
        &serde_json::json!({
            "version": 1,
            "source_hashes_path": honest_hashes_path,
            "corrupted_hashes_path": hashes_path,
            "checkpoint_index": index,
            "checkpoint_number": index + 1,
            "stage": stage,
            "original_hash": honest_hashes[index],
            "corrupted_hash": hashes[index],
        }),
    )?;
    Ok(CorruptedClaim {
        submission_path,
        hashes_path,
        manifest_path,
        checkpoint_index: index,
        stage,
        original_hash: honest_hashes[index].clone(),
        corrupted_hash: hashes[index].clone(),
    })
}

#[derive(Debug, Clone)]
pub struct FetchClaimOptions {
    /// The claim's `claim_submission.json`, as it was submitted: with where
    /// its data is.
    pub submission_path: PathBuf,
    /// The reader's own run spec: the pipeline, the models and the
    /// parameters of the configuration the claim names. Its prompt, and the
    /// documents of its source, are replaced by the ones the claim posted.
    pub run_spec_path: PathBuf,
    pub pins_path: Option<PathBuf>,
    pub da_dir: Option<PathBuf>,
    pub out_dir: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FetchedClaim {
    pub submission_path: PathBuf,
    pub run_spec_path: PathBuf,
    pub blobs: usize,
    pub bytes: u64,
    pub checkpoints: usize,
    pub input_manifests: usize,
}

/// A stage name as a directory name. The names in a posted claim are the
/// claimer's, and the directory is the reader's.
fn stage_dir_name(stage: &str) -> Result<&str> {
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
/// everything a challenger needs that its configuration and its models do
/// not fix.
///
/// The directory holds the submission, the claim's `checkpoints.txt` and
/// `checkpoint_trace.json`, every stage's `input_manifest.json`, and
/// `inference.toml`: the reader's run spec over the prompt text, or the
/// source's documents, that the claim posted. Nothing is written unless the
/// blobs are the claimer's and hold the new information the statement names
/// and the checkpoint list the claim names.
pub fn fetch_claim(options: FetchClaimOptions) -> Result<FetchedClaim> {
    let submission: ClaimSubmission = read_json(&options.submission_path)?;
    ensure!(
        !submission.availability.is_empty(),
        "{} names no posted data: the claim was not posted",
        options.submission_path.display()
    );
    let configuration =
        crate::contract::spec_configuration(&options.run_spec_path, options.pins_path.as_deref())?;
    ensure!(
        configuration.id() == submission.configuration,
        "the claim is of configuration {}, and {} describes {}",
        submission.configuration,
        options.run_spec_path.display(),
        configuration.id()
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

    let spec_dir = parent(&options.run_spec_path)?;
    let mut spec = read_run_spec(&options.run_spec_path)?;
    spec.model_manifest = spec
        .model_manifest
        .map(|manifest| absolute(spec_dir, &manifest));
    for model in &mut spec.models {
        model.manifest = absolute(spec_dir, &model.manifest);
    }
    spec.prompt = None;
    spec.prompt_file = None;

    fresh_dir(&options.out_dir)?;
    let out = fs::canonicalize(&options.out_dir)
        .with_context(|| format!("failed to resolve {}", options.out_dir.display()))?;
    let sourced = submission.source_sha256 != inference_artifacts::inference::no_source_id();
    if sourced {
        let posted = entries
            .iter()
            .find_map(|entry| match entry {
                BlobEntry::Source(source)
                    if source_record_id(&source.record()) == submission.source_sha256 =>
                {
                    Some(source)
                }
                _ => None,
            })
            .context("the claim's source is not among what it posted")?;
        // The recipe and the signer's key are the configuration's, so they
        // stay the reader's own. The documents are the new information.
        let source = spec.source.as_mut().with_context(|| {
            format!(
                "the claim has a source, and {} names none: the recipe and the signer's key \
                 are the configuration's",
                options.run_spec_path.display()
            )
        })?;
        source.recipe = absolute(spec_dir, &source.recipe);
        source.signer = absolute(spec_dir, &source.signer);
        source.documents.clear();
        for (index, document) in posted.documents.iter().enumerate() {
            let body = out.join(format!("posted-document-{index}.bin"));
            fs::write(&body, &document.body)
                .with_context(|| format!("failed to write {}", body.display()))?;
            let attestation = out.join(format!("posted-attestation-{index}.json"));
            write_json(&attestation, &document.attestation)?;
            source.documents.push(SourceDocumentSpec {
                document: body,
                attestation,
            });
        }
    } else {
        let wanted = &submission.input_text_sha256;
        let text = entries
            .iter()
            .find_map(|entry| match entry {
                BlobEntry::PromptText { text }
                    if hex::encode(Sha256::digest(text.as_bytes()))
                        .eq_ignore_ascii_case(wanted) =>
                {
                    Some(text)
                }
                _ => None,
            })
            .context("the claim's prompt text is not among what it posted")?;
        spec.source = None;
        // In the spec itself, so that the text is the posted one exactly.
        spec.prompt = Some(text.clone());
    }

    write_package(&out, package)?;
    let submission_path = out.join(CLAIM_SUBMISSION_JSON);
    write_json(&submission_path, &submission)?;
    let run_spec_path = out.join(INFERENCE_RUN_SPEC_TOML);
    fs::write(
        &run_spec_path,
        toml::to_string(&spec).context("the run spec does not encode")?,
    )
    .with_context(|| format!("failed to write {}", run_spec_path.display()))?;
    let trace: CheckpointTrace = serde_json::from_slice(&package.trace)?;
    Ok(FetchedClaim {
        submission_path,
        run_spec_path,
        blobs: submission.availability.len(),
        bytes: bytes_read,
        checkpoints: trace.checkpoints.len(),
        input_manifests: package.input_manifests.len(),
    })
}

/// A checkpoint package as the files of a claim's directory: the list, the
/// checkpoints, and each stage's input manifest in the stage's directory.
fn write_package(dir: &Path, package: &CheckpointPackage) -> Result<()> {
    let write = |path: PathBuf, bytes: &[u8]| {
        fs::write(&path, bytes).with_context(|| format!("failed to write {}", path.display()))
    };
    write(dir.join(CHECKPOINTS_TXT), &package.list)?;
    write(dir.join(CHECKPOINT_TRACE_JSON), &package.trace)?;
    let manifests: BTreeMap<String, &Vec<u8>> = package
        .input_manifests
        .iter()
        .map(|manifest| (hex::encode(Sha256::digest(manifest)), manifest))
        .collect();
    let trace: CheckpointTrace = serde_json::from_slice(&package.trace)?;
    for checkpoint in &trace.checkpoints {
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
