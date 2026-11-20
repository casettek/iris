//! A claim and a challenge as the other parties meet them: read back from
//! the data-availability layer, with nothing taken from the directory they
//! were built in.
//!
//! A challenger has the configuration and the models. `fetch_claim` gives it
//! the rest: the claim's checkpoint list, checkpoints and input manifests,
//! and the inputs the claim's run was given, with a run spec over them. A
//! defender has its own run of the disputed stage. `fetch_challenge` gives it
//! the trace the challenge names.
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
    is_address, ChallengeSubmission, ClaimSubmission, CLAIM_SUBMISSION_JSON,
};
use inference_artifacts::{
    checkpoint_hashes, checkpoints_root, read_checkpoint_hashes, read_checkpoint_trace, read_json,
    read_run_spec, write_checkpoint_hashes, write_json, CheckpointTrace, SourceDocumentSpec,
    CHECKPOINTS_TXT, CHECKPOINT_TRACE_JSON, INFERENCE_RUN_SPEC_TOML,
};
use sha2::{Digest, Sha256};

use crate::availability::{contents_of, open_layer, trace_header_digest, PreparedPosting};
use crate::claim::{corrupt_hash, select_checkpoint_index, CorruptCheckpointSelection};

/// A stage's input list, in the stage's directory.
const INPUT_MANIFEST_JSON: &str = "input_manifest.json";

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

/// What a lying claim says falsely of the checkpoint it lies about.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum CorruptFault {
    /// Another output than the stage's: a lie about the stage's execution,
    /// which a replay of the stage shows.
    #[default]
    Output,
    /// A program outside the approved set.
    Program,
    /// An input that is not from where the configuration says it comes: not
    /// what its producer made, not a pinned commitment, or not a value the
    /// claim posted, by the kind of input. The named input, or the first the
    /// checkpoint names a producer for.
    Input(Option<String>),
}

#[derive(Debug, Clone)]
pub struct CorruptClaimOptions {
    /// The honest claim's `claim_submission.json`, in its directory.
    pub submission_path: PathBuf,
    pub selection: CorruptCheckpointSelection,
    pub fault: CorruptFault,
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
/// and the same results, with one checkpoint false. By default it states
/// another output. It can instead name a program outside the approved set,
/// or commit to an input list in which one input is not its producer's
/// output; neither is a lie a replay shows, and a contract takes the claim
/// all the same unless the checkpoint is one its records are read from.
///
/// A checkpoint list is posted with the checkpoints it names, so a changed
/// hash alone is no claim: the lie is a checkpoint that states something
/// false, and the list and its root follow from it. The directory written
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
    // Input lists the lying claim posts beside the honest ones, with the
    // input each one lies about.
    let mut lying_manifests: Vec<Vec<u8>> = Vec::new();
    let mut lied_input = None;
    match &options.fault {
        CorruptFault::Output => {
            trace.checkpoints[index].output_sha256 =
                corrupt_hash(&trace.checkpoints[index].output_sha256);
        }
        CorruptFault::Program => {
            trace.checkpoints[index].program_commitment =
                corrupt_hash(&trace.checkpoints[index].program_commitment);
        }
        CorruptFault::Input(named) => {
            let checkpoint = &mut trace.checkpoints[index];
            let input = match named {
                Some(input) => input.clone(),
                None => checkpoint
                    .input_sources
                    .keys()
                    .next()
                    .cloned()
                    .with_context(|| {
                        format!(
                        "checkpoint {index} (`{stage}`) names no producer for any input: pick a \
                         later stage, or name the input"
                    )
                    })?,
            };
            let manifest_path = dir.join(stage_dir_name(&stage)?).join(INPUT_MANIFEST_JSON);
            let honest_manifest = fs::read(&manifest_path)
                .with_context(|| format!("failed to read {}", manifest_path.display()))?;
            ensure!(
                hex::encode(Sha256::digest(&honest_manifest)) == checkpoint.input_commitment,
                "{} is not the input list checkpoint {index} commits to",
                manifest_path.display()
            );
            let mut entries: BTreeMap<String, BTreeMap<String, String>> =
                serde_json::from_slice(&honest_manifest)
                    .with_context(|| format!("failed to parse {}", manifest_path.display()))?;
            let loaded = entries
                .get_mut(&input)
                .and_then(|entry| entry.get_mut("commitment"))
                .with_context(|| format!("stage `{stage}` has no input `{input}`"))?;
            *loaded = corrupt_hash(loaded);
            let lying = serde_json::to_vec_pretty(&entries)?;
            checkpoint.input_commitment = hex::encode(Sha256::digest(&lying));
            lying_manifests.push(lying);
            lied_input = Some(input);
        }
    }
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
    for manifest in &lying_manifests {
        let stage_dir = options.out_dir.join(stage_dir_name(&stage)?);
        fs::create_dir_all(&stage_dir)
            .with_context(|| format!("failed to create {}", stage_dir.display()))?;
        fs::write(stage_dir.join(INPUT_MANIFEST_JSON), manifest)?;
    }
    let posting = PreparedPosting::read(dir)?
        .with_checkpoints(&claimer, &fs::read(&hashes_path)?, &fs::read(&trace_path)?)?
        .with_input_manifests(lying_manifests)?;
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
            "fault": match &options.fault {
                CorruptFault::Output => "output",
                CorruptFault::Program => "program",
                CorruptFault::Input(_) => "input",
            },
            "input": lied_input,
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
/// `checkpoint_trace.json`, every stage's `input_manifest.json`, the inputs
/// the claim's run was given (`posted_run.bin`), and `inference.toml`: the
/// reader's run spec over the prompt text, or the source's documents, that
/// those inputs hold. Nothing is written unless the blobs are the claimer's
/// and hold the checkpoint list the claim names and the inputs of one run.
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
    if let Some(posted) = &run.source {
        // The recipe and the signer's key are the configuration's, so they
        // stay the reader's own in the spec. The stages run on the posted
        // ones all the same.
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
        for (index, (document, attested)) in crate::run_inputs::documents_of(posted)
            .into_iter()
            .enumerate()
        {
            let body = out.join(format!("posted-document-{index}.bin"));
            fs::write(&body, &document)
                .with_context(|| format!("failed to write {}", body.display()))?;
            let attestation = out.join(format!("posted-attestation-{index}.json"));
            write_json(&attestation, attested)?;
            source.documents.push(SourceDocumentSpec {
                document: body,
                attestation,
            });
        }
    } else {
        spec.source = None;
        // In the spec itself: the text the posted slices concatenate to.
        spec.prompt = Some(crate::run_inputs::text_of(first));
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

/// What a submission put on the layer, measured there: each blob it names is
/// read back, and its size is split among what it holds.
///
/// `bytes` is everything posted: the whole of every blob. Each blob's parts
/// are sized as the blob encodes them, and `framing_bytes` is what is left:
/// the magic, the submitter and the lengths. A run's inputs are split into
/// the text, the token marks, the claimed ids, the documents' bytes and the
/// rest (the run's settings, the forms, and a source's attestations, recipe
/// and signer's key), and a claim's checkpoints into the list, the
/// checkpoints and the input manifests.
pub fn measure_posting(submission_path: &Path, da_dir: Option<&Path>) -> Result<serde_json::Value> {
    #[derive(serde::Deserialize)]
    struct Posted {
        #[serde(default)]
        availability: Vec<inference_artifacts::availability::PostedBlob>,
    }
    let submission: Posted = read_json(submission_path)?;
    ensure!(
        !submission.availability.is_empty(),
        "{} names no posted data: nothing was posted",
        submission_path.display()
    );
    let layer = open_layer(da_dir)?;
    let encoded =
        |entry: &BlobEntry| -> Result<u64> { Ok(postcard::to_allocvec(entry)?.len() as u64) };
    let mut blobs = Vec::new();
    let mut total = 0u64;
    for (index, posted) in submission.availability.iter().enumerate() {
        let bytes = layer
            .read(&posted.pointer)
            .with_context(|| format!("blob {index} could not be read from the layer"))?;
        ensure!(
            bytes.len() as u64 == posted.length,
            "blob {index} is {} bytes on the layer, and the submission says {}",
            bytes.len(),
            posted.length
        );
        let blob =
            Blob::decode(&bytes).map_err(|fault| anyhow::anyhow!("blob {index}: {fault}"))?;
        let mut parts = Vec::new();
        let mut held = 0u64;
        for entry in &blob.entries {
            let size = encoded(entry)?;
            held += size;
            parts.push(match entry {
                BlobEntry::Run(run) => {
                    let sized = |value: &dyn Fn() -> postcard::Result<Vec<u8>>| -> Result<u64> {
                        Ok(value()?.len() as u64)
                    };
                    let (mut text, mut marks, mut claimed) = (0u64, 0u64, 0u64);
                    for inference in &run.inferences {
                        text += sized(&|| postcard::to_allocvec(&inference.text))?;
                        marks += sized(&|| postcard::to_allocvec(&inference.marks))?;
                        claimed += sized(&|| postcard::to_allocvec(&inference.claimed))?;
                    }
                    let (documents, bodies) = match &run.source {
                        Some(source) => (
                            source.documents.attestations.len(),
                            source
                                .documents
                                .chunks
                                .iter()
                                .map(|chunk| chunk.bytes.as_slice().len() as u64)
                                .sum(),
                        ),
                        None => (0, 0u64),
                    };
                    serde_json::json!({
                        "kind": "run_inputs",
                        "bytes": size,
                        "inferences": run.inferences.len(),
                        "values": run.values().len(),
                        "text_bytes": text,
                        "marks_bytes": marks,
                        "claimed_bytes": claimed,
                        "documents": documents,
                        "document_bytes": bodies,
                        "other_bytes": size.saturating_sub(text + marks + claimed + bodies),
                    })
                }
                BlobEntry::Checkpoints(package) => serde_json::json!({
                    "kind": "checkpoints",
                    "bytes": size,
                    "list_bytes": package.list.len(),
                    "trace_bytes": package.trace.len(),
                    "input_manifests": package.input_manifests.len(),
                    "input_manifest_bytes":
                        package.input_manifests.iter().map(Vec::len).sum::<usize>(),
                }),
                BlobEntry::TraceBody { bytes: body } => serde_json::json!({
                    "kind": "trace_body",
                    "bytes": size,
                    "trace_bytes": body.len(),
                }),
            });
        }
        total += posted.length;
        blobs.push(serde_json::json!({
            "height": posted.pointer.height,
            "commitment": posted.pointer.commitment,
            "bytes": posted.length,
            "framing_bytes": posted.length - held,
            "parts": parts,
        }));
    }
    Ok(serde_json::json!({ "bytes": total, "blobs": blobs }))
}

#[derive(Debug, Clone)]
pub struct AuditClaimOptions {
    /// The claim's `claim_submission.json`, in a directory that holds its
    /// checkpoint list, checkpoints and input lists: the one `fetch_claim`
    /// wrote.
    pub submission_path: PathBuf,
    /// A run spec of the configuration the claim names.
    pub run_spec_path: PathBuf,
    pub pins_path: Option<PathBuf>,
}

/// One thing wrong with a claim that no replay shows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuditFault {
    /// The position in [`ClaimAudit::calls`] of the call that points at it.
    pub call: usize,
    pub checkpoint: u32,
    pub stage: String,
    /// The input, for a fault of an input.
    pub input: Option<String>,
    /// The fault a contract records: `IrisClaims.ClaimFault`.
    pub fault: String,
}

#[derive(Debug, Clone)]
pub struct ClaimAudit {
    pub checkpoints: usize,
    pub programs_approved: usize,
    /// Inputs that are what the configuration says they must be: an earlier
    /// stage's output, a pinned commitment, or a value the claim posted.
    pub inputs_linked: usize,
    pub faults: Vec<AuditFault>,
    /// Every question a contract can be asked of the claim with no replay:
    /// for each checkpoint whether its program is approved, and for each
    /// input its program takes from an earlier stage whether it is that
    /// stage's output. Each is the contract call with `expected`: the fault
    /// it records (`void`) or the error it reverts with (`revert`).
    pub calls: Vec<serde_json::Value>,
}

/// What is wrong with a claim that no replay shows: a checkpoint that names
/// a program outside the approved set, or an input that is not what an
/// earlier checkpoint produced, not a pinned commitment, or not a value the
/// claim posted.
///
/// A replay runs the program a checkpoint names on the inputs it names, so a
/// challenger asks these two things first, of every checkpoint, from the
/// claim's own files and the configuration alone. Nothing is run. A fault
/// found here is pointed at with `voidForProgram` or `voidForInput`, and the
/// claim is void: there is no challenge to post and nothing to answer.
pub fn audit_claim(options: AuditClaimOptions) -> Result<ClaimAudit> {
    use inference_artifacts::contract::{CheckpointList, LinkCheck, StageAdmission};
    use inference_artifacts::evm::{abi_encode, claim_id, input_fault_token, program_fault_token};

    let submission: ClaimSubmission = read_json(&options.submission_path)?;
    let configuration =
        crate::contract::spec_configuration(&options.run_spec_path, options.pins_path.as_deref())?;
    ensure!(
        configuration.id() == submission.configuration,
        "the claim is of configuration {}, and {} describes {}: give the run spec of the \
         configuration the claim names",
        submission.configuration,
        options.run_spec_path.display(),
        configuration.id()
    );
    let pins = &configuration.programs;
    let dir = parent(&options.submission_path)?;
    let hashes_path = dir.join(CHECKPOINTS_TXT);
    let list = CheckpointList::open(
        &submission,
        &fs::read(&hashes_path)
            .with_context(|| format!("failed to read {}", hashes_path.display()))?,
    )
    .map_err(|refusal| anyhow::anyhow!("{}: {refusal}", hashes_path.display()))?;
    let hashes = read_checkpoint_hashes(&hashes_path)?;
    let trace = read_checkpoint_trace(&dir.join(CHECKPOINT_TRACE_JSON))?;
    ensure!(
        trace.checkpoints.len() == hashes.len(),
        "{} lists {} checkpoints, and {CHECKPOINT_TRACE_JSON} holds {}",
        hashes_path.display(),
        hashes.len(),
        trace.checkpoints.len()
    );
    let claim = claim_id(&submission)?;
    let posted = submission.posted_run_inputs();

    let mut audit = ClaimAudit {
        checkpoints: hashes.len(),
        programs_approved: 0,
        inputs_linked: 0,
        faults: Vec::new(),
        calls: Vec::new(),
    };
    for (index, checkpoint) in trace.checkpoints.iter().enumerate() {
        let position = index as u32;
        let path = inference_artifacts::checkpoint_opening(&hashes, index)?;
        let refused =
            |refusal| anyhow::anyhow!("checkpoint {index} (`{}`): {refusal}", checkpoint.stage);
        let approved = matches!(
            list.admit_stage_challenge(pins, position, checkpoint)
                .map_err(refused)?,
            StageAdmission::Disputed { .. }
        );
        let mut asked = crate::evm::call(
            "voidForProgram",
            &abi_encode(&program_fault_token(&claim, position, checkpoint, &path)?),
        );
        asked["checkpoint"] = serde_json::json!(position);
        asked["stage"] = serde_json::json!(checkpoint.stage);
        asked["input"] = serde_json::Value::Null;
        asked["expected"] = if approved {
            serde_json::json!({ "revert": "ProgramApproved" })
        } else {
            serde_json::json!({ "void": "UnapprovedProgram" })
        };
        audit.calls.push(asked);
        let Some(program) = pins.program(&checkpoint.program_commitment) else {
            audit.faults.push(AuditFault {
                call: audit.calls.len() - 1,
                checkpoint: position,
                stage: checkpoint.stage.clone(),
                input: None,
                fault: String::from("UnapprovedProgram"),
            });
            // What an unapproved program takes from earlier stages is not
            // defined, so its inputs are not asked about.
            continue;
        };
        audit.programs_approved += 1;
        // Every input of the program: the ones taken from an earlier stage,
        // the ones bound to fixed data, and the ones the run gives.
        let held: std::collections::BTreeSet<&str> = program
            .chained_inputs
            .iter()
            .map(|input| input.name.as_str())
            .chain(
                program
                    .pinned_inputs
                    .iter()
                    .map(|input| input.name.as_str()),
            )
            .chain(program.run_inputs.iter().map(String::as_str))
            .collect();
        if held.is_empty() {
            continue;
        }
        let manifest_path = dir
            .join(stage_dir_name(&checkpoint.stage)?)
            .join(INPUT_MANIFEST_JSON);
        let manifest = fs::read(&manifest_path).with_context(|| {
            format!(
                "failed to read {}: audit the directory `iris claim fetch` wrote",
                manifest_path.display()
            )
        })?;
        for input in held {
            let producer = checkpoint.input_sources.get(input).and_then(|producer| {
                let listed = trace.checkpoints.get(*producer as usize)?;
                Some((*producer as usize, listed))
            });
            let verdict = list.check_link(
                pins,
                position,
                checkpoint,
                &manifest,
                input,
                producer.map(|(_, listed)| listed),
            );
            let producer_opening = producer
                .map(|(producer, listed)| {
                    inference_artifacts::checkpoint_opening(&hashes, producer)
                        .map(|path| (listed, path))
                })
                .transpose()?;
            let mut asked = crate::evm::call(
                "voidForInput",
                &abi_encode(&input_fault_token(
                    &claim,
                    position,
                    checkpoint,
                    &path,
                    &manifest,
                    input,
                    producer_opening
                        .as_ref()
                        .map(|(listed, path)| (*listed, path.as_slice())),
                    &posted,
                )?),
            );
            asked["checkpoint"] = serde_json::json!(position);
            asked["stage"] = serde_json::json!(checkpoint.stage);
            asked["input"] = serde_json::json!(input);
            asked["expected"] = match &verdict {
                Ok(LinkCheck::Linked) => {
                    audit.inputs_linked += 1;
                    serde_json::json!({ "revert": "Linked" })
                }
                Ok(LinkCheck::Void(fault)) => {
                    audit.faults.push(AuditFault {
                        call: audit.calls.len(),
                        checkpoint: position,
                        stage: checkpoint.stage.clone(),
                        input: Some(input.to_string()),
                        fault: fault.name().to_string(),
                    });
                    serde_json::json!({ "void": fault.name() })
                }
                Err(refusal) => serde_json::json!({ "revert": refusal.name() }),
            };
            audit.calls.push(asked);
        }
    }
    Ok(audit)
}
