//! The host's stand-ins for the data-availability layer: a Celestia that
//! keeps blobs in a directory, a Blobstream that attests to every block it
//! makes, and an availability proof the host checks by reading the blob.
//!
//! The rules a contract applies are in `inference_artifacts::availability`.
//! This module posts a claim's data and a challenger's trace body, and answers
//! the two questions a contract asks of the layer. Every function a real
//! network replaces is here and behind `AvailabilityLayer`; nothing else in
//! iris knows the layer is a directory.
//!
//! The directory is `IRIS_DA_DIR`, or `.cache/iris/stub-celestia` in the home
//! directory: one directory is one network, so a claimer and a challenger
//! working in different places see the same blobs. `IRIS_DA_NOW` (Unix
//! seconds) fixes the clock new blocks are stamped with, for tests.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{bail, Context, Result};
use inference_artifacts::availability::{
    stub_blob_commitment, AvailabilityLayer, AvailabilityProof, Blob, BlobContents, BlobEntry,
    BlobPointer, CheckpointPackage, OutputReader, PostedBlob, PostedDocument, PostedSource,
    WitnessedOutput, NAMESPACE_BYTES,
};
use inference_artifacts::contract::{ResultInOutput, SubmittedRecord};
use inference_artifacts::{
    read_run_spec, CheckpointTrace, PreparedRun, CHECKPOINTS_TXT, CHECKPOINT_TRACE_JSON,
};
use raster::core::trace::TraceCommitment;
use raster_prover::trace::TraceCommitmentExt;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Where the stand-in layer keeps its blocks, under the home directory.
pub const STUB_DA_DIR: &str = ".cache/iris/stub-celestia";
/// The stand-in Blobstream's attested data roots, in the layer's directory.
const BLOBSTREAM_JSON: &str = "blobstream.json";
/// The most iris puts in one blob. A stand-in for the layer's own limit.
const MAX_BLOB_BYTES: usize = 1 << 20;

/// A version-0 namespace with this 10-byte id: 19 zero bytes, then the id.
const fn namespace(id: &[u8; 10]) -> [u8; NAMESPACE_BYTES] {
    let mut namespace = [0; NAMESPACE_BYTES];
    let mut i = 0;
    while i < 10 {
        namespace[19 + i] = id[i];
        i += 1;
    }
    namespace
}

/// Where claims post their data.
pub const CLAIM_NAMESPACE: [u8; NAMESPACE_BYTES] = namespace(b"iris-claim");
/// Where challengers post their trace bodies.
pub const CHALLENGE_NAMESPACE: [u8; NAMESPACE_BYTES] = namespace(b"iris-chall");

/// The layer's directory.
pub fn da_dir() -> PathBuf {
    std::env::var_os("IRIS_DA_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            std::env::var_os("HOME")
                .map(PathBuf::from)
                .unwrap_or_else(std::env::temp_dir)
                .join(STUB_DA_DIR)
        })
}

/// Unix seconds now, or `IRIS_DA_NOW` when it is set.
pub fn now() -> Result<u64> {
    if let Some(value) = std::env::var_os("IRIS_DA_NOW") {
        let value = value.to_string_lossy();
        return value
            .parse()
            .with_context(|| format!("IRIS_DA_NOW is not a number of seconds: {value}"));
    }
    Ok(SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("the clock is before 1970")?
        .as_secs())
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct Blobstream {
    blocks: Vec<AttestedBlock>,
}

/// One block's data root as the stand-in Blobstream attests to it.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct AttestedBlock {
    height: u64,
    time: u64,
    data_root: String,
}

/// Held while a block is made, so two posters never make the same height.
struct Lock(PathBuf);

impl Lock {
    fn take(dir: &Path) -> Result<Self> {
        fs::create_dir_all(dir).with_context(|| format!("failed to create {}", dir.display()))?;
        let path = dir.join("post.lock");
        for _ in 0..600 {
            match fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)
            {
                Ok(_) => return Ok(Self(path)),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                    std::thread::sleep(std::time::Duration::from_millis(100));
                }
                Err(error) => {
                    return Err(error)
                        .with_context(|| format!("failed to create {}", path.display()))
                }
            }
        }
        bail!(
            "{} has been held for a minute; remove it if no other post is running",
            path.display()
        )
    }
}

impl Drop for Lock {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

/// The stand-in layer, kept in a directory.
pub struct StubLayer {
    dir: PathBuf,
}

impl StubLayer {
    pub fn open(dir: impl Into<PathBuf>) -> Self {
        Self { dir: dir.into() }
    }

    fn blobstream(&self) -> Result<Blobstream> {
        let path = self.dir.join(BLOBSTREAM_JSON);
        if !path.exists() {
            return Ok(Blobstream::default());
        }
        inference_artifacts::read_json(&path)
    }

    fn block_dir(&self, height: u64) -> PathBuf {
        self.dir.join("blocks").join(height.to_string())
    }

    fn blob_path(&self, pointer: &BlobPointer) -> PathBuf {
        self.block_dir(pointer.height)
            .join(format!("{}-{}.blob", pointer.namespace, pointer.commitment))
    }

    /// The data root of a block: a hash over every blob in it, by namespace
    /// and commitment in order. Celestia's is a Merkle root over the block's
    /// shares; a real Blobstream proves a blob under it.
    fn data_root(&self, height: u64) -> Result<String> {
        let mut names = BTreeSet::new();
        for entry in fs::read_dir(self.block_dir(height))
            .with_context(|| format!("block {height} has no directory"))?
        {
            names.insert(entry?.file_name().to_string_lossy().into_owned());
        }
        let mut hasher = Sha256::new();
        for name in names {
            hasher.update(name.as_bytes());
            hasher.update([0]);
        }
        Ok(hex::encode(hasher.finalize()))
    }

    /// Include these blobs in one new block, have the stand-in Blobstream
    /// attest to it, and say where each one is.
    pub fn post(
        &self,
        namespace: &[u8; NAMESPACE_BYTES],
        blobs: &[Vec<u8>],
    ) -> Result<Vec<BlobPointer>> {
        let _lock = Lock::take(&self.dir)?;
        let mut blobstream = self.blobstream()?;
        let height = blobstream.blocks.last().map_or(1, |block| block.height + 1);
        let block_dir = self.block_dir(height);
        if block_dir.exists() {
            bail!(
                "{} exists, and the stand-in Blobstream attests to no block {height}",
                block_dir.display()
            );
        }
        fs::create_dir_all(&block_dir)
            .with_context(|| format!("failed to create {}", block_dir.display()))?;
        let pointers = blobs
            .iter()
            .map(|bytes| {
                let pointer = BlobPointer {
                    height,
                    namespace: hex::encode(namespace),
                    commitment: stub_blob_commitment(namespace, bytes),
                };
                let path = self.blob_path(&pointer);
                fs::write(&path, bytes)
                    .with_context(|| format!("failed to write {}", path.display()))?;
                Ok(pointer)
            })
            .collect::<Result<Vec<_>>>()?;
        blobstream.blocks.push(AttestedBlock {
            height,
            time: now()?,
            data_root: self.data_root(height)?,
        });
        inference_artifacts::write_json(&self.dir.join(BLOBSTREAM_JSON), &blobstream)?;
        Ok(pointers)
    }

    /// The bytes of a blob, from the layer, as a challenger or a defender
    /// gets them.
    pub fn read(&self, pointer: &BlobPointer) -> Result<Vec<u8>> {
        let path = self.blob_path(pointer);
        fs::read(&path).with_context(|| format!("failed to read {}", path.display()))
    }

    /// The bytes of the blob a pointer names, when the layer holds a blob
    /// under that commitment in that block.
    fn included(&self, pointer: &BlobPointer) -> Option<Vec<u8>> {
        let namespace = hex::decode(&pointer.namespace).ok()?;
        let bytes = self.read(pointer).ok()?;
        (stub_blob_commitment(&namespace, &bytes) == pointer.commitment).then_some(bytes)
    }
}

impl AvailabilityLayer for StubLayer {
    fn included_at(&self, pointer: &BlobPointer, length: u64) -> Option<u64> {
        let block = self
            .blobstream()
            .ok()?
            .blocks
            .into_iter()
            .find(|block| block.height == pointer.height)?;
        let bytes = self.included(pointer)?;
        let root = self.data_root(pointer.height).ok()?;
        (bytes.len() as u64 == length && root == block.data_root).then_some(block.time)
    }

    fn proves(&self, blob: &PostedBlob) -> bool {
        match &blob.proof {
            AvailabilityProof::HostChecked { outputs } => {
                self.included(&blob.pointer).is_some_and(|bytes| {
                    contents_of(&bytes, outputs).ok().as_ref() == Some(&blob.contents)
                })
            }
        }
    }
}

/// What a blob holds, as the stand-in proof establishes it by reading the
/// blob: the record the source step gives over each source, the hash of each
/// prompt text, each checkpoint package whole, and each trace body a
/// well-formed Raster trace commitment. With the stage outputs a claimer
/// gives, also the records its checkpoints commit to.
pub fn contents_of(bytes: &[u8], outputs: &[WitnessedOutput]) -> Result<BlobContents> {
    Blob::decode(bytes)
        .and_then(|blob| blob.contents(&HostReader, outputs))
        .map_err(|fault| anyhow::anyhow!("{fault}"))
}

/// The host's reader of Raster formats: the code the stages' outputs are
/// read with everywhere else in this crate.
struct HostReader;

impl OutputReader for HostReader {
    fn trace_header(&self, body: &[u8]) -> Option<String> {
        trace_header_digest(body)
    }

    fn check_record(&self, payload: &[u8], index: &[u8]) -> Option<SubmittedRecord> {
        let edge = crate::contract::output_json(payload, index).ok()?;
        crate::contract::claim_record(edge.get("claim")?).ok()
    }

    fn results(&self, payload: &[u8], index: &[u8]) -> Option<ResultInOutput> {
        let stated = crate::contract::output_json(payload, index).ok()?;
        // The two pipelines' result records have no field in common at the
        // top, so an output reads as one of them or as neither.
        [
            inference_artifacts::Pipeline::Consensus,
            inference_artifacts::Pipeline::Single,
        ]
        .into_iter()
        .find_map(|pipeline| crate::contract::result_in_output(pipeline, &stated).ok())
    }
}

/// The digest of the header of a Raster trace commitment body, as a stage
/// challenge names it. `None` unless the body decodes and is well formed.
pub fn trace_header_digest(bytes: &[u8]) -> Option<String> {
    let body: TraceCommitment = postcard::from_bytes(bytes).ok()?;
    body.validate().ok()?;
    Some(hex::encode(body.header().digest()))
}

/// Pack entries into blobs of at most [`MAX_BLOB_BYTES`], keeping their
/// order. An entry larger than that goes alone.
fn pack(submitter: &str, entries: Vec<BlobEntry>) -> Vec<Blob> {
    let blob = |entries| Blob {
        submitter: submitter.to_string(),
        entries,
    };
    let mut blobs = Vec::new();
    let mut current = blob(Vec::new());
    for entry in entries {
        let mut next = current.clone();
        next.entries.push(entry.clone());
        if !current.entries.is_empty() && next.encode().len() > MAX_BLOB_BYTES {
            blobs.push(std::mem::replace(&mut current, blob(vec![entry])));
        } else {
            current = next;
        }
    }
    if !current.entries.is_empty() {
        blobs.push(current);
    }
    blobs
}

/// Post blobs and state each one as a claim or a challenge submits it.
/// `outputs` are the stage outputs a claim's records are read from: they go
/// with the proof of the blob that holds its checkpoints, and are not posted.
fn post_blobs(
    layer: &StubLayer,
    namespace: &[u8; NAMESPACE_BYTES],
    blobs: Vec<Blob>,
    outputs: &[WitnessedOutput],
) -> Result<Vec<PostedBlob>> {
    let encoded: Vec<Vec<u8>> = blobs.iter().map(Blob::encode).collect();
    let given: Vec<&[WitnessedOutput]> = blobs
        .iter()
        .map(|blob| {
            let holds_checkpoints = blob
                .entries
                .iter()
                .any(|entry| matches!(entry, BlobEntry::Checkpoints(_)));
            if holds_checkpoints {
                outputs
            } else {
                &[]
            }
        })
        .collect();
    // What each blob holds, before anything is posted: a blob that holds
    // nothing it can state is never posted.
    let contents = encoded
        .iter()
        .zip(&given)
        .map(|(bytes, outputs)| contents_of(bytes, outputs))
        .collect::<Result<Vec<_>>>()?;
    let pointers = layer.post(namespace, &encoded)?;
    Ok(pointers
        .into_iter()
        .zip(encoded)
        .zip(contents)
        .zip(given)
        .map(|(((pointer, bytes), contents), outputs)| PostedBlob {
            contents,
            length: bytes.len() as u64,
            pointer,
            proof: AvailabilityProof::HostChecked {
                outputs: outputs.to_vec(),
            },
        })
        .collect())
}

/// The new information a prepared run was given, which no configuration or
/// model fixes: its source (the documents with their attestations, the
/// recipe, and the configuration's signer key), or, for a run with no source,
/// its prompt text as given.
pub fn new_information(prepared: &PreparedRun) -> Result<BlobEntry> {
    let run_spec = read_run_spec(&prepared.run_spec_path)?;
    let Some(spec) = &run_spec.source else {
        return Ok(BlobEntry::PromptText {
            text: prepared.first()?.prompt.resolved_prompt.clone(),
        });
    };
    let spec_dir = prepared
        .run_spec_path
        .parent()
        .context("the run spec has no directory")?;
    let loaded = run_prep::source::load_source(spec_dir, spec)?;
    Ok(BlobEntry::Source(PostedSource {
        signer: loaded.signer,
        recipe: loaded.recipe.recipe(),
        documents: loaded
            .documents
            .into_iter()
            .map(|document| PostedDocument {
                attestation: document.attestation,
                body: document.body,
            })
            .collect(),
    }))
}

/// Post what a claim in `claim_dir` must make available, for the address
/// `claimer`: the new information its run was given, and its checkpoint list
/// with every checkpoint and every input manifest. No file a stage loads is
/// posted, and no index: a challenger makes those itself.
pub fn post_claim(
    layer: &StubLayer,
    claim_dir: &Path,
    claimer: &str,
    new_information: Option<BlobEntry>,
) -> Result<Vec<PostedBlob>> {
    let list = fs::read(claim_dir.join(CHECKPOINTS_TXT))
        .with_context(|| format!("{} has no {CHECKPOINTS_TXT}", claim_dir.display()))?;
    let trace_bytes = fs::read(claim_dir.join(CHECKPOINT_TRACE_JSON))
        .with_context(|| format!("{} has no {CHECKPOINT_TRACE_JSON}", claim_dir.display()))?;
    let trace: CheckpointTrace = serde_json::from_slice(&trace_bytes)?;

    let mut manifests = Vec::new();
    let mut manifest_hashes = BTreeSet::new();
    for checkpoint in &trace.checkpoints {
        let manifest_path = claim_dir
            .join(&checkpoint.stage)
            .join("input_manifest.json");
        let manifest = fs::read(&manifest_path)
            .with_context(|| format!("failed to read {}", manifest_path.display()))?;
        if manifest_hashes.insert(hex::encode(Sha256::digest(&manifest))) {
            manifests.push(manifest);
        }
    }
    // The outputs the claim's records are read from: its last stage's, and
    // those of the stages that one read.
    let last = trace
        .checkpoints
        .last()
        .context("the claim has no checkpoints")?;
    let mut positions: BTreeSet<u32> = last.input_sources.values().copied().collect();
    positions.insert(trace.checkpoints.len() as u32 - 1);
    let outputs = positions
        .into_iter()
        .filter_map(|position| Some((position, trace.checkpoints.get(position as usize)?)))
        .map(|(position, checkpoint)| {
            let read = |file: &str| {
                let path = claim_dir.join(&checkpoint.stage).join(file);
                fs::read(&path).with_context(|| format!("failed to read {}", path.display()))
            };
            Ok(WitnessedOutput {
                checkpoint: position,
                payload: hex::encode(read("output.bin")?),
                index: hex::encode(read("output.rindex")?),
            })
        })
        .collect::<Result<Vec<_>>>()?;

    let mut entries: Vec<BlobEntry> = new_information.into_iter().collect();
    entries.push(BlobEntry::Checkpoints(CheckpointPackage {
        list,
        trace: trace_bytes,
        input_manifests: manifests,
    }));
    post_blobs(layer, &CLAIM_NAMESPACE, pack(claimer, entries), &outputs)
}

/// Post a challenger's trace body (`commit.bin`), for the address
/// `challenger`.
pub fn post_trace_body(
    layer: &StubLayer,
    challenger: &str,
    body: Vec<u8>,
) -> Result<Vec<PostedBlob>> {
    post_blobs(
        layer,
        &CHALLENGE_NAMESPACE,
        vec![Blob {
            submitter: challenger.to_string(),
            entries: vec![BlobEntry::TraceBody { bytes: body }],
        }],
        &[],
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use inference_artifacts::availability::check_posted;

    fn layer(label: &str) -> StubLayer {
        let dir = std::env::temp_dir().join(format!(
            "iris-stub-celestia-{label}-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        StubLayer::open(dir)
    }

    #[test]
    fn a_posted_prompt_is_included_proven_and_named_by_its_hash() {
        let layer = layer("prompt");
        let alice = "0x00000000000000000000000000000000000a11ce";
        let blob = Blob {
            submitter: alice.to_string(),
            entries: vec![BlobEntry::PromptText {
                text: String::from("Which section was about a bakery?"),
            }],
        };
        let posted = post_blobs(&layer, &CLAIM_NAMESPACE, vec![blob], &[]).unwrap();
        assert_eq!(
            posted[0].contents.prompt_texts,
            vec![hex::encode(Sha256::digest(
                b"Which section was about a bakery?"
            ))]
        );
        assert!(layer.proves(&posted[0]));
        assert!(layer
            .included_at(&posted[0].pointer, posted[0].length)
            .is_some());
        assert_eq!(posted[0].contents.submitter, alice);
        assert!(check_posted(&layer, 1, &posted, alice, 0).is_ok());
        // Posted for one address, it serves no other.
        assert!(check_posted(
            &layer,
            1,
            &posted,
            "0x0000000000000000000000000000000000000b0b",
            0
        )
        .is_err());

        // Another length, another commitment, or other stated contents: not
        // this blob.
        assert!(layer
            .included_at(&posted[0].pointer, posted[0].length + 1)
            .is_none());
        let mut moved = posted[0].clone();
        moved.pointer.commitment = "00".repeat(32);
        assert!(layer.included_at(&moved.pointer, moved.length).is_none());
        assert!(!layer.proves(&moved));
        let mut overstated = posted[0].clone();
        overstated.contents.prompt_texts.push("11".repeat(32));
        assert!(!layer.proves(&overstated));

        // A second post is a later block.
        let later = post_trace_body(&layer, alice, vec![1, 2, 3]);
        assert!(later.is_err(), "three bytes are no trace commitment");
    }

    #[test]
    fn blobs_stay_under_the_size_and_keep_their_order() {
        let entry = |byte: u8| BlobEntry::TraceBody {
            bytes: vec![byte; MAX_BLOB_BYTES / 3],
        };
        let blobs = pack(
            "0x00000000000000000000000000000000000a11ce",
            (0..5).map(entry).collect(),
        );
        assert_eq!(blobs.len(), 3);
        assert!(blobs
            .iter()
            .all(|blob| blob.encode().len() <= MAX_BLOB_BYTES));
        let order: Vec<u8> = blobs
            .iter()
            .flat_map(|blob| &blob.entries)
            .map(|entry| match entry {
                BlobEntry::TraceBody { bytes } => bytes[0],
                _ => unreachable!(),
            })
            .collect();
        assert_eq!(order, vec![0, 1, 2, 3, 4]);
    }
}
