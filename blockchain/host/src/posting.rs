//! The host's stand-ins for the data-availability layer: a Celestia that
//! keeps blobs in a directory, a Blobstream that attests to every block it
//! makes, and an availability proof the host checks by reading the blob.
//!
//! The rules a contract applies are in `blockchain_rules::availability`.
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
use blockchain_rules::availability::{
    stub_blob_commitment, AvailabilityLayer, AvailabilityProof, Blob, BlobContents, BlobEntry,
    BlobPointer, CheckpointPackage, OutputReader, PostedBlob, PostedRun, WitnessedOutput,
    NAMESPACE_BYTES,
};
use blockchain_rules::checks::{ResultInOutput, SubmittedRecord};
use procedure_definition::{CheckpointList, PreparedRun, CHECKPOINTS_TXT, CHECKPOINT_LIST_JSON};
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

/// The stand-in layer's directory, when none is given.
fn da_dir_default() -> PathBuf {
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

/// A layer blobs are posted to and read from, besides what a contract asks
/// of it. Everything in iris that posts or reads takes one of these, so it
/// does not know which layer it has.
pub trait PostingLayer: AvailabilityLayer {
    /// Include these blobs and say where each one is.
    fn post(
        &self,
        namespace: &[u8; NAMESPACE_BYTES],
        blobs: &[Vec<u8>],
    ) -> Result<Vec<BlobPointer>>;
    /// The bytes of a blob, as a challenger or a defender gets them.
    fn read(&self, pointer: &BlobPointer) -> Result<Vec<u8>>;
    /// The root and time of the block at `height`: what a relayer tells a
    /// contract's oracle, and what an attestation of a blob in it names.
    fn block_root(&self, height: u64) -> Result<BlockRoot>;
}

/// A block as a contract's oracle is told of it: its height, the root of its
/// data and its time in Unix seconds.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BlockRoot {
    pub height: u64,
    /// In lowercase hex.
    pub data_root: String,
    pub time: u64,
}

/// The layer this process posts to and reads from. A directory given names
/// the stand-in there. Otherwise it is the Celestia node `IRIS_CELESTIA_RPC`
/// names, and with none named, the stand-in in its default directory.
pub fn open_layer(da_dir: Option<&Path>) -> Result<Box<dyn PostingLayer>> {
    if let Some(dir) = da_dir {
        return Ok(Box::new(StubLayer::open(dir)));
    }
    Ok(
        match std::env::var(crate::blockchain::celestia::CELESTIA_RPC_ENV) {
            Ok(url) if !url.is_empty() => {
                Box::new(crate::blockchain::celestia::CelestiaLayer::open(&url)?)
            }
            _ => Box::new(StubLayer::open(da_dir_default())),
        },
    )
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
        procedure_definition::read_json(&path)
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

    /// The bytes of the blob a pointer names, when the layer holds a blob
    /// under that commitment in that block.
    fn included(&self, pointer: &BlobPointer) -> Option<Vec<u8>> {
        let namespace = hex::decode(&pointer.namespace).ok()?;
        let bytes = self.read(pointer).ok()?;
        (stub_blob_commitment(&namespace, &bytes) == pointer.commitment).then_some(bytes)
    }
}

impl PostingLayer for StubLayer {
    /// One new block holding these blobs, attested by the stand-in
    /// Blobstream.
    fn post(
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
        procedure_definition::write_json(&self.dir.join(BLOBSTREAM_JSON), &blobstream)?;
        Ok(pointers)
    }

    fn read(&self, pointer: &BlobPointer) -> Result<Vec<u8>> {
        let path = self.blob_path(pointer);
        fs::read(&path).with_context(|| format!("failed to read {}", path.display()))
    }

    fn block_root(&self, height: u64) -> Result<BlockRoot> {
        self.block(height)
    }
}

impl StubLayer {
    fn block(&self, height: u64) -> Result<BlockRoot> {
        let block = self
            .blobstream()?
            .blocks
            .into_iter()
            .find(|block| block.height == height)
            .with_context(|| format!("the stand-in layer has no block {height}"))?;
        Ok(BlockRoot {
            height,
            data_root: block.data_root,
            time: block.time,
        })
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
            AvailabilityProof::HostChecked { outputs, .. } => {
                self.included(&blob.pointer).is_some_and(|bytes| {
                    contents_of(&bytes, outputs).ok().as_ref() == Some(&blob.contents)
                })
            }
        }
    }
}

/// What a blob holds, as the stand-in proof establishes it by reading the
/// blob: the commitment a stage names each run input by, each checkpoint
/// package whole, and each trace body a well-formed Raster trace commitment.
/// With the stage outputs a claimer gives, also the records its checkpoints
/// commit to.
pub fn contents_of(bytes: &[u8], outputs: &[WitnessedOutput]) -> Result<BlobContents> {
    Blob::decode(bytes)
        .and_then(|blob| blob.contents(&HostReader, outputs))
        .map_err(|fault| anyhow::anyhow!("{fault}"))
}

/// The host's reader of Raster formats: the code the stages' outputs are
/// read with everywhere else in this crate.
struct HostReader;

impl OutputReader for HostReader {
    fn run_inputs(&self, run: &PostedRun) -> Option<Vec<String>> {
        crate::run_inputs::commitments(run).ok()
    }

    fn trace_header(&self, body: &[u8]) -> Option<String> {
        trace_header_digest(body)
    }

    fn check_record(&self, payload: &[u8], index: &[u8]) -> Option<SubmittedRecord> {
        let edge = crate::claim_check::output_json(payload, index).ok()?;
        crate::claim_check::claim_record(edge.get("claim")?).ok()
    }

    fn results(&self, payload: &[u8], index: &[u8]) -> Option<ResultInOutput> {
        let stated = crate::claim_check::output_json(payload, index).ok()?;
        // The result kinds' records have no field in common at the top, so an
        // output reads as one of them or as neither.
        procedure_definition::ResultKind::ALL
            .iter()
            .find_map(|result| crate::claim_check::result_in_output(*result, &stated).ok())
    }
}

/// What a result stage's output states of its claim, as a contract reads it.
/// `None` when the output is no result record.
pub fn stated_results(payload: &[u8], index: &[u8]) -> Option<ResultInOutput> {
    HostReader.results(payload, index)
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

/// The outputs a blob's proof is given: a claim's, for the blob that holds
/// its checkpoints, and none for any other.
fn outputs_for<'a>(blob: &Blob, outputs: &'a [WitnessedOutput]) -> &'a [WitnessedOutput] {
    let holds_checkpoints = blob
        .entries
        .iter()
        .any(|entry| matches!(entry, BlobEntry::Checkpoints(_)));
    if holds_checkpoints {
        outputs
    } else {
        &[]
    }
}

/// Post blobs and state each one as a claim or a challenge submits it.
/// `outputs` are the stage outputs a claim's records are read from: they go
/// with the proof of the blob that holds its checkpoints, and are not posted.
fn post_blobs(
    layer: &dyn PostingLayer,
    namespace: &[u8; NAMESPACE_BYTES],
    blobs: &[Blob],
    outputs: &[WitnessedOutput],
) -> Result<Vec<PostedBlob>> {
    let encoded: Vec<Vec<u8>> = blobs.iter().map(Blob::encode).collect();
    let given: Vec<&[WitnessedOutput]> = blobs
        .iter()
        .map(|blob| outputs_for(blob, outputs))
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
                attestation: None,
            },
        })
        .collect())
}

/// The inputs a prepared run's stages were given that neither an earlier
/// stage nor the procedure fixes, as run preparation wrote them beside
/// the run's manifest: the values its files were written from.
pub fn posted_run(prepared: &PreparedRun) -> Result<BlobEntry> {
    let run_dir = prepared
        .run_manifest_path
        .as_deref()
        .and_then(Path::parent)
        .context("the prepared run has no manifest")?;
    Ok(BlobEntry::Run(crate::run_inputs::load(run_dir)?))
}

/// What a claim is ready to post: its blobs, and the stage outputs its
/// records are read from, which go to the proof and are not posted. `claim
/// build` writes it beside the claim; `claim post` posts it, any number of
/// times, without the claim being built again.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedPosting {
    pub blobs: Vec<Blob>,
    pub outputs: Vec<WitnessedOutput>,
}

/// The file that lists the outputs a claim's proof is given, written beside
/// its submission with the blobs to post (`posting-blob-<n>.bin`). They are
/// plain files in the claim's directory: every subdirectory there is a stage.
pub const POSTING_OUTPUTS_JSON: &str = "posting-outputs.json";

fn posting_blob(claim_dir: &Path, index: usize) -> PathBuf {
    claim_dir.join(format!("posting-blob-{index}.bin"))
}

impl PreparedPosting {
    /// What the blobs hold together, as their proofs will state it.
    pub fn contents(&self) -> Result<BlobContents> {
        let mut contents = BlobContents::default();
        for blob in &self.blobs {
            let held = contents_of(&blob.encode(), outputs_for(blob, &self.outputs))?;
            contents.submitter = held.submitter.clone();
            contents.extend(&held);
        }
        Ok(contents)
    }

    /// Write the blobs and the outputs beside the claim, in place of any
    /// written there before.
    pub fn write(&self, claim_dir: &Path) -> Result<()> {
        fs::create_dir_all(claim_dir)
            .with_context(|| format!("failed to create {}", claim_dir.display()))?;
        let mut stale = self.blobs.len();
        while posting_blob(claim_dir, stale).exists() {
            fs::remove_file(posting_blob(claim_dir, stale))?;
            stale += 1;
        }
        for (index, blob) in self.blobs.iter().enumerate() {
            let path = posting_blob(claim_dir, index);
            fs::write(&path, blob.encode())
                .with_context(|| format!("failed to write {}", path.display()))?;
        }
        procedure_definition::write_json(&claim_dir.join(POSTING_OUTPUTS_JSON), &self.outputs)
    }

    /// Read what `claim build` wrote beside the claim.
    pub fn read(claim_dir: &Path) -> Result<Self> {
        let mut blobs = Vec::new();
        loop {
            let path = posting_blob(claim_dir, blobs.len());
            if !path.exists() {
                break;
            }
            let bytes =
                fs::read(&path).with_context(|| format!("failed to read {}", path.display()))?;
            blobs.push(
                Blob::decode(&bytes)
                    .map_err(|fault| anyhow::anyhow!("{}: {fault}", path.display()))?,
            );
        }
        anyhow::ensure!(
            !blobs.is_empty(),
            "{} holds no blobs to post: build the claim first",
            claim_dir.display()
        );
        Ok(Self {
            blobs,
            outputs: procedure_definition::read_json(&claim_dir.join(POSTING_OUTPUTS_JSON))?,
        })
    }

    /// This posting for another claim of the same run: `hashes` and `list` in
    /// place of the checkpoint hashes and the checkpoints, and every blob
    /// posted for `submitter`. The new information and the input manifests
    /// stay. It is what a claimer posts who states another output for a
    /// checkpoint of an honest run.
    pub fn with_checkpoints(mut self, submitter: &str, hashes: &[u8], list: &[u8]) -> Result<Self> {
        let mut replaced = 0;
        for blob in &mut self.blobs {
            blob.submitter = submitter.to_string();
            for entry in &mut blob.entries {
                if let BlobEntry::Checkpoints(package) = entry {
                    package.hashes = hashes.to_vec();
                    package.list = list.to_vec();
                    replaced += 1;
                }
            }
        }
        anyhow::ensure!(
            replaced == 1,
            "the posting holds {replaced} checkpoint lists, and a claim posts one"
        );
        Ok(self)
    }

    /// The same posting with more input manifests beside the ones it holds:
    /// those of checkpoints that commit to another input list than the
    /// claim's own.
    pub fn with_input_manifests(mut self, manifests: Vec<Vec<u8>>) -> Result<Self> {
        if manifests.is_empty() {
            return Ok(self);
        }
        let package = self
            .blobs
            .iter_mut()
            .flat_map(|blob| &mut blob.entries)
            .find_map(|entry| match entry {
                BlobEntry::Checkpoints(package) => Some(package),
                _ => None,
            })
            .context("the posting holds no checkpoints")?;
        package.input_manifests.extend(manifests);
        Ok(self)
    }

    /// Post the blobs to `layer`, and state each as a claim submits it.
    pub fn post(&self, layer: &dyn PostingLayer) -> Result<Vec<PostedBlob>> {
        post_blobs(layer, &CLAIM_NAMESPACE, &self.blobs, &self.outputs)
    }
}

/// Post what a claim in `claim_dir` must make available, for the address
/// `claimer`: [`claim_blobs`], posted.
pub fn post_claim(
    layer: &dyn PostingLayer,
    claim_dir: &Path,
    claimer: &str,
    new_information: Option<BlobEntry>,
) -> Result<Vec<PostedBlob>> {
    claim_blobs(claim_dir, claimer, new_information)?.post(layer)
}

/// What a claim in `claim_dir` must make available, for the address
/// `claimer`: the inputs its run was given, and its checkpoint list with
/// every checkpoint and every input manifest. A run input is posted as its
/// value, and anyone writes the file a stage loads from it.
pub fn claim_blobs(
    claim_dir: &Path,
    claimer: &str,
    new_information: Option<BlobEntry>,
) -> Result<PreparedPosting> {
    let hashes = fs::read(claim_dir.join(CHECKPOINTS_TXT))
        .with_context(|| format!("{} has no {CHECKPOINTS_TXT}", claim_dir.display()))?;
    let list_bytes = fs::read(claim_dir.join(CHECKPOINT_LIST_JSON))
        .with_context(|| format!("{} has no {CHECKPOINT_LIST_JSON}", claim_dir.display()))?;
    let list: CheckpointList = serde_json::from_slice(&list_bytes)?;

    let mut manifests = Vec::new();
    let mut manifest_hashes = BTreeSet::new();
    for checkpoint in &list.checkpoints {
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
    let last = list
        .checkpoints
        .last()
        .context("the claim has no checkpoints")?;
    let mut positions: BTreeSet<u32> = last.input_sources.values().copied().collect();
    positions.insert(list.checkpoints.len() as u32 - 1);
    let outputs = positions
        .into_iter()
        .filter_map(|position| Some((position, list.checkpoints.get(position as usize)?)))
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
        hashes,
        list: list_bytes,
        input_manifests: manifests,
    }));
    Ok(PreparedPosting {
        blobs: pack(claimer, entries),
        outputs,
    })
}

/// Post a challenger's trace body (`commit.bin`), for the address
/// `challenger`.
pub fn post_trace_body(
    layer: &dyn PostingLayer,
    challenger: &str,
    body: Vec<u8>,
) -> Result<Vec<PostedBlob>> {
    post_blobs(
        layer,
        &CHALLENGE_NAMESPACE,
        &[Blob {
            submitter: challenger.to_string(),
            entries: vec![BlobEntry::TraceBody { bytes: body }],
        }],
        &[],
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use blockchain_rules::availability::check_posted;

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

    /// The inputs of a run of one inference over one prompt range.
    fn run() -> PostedRun {
        use blockchain_rules::availability::{PostedInference, PromptText};
        PostedRun {
            source: None,
            forms: Default::default(),
            inferences: vec![PostedInference {
                config: Default::default(),
                schedule: Default::default(),
                text_digest: Default::default(),
                text: vec![PromptText {
                    chunks: vec![String::from("Which section was about a bakery?")].into(),
                }],
                marks: vec![vec![1, 0, 1]],
                claimed: vec![vec![7]],
            }],
            relay: None,
        }
    }

    #[test]
    fn posted_run_inputs_are_included_proven_and_named_by_their_commitments() {
        let layer = layer("prompt");
        let alice = "0x00000000000000000000000000000000000a11ce";
        let blob = Blob {
            submitter: alice.to_string(),
            entries: vec![BlobEntry::Run(run())],
        };
        let posted = post_blobs(&layer, &CLAIM_NAMESPACE, &[blob], &[]).unwrap();
        // Each input by the commitment a stage's input list names it by: the
        // one its Raster file has.
        assert_eq!(
            posted[0].contents.run_inputs,
            crate::run_inputs::commitments(&run()).unwrap()
        );
        assert_eq!(posted[0].contents.run_inputs.len(), 7);
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
        overstated.contents.run_inputs.push("11".repeat(32));
        assert!(!layer.proves(&overstated));

        // A second post is a later block.
        let later = post_trace_body(&layer, alice, vec![1, 2, 3]);
        assert!(later.is_err(), "three bytes are no trace commitment");
    }

    #[test]
    fn a_prepared_posting_is_written_once_and_posted_as_often_as_needed() {
        let layer = layer("prepared");
        let alice = "0x00000000000000000000000000000000000a11ce";
        let prepared = PreparedPosting {
            blobs: vec![Blob {
                submitter: alice.to_string(),
                entries: vec![BlobEntry::Run(run())],
            }],
            outputs: Vec::new(),
        };
        let claim_dir = layer.dir.join("claim");
        prepared.write(&claim_dir).unwrap();
        let read = PreparedPosting::read(&claim_dir).unwrap();
        assert_eq!(read, prepared);
        // What it holds is known before anything is posted, and each post is
        // a block of its own with the same contents.
        let contents = read.contents().unwrap();
        assert_eq!(contents.submitter, alice);
        let first = read.post(&layer).unwrap();
        let second = read.post(&layer).unwrap();
        assert_eq!(first[0].contents, contents);
        assert_eq!(second[0].contents, contents);
        assert_eq!(second[0].pointer.height, first[0].pointer.height + 1);
        assert_eq!(second[0].pointer.commitment, first[0].pointer.commitment);
        // A directory nothing was prepared in has nothing to post.
        assert!(PreparedPosting::read(&layer.dir.join("nothing")).is_err());
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
