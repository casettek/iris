//! What a claimer and a challenger make available, and what a contract checks
//! of it before it accepts a claim or a challenge.
//!
//! A claim names its inputs and its checkpoints by commitment. A challenger
//! who cannot get the bytes behind those commitments cannot replay anything,
//! so a claim is only accepted with its data posted to the data-availability
//! layer (Celestia), and a challenge only with its trace body posted.
//!
//! A claimer posts everything its stages were given that neither an earlier
//! stage nor the procedure fixes: the run's inputs ([`PostedRun`]), each
//! as the value a stage loads, and its own checkpoints with their input
//! manifests. A stage's input is then one of three things, and a challenger
//! holds all three: an earlier checkpoint's output, a commitment the
//! procedure pins, or a value posted here. So whenever two parties
//! disagree about a stage they ran it on the same inputs.
//!
//! Every posted blob names the address it was posted for and carries an
//! availability proof made by the party posting it: these bytes, which the
//! layer commits to as the blob, hold data with these commitments, the ones
//! the claim or the challenge names. For a run's inputs those are the
//! commitments a stage's input list names each one by. A contract checks the
//! proof and the blob's inclusion (Blobstream) up front. It does not judge
//! what a posted input says: the stages do that as they run, and a replay on
//! the posted input shows a stage that said otherwise.
//!
//! Here are the formats and the rules. The layer itself, Blobstream and the
//! proof are behind [`AvailabilityLayer`]; today the host stands in for all
//! three (`iris` CLI, `availability` module).

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub use prompt_core::{PromptText, RunConfig, RunSchedule, TextClaim};

use crate::checkpoint::{checkpoint_hash, checkpoints_root, CheckpointList};
use crate::checks::{is_address, ClaimSubmission, ResultInOutput, SubmittedRecord};

/// The first bytes of every blob iris posts.
pub const BLOB_MAGIC: &[u8] = b"iris/blob/v3\n";
/// A Celestia namespace: one version byte and 28 bytes of id.
/// Blockchain-specific: the size is Celestia's.
pub const NAMESPACE_BYTES: usize = 29;
/// About how long Celestia keeps a blob retrievable after its block: the
/// sampling window pruned nodes keep. To be confirmed against the network a
/// contract settles on; it is a contract parameter.
pub const DEFAULT_RETENTION_SECS: u64 = 30 * 24 * 60 * 60;
/// How long a claim stays open to challenge, by default.
pub const DEFAULT_CHALLENGE_PERIOD_SECS: u64 = 7 * 24 * 60 * 60;
/// How long a defender has to answer an admitted challenge, by default.
pub const DEFAULT_FAULT_PROOF_PERIOD_SECS: u64 = 2 * 24 * 60 * 60;

/// Where a blob is: the block it was included in, its namespace and the
/// layer's commitment to its bytes, all in lowercase hex where they are
/// bytes. This is what Blobstream's inclusion proofs are about.
/// Blockchain-specific: a height, a namespace and a commitment are how
/// Celestia names a blob.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BlobPointer {
    pub height: u64,
    pub namespace: String,
    pub commitment: String,
}

/// What a blob holds, by the commitments a claim or a challenge uses for it,
/// and who it was posted for. This is what an availability proof states.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct BlobContents {
    /// The address the blob names: the claimer or the challenger it was
    /// posted for. A submission under another address cannot use the blob.
    pub submitter: String,
    /// The commitment of each run input in the blob: what a stage's input
    /// list names it by ([`PostedRun::values`] gives their order).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub run_inputs: Vec<String>,
    /// The root of each checkpoint list in the blob
    /// ([`crate::checkpoint::checkpoints_root`]), posted with every
    /// checkpoint it lists and every input manifest those commit to.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub checkpoint_lists: Vec<String>,
    /// The digest of the header of each Raster trace commitment body in the
    /// blob: what a stage challenge names the challenger's trace by.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub trace_bodies: Vec<String>,
    /// For a checkpoint list in the blob, what the outputs of its last
    /// stage and of the checks that stage read hold: the records a
    /// submission of that claim must state.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub records: Vec<StatedRecords>,
}

/// What a claim's own checkpoints commit to, read out of two kinds of output
/// by the proof: what its last stage states, and the claim record in each
/// check that stage read. Each output is held to the hash its checkpoint
/// commits to before it is read.
///
/// A contract compares these with the submission when a claim arrives
/// ([`crate::checks::check_stated_records`]), so neither a contract nor a
/// challenger ever decodes a stage's output.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct StatedRecords {
    /// The list these are the records of.
    pub checkpoints_root: String,
    /// The program the list's last checkpoint names.
    pub result_program: String,
    /// The checkpoints the last one names as its inputs' producers, by input.
    pub result_inputs: std::collections::BTreeMap<String, u32>,
    /// What the last checkpoint's output states.
    pub results: ResultInOutput,
    /// The claim record in the output of each of those producers that holds
    /// one, with the program its checkpoint names.
    pub checks: Vec<StatedCheck>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct StatedCheck {
    /// The check's position in the list.
    pub checkpoint: u32,
    pub program: String,
    pub record: SubmittedRecord,
}

/// A stage's output as a claimer gives it to its proof: not posted, since a
/// challenger makes its own, and held by the proof to the hash the claim's
/// checkpoint at that position commits to.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct WitnessedOutput {
    pub checkpoint: u32,
    /// The stage's `output.bin`, in lowercase hex.
    pub payload: String,
    /// The stage's `output.rindex`, in lowercase hex.
    pub index: String,
}

/// What reads the Raster formats this crate does not link: a trace body's
/// header, the records in two kinds of stage output, and the commitment a
/// stage names a run input by.
pub trait OutputReader {
    /// The commitment of each of a run's inputs, in the order of
    /// [`PostedRun::values`], or `None` when one does not encode.
    fn run_inputs(&self, run: &PostedRun) -> Option<Vec<String>>;
    /// The digest of the header of a trace commitment body, or `None` when
    /// the bytes are no well-formed one.
    fn trace_header(&self, body: &[u8]) -> Option<String>;
    /// The claim record in a check stage's output.
    fn check_record(&self, payload: &[u8], index: &[u8]) -> Option<SubmittedRecord>;
    /// What a result stage's output states, of either procedure kind.
    fn results(&self, payload: &[u8], index: &[u8]) -> Option<ResultInOutput>;
}

impl BlobContents {
    /// Add what another blob holds to what these hold together.
    pub fn extend(&mut self, other: &BlobContents) {
        self.run_inputs.extend(other.run_inputs.iter().cloned());
        self.checkpoint_lists
            .extend(other.checkpoint_lists.iter().cloned());
        self.trace_bodies.extend(other.trace_bodies.iter().cloned());
        self.records.extend(other.records.iter().cloned());
    }
}

fn holds(values: &[String], wanted: &str) -> bool {
    values
        .iter()
        .any(|value| value.eq_ignore_ascii_case(wanted))
}

/// The proof that a blob holds what its [`BlobContents`] state.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum AvailabilityProof {
    /// A stand-in: the host reads the blob and recomputes its contents. A
    /// real proof is a receipt of a program that does the same, which a
    /// contract verifies against the program's pinned image id.
    ///
    /// `outputs` are the stage outputs the records were read from: a
    /// receipt's private inputs, which the stand-in carries in the open so
    /// that the host can read them again.
    HostChecked {
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        outputs: Vec<WitnessedOutput>,
        /// The attester's signature, for a contract: the host that read the
        /// blob vouches for it there.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        attestation: Option<HostAttestation>,
    },
}

/// A trusted host's word to a contract that a blob is in its block and holds
/// what its contents state: the stand-in for a receipt a contract verifies.
/// Blockchain-specific: the data root is a Celestia block's, and the
/// signature is in the form the EVM contract checks.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct HostAttestation {
    /// The data root of the block the blob is in, in lowercase hex. A
    /// contract holds it to the root its oracle has for that height.
    pub data_root: String,
    /// The attester's signature over `iris_evm::attestation_digest`: 65
    /// bytes in hex, `r`, `s` and the recovery byte (27 or 28).
    pub signature: String,
}

impl AvailabilityProof {
    /// The stand-in proof of a blob no records were read for.
    pub fn host_checked() -> Self {
        Self::HostChecked {
            outputs: Vec::new(),
            attestation: None,
        }
    }
}

/// One blob a claimer or a challenger posted, as it submits it.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PostedBlob {
    pub pointer: BlobPointer,
    /// The blob's length in bytes.
    pub length: u64,
    pub contents: BlobContents,
    pub proof: AvailabilityProof,
}

/// A blob's body: what [`BLOB_MAGIC`] is followed by, encoded with postcard.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Blob {
    /// The address the blob is posted for: `0x` and 40 hex digits.
    /// Blockchain-specific: an EVM address.
    pub submitter: String,
    pub entries: Vec<BlobEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum BlobEntry {
    /// A run's inputs: every value its stages were given that neither an
    /// earlier stage nor the procedure fixes.
    Run(PostedRun),
    Checkpoints(CheckpointPackage),
    /// A Raster trace commitment body (`commit.bin`), postcard-encoded.
    TraceBody {
        bytes: Vec<u8>,
    },
}

/// A run's inputs as a claimer posts them: each value a stage is given that
/// is neither an earlier stage's output nor the procedure's. A stage
/// loads each as a Raster file, which anyone writes from the value, and its
/// input list names it by that file's commitment.
///
/// The text is posted once, as the slices the binds are given: they
/// concatenate to the prompt. A source's documents are posted once, as the
/// chunks the source stage is given.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PostedRun {
    /// The source stage's inputs, in a run that has a source.
    pub source: Option<PostedSourceInputs>,
    /// The result stage's forms.
    pub forms: source_core::ResultForms,
    /// One per inference, in the order the procedure kind runs them.
    pub inferences: Vec<PostedInference>,
}

/// What the source stage is given: the key attestations are checked against,
/// the recipe, and the documents with their attestations.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PostedSourceInputs {
    pub signer: source_core::SignerKey,
    pub recipe: source_core::Recipe,
    pub documents: source_core::SourceDocuments,
}

/// What one inference's stages are given.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PostedInference {
    pub config: prompt_core::RunConfig,
    pub schedule: prompt_core::RunSchedule,
    /// What the run claims of the whole text.
    pub claim: prompt_core::TextClaim,
    /// One slice of the text per prompt range.
    pub text: Vec<prompt_core::PromptText>,
    /// One set of token marks per prompt range.
    pub marks: Vec<Vec<u32>>,
    /// One set of claimed output ids per claimed range.
    pub claimed: Vec<Vec<u32>>,
}

/// One of a run's inputs, with the stage input it is given as.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunInput<'a> {
    Signer(&'a source_core::SignerKey),
    Recipe(&'a source_core::Recipe),
    Documents(&'a source_core::SourceDocuments),
    Forms(&'a source_core::ResultForms),
    Config(&'a prompt_core::RunConfig),
    Schedule(&'a prompt_core::RunSchedule),
    Claim(&'a prompt_core::TextClaim),
    Text(&'a prompt_core::PromptText),
    Marks(&'a [u32]),
    Claimed(&'a [u32]),
}

impl RunInput<'_> {
    /// The name of the stage input this is given as.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Signer(_) => "signer",
            Self::Recipe(_) => "recipe",
            Self::Documents(_) => "documents",
            Self::Forms(_) => "forms",
            Self::Config(_) => "config",
            Self::Schedule(_) => "schedule",
            Self::Claim(_) => "claim",
            Self::Text(_) => "text",
            Self::Marks(_) => "marks",
            Self::Claimed(_) => "claimed",
        }
    }
}

impl PostedRun {
    /// Every input of the run, in one order: the source's, the forms, then
    /// each inference's procedure, schedule, text claim, text slices,
    /// marks and claimed ids.
    pub fn values(&self) -> Vec<RunInput<'_>> {
        let mut values = Vec::new();
        if let Some(source) = &self.source {
            values.push(RunInput::Signer(&source.signer));
            values.push(RunInput::Recipe(&source.recipe));
            values.push(RunInput::Documents(&source.documents));
        }
        values.push(RunInput::Forms(&self.forms));
        for inference in &self.inferences {
            values.push(RunInput::Config(&inference.config));
            values.push(RunInput::Schedule(&inference.schedule));
            values.push(RunInput::Claim(&inference.claim));
            values.extend(inference.text.iter().map(RunInput::Text));
            values.extend(inference.marks.iter().map(|marks| RunInput::Marks(marks)));
            values.extend(inference.claimed.iter().map(|ids| RunInput::Claimed(ids)));
        }
        values
    }
}

/// The digest a contract keeps of the run inputs a claim posted: the SHA-256
/// of their commitments, in the order its blobs state them. A challenger
/// that says an input was not posted presents the commitments, and a
/// contract holds them to this.
pub fn posted_inputs_digest(commitments: &[String]) -> Option<String> {
    let mut hasher = Sha256::new();
    for commitment in commitments {
        let bytes = hex::decode(commitment)
            .ok()
            .filter(|bytes| bytes.len() == 32)?;
        hasher.update(bytes);
    }
    Some(hex::encode(hasher.finalize()))
}

/// A claim's checkpoint list with what a challenger needs to open any
/// checkpoint in it and rule on its inputs: the checkpoints, and every input
/// manifest they commit to.
///
/// The checkpoints fix which step each position in the list is: its stage,
/// its program, its producers and the hash of its input manifest. A challenge
/// is built from the claimer's checkpoint, so a challenger cannot dispute a
/// position with an honest run of another step. The manifests say what each
/// stage loaded: without one, a stage whose inputs are not the challenger's
/// could be neither replayed nor held to its producers.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CheckpointPackage {
    /// The claim's `checkpoints.txt`, exactly.
    pub hashes: Vec<u8>,
    /// The claim's `checkpoint_list.json`, exactly.
    pub list: Vec<u8>,
    /// Each distinct `input_manifest.json` a checkpoint commits to, exactly.
    pub input_manifests: Vec<Vec<u8>>,
}

/// Why a blob's bytes are not a blob of iris's, or do not hold what they
/// should.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BlobFault {
    NotABlob,
    /// The blob names no address.
    NoSubmitter,
    /// A checkpoint package is not hashes, the checkpoints they name and
    /// their input manifests.
    MalformedCheckpoints(String),
    /// A trace body is not a well-formed Raster trace commitment.
    MalformedTraceBody,
    /// A run input is not a value a stage can be given.
    MalformedRunInput,
    /// An output given to read records from is not the one the claim's
    /// checkpoint at that position commits to, or does not hold what a last
    /// stage's output does.
    NotTheOutput {
        checkpoint: u32,
    },
}

impl std::fmt::Display for BlobFault {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotABlob => write!(f, "the bytes are not an iris blob"),
            Self::NoSubmitter => write!(f, "the blob names no address it was posted for"),
            Self::MalformedCheckpoints(why) => write!(f, "the checkpoint package {why}"),
            Self::MalformedTraceBody => {
                write!(
                    f,
                    "a trace body is not a well-formed Raster trace commitment"
                )
            }
            Self::MalformedRunInput => {
                write!(f, "a run input is not a value a stage can be given")
            }
            Self::NotTheOutput { checkpoint } => write!(
                f,
                "the output given for checkpoint {checkpoint} is not the one it commits to, or \
                 holds no records"
            ),
        }
    }
}

impl Blob {
    pub fn encode(&self) -> Vec<u8> {
        let mut bytes = BLOB_MAGIC.to_vec();
        bytes.extend(postcard::to_allocvec(self).expect("a blob encodes"));
        bytes
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, BlobFault> {
        let body = bytes.strip_prefix(BLOB_MAGIC).ok_or(BlobFault::NotABlob)?;
        let (blob, rest) =
            postcard::take_from_bytes::<Blob>(body).map_err(|_| BlobFault::NotABlob)?;
        if !rest.is_empty() {
            return Err(BlobFault::NotABlob);
        }
        // One encoding per blob: a reader must not take one set of bytes for
        // two different blobs.
        if blob.encode() != bytes {
            return Err(BlobFault::NotABlob);
        }
        Ok(blob)
    }

    /// What the blob holds, by the commitments a claim or a challenge names
    /// it by. `reader` reads the Raster formats this crate does not link.
    /// `outputs` are the stage outputs a claimer gives to have its records
    /// read: with none, the blob states no records.
    pub fn contents(
        &self,
        reader: &dyn OutputReader,
        outputs: &[WitnessedOutput],
    ) -> Result<BlobContents, BlobFault> {
        if !is_address(&self.submitter) {
            return Err(BlobFault::NoSubmitter);
        }
        let mut contents = BlobContents {
            submitter: self.submitter.clone(),
            ..BlobContents::default()
        };
        for entry in &self.entries {
            match entry {
                BlobEntry::Run(run) => {
                    contents
                        .run_inputs
                        .extend(reader.run_inputs(run).ok_or(BlobFault::MalformedRunInput)?);
                }
                BlobEntry::Checkpoints(package) => {
                    let root = package.check()?;
                    if !outputs.is_empty() {
                        contents
                            .records
                            .push(package.records(&root, reader, outputs)?);
                    }
                    contents.checkpoint_lists.push(root);
                }
                BlobEntry::TraceBody { bytes } => {
                    contents.trace_bodies.push(
                        reader
                            .trace_header(bytes)
                            .ok_or(BlobFault::MalformedTraceBody)?,
                    );
                }
            }
        }
        Ok(contents)
    }
}

impl CheckpointPackage {
    /// What the outputs of this list's last stage and of the checks it read
    /// hold. Every output given is held to the hash its checkpoint commits
    /// to, and the last stage's must be among them.
    fn records(
        &self,
        root: &str,
        reader: &dyn OutputReader,
        outputs: &[WitnessedOutput],
    ) -> Result<StatedRecords, BlobFault> {
        let list: CheckpointList = serde_json::from_slice(&self.list).map_err(|_| {
            BlobFault::MalformedCheckpoints(String::from("the checkpoint list is not JSON"))
        })?;
        // The bytes of the output given for a position, when they are the
        // ones its checkpoint commits to.
        let output = |position: u32| -> Result<Option<(Vec<u8>, Vec<u8>)>, BlobFault> {
            let Some(given) = outputs.iter().find(|given| given.checkpoint == position) else {
                return Ok(None);
            };
            let fault = BlobFault::NotTheOutput {
                checkpoint: position,
            };
            let checkpoint = list
                .checkpoints
                .get(position as usize)
                .ok_or(fault.clone())?;
            let payload = hex::decode(&given.payload).map_err(|_| fault.clone())?;
            let index = hex::decode(&given.index).map_err(|_| fault.clone())?;
            if hex::encode(Sha256::digest(&payload)) != checkpoint.output_sha256 {
                return Err(fault);
            }
            Ok(Some((payload, index)))
        };
        let last = list.checkpoints.len().saturating_sub(1) as u32;
        let fault = BlobFault::NotTheOutput { checkpoint: last };
        let last_checkpoint = list.checkpoints.last().ok_or(fault.clone())?;
        let (payload, index) = output(last)?.ok_or(fault.clone())?;
        let results = reader.results(&payload, &index).ok_or(fault)?;
        let mut checks = Vec::new();
        for position in last_checkpoint.input_sources.values() {
            let Some((payload, index)) = output(*position)? else {
                continue;
            };
            // A producer that is no check holds no record, and states none.
            if let Some(record) = reader.check_record(&payload, &index) {
                checks.push(StatedCheck {
                    checkpoint: *position,
                    program: list.checkpoints[*position as usize]
                        .program_commitment
                        .clone(),
                    record,
                });
            }
        }
        Ok(StatedRecords {
            checkpoints_root: root.to_string(),
            result_program: last_checkpoint.program_commitment.clone(),
            result_inputs: last_checkpoint.input_sources.clone(),
            results,
            checks,
        })
    }

    /// The root of the list, when the package holds every checkpoint the
    /// list names, in order, and an input manifest for each.
    pub fn check(&self) -> Result<String, BlobFault> {
        let fault = |why: &str| BlobFault::MalformedCheckpoints(why.to_string());
        let hashes =
            std::str::from_utf8(&self.hashes).map_err(|_| fault("the hashes are not text"))?;
        let lines: Vec<String> = hashes.lines().map(String::from).collect();
        let list: CheckpointList = serde_json::from_slice(&self.list)
            .map_err(|_| fault("the checkpoint list is not JSON"))?;
        if lines.len() != list.checkpoints.len() {
            return Err(fault("lists another number of checkpoints than it holds"));
        }
        let manifests: Vec<String> = self
            .input_manifests
            .iter()
            .map(|manifest| hex::encode(Sha256::digest(manifest)))
            .collect();
        for (line, checkpoint) in lines.iter().zip(&list.checkpoints) {
            let hash = checkpoint_hash(checkpoint)
                .map_err(|_| fault("holds a checkpoint that does not encode"))?;
            if *line != hash {
                return Err(fault("holds a checkpoint the list does not name"));
            }
            if !manifests.contains(&checkpoint.input_commitment) {
                return Err(fault(&format!(
                    "holds no input manifest for stage `{}`",
                    checkpoint.stage
                )));
            }
        }
        checkpoints_root(&lines).map_err(|_| fault("the hashes are not one hash per line"))
    }
}

/// The stand-in for Celestia's blob commitment: a hash over the namespace and
/// the bytes. Celestia's own is a Merkle root over the blob's shares; the
/// integration with a real network replaces this function and nothing that
/// calls it.
/// Blockchain-specific: it fills the commitment of a [`BlobPointer`], which
/// is Celestia's.
pub fn stub_blob_commitment(namespace: &[u8], bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"iris/stub-celestia-blob/v1\0");
    hasher.update(namespace);
    hasher.update((bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
    hex::encode(hasher.finalize())
}

/// The data-availability layer as a contract sees it: Blobstream's attested
/// data roots and the verifier of availability proofs.
pub trait AvailabilityLayer {
    /// The time, in Unix seconds, of the block a blob was included in, when a
    /// data root Blobstream has attested to includes the blob `pointer` names
    /// with this length. `None` otherwise.
    fn included_at(&self, pointer: &BlobPointer, length: u64) -> Option<u64>;
    /// Whether the blob's proof shows that the blob holds what its contents
    /// state.
    fn proves(&self, blob: &PostedBlob) -> bool;
}

/// What a contract requires of posted data besides the posting itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AvailabilityParameters {
    /// How long the layer keeps a blob retrievable after its block.
    pub retention_secs: u64,
    /// How long a claim stays open to challenge after it is submitted.
    pub challenge_period_secs: u64,
    /// How long a defender has to answer a challenge after it is submitted.
    pub fault_proof_period_secs: u64,
}

impl Default for AvailabilityParameters {
    fn default() -> Self {
        Self {
            retention_secs: DEFAULT_RETENTION_SECS,
            challenge_period_secs: DEFAULT_CHALLENGE_PERIOD_SECS,
            fault_proof_period_secs: DEFAULT_FAULT_PROOF_PERIOD_SECS,
        }
    }
}

/// Why a contract does not accept what was posted for a claim or a challenge.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AvailabilityRejection {
    /// The submission names no well-formed address.
    NoSubmitter {
        named: String,
    },
    NothingPosted,
    /// Blobstream has attested to no data root that includes the blob.
    NotIncluded {
        blob: usize,
    },
    /// The blob may be gone before the period it has to serve ends.
    ExpiresTooSoon {
        blob: usize,
        retrievable_until: u64,
        needed_until: u64,
    },
    /// The blob's proof does not show it holds what was stated.
    NotProven {
        blob: usize,
    },
    /// The blob was posted for another address than the submission names.
    AnotherSubmitter {
        blob: usize,
        posted_for: String,
        submitted_by: String,
    },
    /// No posted blob holds the claim's checkpoint list.
    CheckpointsNotPosted {
        checkpoints_root: String,
    },
    /// No posted blob holds the challenge's trace body.
    TraceBodyNotPosted {
        header_digest: String,
    },
}

impl std::fmt::Display for AvailabilityRejection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoSubmitter { named } => {
                write!(f, "`{named}` is not an address a submission can name")
            }
            Self::NothingPosted => write!(f, "nothing was posted to the data-availability layer"),
            Self::NotIncluded { blob } => write!(
                f,
                "blob {blob} is in no data root Blobstream has attested to"
            ),
            Self::ExpiresTooSoon {
                blob,
                retrievable_until,
                needed_until,
            } => write!(
                f,
                "blob {blob} is retrievable until {retrievable_until}, and must be until \
                 {needed_until}"
            ),
            Self::NotProven { blob } => write!(
                f,
                "blob {blob}'s availability proof does not show it holds what it states"
            ),
            Self::AnotherSubmitter {
                blob,
                posted_for,
                submitted_by,
            } => write!(
                f,
                "blob {blob} was posted for {posted_for}, and the submission names {submitted_by}"
            ),
            Self::CheckpointsNotPosted { checkpoints_root } => write!(
                f,
                "no posted blob holds the checkpoint list {checkpoints_root}"
            ),
            Self::TraceBodyNotPosted { header_digest } => write!(
                f,
                "no posted blob holds the trace body with header {header_digest}"
            ),
        }
    }
}

/// The checks every posting is held to: each blob is included in an attested
/// data root, stays retrievable until `needed_until`, its proof holds, and it
/// was posted for the address the submission names. Returns what the blobs
/// hold together.
pub fn check_posted(
    layer: &dyn AvailabilityLayer,
    retention_secs: u64,
    blobs: &[PostedBlob],
    submitter: &str,
    needed_until: u64,
) -> Result<BlobContents, AvailabilityRejection> {
    if !is_address(submitter) {
        return Err(AvailabilityRejection::NoSubmitter {
            named: submitter.to_string(),
        });
    }
    if blobs.is_empty() {
        return Err(AvailabilityRejection::NothingPosted);
    }
    let mut contents = BlobContents {
        submitter: submitter.to_string(),
        ..BlobContents::default()
    };
    for (index, blob) in blobs.iter().enumerate() {
        let included = layer
            .included_at(&blob.pointer, blob.length)
            .ok_or(AvailabilityRejection::NotIncluded { blob: index })?;
        let retrievable_until = included.saturating_add(retention_secs);
        if retrievable_until < needed_until {
            return Err(AvailabilityRejection::ExpiresTooSoon {
                blob: index,
                retrievable_until,
                needed_until,
            });
        }
        if !layer.proves(blob) {
            return Err(AvailabilityRejection::NotProven { blob: index });
        }
        if blob.contents.submitter != submitter {
            return Err(AvailabilityRejection::AnotherSubmitter {
                blob: index,
                posted_for: blob.contents.submitter.clone(),
                submitted_by: submitter.to_string(),
            });
        }
        contents.extend(&blob.contents);
    }
    Ok(contents)
}

/// What a claim must have posted, and for whom: its checkpoint list. Its
/// run inputs are not required here. A stage input that was not posted is a
/// fault of the checkpoint that names it, which a challenger points at
/// ([`crate::checks::CheckpointHashes::check_link`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequiredPosting {
    /// The address the claim names as its claimer.
    pub submitter: String,
    pub checkpoints_root: String,
}

impl RequiredPosting {
    pub fn of(submission: &ClaimSubmission) -> Self {
        Self {
            submitter: submission.claimer.clone(),
            checkpoints_root: submission.checkpoints_root.clone(),
        }
    }
}

/// The availability check on a claim submitted at `submitted_at`: what it
/// posted was posted for its claimer, is retrievable for the whole challenge
/// period, and holds its checkpoint list.
pub fn check_claim_availability(
    layer: &dyn AvailabilityLayer,
    parameters: AvailabilityParameters,
    blobs: &[PostedBlob],
    required: &RequiredPosting,
    submitted_at: u64,
) -> Result<BlobContents, AvailabilityRejection> {
    let contents = check_posted(
        layer,
        parameters.retention_secs,
        blobs,
        &required.submitter,
        submitted_at.saturating_add(parameters.challenge_period_secs),
    )?;
    check_holds(&contents, required)?;
    Ok(contents)
}

/// Whether blobs that hold `contents` together are a claim's posting: they
/// name its claimer and hold its checkpoint list. A claimer makes this check
/// on the blobs it is about to post; a contract makes it on what was posted.
pub fn check_holds(
    contents: &BlobContents,
    required: &RequiredPosting,
) -> Result<(), AvailabilityRejection> {
    if !is_address(&required.submitter) {
        return Err(AvailabilityRejection::NoSubmitter {
            named: required.submitter.clone(),
        });
    }
    if contents.submitter != required.submitter {
        return Err(AvailabilityRejection::AnotherSubmitter {
            blob: 0,
            posted_for: contents.submitter.clone(),
            submitted_by: required.submitter.clone(),
        });
    }
    if !holds(&contents.checkpoint_lists, &required.checkpoints_root) {
        return Err(AvailabilityRejection::CheckpointsNotPosted {
            checkpoints_root: required.checkpoints_root.clone(),
        });
    }
    Ok(())
}

/// The availability check on a challenge submitted at `challenged_at`: the
/// body of the trace commitment it names was posted for its challenger and is
/// retrievable for the whole time a defender has to answer.
pub fn check_challenge_availability(
    layer: &dyn AvailabilityLayer,
    parameters: AvailabilityParameters,
    blobs: &[PostedBlob],
    challenger: &str,
    header_digest: &str,
    challenged_at: u64,
) -> Result<(), AvailabilityRejection> {
    let contents = check_posted(
        layer,
        parameters.retention_secs,
        blobs,
        challenger,
        challenged_at.saturating_add(parameters.fault_proof_period_secs),
    )?;
    if !holds(&contents.trace_bodies, header_digest) {
        return Err(AvailabilityRejection::TraceBodyNotPosted {
            header_digest: header_digest.to_string(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checkpoint::{checkpoint_hashes, Checkpoint};

    const ALICE: &str = "0x00000000000000000000000000000000000a11ce";

    /// A reader for outputs written as JSON, and trace bodies with the
    /// header it is given.
    struct Reader(Option<String>);

    impl OutputReader for Reader {
        fn run_inputs(&self, run: &PostedRun) -> Option<Vec<String>> {
            // Each value by its name and place: enough to tell them apart.
            Some(
                run.values()
                    .iter()
                    .enumerate()
                    .map(|(index, value)| {
                        hex::encode(Sha256::digest(format!("{}/{index}", value.name())))
                    })
                    .collect(),
            )
        }
        fn trace_header(&self, _body: &[u8]) -> Option<String> {
            self.0.clone()
        }
        fn check_record(&self, payload: &[u8], _index: &[u8]) -> Option<SubmittedRecord> {
            serde_json::from_slice(payload).ok()
        }
        fn results(&self, payload: &[u8], _index: &[u8]) -> Option<ResultInOutput> {
            serde_json::from_slice(payload).ok()
        }
    }
    const BOB: &str = "0x0000000000000000000000000000000000000b0b";

    fn package() -> CheckpointPackage {
        let manifest = b"{}".to_vec();
        let list = CheckpointList {
            checkpoints: vec![Checkpoint {
                stage: String::from("source"),
                input_commitment: hex::encode(Sha256::digest(&manifest)),
                ..Checkpoint::default()
            }],
        };
        let hashes = checkpoint_hashes(&list).unwrap().join("\n") + "\n";
        CheckpointPackage {
            hashes: hashes.into_bytes(),
            list: serde_json::to_vec(&list).unwrap(),
            input_manifests: vec![manifest],
        }
    }

    fn blob(entries: Vec<BlobEntry>) -> Blob {
        Blob {
            submitter: ALICE.to_string(),
            entries,
        }
    }

    #[test]
    fn a_blob_reads_back_only_from_its_one_encoding() {
        let blob = blob(vec![
            BlobEntry::Checkpoints(package()),
            BlobEntry::TraceBody {
                bytes: vec![1, 2, 3],
            },
        ]);
        let bytes = blob.encode();
        assert_eq!(Blob::decode(&bytes), Ok(blob));
        let mut longer = bytes.clone();
        longer.push(0);
        assert_eq!(Blob::decode(&longer), Err(BlobFault::NotABlob));
        assert_eq!(Blob::decode(&bytes[1..]), Err(BlobFault::NotABlob));
    }

    #[test]
    fn a_blob_states_the_commitments_it_holds_and_who_it_is_for() {
        let package = package();
        let root = checkpoints_root(
            &std::str::from_utf8(&package.hashes)
                .unwrap()
                .lines()
                .map(String::from)
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let posted = blob(vec![
            BlobEntry::Checkpoints(package.clone()),
            BlobEntry::TraceBody { bytes: vec![7] },
        ]);
        let contents = posted
            .contents(&Reader(Some(String::from("aa"))), &[])
            .unwrap();
        assert_eq!(contents.submitter, ALICE);
        assert_eq!(contents.checkpoint_lists, vec![root]);
        assert_eq!(contents.trace_bodies, vec![String::from("aa")]);
        assert_eq!(
            posted.contents(&Reader(None), &[]),
            Err(BlobFault::MalformedTraceBody)
        );
        // A blob that names nobody states nothing.
        let mut unnamed = posted;
        unnamed.submitter = String::new();
        assert_eq!(
            unnamed.contents(&Reader(Some(String::new())), &[]),
            Err(BlobFault::NoSubmitter)
        );

        // A package missing a manifest, or holding a checkpoint its hashes do
        // not name, holds no checkpoint list.
        let mut no_manifest = package.clone();
        no_manifest.input_manifests.clear();
        assert!(no_manifest.check().is_err());
        let mut other_hashes = package;
        other_hashes.hashes = format!("{}\n", "00".repeat(32)).into_bytes();
        assert!(other_hashes.check().is_err());
    }

    /// A run of one inference over two prompt ranges and one claimed range.
    fn run() -> PostedRun {
        use prompt_core::PromptText;
        let slice = |text: &str| PromptText {
            chunks: vec![text.to_string()].into(),
        };
        PostedRun {
            source: None,
            forms: source_core::ResultForms::default(),
            inferences: vec![PostedInference {
                config: Default::default(),
                schedule: Default::default(),
                claim: Default::default(),
                text: vec![slice("hello "), slice("there")],
                marks: vec![vec![1, 0, 1], vec![1]],
                claimed: vec![vec![7, 8]],
            }],
        }
    }

    #[test]
    fn a_blob_states_the_commitment_of_every_run_input_it_holds() {
        let run = run();
        let names: Vec<&str> = run.values().iter().map(RunInput::name).collect();
        assert_eq!(
            names,
            ["forms", "config", "schedule", "claim", "text", "text", "marks", "marks", "claimed"]
        );
        let posted = blob(vec![BlobEntry::Run(run.clone())]);
        assert_eq!(Blob::decode(&posted.encode()), Ok(posted.clone()));
        let contents = posted.contents(&Reader(None), &[]).unwrap();
        assert_eq!(contents.run_inputs.len(), names.len());
        // The digest a contract keeps is over the commitments in order, and
        // of 32-byte commitments only.
        let digest = posted_inputs_digest(&contents.run_inputs).unwrap();
        let mut swapped = contents.run_inputs.clone();
        swapped.swap(0, 1);
        assert_ne!(posted_inputs_digest(&swapped).unwrap(), digest);
        assert_eq!(posted_inputs_digest(&[String::from("aa")]), None);
        assert_eq!(
            posted_inputs_digest(&[]).unwrap(),
            hex::encode(Sha256::digest(b""))
        );
    }

    #[test]
    fn a_proof_states_the_records_its_checkpoints_commit_to() {
        use crate::checks::{StatedInference, SubmittedResult};

        let record = |valid| SubmittedRecord {
            statement_sha256: "aa".repeat(32),
            output_text_sha256: "bb".repeat(32),
            valid,
        };
        let results = ResultInOutput {
            inferences: vec![StatedInference {
                record: record(true),
                result: SubmittedResult::default(),
            }],
            consensus: None,
        };
        let check_output = serde_json::to_vec(&record(true)).unwrap();
        let last_output = serde_json::to_vec(&results).unwrap();
        let manifest = b"{}".to_vec();
        let checkpoint = |stage: &str, output: &[u8], sources: &[(&str, u32)]| Checkpoint {
            stage: stage.to_string(),
            program_commitment: hex::encode(Sha256::digest(stage.as_bytes())),
            input_commitment: hex::encode(Sha256::digest(&manifest)),
            input_sources: sources
                .iter()
                .map(|(name, index)| (name.to_string(), *index))
                .collect(),
            output_sha256: hex::encode(Sha256::digest(output)),
            ..Checkpoint::default()
        };
        let list = CheckpointList {
            checkpoints: vec![
                checkpoint("check_c0", &check_output, &[]),
                checkpoint("result", &last_output, &[("check", 0)]),
            ],
        };
        let hashes = checkpoint_hashes(&list).unwrap().join("\n") + "\n";
        let package = CheckpointPackage {
            hashes: hashes.into_bytes(),
            list: serde_json::to_vec(&list).unwrap(),
            input_manifests: vec![manifest],
        };
        let posted = blob(vec![BlobEntry::Checkpoints(package)]);
        let given = |checkpoint: u32, bytes: &[u8]| WitnessedOutput {
            checkpoint,
            payload: hex::encode(bytes),
            index: String::new(),
        };
        let read = |outputs: &[WitnessedOutput]| posted.contents(&Reader(None), outputs);

        // With no outputs, the blob holds the list and states no records.
        let bare = read(&[]).unwrap();
        assert!(bare.records.is_empty());
        let stated = read(&[given(0, &check_output), given(1, &last_output)]).unwrap();
        assert_eq!(
            stated.records,
            vec![StatedRecords {
                checkpoints_root: bare.checkpoint_lists[0].clone(),
                result_program: list.checkpoints[1].program_commitment.clone(),
                result_inputs: list.checkpoints[1].input_sources.clone(),
                results,
                checks: vec![StatedCheck {
                    checkpoint: 0,
                    program: list.checkpoints[0].program_commitment.clone(),
                    record: record(true),
                }],
            }]
        );
        // An output that is not the one its checkpoint commits to states
        // nothing, whichever stage it is given for, and the last stage's
        // output has to be among those given.
        let other = serde_json::to_vec(&record(false)).unwrap();
        assert_eq!(
            read(&[given(0, &other), given(1, &last_output)]),
            Err(BlobFault::NotTheOutput { checkpoint: 0 })
        );
        assert_eq!(
            read(&[given(0, &check_output), given(1, &check_output)]),
            Err(BlobFault::NotTheOutput { checkpoint: 1 })
        );
        assert_eq!(
            read(&[given(0, &check_output)]),
            Err(BlobFault::NotTheOutput { checkpoint: 1 })
        );
        // Without the check's output, the check's record is not stated.
        assert!(read(&[given(1, &last_output)]).unwrap().records[0]
            .checks
            .is_empty());
    }

    /// A layer where every blob was included at time 100, and a proof holds
    /// unless the blob's commitment is `bad`.
    struct Layer;

    impl AvailabilityLayer for Layer {
        fn included_at(&self, pointer: &BlobPointer, _length: u64) -> Option<u64> {
            (pointer.commitment != "missing").then_some(100)
        }

        fn proves(&self, blob: &PostedBlob) -> bool {
            blob.pointer.commitment != "bad"
        }
    }

    fn posted(commitment: &str, contents: BlobContents) -> PostedBlob {
        PostedBlob {
            pointer: BlobPointer {
                height: 1,
                namespace: "00".repeat(NAMESPACE_BYTES),
                commitment: commitment.to_string(),
            },
            length: 1,
            contents: BlobContents {
                submitter: ALICE.to_string(),
                ..contents
            },
            proof: AvailabilityProof::host_checked(),
        }
    }

    #[test]
    fn a_claim_is_accepted_only_with_its_data_posted_for_the_whole_period() {
        let parameters = AvailabilityParameters {
            retention_secs: 1_000,
            challenge_period_secs: 500,
            fault_proof_period_secs: 200,
        };
        let list = BlobContents {
            checkpoint_lists: vec![String::from("cc")],
            run_inputs: vec![String::from("tt")],
            ..BlobContents::default()
        };
        let good = posted("ok", list.clone());
        let text = RequiredPosting {
            submitter: ALICE.to_string(),
            checkpoints_root: String::from("cc"),
        };
        let check = |blobs: &[PostedBlob], at| {
            check_claim_availability(&Layer, parameters, blobs, &text, at)
        };
        assert_eq!(
            check(std::slice::from_ref(&good), 600),
            Ok(good.contents.clone())
        );
        // Blob at 100, kept until 1,100: a claim at 601 needs it until 1,101.
        assert_eq!(
            check(std::slice::from_ref(&good), 601),
            Err(AvailabilityRejection::ExpiresTooSoon {
                blob: 0,
                retrievable_until: 1_100,
                needed_until: 1_101,
            })
        );
        assert_eq!(check(&[], 0), Err(AvailabilityRejection::NothingPosted));
        assert_eq!(
            check(&[good.clone(), posted("missing", list.clone())], 0),
            Err(AvailabilityRejection::NotIncluded { blob: 1 })
        );
        assert_eq!(
            check(&[posted("bad", list.clone())], 0),
            Err(AvailabilityRejection::NotProven { blob: 0 })
        );
        // The checkpoint list is what a claim must have posted. Its run
        // inputs are not required when it arrives: one that is missing is a
        // fault of the checkpoint that names it.
        let checkpoints_only = BlobContents {
            checkpoint_lists: vec![String::from("cc")],
            ..BlobContents::default()
        };
        assert!(check(&[posted("ok", checkpoints_only)], 0).is_ok());
        let inputs_only = BlobContents {
            run_inputs: vec![String::from("tt")],
            ..BlobContents::default()
        };
        assert_eq!(
            check(&[posted("ok", inputs_only)], 0),
            Err(AvailabilityRejection::CheckpointsNotPosted {
                checkpoints_root: String::from("cc"),
            })
        );

        // Another address cannot submit a claim over these blobs, and a
        // submission has to name one.
        let mut copied = text.clone();
        copied.submitter = BOB.to_string();
        assert_eq!(
            check_claim_availability(&Layer, parameters, std::slice::from_ref(&good), &copied, 0),
            Err(AvailabilityRejection::AnotherSubmitter {
                blob: 0,
                posted_for: ALICE.to_string(),
                submitted_by: BOB.to_string(),
            })
        );
        copied.submitter = String::from("bob");
        assert!(matches!(
            check_claim_availability(&Layer, parameters, std::slice::from_ref(&good), &copied, 0),
            Err(AvailabilityRejection::NoSubmitter { .. })
        ));

        // A challenge's body must last through the defender's window, and be
        // its challenger's.
        let body = posted(
            "ok",
            BlobContents {
                trace_bodies: vec![String::from("hh")],
                ..BlobContents::default()
            },
        );
        let challenge = |by: &str, header: &str, at| {
            check_challenge_availability(
                &Layer,
                parameters,
                std::slice::from_ref(&body),
                by,
                header,
                at,
            )
        };
        assert_eq!(challenge(ALICE, "hh", 900), Ok(()));
        assert!(matches!(
            challenge(ALICE, "hh", 901),
            Err(AvailabilityRejection::ExpiresTooSoon { .. })
        ));
        assert_eq!(
            challenge(ALICE, "ii", 0),
            Err(AvailabilityRejection::TraceBodyNotPosted {
                header_digest: String::from("ii"),
            })
        );
        assert!(matches!(
            challenge(BOB, "hh", 0),
            Err(AvailabilityRejection::AnotherSubmitter { .. })
        ));
    }
}
