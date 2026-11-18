//! What a claimer and a challenger make available, and what a contract checks
//! of it before it accepts a claim or a challenge.
//!
//! A claim names its inputs and its checkpoints by commitment. A challenger
//! who cannot get the bytes behind those commitments cannot replay anything,
//! so a claim is only accepted with its data posted to the data-availability
//! layer (Celestia), and a challenge only with its trace body posted.
//!
//! A claimer posts only what a challenger cannot make itself: the new
//! information given to the pipeline, which no configuration or model fixes,
//! and its own checkpoints. The new information is a source's documents with
//! their attestations and its recipe, or, for a run with no source, the prompt
//! text. Every file a stage loads is made from it, the configuration and the
//! model, the same way by anyone, so a challenger reruns everything itself.
//!
//! Every posted blob carries an availability proof made by the party posting
//! it: these bytes, which the layer commits to as the blob, hold data with
//! these commitments, the ones the claim or the challenge names. For the new
//! information that is the claim's statement itself: the source step run over
//! the posted signer, recipe and documents gives the source record whose digest
//! the statement names, and the posted prompt text hashes to the statement's
//! input text. A contract checks the proof and the blob's inclusion
//! (Blobstream) up front.
//!
//! Here are the formats and the rules. The layer itself, Blobstream and the
//! proof are behind [`AvailabilityLayer`]; today the host stands in for all
//! three (`iris` CLI, `availability` module).

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::checkpoint::{checkpoint_hash, CheckpointTrace};
use crate::contract::{source_record_id, ClaimSubmission};

/// The first bytes of every blob iris posts.
pub const BLOB_MAGIC: &[u8] = b"iris/blob/v1\n";
/// A Celestia namespace: one version byte and 28 bytes of id.
pub const NAMESPACE_BYTES: usize = 29;
/// About how long Celestia keeps a blob retrievable after its block: the
/// sampling window pruned nodes keep. To be confirmed against the network a
/// contract settles on; it is a contract parameter.
pub const DEFAULT_RETENTION_SECS: u64 = 30 * 24 * 60 * 60;
/// How long a claim stays open to challenge, by default.
pub const DEFAULT_CHALLENGE_PERIOD_SECS: u64 = 7 * 24 * 60 * 60;
/// How long a defender has to answer an admitted challenge, by default.
pub const DEFAULT_FAULT_PROOF_WINDOW_SECS: u64 = 2 * 24 * 60 * 60;

/// Where a blob is: the block it was included in, its namespace and the
/// layer's commitment to its bytes, all in lowercase hex where they are
/// bytes. This is what Blobstream's inclusion proofs are about.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BlobPointer {
    pub height: u64,
    pub namespace: String,
    pub commitment: String,
}

/// What a blob holds, by the commitments a claim or a challenge uses for it.
/// This is what an availability proof states.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct BlobContents {
    /// For each source in the blob, the digest of the record the source step
    /// gives over it: what a statement names a source by.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub sources: Vec<String>,
    /// The SHA-256 of each prompt text in the blob: what a statement names
    /// its input text by.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub prompt_texts: Vec<String>,
    /// The SHA-256 of each checkpoint list in the blob, posted with every
    /// checkpoint it lists.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub checkpoint_lists: Vec<String>,
    /// The digest of the header of each Raster trace commitment body in the
    /// blob: what a stage challenge names the challenger's trace by.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub trace_bodies: Vec<String>,
}

impl BlobContents {
    fn extend(&mut self, other: &BlobContents) {
        self.sources.extend(other.sources.iter().cloned());
        self.prompt_texts.extend(other.prompt_texts.iter().cloned());
        self.checkpoint_lists
            .extend(other.checkpoint_lists.iter().cloned());
        self.trace_bodies.extend(other.trace_bodies.iter().cloned());
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
    HostChecked,
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
    pub entries: Vec<BlobEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum BlobEntry {
    /// A run's source: the documents with their attestations, the recipe,
    /// and the key they were checked against.
    Source(PostedSource),
    /// The prompt text of a run with no source, exactly as given.
    PromptText {
        text: String,
    },
    Checkpoints(CheckpointPackage),
    /// A Raster trace commitment body (`commit.bin`), postcard-encoded.
    TraceBody {
        bytes: Vec<u8>,
    },
}

/// A source as a claimer posts it. The key is the configuration's, posted so
/// that the source step can be run over the blob alone.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PostedSource {
    pub signer: source_core::SignerKey,
    pub recipe: source_core::Recipe,
    /// In the order the recipe names them.
    pub documents: Vec<PostedDocument>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PostedDocument {
    pub attestation: source_core::Attestation,
    /// The document's bytes, as the server sent them.
    pub body: Vec<u8>,
}

impl PostedSource {
    /// The record the source step states over this source: what the
    /// statement of a claim built from it names.
    pub fn record(&self) -> source_core::SourceRecord {
        let documents: Vec<source_core::SourceDocument> = self
            .documents
            .iter()
            .map(|document| source_core::SourceDocument {
                attestation: &document.attestation,
                body: &document.body,
            })
            .collect();
        source_core::run_source(&self.signer, &self.recipe, &documents).record
    }
}

/// A claim's checkpoint list with every checkpoint it lists.
///
/// The checkpoints fix which step each position in the list is: its stage,
/// its program, its producers and the hash of its input manifest. A challenge
/// is built from the claimer's checkpoint, so a challenger cannot dispute a
/// position with an honest run of another step. The input manifests are not
/// posted: an honest claimer's are the ones a challenger makes, and a
/// replay is about a checkpoint whose manifest hash is the challenger's own.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CheckpointPackage {
    /// The claim's `checkpoints.txt`, exactly.
    pub list: Vec<u8>,
    /// The claim's `checkpoint_trace.json`, exactly.
    pub trace: Vec<u8>,
}

/// Why a blob's bytes are not a blob of iris's, or do not hold what they
/// should.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BlobFault {
    NotABlob,
    /// A checkpoint package is not a list and the checkpoints it lists.
    MalformedCheckpoints(String),
    /// A trace body is not a well-formed Raster trace commitment.
    MalformedTraceBody,
}

impl std::fmt::Display for BlobFault {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotABlob => write!(f, "the bytes are not an iris blob"),
            Self::MalformedCheckpoints(why) => write!(f, "the checkpoint package {why}"),
            Self::MalformedTraceBody => {
                write!(
                    f,
                    "a trace body is not a well-formed Raster trace commitment"
                )
            }
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
    /// it by, as far as this crate can compute them. A trace body is listed
    /// by the header digest `trace_header` gives for its bytes, or refused
    /// when it gives none: computing it is Raster's prover's, which this
    /// crate does not link.
    pub fn contents(
        &self,
        trace_header: impl Fn(&[u8]) -> Option<String>,
    ) -> Result<BlobContents, BlobFault> {
        let mut contents = BlobContents::default();
        for entry in &self.entries {
            match entry {
                BlobEntry::Source(source) => {
                    contents.sources.push(source_record_id(&source.record()));
                }
                BlobEntry::PromptText { text } => {
                    contents
                        .prompt_texts
                        .push(hex::encode(Sha256::digest(text.as_bytes())));
                }
                BlobEntry::Checkpoints(package) => {
                    contents.checkpoint_lists.push(package.check()?);
                }
                BlobEntry::TraceBody { bytes } => {
                    contents
                        .trace_bodies
                        .push(trace_header(bytes).ok_or(BlobFault::MalformedTraceBody)?);
                }
            }
        }
        Ok(contents)
    }
}

impl CheckpointPackage {
    /// The SHA-256 of the list, when the package holds every checkpoint the
    /// list names, in order.
    pub fn check(&self) -> Result<String, BlobFault> {
        let fault = |why: &str| BlobFault::MalformedCheckpoints(why.to_string());
        let list = std::str::from_utf8(&self.list).map_err(|_| fault("list is not text"))?;
        let lines: Vec<&str> = list.lines().collect();
        let trace: CheckpointTrace =
            serde_json::from_slice(&self.trace).map_err(|_| fault("trace is not JSON"))?;
        if lines.len() != trace.checkpoints.len() {
            return Err(fault("lists another number of checkpoints than it holds"));
        }
        for (line, checkpoint) in lines.iter().zip(&trace.checkpoints) {
            let hash = checkpoint_hash(checkpoint)
                .map_err(|_| fault("holds a checkpoint that does not encode"))?;
            if *line != hash {
                return Err(fault("holds a checkpoint the list does not name"));
            }
        }
        Ok(hex::encode(Sha256::digest(&self.list)))
    }
}

/// The stand-in for Celestia's blob commitment: a hash over the namespace and
/// the bytes. Celestia's own is a Merkle root over the blob's shares; the
/// integration with a real network replaces this function and nothing that
/// calls it.
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
    pub fault_proof_window_secs: u64,
}

impl Default for AvailabilityParameters {
    fn default() -> Self {
        Self {
            retention_secs: DEFAULT_RETENTION_SECS,
            challenge_period_secs: DEFAULT_CHALLENGE_PERIOD_SECS,
            fault_proof_window_secs: DEFAULT_FAULT_PROOF_WINDOW_SECS,
        }
    }
}

/// Why a contract does not accept what was posted for a claim or a challenge.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AvailabilityRejection {
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
    /// No posted blob holds the claim's checkpoint list.
    CheckpointsNotPosted {
        checkpoints_sha256: String,
    },
    /// No posted blob holds a source whose record is the one the statement
    /// names.
    SourceNotPosted {
        source_sha256: String,
    },
    /// No posted blob holds the prompt text the statement names.
    PromptTextNotPosted {
        input_text_sha256: String,
    },
    /// No posted blob holds the challenge's trace body.
    TraceBodyNotPosted {
        header_digest: String,
    },
}

impl std::fmt::Display for AvailabilityRejection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
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
            Self::CheckpointsNotPosted { checkpoints_sha256 } => write!(
                f,
                "no posted blob holds the checkpoint list {checkpoints_sha256}"
            ),
            Self::SourceNotPosted { source_sha256 } => write!(
                f,
                "no posted blob holds a source whose record is {source_sha256}, the one the \
                 statement names"
            ),
            Self::PromptTextNotPosted { input_text_sha256 } => write!(
                f,
                "no posted blob holds the prompt text {input_text_sha256} the statement names"
            ),
            Self::TraceBodyNotPosted { header_digest } => write!(
                f,
                "no posted blob holds the trace body with header {header_digest}"
            ),
        }
    }
}

/// The checks every posting is held to: each blob is included in an attested
/// data root, stays retrievable until `needed_until`, and its proof holds.
/// Returns what the blobs hold together.
pub fn check_posted(
    layer: &dyn AvailabilityLayer,
    retention_secs: u64,
    blobs: &[PostedBlob],
    needed_until: u64,
) -> Result<BlobContents, AvailabilityRejection> {
    if blobs.is_empty() {
        return Err(AvailabilityRejection::NothingPosted);
    }
    let mut contents = BlobContents::default();
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
        contents.extend(&blob.contents);
    }
    Ok(contents)
}

/// What a claim must have posted: its new information, as its statement
/// names it, and its checkpoint list.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequiredPosting {
    pub new_information: NewInformation,
    pub checkpoints_sha256: String,
}

/// The new information a statement names.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NewInformation {
    /// A source whose record has this digest.
    Source { source_sha256: String },
    /// A prompt text with this SHA-256, for a run with no source.
    PromptText { input_text_sha256: String },
}

impl RequiredPosting {
    pub fn of(submission: &ClaimSubmission) -> Self {
        let new_information = if submission.source_sha256 == crate::inference::no_source_id() {
            NewInformation::PromptText {
                input_text_sha256: submission.input_text_sha256.clone(),
            }
        } else {
            NewInformation::Source {
                source_sha256: submission.source_sha256.clone(),
            }
        };
        Self {
            new_information,
            checkpoints_sha256: submission.checkpoints_sha256.clone(),
        }
    }
}

/// The availability check on a claim submitted at `submitted_at`: what it
/// posted is retrievable for the whole challenge period, and holds the new
/// information its statement names and its checkpoint list.
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
        submitted_at.saturating_add(parameters.challenge_period_secs),
    )?;
    match &required.new_information {
        NewInformation::Source { source_sha256 } => {
            if !holds(&contents.sources, source_sha256) {
                return Err(AvailabilityRejection::SourceNotPosted {
                    source_sha256: source_sha256.clone(),
                });
            }
        }
        NewInformation::PromptText { input_text_sha256 } => {
            if !holds(&contents.prompt_texts, input_text_sha256) {
                return Err(AvailabilityRejection::PromptTextNotPosted {
                    input_text_sha256: input_text_sha256.clone(),
                });
            }
        }
    }
    if !holds(&contents.checkpoint_lists, &required.checkpoints_sha256) {
        return Err(AvailabilityRejection::CheckpointsNotPosted {
            checkpoints_sha256: required.checkpoints_sha256.clone(),
        });
    }
    Ok(contents)
}

/// The availability check on a challenge submitted at `challenged_at`: the
/// body of the trace commitment it names is retrievable for the whole time a
/// defender has to answer.
pub fn check_challenge_availability(
    layer: &dyn AvailabilityLayer,
    parameters: AvailabilityParameters,
    blobs: &[PostedBlob],
    header_digest: &str,
    challenged_at: u64,
) -> Result<(), AvailabilityRejection> {
    let contents = check_posted(
        layer,
        parameters.retention_secs,
        blobs,
        challenged_at.saturating_add(parameters.fault_proof_window_secs),
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

    fn package() -> CheckpointPackage {
        let trace = CheckpointTrace {
            checkpoints: vec![Checkpoint {
                stage: String::from("source"),
                input_commitment: hex::encode(Sha256::digest(b"{}")),
                ..Checkpoint::default()
            }],
        };
        let list = checkpoint_hashes(&trace).unwrap().join("\n") + "\n";
        CheckpointPackage {
            list: list.into_bytes(),
            trace: serde_json::to_vec(&trace).unwrap(),
        }
    }

    #[test]
    fn a_blob_reads_back_only_from_its_one_encoding() {
        let blob = Blob {
            entries: vec![
                BlobEntry::Checkpoints(package()),
                BlobEntry::TraceBody {
                    bytes: vec![1, 2, 3],
                },
            ],
        };
        let bytes = blob.encode();
        assert_eq!(Blob::decode(&bytes), Ok(blob));
        let mut longer = bytes.clone();
        longer.push(0);
        assert_eq!(Blob::decode(&longer), Err(BlobFault::NotABlob));
        assert_eq!(Blob::decode(&bytes[1..]), Err(BlobFault::NotABlob));
    }

    #[test]
    fn a_blob_states_the_commitments_it_holds() {
        let package = package();
        let list_sha = hex::encode(Sha256::digest(&package.list));
        let blob = Blob {
            entries: vec![
                BlobEntry::Checkpoints(package.clone()),
                BlobEntry::TraceBody { bytes: vec![7] },
            ],
        };
        let contents = blob.contents(|_| Some(String::from("aa"))).unwrap();
        assert_eq!(contents.checkpoint_lists, vec![list_sha]);
        assert_eq!(contents.trace_bodies, vec![String::from("aa")]);
        assert_eq!(blob.contents(|_| None), Err(BlobFault::MalformedTraceBody));

        // A package holding a checkpoint the list does not name, or missing
        // one, holds no list.
        let mut missing = package.clone();
        missing.trace = b"[]".to_vec();
        assert!(missing.check().is_err());
        let mut other_list = package;
        other_list.list = format!("{}\n", "00".repeat(32)).into_bytes();
        assert!(other_list.check().is_err());

        // A prompt text is named by its SHA-256.
        let prompt = Blob {
            entries: vec![BlobEntry::PromptText {
                text: String::from("hello"),
            }],
        };
        assert_eq!(
            prompt.contents(|_| None).unwrap().prompt_texts,
            vec![hex::encode(Sha256::digest(b"hello"))]
        );
    }

    /// A source signed by a test key: one document, a recipe that takes one
    /// value from it.
    fn source() -> PostedSource {
        let secret = [7u8; 32];
        let signer = source_core::signer_key(&secret).unwrap();
        let body = br#"{"magnitude":"6.1"}"#.to_vec();
        let mut attestation = source_core::Attestation {
            scheme: signer.scheme,
            time: 1_700_000_000,
            server: String::from("earthquake.usgs.gov"),
            target: String::from("/event"),
            records_sha256: source_core::hex(&source_core::sha256(&body)),
            records_length: body.len() as u64,
            ..source_core::Attestation::default()
        };
        source_core::sign_attestation(&secret, &mut attestation).unwrap();
        PostedSource {
            recipe: source_core::Recipe {
                base: source_core::text_chunks("Magnitude: ").into(),
                documents: vec![source_core::DocumentName {
                    server: attestation.server.clone(),
                    target: attestation.target.clone(),
                }]
                .into(),
                fields: vec![source_core::Field {
                    lead: String::new(),
                    document: 0,
                    path: String::from("magnitude"),
                }]
                .into(),
                tail: String::from("."),
            },
            signer,
            documents: vec![PostedDocument { attestation, body }],
        }
    }

    #[test]
    fn a_posted_source_is_named_by_the_record_the_source_step_gives() {
        let source = source();
        let record = source.record();
        assert_eq!(record.fault, 0, "the test source is not a clean one");
        let blob = Blob {
            entries: vec![BlobEntry::Source(source.clone())],
        };
        assert_eq!(Blob::decode(&blob.encode()), Ok(blob.clone()));
        assert_eq!(
            blob.contents(|_| None).unwrap().sources,
            vec![source_record_id(&record)]
        );
        // Another document is another record.
        let mut other = source;
        other.documents[0].body = br#"{"magnitude":"7.1"}"#.to_vec();
        assert_ne!(source_record_id(&other.record()), source_record_id(&record));
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
            contents,
            proof: AvailabilityProof::HostChecked,
        }
    }

    #[test]
    fn a_claim_is_accepted_only_with_its_data_posted_for_the_whole_period() {
        let parameters = AvailabilityParameters {
            retention_secs: 1_000,
            challenge_period_secs: 500,
            fault_proof_window_secs: 200,
        };
        let list = BlobContents {
            checkpoint_lists: vec![String::from("cc")],
            prompt_texts: vec![String::from("tt")],
            ..BlobContents::default()
        };
        let good = posted("ok", list.clone());
        let required = |new_information| RequiredPosting {
            new_information,
            checkpoints_sha256: String::from("cc"),
        };
        let text = required(NewInformation::PromptText {
            input_text_sha256: String::from("tt"),
        });
        let check = |blobs: &[PostedBlob], at| {
            check_claim_availability(&Layer, parameters, blobs, &text, at)
        };
        assert_eq!(check(std::slice::from_ref(&good), 600), Ok(list.clone()));
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
            check(&[posted("bad", list)], 0),
            Err(AvailabilityRejection::NotProven { blob: 0 })
        );
        let checkpoints_only = BlobContents {
            checkpoint_lists: vec![String::from("cc")],
            ..BlobContents::default()
        };
        assert_eq!(
            check(&[posted("ok", checkpoints_only.clone())], 0),
            Err(AvailabilityRejection::PromptTextNotPosted {
                input_text_sha256: String::from("tt"),
            })
        );
        let text_only = BlobContents {
            prompt_texts: vec![String::from("tt")],
            ..BlobContents::default()
        };
        assert_eq!(
            check(&[posted("ok", text_only)], 0),
            Err(AvailabilityRejection::CheckpointsNotPosted {
                checkpoints_sha256: String::from("cc"),
            })
        );
        // A claim with a source needs the source, not a text.
        let sourced = required(NewInformation::Source {
            source_sha256: String::from("ss"),
        });
        assert_eq!(
            check_claim_availability(&Layer, parameters, std::slice::from_ref(&good), &sourced, 0),
            Err(AvailabilityRejection::SourceNotPosted {
                source_sha256: String::from("ss"),
            })
        );
        let with_source = BlobContents {
            sources: vec![String::from("ss")],
            ..checkpoints_only
        };
        assert!(check_claim_availability(
            &Layer,
            parameters,
            &[posted("ok", with_source)],
            &sourced,
            0
        )
        .is_ok());

        // A challenge's body must last through the defender's window.
        let body = posted(
            "ok",
            BlobContents {
                trace_bodies: vec![String::from("hh")],
                ..BlobContents::default()
            },
        );
        let challenge = |at| {
            check_challenge_availability(&Layer, parameters, std::slice::from_ref(&body), "hh", at)
        };
        assert_eq!(challenge(900), Ok(()));
        assert!(matches!(
            challenge(901),
            Err(AvailabilityRejection::ExpiresTooSoon { .. })
        ));
        assert_eq!(
            check_challenge_availability(&Layer, parameters, std::slice::from_ref(&body), "ii", 0),
            Err(AvailabilityRejection::TraceBodyNotPosted {
                header_digest: String::from("ii"),
            })
        );
    }
}
