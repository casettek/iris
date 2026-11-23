//! A claim's checkpoint list, and what a contract reads from it with no
//! replay: whether a stage names an approved program, and whether each of
//! its inputs is an earlier stage's output, a pinned commitment or a value
//! the claim posted. A claim that fails one of these is void on the spot.

use std::collections::BTreeMap;

use ::model_identity::context::parse_id;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::checkpoint::{checkpoint_hash, checkpoints_root, Checkpoint, CheckpointTrace};

use super::*;

/// Why a challenge is not about this claim, or not one this check can rule
/// on. Not a verdict on the claim.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChallengeRefusal {
    /// The list presented is not the checkpoint list the submission stored.
    AnotherCheckpointList {
        stored: String,
        presented: String,
    },
    /// The list is not one 32-byte hash per line.
    MalformedCheckpointList,
    /// The list has no checkpoints, so no stage and no last output.
    EmptyCheckpointList,
    UnknownStage {
        challenged: u32,
        stages: usize,
    },
    /// The checkpoint presented for a stage is not the one the list holds.
    NotTheCheckpoint {
        stage: u32,
        listed: String,
        presented: String,
    },
    /// The input list presented is not the one the stage's checkpoint commits
    /// to.
    NotTheInputManifest {
        stage: u32,
        committed: String,
        presented: String,
    },
    /// The stage names a program outside the approved set, so its inputs are
    /// not defined here. That is its own challenge, and it voids the claim.
    UnapprovedStage {
        stage: u32,
    },
    /// The input is not one the configuration lists for the stage's program.
    NotAChainedInput {
        stage: u32,
        input: String,
    },
    /// The checkpoint names a producer for the input and none was presented.
    ProducerNotPresented {
        stage: u32,
        producer: u32,
    },
}

impl ChallengeRefusal {
    /// The contract error this is. A list or a checkpoint that is not the
    /// claim's is one error there: the contract holds only the root, and a
    /// checkpoint either opens against it or does not.
    pub fn name(&self) -> &'static str {
        match self {
            Self::AnotherCheckpointList { .. }
            | Self::MalformedCheckpointList
            | Self::EmptyCheckpointList
            | Self::UnknownStage { .. }
            | Self::NotTheCheckpoint { .. } => "NotTheCheckpoint",
            Self::NotTheInputManifest { .. } => "NotTheInputManifest",
            Self::UnapprovedStage { .. } => "UnapprovedStage",
            Self::NotAChainedInput { .. } => "NotAChainedInput",
            Self::ProducerNotPresented { .. } => "ProducerNotPresented",
        }
    }
}

impl std::fmt::Display for ChallengeRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AnotherCheckpointList { stored, presented } => write!(
                f,
                "the checkpoint list presented has root {presented}, the submission stored \
                 {stored}"
            ),
            Self::MalformedCheckpointList => {
                write!(f, "the checkpoint list is not one 32-byte hash per line")
            }
            Self::EmptyCheckpointList => write!(f, "the checkpoint list is empty"),
            Self::UnknownStage { challenged, stages } => {
                write!(f, "stage {challenged} is not one of the claim's {stages}")
            }
            Self::NotTheCheckpoint {
                stage,
                listed,
                presented,
            } => write!(
                f,
                "the checkpoint presented for stage {stage} hashes to {presented}, the claim \
                 lists {listed}"
            ),
            Self::NotTheInputManifest {
                stage,
                committed,
                presented,
            } => write!(
                f,
                "the input list presented for stage {stage} hashes to {presented}, its \
                 checkpoint commits to {committed}"
            ),
            Self::UnapprovedStage { stage } => write!(
                f,
                "stage {stage} names a program outside the approved set, so it has no inputs \
                 to check"
            ),
            Self::NotAChainedInput { stage, input } => write!(
                f,
                "`{input}` is not an input the configuration lists for stage {stage}'s program"
            ),
            Self::ProducerNotPresented { stage, producer } => write!(
                f,
                "stage {stage}'s checkpoint names checkpoint {producer} as a producer, and it \
                 was not presented"
            ),
        }
    }
}

/// What a contract concludes about a challenged stage before any replay.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StageAdmission {
    /// The stage names an approved program: the challenge is an ordinary
    /// disputed one, and a defender must answer it within the window.
    Disputed { program_commitment: String },
    /// The stage names a program outside the approved set, or none: the claim
    /// is void with no replay.
    UnapprovedProgram { program_commitment: String },
}

/// Whether a stage's input is from where the configuration says it comes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LinkCheck {
    /// The input is an earlier checkpoint's output, a pinned value, or a
    /// value the claim posted.
    Linked,
    /// It is not: the claim is void with no replay.
    Void(LinkFault),
}

/// Why a stage's input is not linked to the claim's earlier checkpoints.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LinkFault {
    /// The stage's committed input list cannot be read one way only: it is
    /// not the canonical encoding of what it holds.
    UnreadableInputList,
    /// The input list has no such input.
    InputMissing,
    /// The input is not an empty value and the checkpoint names no producer
    /// for it: a value from outside the claim.
    NoProducer { loaded: String },
    /// The checkpoint names itself or a later checkpoint as the producer.
    ProducerNotEarlier { producer: u32 },
    /// The producer is not a stage program that may produce this input.
    ProducerOfWrongKind {
        producer: u32,
        program_commitment: String,
    },
    /// The input the stage loaded is not what the producer committed.
    NotTheProducersOutput {
        producer: u32,
        loaded: String,
        produced: String,
    },
    /// The input is one the program is given from fixed data, and the stage
    /// loaded a commitment the configuration does not pin for it.
    NotAPinnedValue { loaded: String },
    /// The input is one the run gives the program, and the stage loaded a
    /// commitment that is none of the run inputs the claim posted.
    NotAPostedInput { loaded: String },
}

impl LinkFault {
    /// The fault a contract records for the claim: `IrisClaims.ClaimFault`.
    /// One of [`CLAIM_FAULTS`].
    pub fn name(&self) -> &'static str {
        match self {
            Self::UnreadableInputList => "UnreadableInputList",
            Self::InputMissing => "InputMissing",
            Self::NoProducer { .. } => "NoProducer",
            Self::ProducerNotEarlier { .. } => "ProducerNotEarlier",
            Self::ProducerOfWrongKind { .. } => "ProducerOfWrongKind",
            Self::NotTheProducersOutput { .. } => "NotTheProducersOutput",
            Self::NotAPinnedValue { .. } => "NotAPinnedValue",
            Self::NotAPostedInput { .. } => "NotAPostedInput",
        }
    }
}

impl std::fmt::Display for LinkFault {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnreadableInputList => write!(
                f,
                "the input list the checkpoint commits to is not in its canonical form"
            ),
            Self::InputMissing => write!(f, "the input list does not hold the input"),
            Self::NoProducer { loaded } => write!(
                f,
                "it loaded {loaded}, which is not an empty value, and the checkpoint names no \
                 earlier stage that produced it"
            ),
            Self::ProducerNotEarlier { producer } => write!(
                f,
                "the checkpoint names checkpoint {producer} as its producer, which is not an \
                 earlier one"
            ),
            Self::ProducerOfWrongKind {
                producer,
                program_commitment,
            } => write!(
                f,
                "checkpoint {producer} ran program {program_commitment}, which may not produce \
                 this input"
            ),
            Self::NotTheProducersOutput {
                producer,
                loaded,
                produced,
            } => write!(
                f,
                "it loaded {loaded}, but checkpoint {producer} produced {produced}"
            ),
            Self::NotAPinnedValue { loaded } => write!(
                f,
                "it loaded {loaded}, which is no commitment the configuration pins for it"
            ),
            Self::NotAPostedInput { loaded } => write!(
                f,
                "it loaded {loaded}, which is none of the run inputs the claim posted"
            ),
        }
    }
}

/// One entry of a stage's `input_manifest.json`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct ManifestEntry {
    commitment: String,
    encoding: String,
    r#type: String,
}

/// Text a contract reads as it stands: printable ASCII with no quote and no
/// backslash, so that nothing in it is escaped.
fn plain(text: &str) -> bool {
    text.bytes()
        .all(|byte| (0x20..0x7f).contains(&byte) && byte != b'"' && byte != b'\\')
}

/// The commitment each input of a stage was loaded by, read from the bytes of
/// its `input_manifest.json`. `None` unless the bytes are exactly the
/// canonical encoding of what they hold, so that no two readers can take the
/// same bytes for different inputs: a repeated name, an extra field or other
/// spacing is refused. So is a name or a field that is not plain text: a
/// contract reads the list byte by byte (`InputManifest.sol`) and unescapes
/// nothing.
pub fn read_input_manifest(bytes: &[u8]) -> Option<BTreeMap<String, String>> {
    let entries: BTreeMap<String, ManifestEntry> = serde_json::from_slice(bytes).ok()?;
    let canonical = serde_json::to_vec_pretty(&serde_json::to_value(&entries).ok()?).ok()?;
    let all_plain = entries.iter().all(|(name, entry)| {
        plain(name) && plain(&entry.commitment) && plain(&entry.encoding) && plain(&entry.r#type)
    });
    (canonical == bytes && all_plain).then(|| {
        entries
            .into_iter()
            .map(|(name, entry)| (name, entry.commitment))
            .collect()
    })
}

/// A claim's checkpoint list, held to the root its submission stored. Every
/// challenge-time comparison starts from one.
///
/// The host holds the whole list. A contract holds only the root and opens
/// one checkpoint with a path
/// ([`crate::checkpoint::verify_checkpoint_opening`]); every comparison
/// here reads a checkpoint through [`CheckpointList::opened`], which is that
/// opening with the list in hand.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckpointList {
    hashes: Vec<String>,
    /// The commitment of every run input the claim posted. A contract holds
    /// their digest, and is presented the commitments with a challenge.
    posted: Vec<String>,
}

impl CheckpointList {
    /// Open the list a challenge presents: `checkpoints_txt` is the bytes of
    /// the claim's `checkpoints.txt`.
    pub fn open(
        submission: &ClaimSubmission,
        checkpoints_txt: &[u8],
    ) -> std::result::Result<Self, ChallengeRefusal> {
        let text = std::str::from_utf8(checkpoints_txt)
            .map_err(|_| ChallengeRefusal::MalformedCheckpointList)?;
        let hashes = text
            .lines()
            .map(|line| parse_id(line).map(|_| line.to_string()))
            .collect::<Option<Vec<_>>>()
            .ok_or(ChallengeRefusal::MalformedCheckpointList)?;
        if hashes.is_empty() {
            return Err(ChallengeRefusal::EmptyCheckpointList);
        }
        let presented =
            checkpoints_root(&hashes).map_err(|_| ChallengeRefusal::MalformedCheckpointList)?;
        if submission.checkpoints_root != presented
            || submission.checkpoint_count as usize != hashes.len()
        {
            return Err(ChallengeRefusal::AnotherCheckpointList {
                stored: submission.checkpoints_root.clone(),
                presented,
            });
        }
        Ok(Self {
            hashes,
            posted: submission.posted_run_inputs(),
        })
    }

    pub fn len(&self) -> usize {
        self.hashes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.hashes.is_empty()
    }

    /// Hold a presented checkpoint to the hash the list has at `stage`.
    fn opened<'a>(
        &self,
        stage: u32,
        checkpoint: &'a Checkpoint,
    ) -> std::result::Result<&'a Checkpoint, ChallengeRefusal> {
        let listed = self
            .hashes
            .get(stage as usize)
            .ok_or(ChallengeRefusal::UnknownStage {
                challenged: stage,
                stages: self.hashes.len(),
            })?;
        // A checkpoint is a struct of strings and always encodes.
        let presented = checkpoint_hash(checkpoint).unwrap_or_default();
        if *listed != presented {
            return Err(ChallengeRefusal::NotTheCheckpoint {
                stage,
                listed: listed.clone(),
                presented,
            });
        }
        Ok(checkpoint)
    }

    /// What a contract reads when a stage challenge arrives: the program id
    /// the claimer's own checkpoint names for that stage, compared with the
    /// approved set. The challenger states no program.
    pub fn admit_stage_challenge(
        &self,
        pins: &PipelinePins,
        challenged_stage: u32,
        checkpoint: &Checkpoint,
    ) -> std::result::Result<StageAdmission, ChallengeRefusal> {
        let checkpoint = self.opened(challenged_stage, checkpoint)?;
        let program_commitment = checkpoint.program_commitment.clone();
        Ok(if pins.approves(&program_commitment) {
            StageAdmission::Disputed { program_commitment }
        } else {
            StageAdmission::UnapprovedProgram { program_commitment }
        })
    }

    /// Whether one input of a stage is from where the configuration says it
    /// comes: an earlier stage's output for an input the program takes from
    /// one, a pinned commitment for an input bound to fixed data, a value
    /// the claim posted for an input the run gives.
    /// `input_manifest` is the bytes of the stage's
    /// `input_manifest.json`; `producer` is the checkpoint the stage's own
    /// checkpoint names as the input's producer, when it names one.
    ///
    /// A challenger points at the stage and the input. The claimer's
    /// checkpoint says where the input came from, so nothing here is the
    /// challenger's word. Which earlier stage is the right one is not judged:
    /// every output says which step produced it and the consuming stage
    /// checks that. This holds only that the input is a real earlier output
    /// of the right kind, or an empty value.
    pub fn check_link(
        &self,
        pins: &PipelinePins,
        stage: u32,
        checkpoint: &Checkpoint,
        input_manifest: &[u8],
        input: &str,
        producer: Option<&Checkpoint>,
    ) -> std::result::Result<LinkCheck, ChallengeRefusal> {
        let checkpoint = self.opened(stage, checkpoint)?;
        let presented = hex::encode(Sha256::digest(input_manifest));
        if checkpoint.input_commitment != presented {
            return Err(ChallengeRefusal::NotTheInputManifest {
                stage,
                committed: checkpoint.input_commitment.clone(),
                presented,
            });
        }
        let program = pins
            .program(&checkpoint.program_commitment)
            .ok_or(ChallengeRefusal::UnapprovedStage { stage })?;
        let chained = program.chained_input(input);
        let pinned = program.pinned_input(input);
        let from_run = program.run_inputs.iter().any(|run| run == input);
        // A name the program's entry does not list is not an input of it.
        if chained.is_none() && pinned.is_none() && !from_run {
            return Err(ChallengeRefusal::NotAChainedInput {
                stage,
                input: input.to_string(),
            });
        }

        let void = |fault| Ok(LinkCheck::Void(fault));
        let Some(inputs) = read_input_manifest(input_manifest) else {
            return void(LinkFault::UnreadableInputList);
        };
        let Some(loaded) = inputs.get(input) else {
            return void(LinkFault::InputMissing);
        };
        let Some(chained) = chained else {
            // Bound to fixed data: the model's, or another pinned identity.
            // The stage checks what it loads against that identity as it
            // runs, and a challenger can show a lie about that check only by
            // running the stage on the same input, which it holds only if
            // the input is a pinned one.
            if pinned.is_some_and(|pinned| pinned.values.contains(loaded)) {
                return Ok(LinkCheck::Linked);
            }
            // Given by the run: the claimer posted it, or a challenger has
            // nothing to run the stage on. What the value says is not judged
            // here. The stage checks it as it runs, and a replay on the
            // posted value shows a stage that said otherwise.
            if from_run {
                if self.posted.contains(loaded) {
                    return Ok(LinkCheck::Linked);
                }
                return void(LinkFault::NotAPostedInput {
                    loaded: loaded.clone(),
                });
            }
            return void(LinkFault::NotAPinnedValue {
                loaded: loaded.clone(),
            });
        };
        if chained.empty_values.contains(loaded) {
            return Ok(LinkCheck::Linked);
        }
        let Some(&index) = checkpoint.input_sources.get(input) else {
            return void(LinkFault::NoProducer {
                loaded: loaded.clone(),
            });
        };
        if index >= stage {
            return void(LinkFault::ProducerNotEarlier { producer: index });
        }
        let producer = self.opened(
            index,
            producer.ok_or(ChallengeRefusal::ProducerNotPresented {
                stage,
                producer: index,
            })?,
        )?;
        let of_right_kind = pins
            .program(&producer.program_commitment)
            .is_some_and(|program| chained.producers.contains(&program.stage));
        if !of_right_kind {
            return void(LinkFault::ProducerOfWrongKind {
                producer: index,
                program_commitment: producer.program_commitment.clone(),
            });
        }
        if producer.output_commitment != *loaded {
            return void(LinkFault::NotTheProducersOutput {
                producer: index,
                loaded: loaded.clone(),
                produced: producer.output_commitment.clone(),
            });
        }
        Ok(LinkCheck::Linked)
    }
}

/// The first stage of a claim that names a program outside the approved set,
/// which is the stage a challenger would point a contract at.
pub fn first_unapproved_stage(pins: &PipelinePins, trace: &CheckpointTrace) -> Option<u32> {
    trace
        .checkpoints
        .iter()
        .position(|checkpoint| !pins.approves(&checkpoint.program_commitment))
        .map(|index| index as u32)
}

/// `IrisClaims.ClaimFault` in the contract's order: the number the contract
/// holds for a void claim names the fault at that place. The first is no
/// fault. The second is [`StageAdmission::UnapprovedProgram`], and the rest
/// are the [`LinkFault`]s by [`LinkFault::name`].
pub const CLAIM_FAULTS: [&str; 10] = [
    "None",
    "UnapprovedProgram",
    "UnreadableInputList",
    "InputMissing",
    "NoProducer",
    "ProducerNotEarlier",
    "ProducerOfWrongKind",
    "NotTheProducersOutput",
    "NotAPinnedValue",
    "NotAPostedInput",
];

/// Which rule settles a difference between a claimed checkpoint and the one a
/// challenger recomputed for the same stage. Only a differing output is a
/// matter for a replay: a replay runs the claimer's own program on the
/// claimer's own inputs, so it says nothing about either.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Settlement {
    /// The checkpoints are for different stages: the lists do not line up.
    StageOrder,
    /// The claim names another program: the approved-set comparison.
    ProgramCheck,
    /// The claim ran the stage on other inputs than the challenger did, or
    /// names other producers. The challenger's own replay settles nothing
    /// here, since it is a run on inputs the claimer did not use. The link
    /// comparison voids the claim when an input taken from an earlier stage is
    /// not that stage's output. Otherwise the input is model data, a run input
    /// or a real earlier output, which the stage's own checks judge, and what
    /// is left to dispute is whether the claimed output follows from the
    /// claimer's inputs: a replay on those.
    Inputs,
    /// Same program and inputs, another output: a replay of the stage.
    Replay,
    /// Nothing differs.
    Nothing,
}

pub fn settlement(claimed: &Checkpoint, recomputed: &Checkpoint) -> Settlement {
    if claimed.stage != recomputed.stage {
        Settlement::StageOrder
    } else if claimed.program_commitment != recomputed.program_commitment {
        Settlement::ProgramCheck
    } else if claimed.input_commitment != recomputed.input_commitment
        || claimed.input_sources != recomputed.input_sources
    {
        Settlement::Inputs
    } else if claimed.output_commitment != recomputed.output_commitment
        || claimed.output_sha256 != recomputed.output_sha256
    {
        Settlement::Replay
    } else {
        Settlement::Nothing
    }
}
