//! A dispute about one stage's execution: the Raster commitment a checkpoint
//! stands as, the challenge a challenger submits against it, and the rule
//! for when a fault proof answers that challenge.

use std::collections::BTreeMap;

use ::common_identity::context::parse_id;
use raster_core::pipeline::{
    defend, DefenceInput, DefenceJournal, DefenceRefusal, InputBindingSource, PipelineCommitment,
    PipelineShape, StageChallenge, StageCheckpoint,
};
use raster_core::transition::{TraceCommitmentHeader, TransitionJournal, TransitionState};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::availability::PostedBlob;
use procedure_definition::checkpoint::{Checkpoint, CheckpointList};

/// Where a claim's checkpoints are not the boundaries a Raster pipeline run of
/// the same claim committed, or `None` when every one is: the same stages in
/// the same order, each with Raster's program id, input manifest commitment,
/// producers and output commitments.
pub fn raster_boundary_mismatch(
    list: &CheckpointList,
    pipeline: &PipelineCommitment,
) -> Option<String> {
    if list.checkpoints.len() != pipeline.stages.len() {
        return Some(format!(
            "the claim has {} checkpoints, the Raster run {} stages",
            list.checkpoints.len(),
            pipeline.stages.len()
        ));
    }
    for (index, (checkpoint, stage)) in list.checkpoints.iter().zip(&pipeline.stages).enumerate() {
        let raster_sources = stage
            .input_bindings
            .iter()
            .filter_map(|(input, source)| match source {
                InputBindingSource::Piped { stage } => Some(format!("{input}={stage}")),
                InputBindingSource::External => None,
            })
            .collect::<Vec<_>>()
            .join(" ");
        let claimed_sources = checkpoint
            .input_sources
            .iter()
            .map(|(input, stage)| format!("{input}={stage}"))
            .collect::<Vec<_>>()
            .join(" ");
        for (field, claimed, raster) in [
            ("stage name", checkpoint.stage.clone(), stage.name.clone()),
            (
                "program",
                checkpoint.program_commitment.clone(),
                hex::encode(&stage.program_commitment),
            ),
            (
                "input manifest commitment",
                checkpoint.input_commitment.clone(),
                hex::encode(&stage.input_manifest_commitment),
            ),
            ("producers", claimed_sources, raster_sources),
            (
                "output structural commitment",
                checkpoint.output_commitment.clone(),
                hex::encode(&stage.output_structural_commitment),
            ),
            (
                "output hash",
                checkpoint.output_sha256.clone(),
                hex::encode(&stage.output_payload_commitment),
            ),
        ] {
            if claimed != raster {
                return Some(format!(
                    "stage {index} ({}): the checkpoint's {field} is {claimed}, the Raster \
                     run's is {raster}",
                    checkpoint.stage
                ));
            }
        }
    }
    None
}

/// The Raster pipeline commitment that stands for one checkpoint in a dispute.
///
/// Raster's challenge and defence name a pipeline commitment and a stage of it,
/// and read three things of that stage: its program, its input manifest and
/// the hash of its output. A checkpoint holds exactly those. So a dispute
/// about a checkpoint is a dispute about a pipeline of one stage built from it,
/// with nothing else in it: no bindings, no manifest digest and no repeats.
///
/// A contract recomputes these bytes from the checkpoint it opened, so the
/// digest a Raster proof names says which checkpoint the proof is about. The
/// encoding is postcard, which for this value is a plain concatenation:
///
/// ```text
/// 01                      one stage
/// len name                the stage name, length as a varint
/// 20 program              32 bytes
/// 20 input manifest       32 bytes
/// 00                      no bindings
/// 20 output hash          32 bytes
/// 20 output structural    32 bytes
/// 00                      no manifest digest
/// 00                      no repeats
/// ```
///
/// `None` when a field is not the 32-byte hash it should be, which no honest
/// checkpoint is and no contract accepts.
pub fn dispute_commitment(checkpoint: &Checkpoint) -> Option<Vec<u8>> {
    let hash = |text: &str| parse_id(text).map(|bytes| bytes.to_vec());
    let pipeline = PipelineCommitment {
        stages: vec![StageCheckpoint {
            name: checkpoint.stage.clone(),
            program_commitment: hash(&checkpoint.program_commitment)?,
            input_manifest_commitment: hash(&checkpoint.input_commitment)?,
            input_bindings: BTreeMap::new(),
            output_payload_commitment: hash(&checkpoint.output_sha256)?,
            output_structural_commitment: hash(&checkpoint.output_commitment)?,
        }],
        shape: PipelineShape {
            spec_digest: Vec::new(),
            repeats: Vec::new(),
        },
    };
    postcard::to_allocvec(&pipeline).ok()
}

/// The digest a Raster challenge or defence about this checkpoint names.
pub fn dispute_digest(checkpoint: &Checkpoint) -> Option<[u8; 32]> {
    dispute_commitment(checkpoint).map(|bytes| Sha256::digest(bytes).into())
}

/// The Raster stage challenge that disputes a checkpoint's output:
/// `trace_commitment` is the header of the challenger's own authenticated run
/// of the stage, and `claimed_output_sha256` the hash of the output that run
/// ends in. `None` when the checkpoint or the hash is malformed.
pub fn stage_challenge(
    checkpoint: &Checkpoint,
    trace_commitment: TraceCommitmentHeader,
    claimed_output_sha256: &str,
) -> Option<StageChallenge> {
    Some(StageChallenge {
        pipeline_commitment_bytes: dispute_commitment(checkpoint)?,
        challenged_stage: 0,
        trace_commitment_header: trace_commitment,
        claimed_output: parse_id(claimed_output_sha256)?.to_vec(),
    })
}

/// Written beside a challenge's `challenge_bundle.json`.
pub const CHALLENGE_SUBMISSION_JSON: &str = "challenge_submission.json";

pub const CHALLENGE_SUBMISSION_VERSION: u32 = 2;

/// What a challenger submits: the claim and the stage it disputes, the
/// Raster stage challenge about that stage's checkpoint, and where the body
/// of its trace commitment is posted, so a defender can read it.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChallengeSubmission {
    pub version: u32,
    /// The address the challenge is attributed to, as a claim names its
    /// claimer. The posted trace body names it.
    pub challenger: String,
    /// The disputed claim's checkpoint list, as its submission stored it.
    pub checkpoints_root: String,
    /// The position of the disputed checkpoint in that list.
    pub stage: u32,
    /// The one-stage pipeline commitment built from the claimer's checkpoint,
    /// the header of the challenger's own trace commitment of the stage, and
    /// the output its run ends in ([`stage_challenge`]).
    pub challenge: StageChallenge,
    /// The posted trace body, with its availability proof.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub availability: Vec<PostedBlob>,
}

impl ChallengeSubmission {
    /// The digest of the challenger's trace commitment header: what the
    /// posted body must be the body of.
    pub fn trace_header_digest(&self) -> String {
        hex::encode(self.challenge.trace_commitment_header.digest())
    }
}

/// Whether a stage challenge is about this checkpoint: the one-stage pipeline
/// commitment it carries is the checkpoint's own.
pub fn challenge_is_about(checkpoint: &Checkpoint, challenge: &StageChallenge) -> bool {
    challenge.challenged_stage == 0
        && dispute_commitment(checkpoint)
            .is_some_and(|commitment| commitment == challenge.pipeline_commitment_bytes)
}

/// Why a proven window does not dismiss a challenge to a checkpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DefenceRejection {
    /// The checkpoint is not one a dispute can be built from.
    MalformedCheckpoint,
    /// The challenge is about another checkpoint than this one.
    AnotherCheckpoint,
    /// Raster's defence rule refuses the window for this challenge.
    Refused(DefenceRefusal),
}

impl std::fmt::Display for DefenceRejection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MalformedCheckpoint => {
                write!(
                    f,
                    "the checkpoint's fields are not the hashes a dispute names"
                )
            }
            Self::AnotherCheckpoint => {
                write!(f, "the challenge is not about this checkpoint")
            }
            Self::Refused(refusal) => write!(f, "{refusal}"),
        }
    }
}

/// Whether a proven window dismisses a challenge to a checkpoint.
///
/// A proof that knocks down the challenger's trace is not enough: it has to
/// be a proof about this stage. Otherwise a window proven over another
/// program, or over other inputs, would dismiss an honest challenge, since
/// the challenger's trace always diverges from a run it is not a trace of.
/// The rule is Raster's own (`raster_core::pipeline::defend`), applied to the
/// one-stage commitment built from the checkpoint: the window was proven
/// against the challenger's trace commitment, runs the checkpoint's program
/// on the checkpoint's inputs, and ends either in a divergence or, on an
/// honest trace, in an output other than the one the challenger stated.
///
/// `journal` is the journal of a receipt the caller has already verified
/// against `transition_image_id`. The returned journal is what a defence
/// receipt would commit to; its pipeline commitment digest is this checkpoint's
/// [`dispute_digest`].
pub fn check_defence(
    checkpoint: &Checkpoint,
    challenge: &StageChallenge,
    journal: &TransitionJournal,
    transition_image_id: &[u8],
) -> std::result::Result<DefenceJournal, DefenceRejection> {
    let commitment = dispute_commitment(checkpoint).ok_or(DefenceRejection::MalformedCheckpoint)?;
    if challenge.pipeline_commitment_bytes != commitment || challenge.challenged_stage != 0 {
        return Err(DefenceRejection::AnotherCheckpoint);
    }
    let challenge_bytes =
        postcard::to_allocvec(challenge).map_err(|_| DefenceRejection::MalformedCheckpoint)?;
    defend(&DefenceInput {
        challenge_bytes,
        transition_journal: journal.clone(),
        transition_image_id: transition_image_id.to_vec(),
    })
    .map_err(DefenceRejection::Refused)
}

/// Why a Raster fault proof does not dismiss a challenge a contract holds.
/// Each is the error of the same name in `IrisClaims.submitFaultProof`, which
/// makes these checks in this order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FaultProofRejection {
    /// The challenge's own commitment of the disputed stage does not decode.
    MalformedChallenge,
    /// The proven window has not ended at a step the trace gets wrong.
    NotAFaultVerdict,
    /// The window's earlier steps were checked by another transition program.
    TransitionImageIdMismatch,
    /// The window's inputs were checked by another authorization program.
    AuthorizationImageIdMismatch,
    /// The window refutes another trace than the one the challenge posted.
    AnotherTraceCommitment,
    /// The window runs another program than the disputed checkpoint's.
    AnotherProgram,
    /// The window runs on other inputs than the disputed checkpoint's.
    AnotherInputManifest,
}

impl FaultProofRejection {
    /// The contract error this is.
    pub fn name(&self) -> &'static str {
        match self {
            Self::MalformedChallenge => "MalformedChallenge",
            Self::NotAFaultVerdict => "NotAFaultVerdict",
            Self::TransitionImageIdMismatch => "TransitionImageIdMismatch",
            Self::AuthorizationImageIdMismatch => "AuthorizationImageIdMismatch",
            Self::AnotherTraceCommitment => "AnotherTraceCommitment",
            Self::AnotherProgram => "AnotherProgram",
            Self::AnotherInputManifest => "AnotherInputManifest",
        }
    }
}

impl std::fmt::Display for FaultProofRejection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let text = match self {
            Self::MalformedChallenge => "the challenge's commitment of its stage does not decode",
            Self::NotAFaultVerdict => "the proof has not reached a fault verdict",
            Self::TransitionImageIdMismatch => {
                "the proof's earlier steps were checked by another transition program"
            }
            Self::AuthorizationImageIdMismatch => {
                "the proof's inputs were checked by another authorization program"
            }
            Self::AnotherTraceCommitment => {
                "the proof refutes another trace than the one the challenge posted"
            }
            Self::AnotherProgram => "the proof runs another program than the checkpoint's",
            Self::AnotherInputManifest => "the proof runs on other inputs than the checkpoint's",
        };
        write!(f, "{text} ({})", self.name())
    }
}

/// Whether a Raster fault proof dismisses a challenge, as a contract decides
/// it (`IrisClaims.submitFaultProof`).
///
/// `journal` is the journal of a receipt the caller has verified against the
/// transition program. The two image ids are the ones a contract pins: the
/// journal names the programs its earlier steps and its inputs were checked
/// by, and a proof that names others was checked by nothing known.
///
/// These are the checks of `raster_core::pipeline::defend` for a refutation, on
/// what a contract holds of the challenge: the digest of the trace it posted
/// and the disputed checkpoint's program and inputs. A window that ends by
/// matching the trace (`TerminalMatch`), which Raster takes as a defence when
/// the trace's output is not the one the challenger stated, is not a fault
/// proof and is not taken here.
pub fn check_fault_proof(
    challenge: &StageChallenge,
    journal: &TransitionJournal,
    transition_image_id: &[u8],
    authorization_image_id: &[u8],
) -> std::result::Result<(), FaultProofRejection> {
    let pipeline: PipelineCommitment = postcard::from_bytes(&challenge.pipeline_commitment_bytes)
        .map_err(|_| FaultProofRejection::MalformedChallenge)?;
    let stage = usize::try_from(challenge.challenged_stage)
        .ok()
        .and_then(|index| pipeline.stages.get(index))
        .ok_or(FaultProofRejection::MalformedChallenge)?;
    if !matches!(journal.current_state, TransitionState::Finished) {
        return Err(FaultProofRejection::NotAFaultVerdict);
    }
    if journal.transition_image_id != transition_image_id {
        return Err(FaultProofRejection::TransitionImageIdMismatch);
    }
    if journal.authorization_image_id != authorization_image_id {
        return Err(FaultProofRejection::AuthorizationImageIdMismatch);
    }
    if journal.refuted_trace_commitment != challenge.trace_commitment_header.digest() {
        return Err(FaultProofRejection::AnotherTraceCommitment);
    }
    if journal.program_commitment != stage.program_commitment {
        return Err(FaultProofRejection::AnotherProgram);
    }
    if journal.input_manifest_commitment != stage.input_manifest_commitment {
        return Err(FaultProofRejection::AnotherInputManifest);
    }
    Ok(())
}
