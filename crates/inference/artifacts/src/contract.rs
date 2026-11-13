//! What a contract would hold for a claim, and the checks it would make.
//!
//! No contract exists yet. This module is the reference for one: the values it
//! pins before any claim arrives ([`ContractPins`], [`RequiredParameters`]),
//! what a claimer submits ([`ClaimSubmission`]), the checks it makes on
//! submission ([`check_submission`]) and the comparisons it makes against the
//! claimer's own checkpoint list when a challenge arrives
//! ([`CheckpointList::admit_stage_challenge`], [`CheckpointList::check_link`],
//! [`CheckpointList::check_record_binding`]). It verifies no proof and runs no
//! stage.
//!
//! A claim commits to one checkpoint per stage. A checkpoint is the boundary a
//! Raster run of that stage is bound to: the program, the input manifest and
//! the output. So a dispute is about one stage, started at the claimer's own
//! boundary, and no chain-wide Raster commitment is needed.
//! [`raster_boundary_mismatch`] checks that correspondence against a Raster
//! chain run of the same claim. [`dispute_commitment`] is the one-stage Raster
//! commitment a checkpoint stands as in such a dispute, and [`check_defence`]
//! the rule for when a proof about it counts.
//!
//! Stages keep their own store, so nothing inside a stage can tell where an
//! input came from: a made-up input can carry the same labels a real one
//! does. The link check is what holds a stage's inputs to the outputs of
//! earlier checkpoints. Which earlier stage is the right one is the stage
//! schedule's to enforce, inside the programs.
//!
//! Left to the contract and the settlement layer, and absent here: data
//! availability, bonds, the challenge window, and verifying a defence.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use ::model_identity::context::parse_id;
use ::model_identity::pipeline_core::Statement;
use anyhow::{Context, Result};
use raster_core::chain::{
    defend, ChainCommitment, ChainShape, DefenceInput, DefenceJournal, DefenceRefusal,
    InputBindingSource, StageChallenge, StageCheckpoint,
};
use raster_core::transition::{TraceCommitmentHeader, TransitionJournal};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::checkpoint::{checkpoint_hash, Checkpoint, CheckpointTrace};
use crate::direct::{model_id_of, ModelCommitment, ModelManifest};
use crate::inference::InferenceResult;

/// Written beside a model import's `manifest.json`.
pub const CONTRACT_PINS_JSON: &str = "contract_pins.json";
/// Written beside a claim's `claim_bundle.json`.
pub const CLAIM_SUBMISSION_JSON: &str = "claim_submission.json";
pub const CONTRACT_PINS_VERSION: u32 = 2;
pub const CLAIM_SUBMISSION_VERSION: u32 = 1;

/// What a contract pins before any claim arrives, as two entries it opts
/// into: a pipeline and a model. The model import writes both.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ContractPins {
    pub version: u32,
    pub pipeline: PipelinePins,
    pub model: ModelPins,
}

/// What is true of the stage programs whatever model they run: which
/// programs a claim may name, and which of their inputs must be the output of
/// an earlier stage.
///
/// The program ids are the ones locked in each stage's `Raster.lock` at
/// import time, so re-locking a stage means importing again.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PipelinePins {
    /// The model family the stage programs belong to.
    pub family: String,
    /// The approved stage programs. A set: which stage runs where is the
    /// schedule's to enforce, inside the programs.
    pub stage_programs: Vec<StageProgram>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct StageProgram {
    /// The stage project's directory name.
    pub stage: String,
    /// The program's identity from its `Raster.lock`, in lowercase hex.
    pub program_commitment: String,
    /// The inputs this program takes from earlier stages. Every other input
    /// is model data or a run input, which the program checks itself.
    pub chained_inputs: Vec<ChainedInput>,
}

/// One input a stage program takes from an earlier stage.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChainedInput {
    pub name: String,
    /// The stage programs whose output this input may be, by `stage` name.
    /// It keeps an output of the wrong shape out of a stage, which the stage
    /// could not decode and so could not record a fault about.
    pub producers: Vec<String>,
    /// The commitments of the empty values this input may hold in place of an
    /// earlier output, where the chain starts. The stages accept them only
    /// there.
    pub empty_values: Vec<String>,
}

/// What is true of one model: its identity and the range size it is imported
/// with.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelPins {
    /// The model id, in lowercase hex.
    pub model_id: String,
    /// What the model id is a hash of: every committed model input by role.
    pub model_commitments: Vec<ModelCommitment>,
    pub range_size: u32,
}

/// The inference parameters a contract requires besides the model's own. They
/// are the contract's choice, so no import records them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RequiredParameters {
    pub raw_prompt: bool,
    pub max_new_tokens: u32,
}

/// The three values of a claim record, as submitted: see `ClaimRecord` in the
/// `sequence-check` stage.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SubmittedRecord {
    pub statement_sha256: String,
    pub output_text_sha256: String,
    pub valid: bool,
}

/// What a claimer submits: the plaintext, the record, and the commitment to
/// its checkpoints, which a contract stores without reading.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ClaimSubmission {
    pub version: u32,
    /// The output plaintext: what the contract acts on.
    pub output_text: String,
    /// The claimer's commitment to the input it ran. Today the SHA-256 of the
    /// prompt text. The contract takes it as given.
    pub input_text_sha256: String,
    pub record: SubmittedRecord,
    /// SHA-256 of the claim's `checkpoints.txt`: one checkpoint hash per
    /// stage, in order. This is the claim's commitment to every stage
    /// boundary.
    pub checkpoints_sha256: String,
}

/// Why a contract would not open a challenge window for a submission.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SubmissionRejection {
    /// The pins do not describe a model: the id is not the hash of the
    /// commitments, or is not a 32-byte hash.
    InconsistentPins,
    /// The claimed input commitment is not a 32-byte hash.
    MalformedInputCommitment,
    /// The record's statement is not the pinned model and parameters with the
    /// claimed input.
    StatementMismatch { expected: String, submitted: String },
    /// The plaintext does not hash to the record's output text hash.
    OutputMismatch { expected: String, submitted: String },
    /// The chain found a fault in its own run.
    NotValid,
}

impl std::fmt::Display for SubmissionRejection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InconsistentPins => write!(
                f,
                "the pinned model id is not the hash of the pinned model commitments"
            ),
            Self::MalformedInputCommitment => {
                write!(f, "the claimed input commitment is not a 32-byte hash")
            }
            Self::StatementMismatch {
                expected,
                submitted,
            } => write!(
                f,
                "the record's statement is {submitted}, but the pinned model and parameters \
                 with the claimed input give {expected}"
            ),
            Self::OutputMismatch {
                expected,
                submitted,
            } => write!(
                f,
                "the plaintext hashes to {expected}, but the record commits to {submitted}"
            ),
            Self::NotValid => write!(f, "the record's valid flag is not set"),
        }
    }
}

/// Why a challenge is not about this claim, or not one this check can rule
/// on. Not a verdict on the claim.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChallengeRefusal {
    /// The list presented is not the checkpoint list the submission stored.
    AnotherCheckpointList { stored: String, presented: String },
    /// The list is not one 32-byte hash per line.
    MalformedCheckpointList,
    /// The list has no checkpoints, so no stage and no last output.
    EmptyCheckpointList,
    UnknownStage { challenged: u32, stages: usize },
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
    UnapprovedStage { stage: u32 },
    /// The input is not one the stage's program takes from an earlier stage.
    NotAChainedInput { stage: u32, input: String },
    /// The checkpoint names a producer for the input and none was presented.
    ProducerNotPresented { stage: u32, producer: u32 },
}

impl std::fmt::Display for ChallengeRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AnotherCheckpointList { stored, presented } => write!(
                f,
                "the checkpoint list presented hashes to {presented}, the submission stored \
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
                "`{input}` is not an input stage {stage}'s program takes from an earlier stage"
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

/// Whether a stage's input is what the claim says an earlier stage produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LinkCheck {
    /// The input is an earlier checkpoint's output, or a pinned empty value.
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
    ProducerOfWrongKind { producer: u32, program_commitment: String },
    /// The input the stage loaded is not what the producer committed.
    NotTheProducersOutput {
        producer: u32,
        loaded: String,
        produced: String,
    },
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
        }
    }
}

/// Whether the submitted record is the one the claim's last stage committed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecordBinding {
    Bound,
    /// The output presented is not the one the last checkpoint commits to.
    /// The comparison proves nothing.
    NotTheLastOutput { committed: String, presented: String },
    /// The last stage's output holds a different record: the claim is void.
    AnotherRecord,
}

impl ContractPins {
    /// The pins of an import: its model, and the pipeline its chain template
    /// describes, with the program id each stage project has locked.
    pub fn from_import(manifest: &ModelManifest, template: &str) -> Result<Self> {
        let stage_programs = pipeline_wiring(template)?
            .into_iter()
            .map(|(project, chained_inputs)| {
                Ok(StageProgram {
                    stage: project_name(&project)?,
                    program_commitment: locked_program_commitment(&project)?,
                    chained_inputs,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            version: CONTRACT_PINS_VERSION,
            pipeline: PipelinePins {
                family: manifest.family.clone(),
                stage_programs,
            },
            model: ModelPins {
                model_id: manifest.model_id.clone(),
                model_commitments: manifest.model_commitments.clone(),
                range_size: manifest.range_size,
            },
        })
    }
}

impl ModelPins {
    /// The model id is the hash of the commitments it is pinned with.
    pub fn is_consistent(&self) -> bool {
        parse_id(&self.model_id).is_some() && model_id_of(&self.model_commitments) == self.model_id
    }
}

impl PipelinePins {
    /// The approved program with this id, in lowercase hex.
    pub fn program(&self, program_commitment: &str) -> Option<&StageProgram> {
        self.stage_programs
            .iter()
            .find(|approved| approved.program_commitment == program_commitment)
    }

    pub fn approves(&self, program_commitment: &str) -> bool {
        self.program(program_commitment).is_some()
    }
}

impl StageProgram {
    pub fn chained_input(&self, name: &str) -> Option<&ChainedInput> {
        self.chained_inputs.iter().find(|input| input.name == name)
    }
}

fn project_name(project: &Path) -> Result<String> {
    Ok(project
        .file_name()
        .and_then(|name| name.to_str())
        .with_context(|| format!("stage project {} has no name", project.display()))?
        .to_string())
}

/// A stage name with its indexes blanked, so `range_p3_l{l-1}` and
/// `range_p{j}_l{l}` name the same kind of stage.
fn stage_shape(name: &str) -> String {
    let mut shape = String::new();
    let mut chars = name.chars().peekable();
    while let Some(c) = chars.next() {
        let close = match c {
            '{' => Some('}'),
            '@' => Some('@'),
            _ => None,
        };
        if let Some(close) = close {
            for inner in chars.by_ref() {
                if inner == close {
                    break;
                }
            }
            shape.push('#');
        } else if c.is_ascii_digit() {
            while chars.peek().is_some_and(char::is_ascii_digit) {
                chars.next();
            }
            shape.push('#');
        } else {
            shape.push(c);
        }
    }
    shape
}

/// Each stage project a chain template names, with the inputs it takes from
/// earlier stages: for each, the projects that may produce it and the empty
/// values it holds where the chain starts. Projects are in the order the
/// template first names them.
///
/// An input is chained when any stage of the project binds it `from` another
/// stage. Where another stage of the same project binds it to a fixed
/// external instead, that external is the input's empty value.
pub fn pipeline_wiring(template: &str) -> Result<Vec<(PathBuf, Vec<ChainedInput>)>> {
    let document: toml::Value =
        toml::from_str(template).context("the chain template is not valid TOML")?;
    let chain = document
        .get("chain")
        .context("the chain template has no [chain] table")?;
    let stages_of = |table: &toml::Value| -> Vec<toml::Value> {
        ["stage", "stages"]
            .iter()
            .filter_map(|key| table.get(*key).and_then(toml::Value::as_array))
            .flatten()
            .cloned()
            .collect()
    };
    let mut stages = stages_of(chain);
    for repeat in chain
        .get("repeat")
        .and_then(toml::Value::as_array)
        .into_iter()
        .flatten()
    {
        stages.extend(stages_of(repeat));
    }

    let text = |stage: &toml::Value, key: &str| -> Result<String> {
        Ok(stage
            .get(key)
            .and_then(toml::Value::as_str)
            .with_context(|| format!("a chain stage has no `{key}`"))?
            .to_string())
    };
    // Which project runs each kind of stage.
    let mut project_of_shape = BTreeMap::new();
    let mut projects = Vec::new();
    for stage in &stages {
        let project = PathBuf::from(text(stage, "project")?);
        if !projects.contains(&project) {
            projects.push(project.clone());
        }
        project_of_shape.insert(stage_shape(&text(stage, "name")?), project);
    }

    let mut producers: BTreeMap<(PathBuf, String), BTreeSet<String>> = BTreeMap::new();
    let mut externals: BTreeMap<(PathBuf, String), BTreeSet<String>> = BTreeMap::new();
    for stage in &stages {
        let project = PathBuf::from(text(stage, "project")?);
        let inputs = stage.get("inputs").and_then(toml::Value::as_table);
        for (param, binding) in inputs.into_iter().flatten() {
            let key = (project.clone(), param.clone());
            for source in ["from", "first"] {
                let Some(producer) = binding.get(source).and_then(toml::Value::as_str) else {
                    continue;
                };
                let producer_project =
                    project_of_shape
                        .get(&stage_shape(producer))
                        .with_context(|| {
                            format!("the chain template binds `{param}` from `{producer}`, which is no stage of it")
                        })?;
                producers
                    .entry(key.clone())
                    .or_default()
                    .insert(project_name(producer_project)?);
            }
            if let Some(commitment) = binding
                .get("external")
                .and_then(|external| external.get("commitment"))
                .and_then(toml::Value::as_str)
            {
                externals
                    .entry(key)
                    .or_default()
                    .insert(commitment.to_string());
            }
        }
    }

    Ok(projects
        .into_iter()
        .map(|project| {
            let chained_inputs = producers
                .iter()
                .filter(|((of, _), _)| *of == project)
                .map(|(key, producers)| ChainedInput {
                    name: key.1.clone(),
                    producers: producers.iter().cloned().collect(),
                    empty_values: externals
                        .get(key)
                        .map(|values| values.iter().cloned().collect())
                        .unwrap_or_default(),
                })
                .collect();
            (project, chained_inputs)
        })
        .collect())
}

/// The program identity a stage project's `Raster.lock` records.
pub fn locked_program_commitment(stage_project: &Path) -> Result<String> {
    #[derive(Deserialize)]
    struct Lock {
        program_commitment: String,
    }
    let path = stage_project.join("Raster.lock");
    let lock: Lock = serde_json::from_slice(
        &fs::read(&path).with_context(|| format!("failed to read {}", path.display()))?,
    )
    .with_context(|| format!("failed to parse {}", path.display()))?;
    anyhow::ensure!(
        parse_id(&lock.program_commitment).is_some(),
        "{} does not record a 32-byte program commitment",
        path.display()
    );
    Ok(lock.program_commitment)
}

impl ClaimSubmission {
    /// What a claimer submits for a finished run: the last check's record and
    /// text, with the commitment to the checkpoint list in the run directory.
    pub fn from_run(result: &InferenceResult, chain_dir: &Path) -> Result<Self> {
        let checkpoints_path = chain_dir.join(crate::checkpoint::CHECKPOINTS_TXT);
        let checkpoints = fs::read(&checkpoints_path)
            .with_context(|| format!("failed to read {}", checkpoints_path.display()))?;
        Ok(Self {
            version: CLAIM_SUBMISSION_VERSION,
            output_text: result.generated_text.clone(),
            input_text_sha256: result.input_text_sha256.clone(),
            record: SubmittedRecord {
                statement_sha256: result.statement_sha256.clone(),
                output_text_sha256: result.output_text_sha256.clone(),
                valid: result.valid,
            },
            checkpoints_sha256: hex::encode(Sha256::digest(checkpoints)),
        })
    }
}

/// The statement digest a contract recomputes: its pinned model id and
/// parameters with the claimed input commitment. `None` when an id is not a
/// 32-byte hash.
pub fn expected_statement(
    model: &ModelPins,
    required: RequiredParameters,
    input_text_sha256: &str,
) -> Option<[u8; 32]> {
    Some(
        Statement {
            input_text: parse_id(input_text_sha256)?,
            model: parse_id(&model.model_id)?,
            raw_prompt: required.raw_prompt,
            max_new_tokens: required.max_new_tokens,
            range_size: model.range_size,
        }
        .digest(),
    )
}

/// The checks a contract makes on submission, before it opens a challenge
/// window. One equality covers the model and every parameter; the other two
/// cover the output.
pub fn check_submission(
    pins: &ContractPins,
    required: RequiredParameters,
    submission: &ClaimSubmission,
) -> std::result::Result<(), SubmissionRejection> {
    if !pins.model.is_consistent() {
        return Err(SubmissionRejection::InconsistentPins);
    }
    let expected = expected_statement(&pins.model, required, &submission.input_text_sha256)
        .ok_or(SubmissionRejection::MalformedInputCommitment)?;
    let expected = hex::encode(expected);
    if submission.record.statement_sha256 != expected {
        return Err(SubmissionRejection::StatementMismatch {
            expected,
            submitted: submission.record.statement_sha256.clone(),
        });
    }
    let output = hex::encode(Sha256::digest(submission.output_text.as_bytes()));
    if submission.record.output_text_sha256 != output {
        return Err(SubmissionRejection::OutputMismatch {
            expected: output,
            submitted: submission.record.output_text_sha256.clone(),
        });
    }
    if !submission.record.valid {
        return Err(SubmissionRejection::NotValid);
    }
    Ok(())
}

/// One entry of a stage's `input_manifest.json`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct ManifestEntry {
    commitment: String,
    encoding: String,
    r#type: String,
}

/// The commitment each input of a stage was loaded by, read from the bytes of
/// its `input_manifest.json`. `None` unless the bytes are exactly the
/// canonical encoding of what they hold, so that no two readers can take the
/// same bytes for different inputs: a repeated name, an extra field or other
/// spacing is refused.
fn read_input_manifest(bytes: &[u8]) -> Option<BTreeMap<String, String>> {
    let entries: BTreeMap<String, ManifestEntry> = serde_json::from_slice(bytes).ok()?;
    let canonical = serde_json::to_vec_pretty(&serde_json::to_value(&entries).ok()?).ok()?;
    (canonical == bytes).then(|| {
        entries
            .into_iter()
            .map(|(name, entry)| (name, entry.commitment))
            .collect()
    })
}

/// A claim's checkpoint list, held to the digest its submission stored. Every
/// challenge-time comparison starts from one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckpointList {
    hashes: Vec<String>,
}

impl CheckpointList {
    /// Open the list a challenge presents: `checkpoints_txt` is the bytes of
    /// the claim's `checkpoints.txt`.
    pub fn open(
        submission: &ClaimSubmission,
        checkpoints_txt: &[u8],
    ) -> std::result::Result<Self, ChallengeRefusal> {
        let presented = hex::encode(Sha256::digest(checkpoints_txt));
        if submission.checkpoints_sha256 != presented {
            return Err(ChallengeRefusal::AnotherCheckpointList {
                stored: submission.checkpoints_sha256.clone(),
                presented,
            });
        }
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
        Ok(Self { hashes })
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
        pins: &ContractPins,
        challenged_stage: u32,
        checkpoint: &Checkpoint,
    ) -> std::result::Result<StageAdmission, ChallengeRefusal> {
        let checkpoint = self.opened(challenged_stage, checkpoint)?;
        let program_commitment = checkpoint.program_commitment.clone();
        Ok(if pins.pipeline.approves(&program_commitment) {
            StageAdmission::Disputed { program_commitment }
        } else {
            StageAdmission::UnapprovedProgram { program_commitment }
        })
    }

    /// Whether one input of a stage is what the claim says an earlier stage
    /// produced. `input_manifest` is the bytes of the stage's
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
        pins: &ContractPins,
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
        let chained = pins
            .pipeline
            .program(&checkpoint.program_commitment)
            .ok_or(ChallengeRefusal::UnapprovedStage { stage })?
            .chained_input(input)
            .ok_or_else(|| ChallengeRefusal::NotAChainedInput {
                stage,
                input: input.to_string(),
            })?;

        let void = |fault| Ok(LinkCheck::Void(fault));
        let Some(inputs) = read_input_manifest(input_manifest) else {
            return void(LinkFault::UnreadableInputList);
        };
        let Some(loaded) = inputs.get(input) else {
            return void(LinkFault::InputMissing);
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
            .pipeline
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

    /// Whether the submitted record is the last stage's: `last_checkpoint` is
    /// the claim's last checkpoint, `last_output` that stage's `output.bin`,
    /// and `record_in_output` the claim record decoded from it by the caller.
    pub fn check_record_binding(
        &self,
        submission: &ClaimSubmission,
        last_checkpoint: &Checkpoint,
        last_output: &[u8],
        record_in_output: &SubmittedRecord,
    ) -> std::result::Result<RecordBinding, ChallengeRefusal> {
        let last = self.opened(self.hashes.len() as u32 - 1, last_checkpoint)?;
        let presented = hex::encode(Sha256::digest(last_output));
        if last.output_sha256 != presented {
            return Ok(RecordBinding::NotTheLastOutput {
                committed: last.output_sha256.clone(),
                presented,
            });
        }
        Ok(if *record_in_output == submission.record {
            RecordBinding::Bound
        } else {
            RecordBinding::AnotherRecord
        })
    }
}

/// The first stage of a claim that names a program outside the approved set,
/// which is the stage a challenger would point a contract at.
pub fn first_unapproved_stage(pins: &ContractPins, trace: &CheckpointTrace) -> Option<u32> {
    trace
        .checkpoints
        .iter()
        .position(|checkpoint| !pins.pipeline.approves(&checkpoint.program_commitment))
        .map(|index| index as u32)
}

/// Where a claim's checkpoints are not the boundaries a Raster chain run of
/// the same claim committed, or `None` when every one is: the same stages in
/// the same order, each with Raster's program id, input manifest commitment,
/// producers and output commitments.
pub fn raster_boundary_mismatch(trace: &CheckpointTrace, chain: &ChainCommitment) -> Option<String> {
    if trace.checkpoints.len() != chain.stages.len() {
        return Some(format!(
            "the claim has {} checkpoints, the Raster run {} stages",
            trace.checkpoints.len(),
            chain.stages.len()
        ));
    }
    for (index, (checkpoint, stage)) in trace.checkpoints.iter().zip(&chain.stages).enumerate() {
        let raster_sources = stage
            .input_bindings
            .iter()
            .filter_map(|(input, source)| match source {
                InputBindingSource::Chained { stage } => Some(format!("{input}={stage}")),
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

/// The Raster chain commitment that stands for one checkpoint in a dispute.
///
/// Raster's challenge and defence name a chain commitment and a stage of it,
/// and read three things of that stage: its program, its input manifest and
/// the hash of its output. A checkpoint holds exactly those. So a dispute
/// about a checkpoint is a dispute about a chain of one stage built from it,
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
    let chain = ChainCommitment {
        stages: vec![StageCheckpoint {
            name: checkpoint.stage.clone(),
            program_commitment: hash(&checkpoint.program_commitment)?,
            input_manifest_commitment: hash(&checkpoint.input_commitment)?,
            input_bindings: BTreeMap::new(),
            output_payload_commitment: hash(&checkpoint.output_sha256)?,
            output_structural_commitment: hash(&checkpoint.output_commitment)?,
        }],
        shape: ChainShape {
            spec_digest: Vec::new(),
            repeats: Vec::new(),
        },
    };
    postcard::to_allocvec(&chain).ok()
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
        chain_commitment_bytes: dispute_commitment(checkpoint)?,
        challenged_stage: 0,
        trace_commitment_header: trace_commitment,
        claimed_output: parse_id(claimed_output_sha256)?.to_vec(),
    })
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
                write!(f, "the checkpoint's fields are not the hashes a dispute names")
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
/// The rule is Raster's own (`raster_core::chain::defend`), applied to the
/// one-stage commitment built from the checkpoint: the window was proven
/// against the challenger's trace commitment, runs the checkpoint's program
/// on the checkpoint's inputs, and ends either in a divergence or, on an
/// honest trace, in an output other than the one the challenger stated.
///
/// `journal` is the journal of a receipt the caller has already verified
/// against `transition_image_id`. The returned journal is what a defence
/// receipt would commit to; its chain commitment digest is this checkpoint's
/// [`dispute_digest`].
pub fn check_defence(
    checkpoint: &Checkpoint,
    challenge: &StageChallenge,
    journal: &TransitionJournal,
    transition_image_id: &[u8],
) -> std::result::Result<DefenceJournal, DefenceRejection> {
    let commitment =
        dispute_commitment(checkpoint).ok_or(DefenceRejection::MalformedCheckpoint)?;
    if challenge.chain_commitment_bytes != commitment || challenge.challenged_stage != 0 {
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

#[cfg(test)]
mod tests {
    use super::*;
    use raster_core::chain::DefenceReason;
    use raster_core::fingerprint::{BitPacker, Fingerprint};
    use raster_core::transition::{
        EntrypointAuthorization, InitTransition, OutputAuthorization, SerializableFrontier,
        TransitionState,
    };

    const REQUIRED: RequiredParameters = RequiredParameters {
        raw_prompt: true,
        max_new_tokens: 8,
    };
    const BIND: [u8; 32] = [1; 32];
    const CHECK: [u8; 32] = [2; 32];
    const LAYER: [u8; 32] = [4; 32];

    fn empty_edge() -> String {
        hex::encode([0xee; 32])
    }

    /// Three programs: the first takes nothing from the chain, a layer takes
    /// `activations` from a bind or another layer, and the check takes
    /// `activations` from a layer and `prior` from a check or the empty edge.
    fn pins() -> ContractPins {
        let model_commitments = vec![ModelCommitment {
            role: String::from("tokenizer"),
            commitment: hex::encode([7; 32]),
        }];
        let program = |stage: &str, id: [u8; 32], chained_inputs| StageProgram {
            stage: stage.to_string(),
            program_commitment: hex::encode(id),
            chained_inputs,
        };
        let input = |name: &str, producers: &[&str], empty_values: Vec<String>| ChainedInput {
            name: name.to_string(),
            producers: producers.iter().map(|p| p.to_string()).collect(),
            empty_values,
        };
        ContractPins {
            version: CONTRACT_PINS_VERSION,
            pipeline: PipelinePins {
                family: String::from("gemma-dense"),
                stage_programs: vec![
                    program("prompt-bind", BIND, vec![]),
                    program(
                        "prefill-range",
                        LAYER,
                        vec![input(
                            "activations",
                            &["prompt-bind", "prefill-range"],
                            vec![],
                        )],
                    ),
                    program(
                        "sequence-check",
                        CHECK,
                        vec![
                            input("activations", &["prefill-range"], vec![]),
                            input("prior", &["sequence-check"], vec![empty_edge()]),
                        ],
                    ),
                ],
            },
            model: ModelPins {
                model_id: model_id_of(&model_commitments),
                model_commitments,
                range_size: 512,
            },
        }
    }

    /// A stage's input list as the chain writes it.
    fn manifest(inputs: &[(&str, &str)]) -> Vec<u8> {
        let entries = inputs
            .iter()
            .map(|(name, commitment)| {
                (
                    name.to_string(),
                    serde_json::json!({"type": "sha256", "encoding": "raster", "commitment": commitment}),
                )
            })
            .collect::<serde_json::Map<_, _>>();
        serde_json::to_vec_pretty(&serde_json::Value::Object(entries)).unwrap()
    }

    fn output_commitment(output: &[u8]) -> String {
        hex::encode(Sha256::digest([b"structural:".as_slice(), output].concat()))
    }

    fn checkpoint(
        stage: &str,
        program: [u8; 32],
        inputs: &[u8],
        sources: &[(&str, u32)],
        output: &[u8],
    ) -> Checkpoint {
        Checkpoint {
            stage: stage.to_string(),
            program_commitment: hex::encode(program),
            input_commitment: hex::encode(Sha256::digest(inputs)),
            input_sources: sources
                .iter()
                .map(|(input, index)| (input.to_string(), *index))
                .collect(),
            output_commitment: output_commitment(output),
            output_sha256: hex::encode(Sha256::digest(output)),
        }
    }

    fn checkpoints_txt(trace: &CheckpointTrace) -> Vec<u8> {
        let mut text = crate::checkpoint::checkpoint_hashes(trace)
            .unwrap()
            .join("\n");
        text.push('\n');
        text.into_bytes()
    }

    /// An honest claim of three stages, bind then layer then check, with each
    /// stage's input list.
    struct Claim {
        submission: ClaimSubmission,
        trace: CheckpointTrace,
        list: Vec<u8>,
        manifests: Vec<Vec<u8>>,
    }

    impl Claim {
        fn of(checkpoints: Vec<Checkpoint>, manifests: Vec<Vec<u8>>) -> Self {
            let pins = pins();
            let input = hex::encode([5; 32]);
            let trace = CheckpointTrace { checkpoints };
            let list = checkpoints_txt(&trace);
            let submission = ClaimSubmission {
                version: CLAIM_SUBMISSION_VERSION,
                output_text: String::from("ZK"),
                record: SubmittedRecord {
                    statement_sha256: hex::encode(
                        expected_statement(&pins.model, REQUIRED, &input).unwrap(),
                    ),
                    output_text_sha256: hex::encode(Sha256::digest(b"ZK")),
                    valid: true,
                },
                input_text_sha256: input,
                checkpoints_sha256: hex::encode(Sha256::digest(&list)),
            };
            Self {
                submission,
                trace,
                list,
                manifests,
            }
        }

        fn honest() -> Self {
            let manifests = vec![
                manifest(&[("text", &hex::encode([9; 32]))]),
                manifest(&[
                    ("activations", &output_commitment(b"bound")),
                    ("layer", &hex::encode([8; 32])),
                ]),
                manifest(&[
                    ("activations", &output_commitment(b"layered")),
                    ("prior", &empty_edge()),
                ]),
            ];
            Self::of(
                vec![
                    checkpoint("prompt_bind_p0", BIND, &manifests[0], &[], b"bound"),
                    checkpoint(
                        "range_p0_l0",
                        LAYER,
                        &manifests[1],
                        &[("activations", 0)],
                        b"layered",
                    ),
                    checkpoint(
                        "check_c0",
                        CHECK,
                        &manifests[2],
                        &[("activations", 1)],
                        b"last",
                    ),
                ],
                manifests,
            )
        }

        fn opened(&self) -> CheckpointList {
            CheckpointList::open(&self.submission, &self.list).unwrap()
        }

        /// The link check as a challenger pointing at `stage` and `input`
        /// would have a contract make it.
        fn link(&self, stage: u32, input: &str) -> std::result::Result<LinkCheck, ChallengeRefusal> {
            let checkpoint = &self.trace.checkpoints[stage as usize];
            let producer = checkpoint
                .input_sources
                .get(input)
                .and_then(|index| self.trace.checkpoints.get(*index as usize));
            self.opened().check_link(
                &pins(),
                stage,
                checkpoint,
                &self.manifests[stage as usize],
                input,
                producer,
            )
        }
    }

    #[test]
    fn an_honest_submission_opens_a_window() {
        let claim = Claim::honest();
        assert_eq!(
            check_submission(&pins(), REQUIRED, &claim.submission),
            Ok(())
        );
    }

    #[test]
    fn one_equality_covers_the_model_and_every_parameter() {
        let submission = Claim::honest().submission;
        let mismatch = |pins: &ContractPins, required: RequiredParameters| {
            matches!(
                check_submission(pins, required, &submission),
                Err(SubmissionRejection::StatementMismatch { .. })
            )
        };

        // Another model: a consistent pin of other commitments.
        let mut other_model = pins();
        other_model.model.model_commitments[0].commitment = hex::encode([8; 32]);
        other_model.model.model_id = model_id_of(&other_model.model.model_commitments);
        assert!(mismatch(&other_model, REQUIRED));

        let mut other_range = pins();
        other_range.model.range_size = 256;
        assert!(mismatch(&other_range, REQUIRED));

        assert!(mismatch(
            &pins(),
            RequiredParameters {
                raw_prompt: false,
                ..REQUIRED
            }
        ));
        assert!(mismatch(
            &pins(),
            RequiredParameters {
                max_new_tokens: 9,
                ..REQUIRED
            }
        ));

        // The input is the claimer's to state, but the statement must be over it.
        let mut other_input = submission.clone();
        other_input.input_text_sha256 = hex::encode([6; 32]);
        assert!(matches!(
            check_submission(&pins(), REQUIRED, &other_input),
            Err(SubmissionRejection::StatementMismatch { .. })
        ));
    }

    #[test]
    fn a_submission_is_rejected_for_its_output_or_its_verdict() {
        let submission = Claim::honest().submission;

        let mut other_text = submission.clone();
        other_text.output_text = String::from("ZK!");
        assert!(matches!(
            check_submission(&pins(), REQUIRED, &other_text),
            Err(SubmissionRejection::OutputMismatch { .. })
        ));

        let mut faulted = submission.clone();
        faulted.record.valid = false;
        assert_eq!(
            check_submission(&pins(), REQUIRED, &faulted),
            Err(SubmissionRejection::NotValid)
        );

        let mut malformed = submission;
        malformed.input_text_sha256 = String::from("prompt");
        assert_eq!(
            check_submission(&pins(), REQUIRED, &malformed),
            Err(SubmissionRejection::MalformedInputCommitment)
        );
    }

    #[test]
    fn pins_whose_id_is_not_the_hash_of_their_commitments_reject_everything() {
        let submission = Claim::honest().submission;
        let mut pins = pins();
        pins.model.model_commitments[0].commitment = hex::encode([8; 32]);
        assert_eq!(
            check_submission(&pins, REQUIRED, &submission),
            Err(SubmissionRejection::InconsistentPins)
        );
    }

    #[test]
    fn a_challenged_stage_is_disputed_or_void_by_its_own_program_id() {
        let claim = Claim::honest();
        assert_eq!(
            claim
                .opened()
                .admit_stage_challenge(&pins(), 2, &claim.trace.checkpoints[2]),
            Ok(StageAdmission::Disputed {
                program_commitment: hex::encode(CHECK)
            })
        );
        assert_eq!(first_unapproved_stage(&pins(), &claim.trace), None);

        // A claimer that swapped a program in the middle of its chain, or
        // named none, and committed to that.
        for program in [hex::encode([3; 32]), String::new()] {
            let mut checkpoints = claim.trace.checkpoints.clone();
            checkpoints[1].program_commitment = program.clone();
            let swapped = Claim::of(checkpoints, claim.manifests.clone());
            assert_eq!(
                swapped
                    .opened()
                    .admit_stage_challenge(&pins(), 1, &swapped.trace.checkpoints[1]),
                Ok(StageAdmission::UnapprovedProgram {
                    program_commitment: program
                })
            );
            assert_eq!(first_unapproved_stage(&pins(), &swapped.trace), Some(1));
            // The stages around it are still ordinary disputes.
            assert!(matches!(
                swapped
                    .opened()
                    .admit_stage_challenge(&pins(), 2, &swapped.trace.checkpoints[2]),
                Ok(StageAdmission::Disputed { .. })
            ));
            // Its inputs are not defined, so a link challenge is refused: the
            // program challenge is the one that voids the claim.
            assert_eq!(
                swapped.link(1, "activations"),
                Err(ChallengeRefusal::UnapprovedStage { stage: 1 })
            );
        }
    }

    #[test]
    fn a_challenge_must_present_the_list_and_the_checkpoint_the_claim_committed() {
        let claim = Claim::honest();
        let mut other = claim.list.clone();
        other.extend_from_slice(claim.list.split(|byte| *byte == b'\n').next().unwrap());
        other.push(b'\n');
        assert!(matches!(
            CheckpointList::open(&claim.submission, &other),
            Err(ChallengeRefusal::AnotherCheckpointList { .. })
        ));

        let opened = claim.opened();
        assert_eq!(opened.len(), 3);
        assert_eq!(
            opened.admit_stage_challenge(&pins(), 3, &claim.trace.checkpoints[2]),
            Err(ChallengeRefusal::UnknownStage {
                challenged: 3,
                stages: 3
            })
        );
        // A challenger cannot present a checkpoint of its own for a stage: a
        // program id the claimer did not commit to proves nothing.
        let mut forged = claim.trace.checkpoints[1].clone();
        forged.program_commitment = hex::encode([3; 32]);
        assert!(matches!(
            opened.admit_stage_challenge(&pins(), 1, &forged),
            Err(ChallengeRefusal::NotTheCheckpoint { stage: 1, .. })
        ));

        // A stored digest of something that is not a hash list.
        let mut junk = claim.submission.clone();
        junk.checkpoints_sha256 = hex::encode(Sha256::digest(b"not hashes\n"));
        assert_eq!(
            CheckpointList::open(&junk, b"not hashes\n"),
            Err(ChallengeRefusal::MalformedCheckpointList)
        );
        junk.checkpoints_sha256 = hex::encode(Sha256::digest(b""));
        assert_eq!(
            CheckpointList::open(&junk, b""),
            Err(ChallengeRefusal::EmptyCheckpointList)
        );
    }

    #[test]
    fn every_chained_input_of_an_honest_claim_is_linked() {
        let claim = Claim::honest();
        assert_eq!(claim.link(1, "activations"), Ok(LinkCheck::Linked));
        assert_eq!(claim.link(2, "activations"), Ok(LinkCheck::Linked));
        // The first check has no check before it: the pinned empty edge.
        assert_eq!(claim.link(2, "prior"), Ok(LinkCheck::Linked));
        // Model data and run inputs are the stage's own to check.
        assert_eq!(
            claim.link(1, "layer"),
            Err(ChallengeRefusal::NotAChainedInput {
                stage: 1,
                input: String::from("layer")
            })
        );
    }

    /// The honest claim with the last check's inputs and producers replaced.
    fn with_check(inputs: &[(&str, &str)], sources: &[(&str, u32)]) -> Claim {
        let honest = Claim::honest();
        let mut manifests = honest.manifests;
        manifests[2] = manifest(inputs);
        let mut checkpoints = honest.trace.checkpoints;
        checkpoints[2] = checkpoint("check_c0", CHECK, &manifests[2], sources, b"last");
        Claim::of(checkpoints, manifests)
    }

    #[test]
    fn a_made_up_input_voids_the_claim() {
        let made_up = output_commitment(b"activations nobody computed");
        // Every stage still does what its own checkpoint says. Only the link
        // shows that the check ran on something the layer never produced.
        let claim = with_check(
            &[("activations", &made_up), ("prior", &empty_edge())],
            &[("activations", 1)],
        );
        assert_eq!(
            claim.link(2, "activations"),
            Ok(LinkCheck::Void(LinkFault::NotTheProducersOutput {
                producer: 1,
                loaded: made_up.clone(),
                produced: output_commitment(b"layered"),
            }))
        );

        // Passed off as an input from outside the claim: no producer named.
        let claim = with_check(&[("activations", &made_up), ("prior", &empty_edge())], &[]);
        assert_eq!(
            claim.link(2, "activations"),
            Ok(LinkCheck::Void(LinkFault::NoProducer { loaded: made_up }))
        );

        // An empty value where the program has none to accept.
        let claim = with_check(
            &[("activations", &empty_edge()), ("prior", &empty_edge())],
            &[],
        );
        assert!(matches!(
            claim.link(2, "activations"),
            Ok(LinkCheck::Void(LinkFault::NoProducer { .. }))
        ));

        // Left out of the input list altogether.
        let claim = with_check(&[("prior", &empty_edge())], &[("activations", 1)]);
        assert_eq!(
            claim.link(2, "activations"),
            Ok(LinkCheck::Void(LinkFault::InputMissing))
        );
    }

    #[test]
    fn a_real_output_of_the_wrong_kind_or_from_later_voids_the_claim() {
        // The bind stage's real output, wired where a layer's belongs.
        let claim = with_check(
            &[
                ("activations", &output_commitment(b"bound")),
                ("prior", &empty_edge()),
            ],
            &[("activations", 0)],
        );
        assert_eq!(
            claim.link(2, "activations"),
            Ok(LinkCheck::Void(LinkFault::ProducerOfWrongKind {
                producer: 0,
                program_commitment: hex::encode(BIND),
            }))
        );

        // A stage cannot consume its own output, or a later one's.
        for producer in [2, 7] {
            let claim = with_check(
                &[
                    ("activations", &output_commitment(b"last")),
                    ("prior", &empty_edge()),
                ],
                &[("activations", producer)],
            );
            assert_eq!(
                claim.link(2, "activations"),
                Ok(LinkCheck::Void(LinkFault::ProducerNotEarlier { producer }))
            );
        }
    }

    #[test]
    fn an_input_list_that_reads_two_ways_voids_the_claim() {
        let real = output_commitment(b"layered");
        let made_up = output_commitment(b"activations nobody computed");
        let entry = |commitment: &str| {
            format!(
                "{{\n    \"commitment\": \"{commitment}\",\n    \"encoding\": \"raster\",\n    \"type\": \"sha256\"\n  }}"
            )
        };
        // The same name twice: one reader takes the first, another the last.
        let twice = format!(
            "{{\n  \"activations\": {},\n  \"activations\": {},\n  \"prior\": {}\n}}",
            entry(&real),
            entry(&made_up),
            entry(&empty_edge())
        );
        // Honest content, another spelling.
        let respaced = String::from_utf8(manifest(&[("activations", &real), ("prior", &empty_edge())]))
            .unwrap()
            .replace("\n  \"prior\"", "\n\n  \"prior\"");
        for bytes in [twice.into_bytes(), respaced.into_bytes()] {
            let honest = Claim::honest();
            let mut manifests = honest.manifests;
            manifests[2] = bytes;
            let mut checkpoints = honest.trace.checkpoints;
            checkpoints[2] = checkpoint(
                "check_c0",
                CHECK,
                &manifests[2],
                &[("activations", 1)],
                b"last",
            );
            let claim = Claim::of(checkpoints, manifests);
            assert_eq!(
                claim.link(2, "activations"),
                Ok(LinkCheck::Void(LinkFault::UnreadableInputList))
            );
        }
    }

    #[test]
    fn a_link_challenge_must_present_what_the_claim_committed() {
        let claim = Claim::honest();
        let opened = claim.opened();
        let check = &claim.trace.checkpoints[2];
        // Another stage's input list.
        assert!(matches!(
            opened.check_link(
                &pins(),
                2,
                check,
                &claim.manifests[1],
                "activations",
                Some(&claim.trace.checkpoints[1])
            ),
            Err(ChallengeRefusal::NotTheInputManifest { stage: 2, .. })
        ));
        // The named producer withheld, or another checkpoint in its place.
        assert_eq!(
            opened.check_link(&pins(), 2, check, &claim.manifests[2], "activations", None),
            Err(ChallengeRefusal::ProducerNotPresented {
                stage: 2,
                producer: 1
            })
        );
        assert!(matches!(
            opened.check_link(
                &pins(),
                2,
                check,
                &claim.manifests[2],
                "activations",
                Some(&claim.trace.checkpoints[0])
            ),
            Err(ChallengeRefusal::NotTheCheckpoint { stage: 1, .. })
        ));
    }

    #[test]
    fn the_record_is_held_to_the_last_stage_output() {
        let claim = Claim::honest();
        let opened = claim.opened();
        let last = &claim.trace.checkpoints[2];
        let record = &claim.submission.record;
        assert_eq!(
            opened.check_record_binding(&claim.submission, last, b"last", record),
            Ok(RecordBinding::Bound)
        );

        // The last stage committed a record the claimer did not submit.
        let mut in_output = record.clone();
        in_output.valid = false;
        assert_eq!(
            opened.check_record_binding(&claim.submission, last, b"last", &in_output),
            Ok(RecordBinding::AnotherRecord)
        );

        // Bytes that are not the last stage's output prove nothing either way.
        assert!(matches!(
            opened.check_record_binding(&claim.submission, last, b"bound", record),
            Ok(RecordBinding::NotTheLastOutput { .. })
        ));
        // Nor does another stage's checkpoint stand for the last.
        assert!(matches!(
            opened.check_record_binding(
                &claim.submission,
                &claim.trace.checkpoints[0],
                b"bound",
                record
            ),
            Err(ChallengeRefusal::NotTheCheckpoint { stage: 2, .. })
        ));
    }

    #[test]
    fn a_checkpoint_is_the_boundary_a_raster_run_commits() {
        let trace = Claim::honest().trace;
        let stage = |checkpoint: &Checkpoint| StageCheckpoint {
            name: checkpoint.stage.clone(),
            program_commitment: hex::decode(&checkpoint.program_commitment).unwrap(),
            input_manifest_commitment: hex::decode(&checkpoint.input_commitment).unwrap(),
            input_bindings: checkpoint
                .input_sources
                .iter()
                .map(|(input, index)| {
                    (
                        input.clone(),
                        InputBindingSource::Chained {
                            stage: *index as usize,
                        },
                    )
                })
                .chain([(String::from("layer"), InputBindingSource::External)])
                .collect(),
            output_payload_commitment: hex::decode(&checkpoint.output_sha256).unwrap(),
            output_structural_commitment: hex::decode(&checkpoint.output_commitment).unwrap(),
        };
        let chain = |stages: Vec<StageCheckpoint>| ChainCommitment {
            stages,
            shape: ChainShape {
                spec_digest: vec![9; 32],
                repeats: Vec::new(),
            },
        };
        let raster = chain(trace.checkpoints.iter().map(stage).collect());
        assert_eq!(raster_boundary_mismatch(&trace, &raster), None);

        let mut other_program = raster.clone();
        other_program.stages[2].program_commitment = vec![3; 32];
        assert!(raster_boundary_mismatch(&trace, &other_program)
            .unwrap()
            .contains("stage 2 (check_c0): the checkpoint's program"));

        let mut other_input = raster.clone();
        other_input.stages[0].input_manifest_commitment = vec![3; 32];
        assert!(raster_boundary_mismatch(&trace, &other_input)
            .unwrap()
            .contains("input manifest commitment"));

        let mut other_producer = raster.clone();
        other_producer.stages[2].input_bindings.insert(
            String::from("activations"),
            InputBindingSource::Chained { stage: 0 },
        );
        assert!(raster_boundary_mismatch(&trace, &other_producer)
            .unwrap()
            .contains("producers"));

        let shorter = chain(vec![stage(&trace.checkpoints[0])]);
        assert!(raster_boundary_mismatch(&trace, &shorter)
            .unwrap()
            .contains("3 checkpoints"));
    }

    #[test]
    fn the_wiring_is_read_from_the_chain_template() {
        let template = r#"
[chain]
name = "iris"

[[chain.stage]]
name = "prompt_bind_p0"
project = "/repo/raster-stages/prompt-bind"
inputs.tokenizer = { external = { path = "t", commitment = "aa" } }
inputs.text = { input = "prompt_text_0" }
inputs.previous = { external = { path = "e", commitment = "e0" } }

[[chain.repeat]]
name = "prompt"
index = "j"
start = 1
count = 0

  [[chain.repeat.stage]]
name = "prompt_bind_p{j}"
project = "/repo/raster-stages/prompt-bind"
inputs.tokenizer = { external = { path = "t", commitment = "aa" } }
inputs.text = { input = "prompt_text_{j}" }
inputs.previous = { from = "prompt_bind_p{j-1}", first = "prompt_bind_p0" }

[[chain.repeat]]
name = "layers"
index = "l"
count = 2

  [[chain.repeat.stage]]
name = "range_p{j}_l{l}"
index = "j"
count = 0
project = "/repo/raster-stages/prefill-range"
inputs.activations = { from = "range_p{j}_l{l-1}", first = "prompt_bind_p{j}" }
inputs.prior_kv = { external = { path = "k", commitment = "e1" } }
inputs.layer = { input = "transformer_layer_{l}" }

  [[chain.repeat.stage]]
name = "range_c{k}_l{l}"
index = "k"
count = 0
project = "/repo/raster-stages/prefill-range"
inputs.activations = { from = "range_c{k}_l{l-1}", first = "prompt_bind_p@LAST_PROMPT_RANGE@" }
inputs.prior_kv = { from = "range_c{k-1}_l{l}", first = "range_p@LAST_PROMPT_RANGE@_l{l}" }
inputs.layer = { input = "transformer_layer_{l}" }
"#;
        let wiring = pipeline_wiring(template).unwrap();
        let input = |name: &str, producers: &[&str], empty_values: &[&str]| ChainedInput {
            name: name.to_string(),
            producers: producers.iter().map(|p| p.to_string()).collect(),
            empty_values: empty_values.iter().map(|v| v.to_string()).collect(),
        };
        assert_eq!(
            wiring,
            vec![
                (
                    PathBuf::from("/repo/raster-stages/prompt-bind"),
                    // Model data and run inputs are not chained; the first
                    // bind's fixed `previous` is the input's empty value.
                    vec![input("previous", &["prompt-bind"], &["e0"])],
                ),
                (
                    PathBuf::from("/repo/raster-stages/prefill-range"),
                    vec![
                        input("activations", &["prefill-range", "prompt-bind"], &[]),
                        input("prior_kv", &["prefill-range"], &["e1"]),
                    ],
                ),
            ]
        );

        assert!(pipeline_wiring("[chain]\n[[chain.stage]]\nname = \"a\"\nproject = \"p\"\ninputs.x = { from = \"nowhere\" }\n").is_err());
    }

    const TRANSITION_ID: [u8; 32] = [0x7a; 32];

    fn trace_header() -> TraceCommitmentHeader {
        TraceCommitmentHeader {
            bits_packer: BitPacker::new(1),
            fingerprint_len: 200,
            fingerprint_root: vec![1; 32],
            revealed_items_commitment: vec![2; 32],
            window_size: 128,
            revealed_tail_roots_commitment: vec![3; 32],
        }
    }

    /// The journal of a window proven against `trace_header()` over this
    /// checkpoint's program and inputs, ending in `state` with `output`.
    fn journal(checkpoint: &Checkpoint, state: TransitionState, output: &[u8]) -> TransitionJournal {
        let frontier = || SerializableFrontier {
            position: 0,
            leaf: vec![0; 32],
            ommers: Vec::new(),
        };
        TransitionJournal {
            init_state: InitTransition {
                init_frontier: frontier(),
                init_storage_frontier: frontier(),
                init_storage_root: vec![0; 32],
                init_storage_index_root: vec![0; 32],
                fingerprint: Fingerprint {
                    bits_packer: BitPacker::new(1),
                    bits: Vec::new(),
                    len: 0,
                },
            },
            current_state: state,
            transition_image_id: TRANSITION_ID.to_vec(),
            authorization_image_id: vec![0xa0; 32],
            input_manifest_commitment: hex::decode(&checkpoint.input_commitment).unwrap(),
            program_commitment: hex::decode(&checkpoint.program_commitment).unwrap(),
            refuted_trace_commitment: trace_header().digest(),
            entrypoint_authorization: EntrypointAuthorization::Established,
            output_authorization: OutputAuthorization::Established {
                output_commitment: output.to_vec(),
            },
            window_is_terminal: true,
            final_committed_root: None,
        }
    }

    #[test]
    fn a_dispute_commitment_is_the_checkpoint_and_nothing_else() {
        let checkpoint = Claim::honest().trace.checkpoints[2].clone();
        let bytes = dispute_commitment(&checkpoint).unwrap();

        // The layout a contract writes out by hand.
        let mut expected = vec![1, checkpoint.stage.len() as u8];
        expected.extend_from_slice(checkpoint.stage.as_bytes());
        for (field, last) in [
            (&checkpoint.program_commitment, false),
            (&checkpoint.input_commitment, true),
        ] {
            expected.push(32);
            expected.extend_from_slice(&hex::decode(field).unwrap());
            if last {
                expected.push(0);
            }
        }
        for field in [&checkpoint.output_sha256, &checkpoint.output_commitment] {
            expected.push(32);
            expected.extend_from_slice(&hex::decode(field).unwrap());
        }
        expected.extend_from_slice(&[0, 0]);
        assert_eq!(bytes, expected);
        assert_eq!(
            dispute_digest(&checkpoint).unwrap().as_slice(),
            Sha256::digest(&expected).as_slice()
        );

        // It is a Raster chain commitment of one stage, so Raster's own
        // dispute code reads it.
        let chain: ChainCommitment = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(chain.stages.len(), 1);
        assert_eq!(hex::encode(&chain.stages[0].program_commitment), checkpoint.program_commitment);

        // Each of the three things a dispute reads moves the digest; where the
        // inputs came from does not, since the input manifest hash covers them.
        let digest = |change: fn(&mut Checkpoint)| {
            let mut changed = Claim::honest().trace.checkpoints[2].clone();
            change(&mut changed);
            dispute_digest(&changed)
        };
        let base = digest(|_| {});
        assert_ne!(digest(|c| c.program_commitment = hex::encode([3; 32])), base);
        assert_ne!(digest(|c| c.input_commitment = hex::encode([3; 32])), base);
        assert_ne!(digest(|c| c.output_sha256 = hex::encode([3; 32])), base);
        assert_eq!(digest(|c| c.input_sources.clear()), base);
        // A checkpoint that names no program cannot be disputed by a replay:
        // the program check voids it instead.
        assert_eq!(digest(|c| c.program_commitment.clear()), None);
    }

    #[test]
    fn a_proof_dismisses_a_challenge_only_when_it_is_about_the_checkpoint() {
        let checkpoint = Claim::honest().trace.checkpoints[1].clone();
        let stated = hex::encode([0x13; 32]);
        let challenge = stage_challenge(&checkpoint, trace_header(), &stated).unwrap();
        let defence = |journal: &TransitionJournal| {
            check_defence(&checkpoint, &challenge, journal, &TRANSITION_ID)
        };

        // A window that diverges from the challenger's trace, proven over the
        // checkpoint's program and inputs: the challenge is refuted, and the
        // journal names this checkpoint.
        let refuted = defence(&journal(&checkpoint, TransitionState::Finished, &[])).unwrap();
        assert_eq!(refuted.reason, DefenceReason::Refuted);
        assert_eq!(
            refuted.chain_commitment_digest,
            dispute_digest(&checkpoint).unwrap()
        );
        assert_eq!(refuted.challenged_stage, 0);

        // The same verdict over another program, or over other inputs, says
        // nothing about this stage: any trace diverges from a run it is not of.
        let mut another_program = journal(&checkpoint, TransitionState::Finished, &[]);
        another_program.program_commitment = vec![9; 32];
        assert_eq!(
            defence(&another_program),
            Err(DefenceRejection::Refused(DefenceRefusal::AnotherProgram))
        );
        let mut other_inputs = journal(&checkpoint, TransitionState::Finished, &[]);
        other_inputs.input_manifest_commitment = vec![9; 32];
        assert_eq!(
            defence(&other_inputs),
            Err(DefenceRejection::Refused(DefenceRefusal::AnotherInputManifest))
        );
        // Nor does a window proven against another trace commitment.
        let mut another_trace = journal(&checkpoint, TransitionState::Finished, &[]);
        another_trace.refuted_trace_commitment = vec![9; 32];
        assert_eq!(
            defence(&another_trace),
            Err(DefenceRejection::Refused(DefenceRefusal::AnotherTraceCommitment))
        );
        // Nor one verified against a guest other than the pinned one.
        assert_eq!(
            check_defence(
                &checkpoint,
                &challenge,
                &journal(&checkpoint, TransitionState::Finished, &[]),
                &[9; 32]
            ),
            Err(DefenceRejection::Refused(
                DefenceRefusal::TransitionImageIdMismatch
            ))
        );

        // An honest trace that ends in another output than the challenger
        // stated: a mislabel, which also dismisses the challenge.
        let committed = hex::decode(&checkpoint.output_sha256).unwrap();
        let mislabelled =
            defence(&journal(&checkpoint, TransitionState::TerminalMatch, &committed)).unwrap();
        assert_eq!(mislabelled.reason, DefenceReason::Mislabelled);
        assert_eq!(mislabelled.attested_output, Some(committed));
        // An honest trace that ends where the challenger said: the challenge
        // is true, and no defence exists.
        assert_eq!(
            defence(&journal(
                &checkpoint,
                TransitionState::TerminalMatch,
                &hex::decode(&stated).unwrap()
            )),
            Err(DefenceRejection::Refused(DefenceRefusal::OutputIsTheClaimed))
        );

        // A challenge built for another checkpoint is not this one's to
        // answer, whatever the proof says.
        let other = Claim::honest().trace.checkpoints[2].clone();
        assert_eq!(
            check_defence(
                &other,
                &challenge,
                &journal(&other, TransitionState::Finished, &[]),
                &TRANSITION_ID
            ),
            Err(DefenceRejection::AnotherCheckpoint)
        );
    }

    #[test]
    fn only_a_differing_output_is_settled_by_a_replay() {
        let recomputed = Claim::honest().trace.checkpoints[2].clone();
        let claimed = |change: fn(&mut Checkpoint)| {
            let mut claimed = Claim::honest().trace.checkpoints[2].clone();
            change(&mut claimed);
            settlement(&claimed, &recomputed)
        };
        assert_eq!(claimed(|_| {}), Settlement::Nothing);
        assert_eq!(
            claimed(|c| c.output_sha256 = hex::encode([3; 32])),
            Settlement::Replay
        );
        assert_eq!(
            claimed(|c| c.output_commitment = hex::encode([3; 32])),
            Settlement::Replay
        );
        assert_eq!(
            claimed(|c| c.input_commitment = hex::encode([3; 32])),
            Settlement::Inputs
        );
        assert_eq!(
            claimed(|c| {
                c.input_sources.insert(String::from("activations"), 0);
            }),
            Settlement::Inputs
        );
        // A claim that names another program is settled by the program check
        // even when its output differs too.
        assert_eq!(
            claimed(|c| {
                c.program_commitment = hex::encode([3; 32]);
                c.output_sha256 = hex::encode([3; 32]);
            }),
            Settlement::ProgramCheck
        );
        assert_eq!(
            claimed(|c| c.stage = String::from("check_c1")),
            Settlement::StageOrder
        );
    }
}
