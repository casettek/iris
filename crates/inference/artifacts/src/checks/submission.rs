//! What a claimer submits, and the checks a contract makes when it arrives:
//! the claim is of the procedure's procedure kind, states the statement the
//! contract recomputes, a valid record, the source and result the
//! procedure requires, and the records its own checkpoints commit to.

use std::path::Path;

use ::model_identity::context::parse_id;
use ::model_identity::statement_core::Statement;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::availability::{BlobContents, PostedBlob};
use crate::checkpoint::checkpoints_root;
use crate::inference::InferenceResult;
use crate::procedure_kind::ProcedureKind;

use super::*;

/// Written beside a claim's `claim_bundle.json`.
pub const CLAIM_SUBMISSION_JSON: &str = "claim_submission.json";

pub const CLAIM_SUBMISSION_VERSION: u32 = 8;

/// The three values of a claim record, as submitted: see `ClaimRecord` in the
/// `sequence-check` stage.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SubmittedRecord {
    pub statement_sha256: String,
    pub output_text_sha256: String,
    pub valid: bool,
}

/// What the result stage states of a claim's output, as submitted: see
/// `ResultRecord` in the `claim-result` stage.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct SubmittedResult {
    /// The code of the first of the run's forms the output text is, or zero.
    pub code: u32,
    /// The stage's hash over the forms it compared the output with.
    pub forms_sha256: String,
}

/// What a contract requires of where a claim's prompt came from and of how
/// its output was read. Its own choice, like [`RequiredParameters`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RequiredOrigin {
    /// The source the prompt must be built from, or `None` for a prompt taken
    /// as given, when the claim must name no source.
    pub source: Option<RequiredSource>,
    /// The forms the output must be read by, in order. A contract holds them
    /// and not only their hash: the result step is a comparison per form, so
    /// a contract takes it on the submitted output itself and does not rest
    /// on the result stage, which is too short to be replayed in a dispute.
    pub forms: Vec<result_core::Form>,
}

impl RequiredOrigin {
    /// What the result step states of an output read by these forms.
    pub fn result_of(&self, output_text: &str) -> SubmittedResult {
        let matched = result_core::run_forms(&self.forms, output_text);
        SubmittedResult {
            code: matched.code,
            forms_sha256: result_core::forms_id(&matched),
        }
    }

    /// What the consensus result step states of two outputs read by these
    /// forms, when both claim records are valid and name one input. A
    /// contract has held the submission to both before it asks.
    pub fn consensus_of(&self, first_output: &str, second_output: &str) -> SubmittedConsensus {
        let valid = || result_core::Claimed {
            statement: &[],
            output_text_sha256: &[],
            valid: true,
        };
        let record = result_core::close_consensus(
            result_core::run_consensus(&self.forms, first_output, second_output, true),
            valid(),
            valid(),
        );
        SubmittedConsensus {
            same_input: record.same_input,
            agree: record.agree,
            code: record.code,
        }
    }
}

/// What one inference of a claim submits: its plaintext, its claim record,
/// and what the result stage made of the plaintext.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SubmittedInference {
    /// The output plaintext: what the contract acts on.
    pub output_text: String,
    pub record: SubmittedRecord,
    #[serde(default)]
    pub result: SubmittedResult,
}

/// What a consensus claim's result stage states of its two inferences, as
/// submitted: see `ConsensusRecord` in the `consensus-result` stage.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct SubmittedConsensus {
    /// The two inferences name the same input text and source.
    pub same_input: bool,
    /// Both records are valid, they name the same input, and the outputs
    /// agree.
    pub agree: bool,
    /// The code both outputs have, when they agree and the run names forms.
    pub code: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RequiredSource {
    /// The key every attestation must be signed by.
    pub signer: String,
    /// The hash over the recipe the prompt must be built with: its base
    /// prompt, its documents' servers and paths, and its fields. `None`
    /// accepts any recipe. The claim's statement still names the recipe it
    /// used, through the source record.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recipe_sha256: Option<String>,
}

/// The digest a statement names for a source record: see `source_id` in the
/// shared prompt source, which is where the first bind computes it.
pub fn source_record_id(record: &source_core::SourceRecord) -> String {
    prompt_core::source_id(&prompt_core::SourceRecord {
        present: record.present,
        text_sha256: record.text_sha256.clone(),
        text_total: record.text_total,
        recipe_sha256: record.recipe_sha256.clone(),
        documents: record.documents,
        documents_sha256: record.documents_sha256.clone(),
        signer: record.signer.clone(),
        fault: record.fault,
        fault_position: record.fault_position,
    })
}

/// What a claimer submits: the input it ran, what each inference claims, the
/// procedure kind's own result, and the commitment to its checkpoints, which a
/// contract stores without reading.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ClaimSubmission {
    pub version: u32,
    /// The procedure kind the claim is of. A contract accepts claims of one.
    pub procedure_kind: ProcedureKind,
    /// The id of the registered procedure the claim is of: its
    /// procedure kind's programs, its models, its parameters and what it requires
    /// of the source and the result (`crate::registry`).
    pub procedure: String,
    /// The address the claim is attributed to: `0x` and 40 hex digits. The
    /// claimer's choice. Every blob the claim posted names it, so nobody can
    /// submit this claim's blobs under another address.
    /// Blockchain-specific: an EVM address.
    pub claimer: String,
    /// The claimer's commitment to the input it ran. Today the SHA-256 of the
    /// prompt text. The contract takes it as given. Every inference's
    /// statement is recomputed over it, so all of them ran this input.
    pub input_text_sha256: String,
    /// The digest of the source record the claimer says the text was built
    /// from, or 64 zeros. The statement names it, so a record that is valid
    /// was produced under it.
    #[serde(default = "crate::inference::no_source_id")]
    pub source_sha256: String,
    /// The source record that digest is over, for a claim that names a
    /// source. A contract reads the signer and the recipe from it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<source_core::SourceRecord>,
    /// What each inference claims, in the order the procedure kind runs them: one
    /// for the `single` procedure kind, two for `consensus`.
    pub inferences: Vec<SubmittedInference>,
    /// What a consensus claim states of its two inferences. Absent for a
    /// procedure kind of one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub consensus: Option<SubmittedConsensus>,
    /// How many checkpoints the claim has: one per stage.
    pub checkpoint_count: u32,
    /// The claim's commitment to every stage boundary: the count and the
    /// root of the tree over its checkpoint hashes, in order
    /// ([`crate::checkpoint::checkpoints_root`]). One checkpoint opens
    /// against it with a path.
    pub checkpoints_root: String,
    /// What the claimer posted to the data-availability layer: its run's
    /// inputs and its checkpoint list with every checkpoint and input
    /// manifest, each blob with its availability proof. Empty until posted;
    /// a contract accepts no claim without it (`crate::availability`).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub availability: Vec<PostedBlob>,
}

impl ClaimSubmission {
    /// The commitment of every run input the claim posted, in the order its
    /// blobs state them. A contract keeps their digest
    /// ([`crate::availability::posted_inputs_digest`]).
    pub fn posted_run_inputs(&self) -> Vec<String> {
        self.availability
            .iter()
            .flat_map(|blob| blob.contents.run_inputs.iter().cloned())
            .collect()
    }
}

/// The address of a submission nobody is named for.
pub const NO_ADDRESS: &str = "0x0000000000000000000000000000000000000000";

/// Whether text is an address as a submission states one: `0x` and 40
/// lowercase hex digits.
/// Blockchain-specific: an EVM address.
pub fn is_address(text: &str) -> bool {
    text.strip_prefix("0x").is_some_and(|digits| {
        digits.len() == 40
            && digits
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    })
}

/// Why a contract would not open a challenge period for a submission.
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
    /// The source record submitted is not the one the claimed digest is over.
    AnotherSourceRecord { digest: String, submitted: String },
    /// The contract takes its prompt as given, and the claim names a source.
    UnexpectedSource,
    /// The contract requires a source, and the claim submits no record of one.
    MissingSource,
    /// The source record has a fault.
    SourceFault { fault: u32 },
    /// The documents were signed by another key than the contract's.
    AnotherSigner { expected: String, submitted: String },
    /// The prompt was built with another recipe than the contract's.
    AnotherRecipe { expected: String, submitted: String },
    /// The claim is of another procedure kind than the contract accepts.
    AnotherProcedureKind {
        expected: ProcedureKind,
        submitted: ProcedureKind,
    },
    /// The claim does not hold what a claim of its procedure kind does: another
    /// number of inferences, or a consensus where there is none to state.
    AnotherShape {
        procedure_kind: ProcedureKind,
        inferences: usize,
        consensus: bool,
    },
    /// The submitted agreement is not what the contract's forms give the two
    /// submitted outputs.
    AnotherConsensus {
        expected: SubmittedConsensus,
        submitted: SubmittedConsensus,
    },
    /// The output was compared with other forms than the contract's.
    AnotherForms { expected: String, submitted: String },
    /// The submitted code is not the one the contract's forms give the
    /// submitted output.
    AnotherResult { expected: u32, submitted: u32 },
    /// The claim's proof states no records for its checkpoint list.
    RecordsNotStated,
    /// The claim's last checkpoint does not run its procedure kind's result
    /// program.
    NotTheResultStage { expected: &'static str },
    /// What the claim's last stage states is not what was submitted.
    AnotherStatedResult,
    /// The last stage names no check for an inference, or the proof states
    /// no record for the one it names.
    NoCheck { inference: usize },
    /// The stage the last one read as an inference's check is not one.
    NotACheck { inference: usize },
    /// The record in an inference's check is not the one submitted.
    AnotherStatedRecord { inference: usize },
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
            Self::AnotherSourceRecord { digest, submitted } => write!(
                f,
                "the source record submitted has digest {submitted}, but the claim names {digest}"
            ),
            Self::UnexpectedSource => write!(
                f,
                "the contract takes its prompt as given, and the claim names a source"
            ),
            Self::MissingSource => write!(
                f,
                "the contract requires a source, and the claim submits no source record"
            ),
            Self::SourceFault { fault } => {
                write!(f, "the source record has fault {fault}")
            }
            Self::AnotherSigner {
                expected,
                submitted,
            } => write!(
                f,
                "the documents were signed by {submitted}, but the contract's signer is {expected}"
            ),
            Self::AnotherRecipe {
                expected,
                submitted,
            } => write!(
                f,
                "the prompt was built with recipe {submitted}, but the contract's is {expected}"
            ),
            Self::AnotherProcedureKind {
                expected,
                submitted,
            } => write!(
                f,
                "the claim is of the `{submitted}` procedure kind, but the contract accepts `{expected}`"
            ),
            Self::AnotherShape {
                procedure_kind,
                inferences,
                consensus,
            } => write!(
                f,
                "a claim of the `{procedure_kind}` procedure kind holds {} inference(s){}, and this one holds {inferences}{}",
                procedure_kind.inferences(),
                if procedure_kind.inferences() > 1 { " and what they agree on" } else { "" },
                if *consensus { " and an agreement" } else { "" },
            ),
            Self::AnotherConsensus {
                expected,
                submitted,
            } => write!(
                f,
                "the claim states agreement {} with code {}, but the contract's forms give the \
                 submitted outputs agreement {} with code {}",
                submitted.agree, submitted.code, expected.agree, expected.code
            ),
            Self::AnotherForms {
                expected,
                submitted,
            } => write!(
                f,
                "the output was compared with forms {submitted}, but the contract's are {expected}"
            ),
            Self::AnotherResult {
                expected,
                submitted,
            } => write!(
                f,
                "the claim states result code {submitted}, but the contract's forms give the \
                 submitted output {expected}"
            ),
            Self::RecordsNotStated => write!(
                f,
                "the claim's proof states no records for its checkpoint list"
            ),
            Self::NotTheResultStage { expected } => write!(
                f,
                "the claim's last stage does not run {expected}, so the claim does not end in \
                 its procedure kind's result stage"
            ),
            Self::AnotherStatedResult => write!(
                f,
                "the claim's last stage committed other results than the ones submitted"
            ),
            Self::NoCheck { inference } => write!(
                f,
                "the claim's last stage read no check for inference {inference} that its proof \
                 states a record of"
            ),
            Self::NotACheck { inference } => write!(
                f,
                "the stage the claim's last stage read for inference {inference} does not run \
                 {RECORD_STAGE}"
            ),
            Self::AnotherStatedRecord { inference } => write!(
                f,
                "inference {inference}'s check committed another record than the one submitted"
            ),
        }
    }
}

/// What a result stage's output states: for each inference the claim record
/// it was given and what it made of the output, and for a consensus the
/// agreement.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ResultInOutput {
    pub inferences: Vec<StatedInference>,
    pub consensus: Option<SubmittedConsensus>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct StatedInference {
    pub record: SubmittedRecord,
    pub result: SubmittedResult,
}

/// The stage program that states a claim record. In a claim of the `single`
/// procedure kind the checkpoint before the last is its, and the last is the
/// procedure kind's result program ([`ProcedureKind::result_program`]).
pub const RECORD_STAGE: &str = "sequence-check";

impl ClaimSubmission {
    /// What a claimer submits for a finished run: each inference's last
    /// check's record and text with the result stage's code, what a consensus
    /// states of them, the source record when the run has one, and the
    /// commitment to the checkpoint list in the run directory.
    pub fn from_run(
        procedure_kind: ProcedureKind,
        procedure: String,
        results: &[&InferenceResult],
        consensus: Option<SubmittedConsensus>,
        source: Option<&source_core::SourceRecord>,
        chain_dir: &Path,
    ) -> Result<Self> {
        let first = results.first().context("a claim holds an inference")?;
        let checkpoints = crate::checkpoint::read_checkpoint_hashes(
            &chain_dir.join(crate::checkpoint::CHECKPOINTS_TXT),
        )?;
        Ok(Self {
            version: CLAIM_SUBMISSION_VERSION,
            procedure_kind,
            procedure,
            claimer: NO_ADDRESS.to_string(),
            input_text_sha256: first.input_text_sha256.clone(),
            source_sha256: first.source_sha256.clone(),
            source: source.cloned(),
            inferences: results
                .iter()
                .map(|result| SubmittedInference {
                    output_text: result.generated_text.clone(),
                    record: SubmittedRecord {
                        statement_sha256: result.statement_sha256.clone(),
                        output_text_sha256: result.output_text_sha256.clone(),
                        valid: result.valid,
                    },
                    result: SubmittedResult {
                        code: result.result_code,
                        forms_sha256: result.result_forms_sha256.clone(),
                    },
                })
                .collect(),
            consensus,
            checkpoint_count: checkpoints.len() as u32,
            checkpoints_root: checkpoints_root(&checkpoints)?,
            availability: Vec::new(),
        })
    }

    /// What the result stage's output must state for this submission to be
    /// its claim's.
    pub fn stated(&self) -> ResultInOutput {
        ResultInOutput {
            inferences: self
                .inferences
                .iter()
                .map(|inference| StatedInference {
                    record: inference.record.clone(),
                    result: inference.result.clone(),
                })
                .collect(),
            consensus: self.consensus,
        }
    }
}

/// The statement digest a contract recomputes: its pinned model id and
/// parameters with the claimed input and source commitments. `None` when an
/// id is not a 32-byte hash.
pub fn expected_statement(
    model: &ModelPins,
    required: RequiredParameters,
    input_text_sha256: &str,
    source_sha256: &str,
) -> Option<[u8; 32]> {
    Some(
        Statement {
            input_text: parse_id(input_text_sha256)?,
            source: parse_id(source_sha256)?,
            model: parse_id(&model.model_id)?,
            raw_prompt: required.raw_prompt,
            max_new_tokens: required.max_new_tokens,
            range_size: model.range_size,
        }
        .digest(),
    )
}

/// The first check a contract makes on submission: the claim is of the
/// procedure kind the contract accepts. Everything else a contract checks is that
/// procedure kind's: which records a claim states and which stage states its
/// result.
pub fn check_procedure_kind(
    required: ProcedureKind,
    submission: &ClaimSubmission,
) -> std::result::Result<(), SubmissionRejection> {
    if submission.procedure_kind != required {
        return Err(SubmissionRejection::AnotherProcedureKind {
            expected: required,
            submitted: submission.procedure_kind,
        });
    }
    // A claim of a procedure kind holds one entry per inference it makes, and an
    // agreement exactly when the procedure kind states one.
    let states_consensus = required == ProcedureKind::Consensus;
    if submission.inferences.len() != required.inferences()
        || submission.consensus.is_some() != states_consensus
    {
        return Err(SubmissionRejection::AnotherShape {
            procedure_kind: required,
            inferences: submission.inferences.len(),
            consensus: submission.consensus.is_some(),
        });
    }
    Ok(())
}

/// The checks a contract makes on submission of one inference, before it
/// opens a challenge period: `model` is the model the contract pins for that
/// inference. One equality covers the model and every parameter; the other
/// two cover the output.
///
/// Every inference's statement is recomputed over the submission's one input
/// hash and source digest, so a claim of several inferences that passes this
/// for each ran them all on one input.
pub fn check_submission(
    model: &ModelPins,
    required: RequiredParameters,
    submission: &ClaimSubmission,
    inference: usize,
) -> std::result::Result<(), SubmissionRejection> {
    if !model.is_consistent() {
        return Err(SubmissionRejection::InconsistentPins);
    }
    let claimed =
        submission
            .inferences
            .get(inference)
            .ok_or(SubmissionRejection::AnotherShape {
                procedure_kind: submission.procedure_kind,
                inferences: submission.inferences.len(),
                consensus: submission.consensus.is_some(),
            })?;
    let expected = expected_statement(
        model,
        required,
        &submission.input_text_sha256,
        &submission.source_sha256,
    )
    .ok_or(SubmissionRejection::MalformedInputCommitment)?;
    let expected = hex::encode(expected);
    if claimed.record.statement_sha256 != expected {
        return Err(SubmissionRejection::StatementMismatch {
            expected,
            submitted: claimed.record.statement_sha256.clone(),
        });
    }
    let output = hex::encode(Sha256::digest(claimed.output_text.as_bytes()));
    if claimed.record.output_text_sha256 != output {
        return Err(SubmissionRejection::OutputMismatch {
            expected: output,
            submitted: claimed.record.output_text_sha256.clone(),
        });
    }
    if !claimed.record.valid {
        return Err(SubmissionRejection::NotValid);
    }
    Ok(())
}

/// The checks a contract makes on submission of where the prompt came from
/// and how the output was read.
///
/// The statement already names the source digest, so a valid record was
/// produced under it. This holds the digest to the record submitted, and the
/// record and the forms to what the contract requires.
pub fn check_origin(
    required: &RequiredOrigin,
    submission: &ClaimSubmission,
) -> std::result::Result<(), SubmissionRejection> {
    if let Some(record) = &submission.source {
        let submitted = source_record_id(record);
        if submitted != submission.source_sha256 {
            return Err(SubmissionRejection::AnotherSourceRecord {
                digest: submission.source_sha256.clone(),
                submitted,
            });
        }
    }
    match (&required.source, &submission.source) {
        (None, _) => {
            if submission.source_sha256 != crate::inference::no_source_id() {
                return Err(SubmissionRejection::UnexpectedSource);
            }
        }
        (Some(_), None) => return Err(SubmissionRejection::MissingSource),
        (Some(required), Some(record)) => {
            if !record.present || record.fault != 0 {
                return Err(SubmissionRejection::SourceFault {
                    fault: record.fault,
                });
            }
            if record.signer != required.signer {
                return Err(SubmissionRejection::AnotherSigner {
                    expected: required.signer.clone(),
                    submitted: record.signer.clone(),
                });
            }
            if let Some(recipe) = &required.recipe_sha256 {
                if record.recipe_sha256 != *recipe {
                    return Err(SubmissionRejection::AnotherRecipe {
                        expected: recipe.clone(),
                        submitted: record.recipe_sha256.clone(),
                    });
                }
            }
        }
    }
    // Each output text is held to its record by `check_submission`, so the
    // result taken on it here is the one the claim's own output has.
    for claimed in &submission.inferences {
        let expected = required.result_of(&claimed.output_text);
        if claimed.result.forms_sha256 != expected.forms_sha256 {
            return Err(SubmissionRejection::AnotherForms {
                expected: expected.forms_sha256,
                submitted: claimed.result.forms_sha256.clone(),
            });
        }
        if claimed.result.code != expected.code {
            return Err(SubmissionRejection::AnotherResult {
                expected: expected.code,
                submitted: claimed.result.code,
            });
        }
    }
    // And the agreement a consensus states of them is the one these forms
    // give the two outputs. `check_submission` held both records to be valid
    // and of one input.
    if let (Some(submitted), [first, second]) =
        (submission.consensus, submission.inferences.as_slice())
    {
        let expected = required.consensus_of(&first.output_text, &second.output_text);
        if submitted != expected {
            return Err(SubmissionRejection::AnotherConsensus {
                expected,
                submitted,
            });
        }
    }
    Ok(())
}

/// Whether a submission states what its own checkpoints commit to: each
/// inference's record is the one in the check its last stage read, and its
/// results and agreement are what that last stage states.
///
/// `posted` is what the claim's blobs hold, as their proofs establish it. The
/// proof read the records out of the stages' outputs, each held to the hash
/// its checkpoint commits to, so nothing is decoded here and nothing is left
/// for a challenger to show. The record is held to the check and not only to
/// the result stage after it, so the verdict a contract accepts rests on the
/// stage that scored the output. That the result stage read that check's
/// output and no other value is the link comparison's to hold.
pub fn check_stated_records(
    pins: &ProgramPins,
    submission: &ClaimSubmission,
    posted: &BlobContents,
) -> std::result::Result<(), SubmissionRejection> {
    let stated = posted
        .records
        .iter()
        .find(|stated| stated.checkpoints_root == submission.checkpoints_root)
        .ok_or(SubmissionRejection::RecordsNotStated)?;
    let expected = submission.procedure_kind.result_program();
    if !pins.runs(&stated.result_program, expected) {
        return Err(SubmissionRejection::NotTheResultStage { expected });
    }
    if stated.results != submission.stated() {
        return Err(SubmissionRejection::AnotherStatedResult);
    }
    let inputs = submission.procedure_kind.result_inputs();
    for (inference, claimed) in submission.inferences.iter().enumerate() {
        let check = inputs
            .get(inference)
            .and_then(|input| stated.result_inputs.get(*input))
            .and_then(|position| {
                stated
                    .checks
                    .iter()
                    .find(|check| check.checkpoint == *position)
            })
            .ok_or(SubmissionRejection::NoCheck { inference })?;
        if !pins.runs(&check.program, RECORD_STAGE) {
            return Err(SubmissionRejection::NotACheck { inference });
        }
        if check.record != claimed.record {
            return Err(SubmissionRejection::AnotherStatedRecord { inference });
        }
    }
    Ok(())
}
