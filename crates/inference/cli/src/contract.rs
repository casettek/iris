//! `iris claim check`: the checks a contract would make on a claim, run on the
//! host against the files a claimer and a challenger would post.
//!
//! The rules themselves are in `inference_artifacts::contract`. This module
//! finds their inputs: the pins beside the model manifest a run spec names,
//! the parameters that spec requires, and the claim directory `claim build`
//! wrote, which holds the submission, the checkpoint list, the checkpoints and
//! the last stage's output.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{ensure, Context, Result};
use inference_artifacts::contract::{
    check_submission, first_unapproved_stage, raster_boundary_mismatch, ChallengeRefusal,
    CheckpointList, ClaimSubmission, ContractPins, LinkCheck, RecordBinding, RequiredParameters,
    StageAdmission, SubmittedRecord, CONTRACT_PINS_JSON,
};
use inference_artifacts::{
    read_checkpoint_trace, read_json, read_run_spec, CheckpointTrace, ModelManifest, PreparedRun,
    CHECKPOINTS_TXT, CHECKPOINT_TRACE_JSON,
};
use raster::core::chain::ChainCommitment;

#[derive(Debug, Clone)]
pub struct ClaimCheckOptions {
    pub base_dir: PathBuf,
    /// The run spec the contract's required parameters are read from.
    pub run_spec_path: PathBuf,
    /// The `claim_submission.json` a claim build wrote. The claim's checkpoint
    /// files and stage outputs are read from the directory it is in.
    pub submission_path: PathBuf,
    /// A Raster chain run of the same claim. With it, every checkpoint is held
    /// to the boundary that run committed.
    pub raster_run: Option<PathBuf>,
}

/// One check a contract would make, and how it came out.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckLine {
    pub check: &'static str,
    /// `None` when the check passed, otherwise why the claim is not accepted.
    pub rejection: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClaimCheckReport {
    pub lines: Vec<CheckLine>,
}

impl ClaimCheckReport {
    pub fn accepted(&self) -> bool {
        self.lines.iter().all(|line| line.rejection.is_none())
    }

    /// The first check a contract would reject the claim at.
    pub fn rejection(&self) -> Option<String> {
        self.lines.iter().find_map(|line| {
            line.rejection
                .as_ref()
                .map(|reason| format!("{}: {reason}", line.check))
        })
    }
}

fn absolute(base: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        base.join(path)
    }
}

/// The pins written beside a model manifest, held to that manifest.
pub(crate) fn load_pins(model_manifest_path: &Path) -> Result<ContractPins> {
    let path = model_manifest_path.with_file_name(CONTRACT_PINS_JSON);
    ensure!(
        path.exists(),
        "{} does not exist: import the model again from the iris root, where the stage \
         projects and their Raster.lock files are",
        path.display()
    );
    let pins: ContractPins = read_json(&path)?;
    let manifest = ModelManifest::load_verified(model_manifest_path)?;
    ensure!(
        pins.model.model_id == manifest.model_id && pins.model.range_size == manifest.range_size,
        "{} pins another model than {}",
        path.display(),
        model_manifest_path.display()
    );
    Ok(pins)
}

/// The checks as a claimer applies them to its own claim before posting it:
/// the first one a contract would reject it at. `Ok(None)` when it would be
/// accepted, or when the import has no pins to check against.
pub(crate) fn claim_rejection(
    prepared: &PreparedRun,
    submission_path: &Path,
) -> Result<Option<String>> {
    if !prepared
        .model_manifest_path
        .with_file_name(CONTRACT_PINS_JSON)
        .exists()
    {
        return Ok(None);
    }
    let pins = load_pins(&prepared.model_manifest_path)?;
    let required = RequiredParameters {
        raw_prompt: prepared.raw_prompt,
        max_new_tokens: prepared.tokens,
    };
    Ok(check_claim_dir(&pins, required, submission_path)?.rejection())
}

pub fn check_claim(options: ClaimCheckOptions) -> Result<ClaimCheckReport> {
    let run_spec_path = absolute(&options.base_dir, &options.run_spec_path);
    let run_spec = read_run_spec(&run_spec_path)?;
    let run_spec_dir = run_spec_path
        .parent()
        .context("run spec has no parent directory")?;
    let pins = load_pins(&absolute(run_spec_dir, &run_spec.model_manifest))?;
    let required = RequiredParameters {
        raw_prompt: run_spec.raw_prompt,
        max_new_tokens: run_spec.tokens,
    };
    let submission_path = absolute(&options.base_dir, &options.submission_path);
    let mut report = check_claim_dir(&pins, required, &submission_path)?;

    if let Some(raster_run) = options.raster_run {
        let commitment_path = absolute(&options.base_dir, &raster_run).join("chain-commitment");
        let commitment = fs::read(&commitment_path)
            .with_context(|| format!("failed to read {}", commitment_path.display()))?;
        let chain: ChainCommitment = postcard::from_bytes(&commitment)
            .with_context(|| format!("failed to parse {}", commitment_path.display()))?;
        report.lines.push(CheckLine {
            check: "every checkpoint is the boundary the Raster run committed",
            rejection: raster_boundary_mismatch(&claim_trace(&submission_path)?, &chain),
        });
    }
    Ok(report)
}

fn claim_dir(submission_path: &Path) -> Result<&Path> {
    submission_path
        .parent()
        .context("the claim submission has no parent directory")
}

fn claim_trace(submission_path: &Path) -> Result<CheckpointTrace> {
    read_checkpoint_trace(&claim_dir(submission_path)?.join(CHECKPOINT_TRACE_JSON))
}

/// The submission checks, then the two comparisons a contract makes against
/// the claimer's checkpoint list when a challenge arrives, on the files in the
/// claim's directory.
fn check_claim_dir(
    pins: &ContractPins,
    required: RequiredParameters,
    submission_path: &Path,
) -> Result<ClaimCheckReport> {
    let submission: ClaimSubmission = read_json(submission_path)?;
    let dir = claim_dir(submission_path)?;
    let mut lines = vec![CheckLine {
        check: "submission: statement, output and valid flag",
        rejection: check_submission(pins, required, &submission)
            .err()
            .map(|rejection| rejection.to_string()),
    }];

    let list_path = dir.join(CHECKPOINTS_TXT);
    let list = fs::read(&list_path)
        .with_context(|| format!("failed to read {}", list_path.display()))?;
    let trace = claim_trace(submission_path)?;
    let refusal = |refusal: ChallengeRefusal| {
        anyhow::anyhow!(
            "the files in {} are not this claim's: {refusal}",
            dir.display()
        )
    };
    let list = CheckpointList::open(&submission, &list).map_err(refusal)?;
    ensure!(
        list.len() == trace.checkpoints.len(),
        "{} lists {} checkpoints, {} holds {}",
        list_path.display(),
        list.len(),
        CHECKPOINT_TRACE_JSON,
        trace.checkpoints.len()
    );

    // A challenger would point the contract at the first stage that names a
    // program outside the set. Every other stage is opened too, so a trace
    // that is not the listed one is found here and not at a challenge.
    let unapproved = first_unapproved_stage(pins, &trace);
    let mut rejection = None;
    for (stage, checkpoint) in trace.checkpoints.iter().enumerate() {
        let admission = list
            .admit_stage_challenge(pins, stage as u32, checkpoint)
            .map_err(refusal)?;
        if let StageAdmission::UnapprovedProgram { program_commitment } = admission {
            if unapproved == Some(stage as u32) {
                rejection = Some(if program_commitment.is_empty() {
                    format!("stage {stage} ({}) names no program", checkpoint.stage)
                } else {
                    format!(
                        "stage {stage} ({}) names program {program_commitment}, which is not \
                         in the approved set",
                        checkpoint.stage
                    )
                });
            }
        }
    }
    lines.push(CheckLine {
        check: "challenge admission: every stage names an approved program",
        rejection,
    });

    // Every input a stage takes from an earlier stage is that stage's
    // committed output, or an empty value: the first one that is not is where
    // a challenger would point. A stage naming an unapproved program has no
    // inputs to check and is reported above.
    let mut rejection = None;
    'stages: for (stage, checkpoint) in trace.checkpoints.iter().enumerate() {
        let Some(program) = pins.pipeline.program(&checkpoint.program_commitment) else {
            continue;
        };
        let manifest_path = dir.join(&checkpoint.stage).join("input_manifest.json");
        let manifest = fs::read(&manifest_path)
            .with_context(|| format!("failed to read {}", manifest_path.display()))?;
        for input in &program.chained_inputs {
            let producer = checkpoint
                .input_sources
                .get(&input.name)
                .and_then(|index| trace.checkpoints.get(*index as usize));
            match list.check_link(
                pins,
                stage as u32,
                checkpoint,
                &manifest,
                &input.name,
                producer,
            ) {
                Ok(LinkCheck::Linked) => {}
                Ok(LinkCheck::Void(fault)) => {
                    rejection = Some(format!(
                        "stage {stage} ({}) input `{}`: {fault}",
                        checkpoint.stage, input.name
                    ));
                    break 'stages;
                }
                // A producer index past the end of the list: a challenger
                // has no checkpoint to present, and the claim has none either.
                Err(ChallengeRefusal::ProducerNotPresented { producer, .. }) => {
                    rejection = Some(format!(
                        "stage {stage} ({}) input `{}`: its checkpoint names checkpoint \
                         {producer} as the producer, and the claim has no such checkpoint",
                        checkpoint.stage, input.name
                    ));
                    break 'stages;
                }
                Err(other) => return Err(refusal(other)),
            }
        }
    }
    lines.push(CheckLine {
        check: "challenge admission: every chained input is an earlier checkpoint's output",
        rejection,
    });

    let last = trace
        .checkpoints
        .last()
        .context("the claim has no checkpoints")?;
    let last_dir = dir.join(&last.stage);
    let output_path = last_dir.join("output.bin");
    let output = fs::read(&output_path)
        .with_context(|| format!("failed to read {}", output_path.display()))?;
    let in_output = record_in_output(&output_path, &last_dir.join("output.rindex"))?;
    lines.push(CheckLine {
        check: "challenge admission: the record is the last stage's",
        rejection: match list
            .check_record_binding(&submission, last, &output, &in_output)
            .map_err(refusal)?
        {
            RecordBinding::Bound => None,
            RecordBinding::AnotherRecord => Some(format!(
                "the last stage ({}) committed another record than the one submitted",
                last.stage
            )),
            RecordBinding::NotTheLastOutput {
                committed,
                presented,
            } => Some(format!(
                "{} hashes to {presented}, but the claim's last checkpoint commits to \
                 {committed}",
                output_path.display()
            )),
        },
    });
    Ok(ClaimCheckReport { lines })
}

/// The claim record a last check's `output.bin` holds.
fn record_in_output(data: &Path, index: &Path) -> Result<SubmittedRecord> {
    let value =
        raster_runtime::read_raster_value(data, index, &raster_runtime::ReadLimits::unbounded())?;
    let edge = crate::parity::compare::value_json(&value)?;
    let claim = edge
        .get("claim")
        .with_context(|| format!("{} holds no claim record", data.display()))?;
    let hash = |field: &str| -> Result<String> {
        let bytes = claim
            .get(field)
            .and_then(|page| page.get("bytes"))
            .and_then(serde_json::Value::as_str)
            .with_context(|| format!("the claim record has no `{field}` bytes"))?;
        ensure!(
            bytes.len() == 64,
            "the claim record's `{field}` is not 32 bytes"
        );
        Ok(bytes.to_string())
    };
    Ok(SubmittedRecord {
        statement_sha256: hash("statement")?,
        output_text_sha256: hash("output_text_sha256")?,
        valid: claim
            .get("valid")
            .and_then(serde_json::Value::as_bool)
            .context("the claim record has no `valid` flag")?,
    })
}
