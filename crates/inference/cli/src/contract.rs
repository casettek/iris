//! `iris claim check`: the checks a contract would make on a claim, run on the
//! host against the files a claimer and a challenger would post.
//!
//! The rules themselves are in `inference_artifacts::contract`. This module
//! finds their inputs: the pins beside the model manifest a run spec names,
//! the parameters that spec requires, and the claim directory `claim build`
//! wrote, which holds the submission, the checkpoint list, the checkpoints and
//! the outputs of the last check and of the result stage after it.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{ensure, Context, Result};
use inference_artifacts::contract::{
    check_origin, check_pipeline, check_submission, first_unapproved_stage,
    raster_boundary_mismatch, ChallengeRefusal, CheckpointList, ClaimSubmission, ContractPins,
    LinkCheck, ModelPins, PipelinePins, RecordBinding, RequiredOrigin, RequiredParameters,
    RequiredSource, ResultInOutput, StageAdmission, StatedInference, SubmittedConsensus,
    SubmittedRecord, SubmittedResult, CONTRACT_PINS_JSON,
};
use inference_artifacts::{
    read_checkpoint_trace, read_json, read_run_spec, Checkpoint, CheckpointTrace, InferenceRunSpec,
    ModelManifest, Pipeline, PreparedRun, CHECKPOINTS_TXT, CHECKPOINT_TRACE_JSON,
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

/// What a run spec requires of where a claim's prompt came from and of the
/// forms its output is read by. A contract holds these as it holds the
/// parameters: they are the terms the claim was asked for under.
fn required_origin(run_spec_path: &Path, run_spec: &InferenceRunSpec) -> Result<RequiredOrigin> {
    let Some(source) = &run_spec.source else {
        return Ok(RequiredOrigin {
            source: None,
            forms: Vec::new(),
        });
    };
    let run_spec_dir = run_spec_path
        .parent()
        .context("run spec has no parent directory")?;
    let terms = run_prep::source::source_terms(run_spec_dir, source)?;
    Ok(RequiredOrigin {
        source: Some(RequiredSource {
            signer: terms.signer,
            recipe_sha256: terms.recipe_sha256,
        }),
        forms: terms.forms,
    })
}

/// What a contract over a run would pin and require: the pipeline it
/// accepts claims of, that pipeline's approved stage programs, the model of
/// each inference, and its own parameters.
struct Contract {
    pipeline: Pipeline,
    programs: PipelinePins,
    models: Vec<ModelPins>,
    required: RequiredParameters,
    origin: RequiredOrigin,
}

/// The contract a run spec describes. `Ok(None)` when a model's import has
/// no pins: it was made where the stage projects are not locked.
///
/// A contract of the `single` pipeline pins what the model's import wrote.
/// One of several inferences pins each model as its import identifies it, and
/// the stage programs of the pipeline's chain over those imports, with the
/// program id each stage project has locked now.
fn contract_of(run_spec_path: &Path, run_spec: &InferenceRunSpec) -> Result<Option<Contract>> {
    let run_spec_dir = run_spec_path
        .parent()
        .context("run spec has no parent directory")?;
    let pipeline = run_spec.pipeline;
    let manifests: Vec<PathBuf> = run_spec
        .model_manifests()
        .into_iter()
        .map(|manifest| absolute(run_spec_dir, manifest))
        .collect();
    if manifests
        .iter()
        .any(|manifest| !manifest.with_file_name(CONTRACT_PINS_JSON).exists())
    {
        return Ok(None);
    }
    let (programs, models) = match manifests.as_slice() {
        [manifest] => {
            let pins = load_pins(manifest)?;
            (pins.pipeline, vec![pins.model])
        }
        manifests => {
            let mut models = Vec::new();
            let mut parts = Vec::new();
            let mut families = Vec::new();
            for path in manifests {
                let manifest = ModelManifest::load_verified(path)?;
                parts.push(manifest.verified_raster_template(path)?.1);
                models.push(ModelPins::of(&manifest));
                if !families.contains(&manifest.family) {
                    families.push(manifest.family.clone());
                }
            }
            let parts: Vec<&str> = parts.iter().map(String::as_str).collect();
            let programs =
                PipelinePins::from_template(families.join("+"), &pipeline.chain_template(&parts)?)?;
            (programs, models)
        }
    };
    Ok(Some(Contract {
        pipeline,
        programs,
        models,
        required: RequiredParameters {
            raw_prompt: run_spec.raw_prompt,
            max_new_tokens: run_spec.tokens,
        },
        origin: required_origin(run_spec_path, run_spec)?,
    }))
}

/// The checks as a claimer applies them to its own claim before posting it:
/// the first one a contract would reject it at. `Ok(None)` when it would be
/// accepted, or when an import has no pins to check against.
pub(crate) fn claim_rejection(
    prepared: &PreparedRun,
    submission_path: &Path,
) -> Result<Option<String>> {
    let run_spec = read_run_spec(&prepared.run_spec_path)?;
    let Some(contract) = contract_of(&prepared.run_spec_path, &run_spec)? else {
        return Ok(None);
    };
    Ok(check_claim_dir(&contract, submission_path)?.rejection())
}

pub fn check_claim(options: ClaimCheckOptions) -> Result<ClaimCheckReport> {
    let run_spec_path = absolute(&options.base_dir, &options.run_spec_path);
    let run_spec = read_run_spec(&run_spec_path)?;
    let contract = contract_of(&run_spec_path, &run_spec)?.with_context(|| {
        format!(
            "a model {} names has no {CONTRACT_PINS_JSON}: import it again from the iris root, \
             where the stage projects and their Raster.lock files are",
            run_spec_path.display()
        )
    })?;
    let submission_path = absolute(&options.base_dir, &options.submission_path);
    let mut report = check_claim_dir(&contract, &submission_path)?;

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

/// The submission checks, then the comparisons a contract makes against the
/// claimer's checkpoint list when a challenge arrives, on the files in the
/// claim's directory.
fn check_claim_dir(contract: &Contract, submission_path: &Path) -> Result<ClaimCheckReport> {
    let submission: ClaimSubmission = read_json(submission_path)?;
    let dir = claim_dir(submission_path)?;
    let pins = &contract.programs;
    let several = contract.models.len() > 1;
    let mut lines = vec![CheckLine {
        check: "submission: pipeline",
        rejection: check_pipeline(contract.pipeline, &submission)
            .err()
            .map(|rejection| rejection.to_string()),
    }];
    // Each inference against the model the contract pins for it.
    for (inference, model) in contract.models.iter().enumerate() {
        lines.push(CheckLine {
            check: match (several, inference) {
                (false, _) => "submission: statement, output and valid flag",
                (true, 0) => "submission: the first inference's statement, output and valid flag",
                (true, _) => "submission: the second inference's statement, output and valid flag",
            },
            rejection: check_submission(model, contract.required, &submission, inference)
                .err()
                .map(|rejection| rejection.to_string()),
        });
    }
    lines.push(CheckLine {
        check: if several {
            "submission: source record, forms, result codes and agreement"
        } else {
            "submission: source record, forms and result code"
        },
        rejection: check_origin(&contract.origin, &submission)
            .err()
            .map(|rejection| rejection.to_string()),
    });

    let list_path = dir.join(CHECKPOINTS_TXT);
    let list =
        fs::read(&list_path).with_context(|| format!("failed to read {}", list_path.display()))?;
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
        let Some(program) = pins.program(&checkpoint.program_commitment) else {
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

    // A claim ends in its pipeline's result stage, which names the last check
    // it read of each inference. Each record is held to its check, and the
    // results to the last stage.
    let last = trace
        .checkpoints
        .last()
        .context("the claim has no checkpoints")?;
    let unbound = |bound: RecordBinding, checkpoint: &Checkpoint, output_path: &Path| match bound
    {
        RecordBinding::Bound => None,
        RecordBinding::AnotherRecord => Some(format!(
            "stage {} committed another record than the one submitted",
            checkpoint.stage
        )),
        RecordBinding::AnotherStage { expected } => Some(format!(
            "stage {} does not run {expected}, so the claim does not end in its checks and \
             the result stage",
            checkpoint.stage
        )),
        RecordBinding::NotTheLastOutput {
            committed,
            presented,
        } => Some(format!(
            "{} hashes to {presented}, but the checkpoint of stage {} commits to {committed}",
            output_path.display(),
            checkpoint.stage
        )),
    };

    for (inference, input) in contract.pipeline.result_inputs().iter().enumerate() {
        let check_line = match (several, inference) {
            (false, _) => "challenge admission: the record is the last check's",
            (true, 0) => "challenge admission: the first inference's record is its last check's",
            (true, _) => "challenge admission: the second inference's record is its last check's",
        };
        // The check the result stage's own checkpoint names for this input.
        // A claim whose last stage names none, or one past its list, has no
        // check to present.
        let producer = last
            .input_sources
            .get(*input)
            .map(|index| trace.checkpoints.get(*index as usize));
        let rejection = match producer {
            Some(Some(check)) => {
                let check_dir = dir.join(&check.stage);
                let output_path = check_dir.join("output.bin");
                let output = fs::read(&output_path)
                    .with_context(|| format!("failed to read {}", output_path.display()))?;
                let edge = stage_output(&output_path, &check_dir.join("output.rindex"))?;
                // A stage that is no check is reported by the binding check,
                // so its record is only read when its output holds one.
                let record = edge.get("claim").map(claim_record).transpose()?;
                let in_output = record.clone().or_else(|| {
                    submission
                        .inferences
                        .get(inference)
                        .map(|claimed| claimed.record.clone())
                });
                match in_output {
                    Some(in_output) => {
                        let binding = list
                            .check_record_binding(
                                pins,
                                &submission,
                                inference,
                                last,
                                check,
                                &output,
                                &in_output,
                            )
                            .map_err(refusal)?;
                        ensure!(
                            binding != RecordBinding::Bound || record.is_some(),
                            "{} holds no claim record",
                            output_path.display()
                        );
                        unbound(binding, check, &output_path)
                    }
                    None => Some(String::from("the claim submits no such inference")),
                }
            }
            Some(None) => Some(format!(
                "the claim's last stage ({}) names a checkpoint past the list as its `{input}`",
                last.stage
            )),
            None => Some(format!(
                "the claim's last stage ({}) takes no `{input}` from an earlier stage, so the \
                 claim does not end in its checks and the result stage",
                last.stage
            )),
        };
        lines.push(CheckLine {
            check: check_line,
            rejection,
        });
    }

    let last_dir = dir.join(&last.stage);
    let output_path = last_dir.join("output.bin");
    let output = fs::read(&output_path)
        .with_context(|| format!("failed to read {}", output_path.display()))?;
    let stated = stage_output(&output_path, &last_dir.join("output.rindex"))?;
    // A last stage that is not the result stage is reported by the binding
    // check, so what its output holds is only read when it decodes.
    let decoded = result_in_output(contract.pipeline, &stated);
    let in_output = decoded
        .as_ref()
        .map_or_else(|_| submission.stated(), Clone::clone);
    let binding = list
        .check_result_binding(pins, &submission, last, &output, &in_output)
        .map_err(refusal)?;
    if binding == RecordBinding::Bound {
        // Bound to a decoded record, never to the submission's own.
        decoded.with_context(|| format!("{} holds no result record", output_path.display()))?;
    }
    lines.push(CheckLine {
        check: "challenge admission: the result is the last stage's",
        rejection: unbound(binding, last, &output_path),
    });
    Ok(ClaimCheckReport { lines })
}

/// A stage's `output.bin` as JSON.
fn stage_output(data: &Path, index: &Path) -> Result<serde_json::Value> {
    let value =
        raster_runtime::read_raster_value(data, index, &raster_runtime::ReadLimits::unbounded())?;
    crate::parity::compare::value_json(&value)
}

/// A claim record as a check commits it, or as the result stage restates it:
/// both name the statement, the output hash and the valid flag the same way.
fn claim_record(claim: &serde_json::Value) -> Result<SubmittedRecord> {
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

/// One inference as a result stage states it: the claim record it was given,
/// the code it gave the output and the hash of the forms.
fn stated_inference(stated: &serde_json::Value) -> Result<StatedInference> {
    Ok(StatedInference {
        record: claim_record(stated)?,
        result: SubmittedResult {
            code: stated
                .get("code")
                .and_then(serde_json::Value::as_u64)
                .and_then(|code| u32::try_from(code).ok())
                .context("the result record has no `code`")?,
            forms_sha256: stated
                .get("forms_sha256")
                .and_then(serde_json::Value::as_str)
                .context("the result record has no `forms_sha256`")?
                .to_string(),
        },
    })
}

/// What the result stage of this pipeline states in its output.
fn result_in_output(pipeline: Pipeline, stated: &serde_json::Value) -> Result<ResultInOutput> {
    Ok(match pipeline {
        Pipeline::Single => ResultInOutput {
            inferences: vec![stated_inference(stated)?],
            consensus: None,
        },
        Pipeline::Consensus => {
            let part = |name: &str| {
                stated
                    .get(name)
                    .with_context(|| format!("the consensus record has no `{name}`"))
            };
            let flag = |name: &str| -> Result<bool> {
                part(name)?
                    .as_bool()
                    .with_context(|| format!("the consensus record's `{name}` is no flag"))
            };
            ResultInOutput {
                inferences: vec![
                    stated_inference(part("first")?)?,
                    stated_inference(part("second")?)?,
                ],
                consensus: Some(SubmittedConsensus {
                    same_input: flag("same_input")?,
                    agree: flag("agree")?,
                    code: part("code")?
                        .as_u64()
                        .and_then(|code| u32::try_from(code).ok())
                        .context("the consensus record has no `code`")?,
                }),
            }
        }
    })
}
