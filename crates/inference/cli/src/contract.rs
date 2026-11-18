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
use inference_artifacts::availability::{
    check_claim_availability, AvailabilityParameters, RequiredPosting,
};
use inference_artifacts::contract::{
    check_origin, check_pipeline, check_submission, first_unapproved_stage,
    raster_boundary_mismatch, ChallengeRefusal, CheckpointList, ClaimSubmission, ContractPins,
    LinkCheck, RecordBinding, RequiredOrigin, RequiredParameters, RequiredSource,
    ResultInOutput, StageAdmission, StatedInference, SubmittedConsensus,
    SubmittedRecord, SubmittedResult, CONTRACT_PINS_JSON,
};
use inference_artifacts::registry::{Configuration, Registry};
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
    /// The pins the contract holds, as `iris claim pins` wrote them. Without
    /// it a `single` contract's are the model import's, and those of a
    /// pipeline over several models are made from the imports now.
    pub pins_path: Option<PathBuf>,
    /// The registry the claim's configuration is looked up in, as `iris
    /// registry add` writes it. Without it the contract's configuration is
    /// the one the run spec and the pins describe.
    pub registry_path: Option<PathBuf>,
    /// The stand-in data-availability layer's directory. Without it, the
    /// default one (`crate::availability::da_dir`).
    pub da_dir: Option<PathBuf>,
    /// When the claim is taken to be submitted, in Unix seconds. Without it,
    /// now.
    pub submitted_at: Option<u64>,
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

/// A pins file, held to the pipeline and the models it is used for.
fn load_pins(path: &Path, pipeline: Pipeline, models: &[&ModelManifest]) -> Result<ContractPins> {
    ensure!(
        path.exists(),
        "{} does not exist: write it with `iris claim pins`, or import the model again from \
         the iris root, where the stage projects and their Raster.lock files are",
        path.display()
    );
    let pins: ContractPins = read_json(path).with_context(|| {
        format!(
            "{} is not a contract pins file of this build: write it again with `iris claim \
             pins`, or import the model again",
            path.display()
        )
    })?;
    if let Some(mismatch) = pins.mismatch(pipeline, models) {
        anyhow::bail!("{} are not this run's pins: {mismatch}", path.display());
    }
    Ok(pins)
}

/// The imports a run spec names: each model's manifest path, the manifest
/// and the model's part of a chain.
fn run_imports(
    run_spec_path: &Path,
    run_spec: &InferenceRunSpec,
) -> Result<Vec<(PathBuf, ModelManifest, String)>> {
    let run_spec_dir = run_spec_path
        .parent()
        .context("run spec has no parent directory")?;
    run_spec
        .model_manifests()
        .into_iter()
        .map(|manifest| {
            let path = absolute(run_spec_dir, manifest);
            let manifest = ModelManifest::load_verified(&path)?;
            let part = manifest.verified_raster_template(&path)?.1;
            Ok((path, manifest, part))
        })
        .collect()
}

/// What a contract of a run spec's pipeline over its models pins, made from
/// the imports and the stage projects' locks as they are now.
pub fn run_pins(base_dir: &Path, run_spec_path: &Path) -> Result<ContractPins> {
    let run_spec_path = absolute(base_dir, run_spec_path);
    let run_spec = read_run_spec(&run_spec_path)?;
    let imports = run_imports(&run_spec_path, &run_spec)?;
    let imports: Vec<(&ModelManifest, &str)> = imports
        .iter()
        .map(|(_, manifest, part)| (manifest, part.as_str()))
        .collect();
    ContractPins::of(run_spec.pipeline, &imports)
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
            // The base prompt and the documents' domains are not pinned yet:
            // a configuration accepts any recipe, and the claim's statement
            // names the one it used.
            recipe_sha256: None,
        }),
        forms: terms.forms,
    })
}

/// The contract a run spec describes. `Ok(None)` when no pins file is given
/// and a model's import has none: it was made where the stage projects are
/// not locked.
///
/// The pins are those of the file given. Without one, a contract of the
/// `single` pipeline pins what the model's import wrote, and one of several
/// inferences what [`run_pins`] makes from the imports now.
fn contract_of(
    run_spec_path: &Path,
    run_spec: &InferenceRunSpec,
    pins_path: Option<&Path>,
) -> Result<Option<Configuration>> {
    let pipeline = run_spec.pipeline;
    let imports = run_imports(run_spec_path, run_spec)?;
    let manifests: Vec<&ModelManifest> = imports.iter().map(|(_, manifest, _)| manifest).collect();
    let pins = match (pins_path, imports.as_slice()) {
        (Some(path), _) => load_pins(path, pipeline, &manifests)?,
        (None, imports) => {
            if imports
                .iter()
                .any(|(path, _, _)| !path.with_file_name(CONTRACT_PINS_JSON).exists())
            {
                return Ok(None);
            }
            match imports {
                [(path, _, _)] => {
                    load_pins(&path.with_file_name(CONTRACT_PINS_JSON), pipeline, &manifests)?
                }
                imports => {
                    let imports: Vec<(&ModelManifest, &str)> = imports
                        .iter()
                        .map(|(_, manifest, part)| (manifest, part.as_str()))
                        .collect();
                    ContractPins::of(pipeline, &imports)?
                }
            }
        }
    };
    Ok(Some(Configuration::of(
        pins,
        RequiredParameters {
            raw_prompt: run_spec.raw_prompt,
            max_new_tokens: run_spec.tokens,
        },
        required_origin(run_spec_path, run_spec)?,
    )))
}

/// The configuration a claim of this prepared run is of: what a contract
/// registers for its pipeline, models and parameters. `Ok(None)` when an
/// import has no pins: it was made where the stage projects are not locked.
pub(crate) fn run_configuration(prepared: &PreparedRun) -> Result<Option<Configuration>> {
    let run_spec = read_run_spec(&prepared.run_spec_path)?;
    contract_of(&prepared.run_spec_path, &run_spec, None)
}

/// Where a contract's checks read posted data from, and when the claim is
/// taken to be submitted.
struct Availability<'a> {
    layer: &'a crate::availability::StubLayer,
    parameters: AvailabilityParameters,
    submitted_at: u64,
}

/// The checks as a claimer applies them to its own claim before posting it:
/// the first one a contract would reject it at, against `configuration`.
/// `Ok(None)` when it would be accepted.
pub(crate) fn claim_rejection(
    configuration: &Configuration,
    submission_path: &Path,
    da_dir: &Path,
) -> Result<Option<String>> {
    let layer = crate::availability::StubLayer::open(da_dir);
    let availability = Availability {
        layer: &layer,
        parameters: AvailabilityParameters::default(),
        submitted_at: crate::availability::now()?,
    };
    Ok(check_claim_dir(configuration, None, submission_path, &availability)?.rejection())
}

pub fn check_claim(options: ClaimCheckOptions) -> Result<ClaimCheckReport> {
    let run_spec_path = absolute(&options.base_dir, &options.run_spec_path);
    let run_spec = read_run_spec(&run_spec_path)?;
    let pins_path = options
        .pins_path
        .as_ref()
        .map(|path| absolute(&options.base_dir, path));
    let contract = contract_of(&run_spec_path, &run_spec, pins_path.as_deref())?.with_context(|| {
        format!(
            "a model {} names has no {CONTRACT_PINS_JSON}: import it again from the iris root, \
             where the stage projects and their Raster.lock files are",
            run_spec_path.display()
        )
    })?;
    let submission_path = absolute(&options.base_dir, &options.submission_path);
    // With a registry, the configuration is the one the claim names there.
    let mut unregistered = None;
    let contract = match &options.registry_path {
        None => contract,
        Some(path) => {
            let registry = Registry::read(&absolute(&options.base_dir, path))?;
            let submission: ClaimSubmission = read_json(&submission_path)?;
            match registry.get(&submission.configuration) {
                Some(registered) => registered.clone(),
                None => {
                    unregistered = Some(format!(
                        "the claim names configuration {}, which is not registered",
                        submission.configuration
                    ));
                    contract
                }
            }
        }
    };
    let da_dir = options
        .da_dir
        .as_ref()
        .map(|dir| absolute(&options.base_dir, dir))
        .unwrap_or_else(crate::availability::da_dir);
    let layer = crate::availability::StubLayer::open(da_dir);
    let availability = Availability {
        layer: &layer,
        parameters: AvailabilityParameters::default(),
        submitted_at: match options.submitted_at {
            Some(at) => at,
            None => crate::availability::now()?,
        },
    };
    let mut report = check_claim_dir(&contract, unregistered, &submission_path, &availability)?;

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
fn check_claim_dir(
    contract: &Configuration,
    unregistered: Option<String>,
    submission_path: &Path,
    availability: &Availability,
) -> Result<ClaimCheckReport> {
    let submission: ClaimSubmission = read_json(submission_path)?;
    let dir = claim_dir(submission_path)?;
    let pins = &contract.programs;
    let several = contract.models.len() > 1;
    let required = contract.required();
    let origin = contract.origin();
    let mut lines = vec![
        CheckLine {
            check: "submission: configuration",
            rejection: unregistered.or_else(|| {
                let id = contract.id();
                (submission.configuration != id).then(|| {
                    format!(
                        "the claim names configuration {}, and the contract's is {id}",
                        submission.configuration
                    )
                })
            }),
        },
        CheckLine {
            check: "submission: pipeline",
            rejection: check_pipeline(contract.pipeline, &submission)
                .err()
                .map(|rejection| rejection.to_string()),
        },
    ];
    // Each inference against the model the contract pins for it.
    for (inference, model) in contract.models.iter().enumerate() {
        lines.push(CheckLine {
            check: match (several, inference) {
                (false, _) => "submission: statement, output and valid flag",
                (true, 0) => "submission: the first inference's statement, output and valid flag",
                (true, _) => "submission: the second inference's statement, output and valid flag",
            },
            rejection: check_submission(model, required, &submission, inference)
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
        rejection: check_origin(&origin, &submission)
            .err()
            .map(|rejection| rejection.to_string()),
    });
    // What the claim posted: the new information its statement names and its
    // checkpoints, each proven and retrievable through the challenge period.
    lines.push(CheckLine {
        check: "submission: data posted, proven and retrievable through the challenge period",
        rejection: check_claim_availability(
            availability.layer,
            availability.parameters,
            &submission.availability,
            &RequiredPosting::of(&submission),
            availability.submitted_at,
        )
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

/// The configuration a run spec describes, with the pins of this file or,
/// without one, as `iris claim check` finds them.
pub fn spec_configuration(run_spec_path: &Path, pins_path: Option<&Path>) -> Result<Configuration> {
    let run_spec = read_run_spec(run_spec_path)?;
    contract_of(run_spec_path, &run_spec, pins_path)?.with_context(|| {
        format!(
            "a model {} names has no {CONTRACT_PINS_JSON}: import it again from the iris root, \
             where the stage projects and their Raster.lock files are",
            run_spec_path.display()
        )
    })
}

#[derive(Debug, Clone)]
pub struct ChallengePostOptions {
    pub run_spec_path: PathBuf,
    pub pins_path: Option<PathBuf>,
    /// The `challenge_bundle.json` that `challenge build` wrote.
    pub challenge_bundle_path: PathBuf,
    /// The disputed claim's `claim_submission.json`, in its claim directory.
    pub submission_path: PathBuf,
    pub da_dir: Option<PathBuf>,
    pub submitted_at: Option<u64>,
}

pub struct ChallengePosted {
    pub report: ClaimCheckReport,
    pub submission_path: PathBuf,
}

/// Post a challenge's trace body, write its submission beside the bundle,
/// and make the checks a contract would make when it arrives.
pub fn post_challenge(options: ChallengePostOptions) -> Result<ChallengePosted> {
    let configuration =
        spec_configuration(&options.run_spec_path, options.pins_path.as_deref())?;
    let claim: ClaimSubmission = read_json(&options.submission_path)?;
    let trace = claim_trace(&options.submission_path)?;
    let bundle = inference_artifacts::read_challenge_bundle(&options.challenge_bundle_path)?;
    let replay: inference_artifacts::ChallengeTrace = read_json(&bundle.challenge_trace_path)?;
    let stage = trace
        .checkpoints
        .iter()
        .position(|checkpoint| checkpoint.stage == bundle.stage)
        .with_context(|| format!("the claim has no stage `{}`", bundle.stage))?;
    let layer = crate::availability::StubLayer::open(
        options
            .da_dir
            .clone()
            .unwrap_or_else(crate::availability::da_dir),
    );
    let submission = challenge_submission(
        &claim.checkpoints_sha256,
        stage as u32,
        &trace.checkpoints[stage],
        &bundle.raster_commit_path,
        &replay.raster_output_sha256,
        &layer,
    )?;
    let submission_path = options
        .challenge_bundle_path
        .with_file_name(inference_artifacts::contract::CHALLENGE_SUBMISSION_JSON);
    inference_artifacts::write_json(&submission_path, &submission)?;
    let submitted_at = match options.submitted_at {
        Some(at) => at,
        None => crate::availability::now()?,
    };
    let report = check_challenge(
        &configuration,
        &options.submission_path,
        &submission,
        &layer,
        submitted_at,
    )?;
    Ok(ChallengePosted {
        report,
        submission_path,
    })
}

/// A challenge of the checkpoint at `stage` of the claim whose list hashes
/// to `checkpoints_sha256`: the Raster stage challenge built from that
/// checkpoint and the challenger's own run of the stage (its trace commitment
/// at `commit_path`, the output it ends in), with the trace body posted.
pub(crate) fn challenge_submission(
    checkpoints_sha256: &str,
    stage: u32,
    checkpoint: &Checkpoint,
    commit_path: &Path,
    output_sha256: &str,
    layer: &crate::availability::StubLayer,
) -> Result<inference_artifacts::contract::ChallengeSubmission> {
    let commitment = raster_prover::commitment_file::CommitmentFile::read_memory(commit_path)
        .with_context(|| format!("{} is not a Raster trace commitment", commit_path.display()))?;
    let challenge = inference_artifacts::contract::stage_challenge(
        checkpoint,
        commitment.header(),
        output_sha256,
    )
    .context("the claim's checkpoint is not one a dispute can be built from")?;
    // The body a defender reads: the commitment in the encoding Raster's
    // admission decodes.
    let body = postcard::to_allocvec(&commitment.into_memory())
        .context("the trace commitment does not encode")?;
    Ok(inference_artifacts::contract::ChallengeSubmission {
        version: inference_artifacts::contract::CHALLENGE_SUBMISSION_VERSION,
        checkpoints_sha256: checkpoints_sha256.to_string(),
        stage,
        challenge,
        availability: crate::availability::post_trace_body(layer, body)
            .context("posting the trace body")?,
    })
}

/// The checks a contract makes when a challenge arrives, before any replay:
/// it is about a checkpoint the claim lists, at a stage that names an
/// approved program, Raster admits it, and its trace body is posted for the
/// whole time a defender has to answer.
pub fn check_challenge(
    configuration: &Configuration,
    claim_submission_path: &Path,
    submission: &inference_artifacts::contract::ChallengeSubmission,
    layer: &crate::availability::StubLayer,
    submitted_at: u64,
) -> Result<ClaimCheckReport> {
    let claim: ClaimSubmission = read_json(claim_submission_path)?;
    let dir = claim_dir(claim_submission_path)?;
    let trace = claim_trace(claim_submission_path)?;
    let list_path = dir.join(CHECKPOINTS_TXT);
    let list_bytes =
        fs::read(&list_path).with_context(|| format!("failed to read {}", list_path.display()))?;
    let mut lines = Vec::new();

    let list = (submission.checkpoints_sha256 == claim.checkpoints_sha256)
        .then(|| CheckpointList::open(&claim, &list_bytes).ok())
        .flatten();
    lines.push(CheckLine {
        check: "challenge: of the claim's checkpoint list",
        rejection: list.is_none().then(|| {
            format!(
                "the challenge names checkpoint list {}, the claim stored {}",
                submission.checkpoints_sha256, claim.checkpoints_sha256
            )
        }),
    });
    let checkpoint = trace.checkpoints.get(submission.stage as usize);
    let admission = match (&list, checkpoint) {
        (Some(list), Some(checkpoint)) => Some(list.admit_stage_challenge(
            &configuration.programs,
            submission.stage,
            checkpoint,
        )),
        _ => None,
    };
    lines.push(CheckLine {
        check: "challenge admission: the stage names an approved program, so a replay settles it",
        rejection: match admission {
            Some(Ok(StageAdmission::Disputed { .. })) => None,
            Some(Ok(StageAdmission::UnapprovedProgram { program_commitment })) => Some(format!(
                "the stage names program {program_commitment}, which is not approved: the claim \
                 is void with no replay"
            )),
            Some(Err(refusal)) => Some(refusal.to_string()),
            None => Some(format!(
                "the claim has no checkpoint {} to dispute",
                submission.stage
            )),
        },
    });
    lines.push(CheckLine {
        check: "challenge: about the claim's checkpoint at that stage",
        rejection: (!checkpoint.is_some_and(|checkpoint| {
            inference_artifacts::contract::challenge_is_about(checkpoint, &submission.challenge)
        }))
        .then(|| String::from("the challenge's chain commitment is not this checkpoint's")),
    });
    lines.push(CheckLine {
        check: "challenge: Raster admits it",
        rejection: raster_prover::challenge::verify_challenge(&submission.challenge)
            .err()
            .map(|refusal| format!("{refusal:?}")),
    });
    lines.push(CheckLine {
        check: "challenge: trace body posted, proven and retrievable through the fault-proof window",
        rejection: inference_artifacts::availability::check_challenge_availability(
            layer,
            AvailabilityParameters::default(),
            &submission.availability,
            &submission.trace_header_digest(),
            submitted_at,
        )
        .err()
        .map(|rejection| rejection.to_string()),
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
