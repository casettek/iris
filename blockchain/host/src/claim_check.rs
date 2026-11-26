//! `iwebs claim check`: the checks a contract would make on a claim, run on the
//! host against the files a claimer and a challenger would post.
//!
//! The rules themselves are in `blockchain_rules::checks`. This module
//! finds their inputs: the pins beside the model manifest a specification names,
//! the parameters that specification requires, and the claim directory `claim build`
//! wrote, which holds the submission, the checkpoint hashes, the checkpoints and
//! the outputs of the last check and of the result stage after it.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{ensure, Context, Result};
use blockchain_rules::availability::{
    check_claim_availability, check_holds, AvailabilityLayer, AvailabilityParameters, BlobContents,
    RequiredPosting,
};
use blockchain_rules::checks::{
    check_origin, check_procedure_kind, check_stated_records, check_submission,
    first_unapproved_stage, raster_boundary_mismatch, ChallengeRefusal, CheckpointHashes,
    ClaimSubmission, ContractPins, LinkCheck, RequiredOrigin, RequiredParameters, RequiredSource,
    ResultInOutput, StageAdmission, StatedInference, SubmittedConsensus, SubmittedRecord,
    SubmittedResult, CONTRACT_PINS_JSON,
};
use blockchain_rules::registry::{Procedure, Registry};
use inference_manifest::ModelManifest;
use procedure_definition::{
    read_checkpoint_list, read_json, read_specification, Checkpoint, CheckpointList, PreparedRun,
    ProcedureKind, Specification, CHECKPOINTS_TXT, CHECKPOINT_LIST_JSON,
};
use raster::core::pipeline::PipelineCommitment;

#[derive(Debug, Clone)]
pub struct ClaimCheckOptions {
    pub base_dir: PathBuf,
    /// The specification the contract's required parameters are read from.
    pub specification_path: PathBuf,
    /// The `claim_submission.json` a claim build wrote. The claim's checkpoint
    /// files and stage outputs are read from the directory it is in.
    pub submission_path: PathBuf,
    /// A Raster pipeline run of the same claim. With it, every checkpoint is held
    /// to the boundary that run committed.
    pub raster_run: Option<PathBuf>,
    /// The pins the contract holds, as `iwebs claim pins` wrote them. Without
    /// it a `single` contract's are the model import's, and those of a
    /// procedure kind over several models are made from the imports now.
    pub pins_path: Option<PathBuf>,
    /// The registry the claim's procedure is looked up in, as `iwebs
    /// registry add` writes it. Without it the contract's procedure is
    /// the one the specification and the pins describe.
    pub registry_path: Option<PathBuf>,
    /// The stand-in data-availability layer's directory. Without it, the
    /// default one (`crate::posting::da_dir`).
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

/// A pins file, held to the procedure kind and the models it is used for.
fn load_pins(
    path: &Path,
    procedure_kind: ProcedureKind,
    models: &[&ModelManifest],
) -> Result<ContractPins> {
    ensure!(
        path.exists(),
        "{} does not exist: write it with `iwebs claim pins`, or import the model again from \
         the iwebs root, where the stage projects and their Raster.lock files are",
        path.display()
    );
    let pins: ContractPins = read_json(path).with_context(|| {
        format!(
            "{} is not a contract pins file of this build: write it again with `iwebs claim \
             pins`, or import the model again",
            path.display()
        )
    })?;
    if let Some(mismatch) = pins.mismatch(procedure_kind, models) {
        anyhow::bail!("{} are not this run's pins: {mismatch}", path.display());
    }
    Ok(pins)
}

/// The imports a specification names: each model's manifest path, the manifest
/// and the model's segment of a pipeline.
fn run_imports(
    specification_path: &Path,
    specification: &Specification,
) -> Result<Vec<(PathBuf, ModelManifest, String)>> {
    let specification_dir = specification_path
        .parent()
        .context("specification has no parent directory")?;
    specification
        .model_manifests()
        .into_iter()
        .map(|manifest| {
            let path = absolute(specification_dir, manifest);
            let manifest = ModelManifest::load_verified(&path)?;
            let segment = manifest.verified_raster_template(&path)?.1;
            Ok((path, manifest, segment))
        })
        .collect()
}

/// What a contract of a specification's procedure kind over its models pins, made from
/// the imports and the stage projects' locks as they are now.
pub fn run_pins(base_dir: &Path, specification_path: &Path) -> Result<ContractPins> {
    let specification_path = absolute(base_dir, specification_path);
    let specification = read_specification(&specification_path)?;
    let imports = run_imports(&specification_path, &specification)?;
    let imports: Vec<(&ModelManifest, &str)> = imports
        .iter()
        .map(|(_, manifest, segment)| (manifest, segment.as_str()))
        .collect();
    ContractPins::of(specification.procedure_kind, &imports)
}

/// What a specification requires of where a claim's prompt came from and of the
/// forms its output is read by. A contract holds these as it holds the
/// parameters: they are the terms the claim was asked for under.
fn required_origin(specification: &Specification) -> RequiredOrigin {
    // A relay's recipe is the specification's own text, so a procedure pins
    // it. The hash is over the recipe alone, so the relay step over no
    // output states it.
    let relay_recipe_sha256 = specification.relay.as_ref().map(|relay| {
        source_phase::relay::run_relay_steps(&relay.recipe(), "", &[], &[], true)
            .record
            .recipe_sha256
    });
    let Some(terms) = procedure_prepare::source::source_terms(specification) else {
        return RequiredOrigin {
            source: None,
            forms: Vec::new(),
            relay_recipe_sha256,
        };
    };
    RequiredOrigin {
        source: Some(RequiredSource {
            signer: terms.signer,
            // The base prompt and the documents' domains are not pinned yet:
            // a procedure accepts any recipe, and the claim's statement
            // names the one it used.
            recipe_sha256: None,
        }),
        forms: terms.forms,
        relay_recipe_sha256,
    }
}

/// The contract a specification describes. `Ok(None)` when no pins file is given
/// and a model's import has none: it was made where the stage projects are
/// not locked.
///
/// The pins are those of the file given. Without one, a contract of the
/// `single` procedure kind pins what the model's import wrote, and one of several
/// inferences what [`run_pins`] makes from the imports now.
fn contract_of(
    specification_path: &Path,
    specification: &Specification,
    pins_path: Option<&Path>,
) -> Result<Option<Procedure>> {
    let procedure_kind = specification.procedure_kind;
    let imports = run_imports(specification_path, specification)?;
    let manifests: Vec<&ModelManifest> = imports.iter().map(|(_, manifest, _)| manifest).collect();
    let pins = match (pins_path, imports.as_slice()) {
        (Some(path), _) => load_pins(path, procedure_kind, &manifests)?,
        (None, imports) => {
            if imports
                .iter()
                .any(|(path, _, _)| !path.with_file_name(CONTRACT_PINS_JSON).exists())
            {
                return Ok(None);
            }
            match imports {
                [(path, _, _)] => load_pins(
                    &path.with_file_name(CONTRACT_PINS_JSON),
                    procedure_kind,
                    &manifests,
                )?,
                imports => {
                    let imports: Vec<(&ModelManifest, &str)> = imports
                        .iter()
                        .map(|(_, manifest, segment)| (manifest, segment.as_str()))
                        .collect();
                    ContractPins::of(procedure_kind, &imports)?
                }
            }
        }
    };
    Ok(Some(Procedure::of(
        pins,
        RequiredParameters {
            raw_prompt: specification.raw_prompt,
            max_new_tokens: specification.max_new_tokens,
        },
        required_origin(specification),
    )))
}

/// The procedure a claim of this prepared run is of: what a contract
/// registers for its procedure kind, models and parameters. `Ok(None)` when an
/// import has no pins: it was made where the stage projects are not locked.
pub fn run_procedure(prepared: &PreparedRun) -> Result<Option<Procedure>> {
    let specification = read_specification(&prepared.specification_path)?;
    contract_of(&prepared.specification_path, &specification, None)
}

/// Where a contract's checks read posted data from, and when the claim is
/// taken to be submitted.
struct Availability<'a> {
    layer: &'a dyn AvailabilityLayer,
    parameters: AvailabilityParameters,
    submitted_at: u64,
}

/// What the checks have of a claim's data: what it posted, or, before it
/// posts, what the blobs it prepared hold.
enum Posting<'a> {
    Posted(Availability<'a>),
    Prepared(BlobContents),
}

/// The checks as a claimer applies them to its own claim before posting it:
/// the first one a contract would reject it at, against `procedure`.
/// `prepared` is what the blobs it is about to post hold. `Ok(None)` when the
/// claim would be accepted once they are posted.
pub fn claim_rejection(
    procedure: &Procedure,
    submission_path: &Path,
    prepared: BlobContents,
) -> Result<Option<String>> {
    Ok(check_claim_dir(
        procedure,
        None,
        submission_path,
        &Posting::Prepared(prepared),
    )?
    .rejection())
}

/// What `iwebs claim post` did: where each blob is, and whether a contract
/// would take them as the claim's data.
pub struct ClaimPosted {
    pub blobs: Vec<blockchain_rules::availability::PostedBlob>,
    /// `None` when they are, otherwise why not.
    pub rejection: Option<String>,
}

/// Post the blobs `claim build` prepared beside a submission, write where
/// they are into the submission, and make the availability check a contract
/// would. Posting again posts new blobs and replaces the submission's list.
pub fn post_claim_submission(
    submission_path: &Path,
    da_dir: Option<&Path>,
    submitted_at: Option<u64>,
) -> Result<ClaimPosted> {
    let mut submission: ClaimSubmission = read_json(submission_path)?;
    let layer = crate::posting::open_layer(da_dir)?;
    let prepared = crate::posting::PreparedPosting::read(claim_dir(submission_path)?)?;
    submission.availability = prepared
        .post(layer.as_ref())
        .context("posting the claim's data")?;
    procedure_definition::write_json(submission_path, &submission)?;
    let rejection = check_claim_availability(
        layer.as_ref(),
        AvailabilityParameters::default(),
        &submission.availability,
        &RequiredPosting::of(&submission),
        match submitted_at {
            Some(at) => at,
            None => crate::posting::now()?,
        },
    )
    .err()
    .map(|rejection| rejection.to_string());
    Ok(ClaimPosted {
        blobs: submission.availability,
        rejection,
    })
}

pub fn check_claim(options: ClaimCheckOptions) -> Result<ClaimCheckReport> {
    let specification_path = absolute(&options.base_dir, &options.specification_path);
    let specification = read_specification(&specification_path)?;
    let pins_path = options
        .pins_path
        .as_ref()
        .map(|path| absolute(&options.base_dir, path));
    let contract = contract_of(&specification_path, &specification, pins_path.as_deref())?
        .with_context(|| {
            format!(
            "a model {} names has no {CONTRACT_PINS_JSON}: import it again from the iwebs root, \
             where the stage projects and their Raster.lock files are",
            specification_path.display()
        )
        })?;
    let submission_path = absolute(&options.base_dir, &options.submission_path);
    // With a registry, the procedure is the one the claim names there.
    let mut unregistered = None;
    let contract = match &options.registry_path {
        None => contract,
        Some(path) => {
            let registry = Registry::read(&absolute(&options.base_dir, path))?;
            let submission: ClaimSubmission = read_json(&submission_path)?;
            match registry.get(&submission.procedure) {
                Some(registered) => registered.clone(),
                None => {
                    unregistered = Some(format!(
                        "the claim names procedure {}, which is not registered",
                        submission.procedure
                    ));
                    contract
                }
            }
        }
    };
    let da_dir = options
        .da_dir
        .as_ref()
        .map(|dir| absolute(&options.base_dir, dir));
    let layer = crate::posting::open_layer(da_dir.as_deref())?;
    let availability = Posting::Posted(Availability {
        layer: layer.as_ref(),
        parameters: AvailabilityParameters::default(),
        submitted_at: match options.submitted_at {
            Some(at) => at,
            None => crate::posting::now()?,
        },
    });
    let mut report = check_claim_dir(&contract, unregistered, &submission_path, &availability)?;

    if let Some(raster_run) = options.raster_run {
        let commitment_path = absolute(&options.base_dir, &raster_run).join("pipeline-commitment");
        let commitment = fs::read(&commitment_path)
            .with_context(|| format!("failed to read {}", commitment_path.display()))?;
        let pipeline: PipelineCommitment = postcard::from_bytes(&commitment)
            .with_context(|| format!("failed to parse {}", commitment_path.display()))?;
        report.lines.push(CheckLine {
            check: "every checkpoint is the boundary the Raster run committed",
            rejection: raster_boundary_mismatch(&claim_list(&submission_path)?, &pipeline),
        });
    }
    Ok(report)
}

fn claim_dir(submission_path: &Path) -> Result<&Path> {
    submission_path
        .parent()
        .context("the claim submission has no parent directory")
}

fn claim_list(submission_path: &Path) -> Result<CheckpointList> {
    read_checkpoint_list(&claim_dir(submission_path)?.join(CHECKPOINT_LIST_JSON))
}

/// The submission checks, then the comparisons a contract makes against the
/// claimer's checkpoint list when a challenge arrives, on the files in the
/// claim's directory.
fn check_claim_dir(
    contract: &Procedure,
    unregistered: Option<String>,
    submission_path: &Path,
    posting: &Posting,
) -> Result<ClaimCheckReport> {
    let submission: ClaimSubmission = read_json(submission_path)?;
    let dir = claim_dir(submission_path)?;
    let pins = &contract.programs;
    let several = contract.models.len() > 1;
    let required = contract.required();
    let origin = contract.origin();
    let mut lines = vec![
        CheckLine {
            check: "submission: procedure",
            rejection: unregistered.or_else(|| {
                let id = contract.id();
                (submission.procedure != id).then(|| {
                    format!(
                        "the claim names procedure {}, and the contract's is {id}",
                        submission.procedure
                    )
                })
            }),
        },
        CheckLine {
            check: "submission: procedure kind",
            rejection: check_procedure_kind(contract.procedure_kind, &submission)
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
    let required = RequiredPosting::of(&submission);
    let (check, posted) = match posting {
        Posting::Posted(availability) => (
            "submission: data posted, proven and retrievable through the challenge period",
            check_claim_availability(
                availability.layer,
                availability.parameters,
                &submission.availability,
                &required,
                availability.submitted_at,
            ),
        ),
        // Before a claim posts, the same question of the blobs it prepared:
        // whether they are its claimer's and hold what its statement names.
        Posting::Prepared(contents) => (
            "submission: the data prepared to post is the claim's",
            check_holds(contents, &required).map(|()| contents.clone()),
        ),
    };
    lines.push(CheckLine {
        check,
        rejection: posted.as_ref().err().map(ToString::to_string),
    });
    // What the claim's proof read out of its last stage's output and its
    // checks': each record and the results are the ones submitted. A claim
    // whose submission is not what its own checkpoints commit to is refused
    // here, so no dispute is about a record.
    lines.push(CheckLine {
        check: "submission: records and results are the ones its checkpoints commit to",
        rejection: check_stated_records(pins, &submission, &posted.unwrap_or_default())
            .err()
            .map(|rejection| rejection.to_string()),
    });

    let list_path = dir.join(CHECKPOINTS_TXT);
    let hashes =
        fs::read(&list_path).with_context(|| format!("failed to read {}", list_path.display()))?;
    let list = claim_list(submission_path)?;
    let refusal = |refusal: ChallengeRefusal| {
        anyhow::anyhow!(
            "the files in {} are not this claim's: {refusal}",
            dir.display()
        )
    };
    let hashes = CheckpointHashes::open(&submission, &hashes).map_err(refusal)?;
    ensure!(
        hashes.len() == list.checkpoints.len(),
        "{} lists {} checkpoints, {} holds {}",
        list_path.display(),
        hashes.len(),
        CHECKPOINT_LIST_JSON,
        list.checkpoints.len()
    );

    // A challenger would point the contract at the first stage that names a
    // program outside the set. Every other stage is opened too, so a list
    // that is not the one the hashes name is found here and not at a challenge.
    let unapproved = first_unapproved_stage(pins, &list);
    let mut rejection = None;
    for (stage, checkpoint) in list.checkpoints.iter().enumerate() {
        let admission = hashes
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
    'stages: for (stage, checkpoint) in list.checkpoints.iter().enumerate() {
        let Some(program) = pins.program(&checkpoint.program_commitment) else {
            continue;
        };
        let manifest_path = dir.join(&checkpoint.stage).join("input_manifest.json");
        let manifest = fs::read(&manifest_path)
            .with_context(|| format!("failed to read {}", manifest_path.display()))?;
        for input in &program.piped_inputs {
            let producer = checkpoint
                .input_sources
                .get(&input.name)
                .and_then(|index| list.checkpoints.get(*index as usize));
            match hashes.check_link(
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
        check: "challenge admission: every piped input is an earlier checkpoint's output",
        rejection,
    });

    Ok(ClaimCheckReport { lines })
}

/// The procedure a specification describes, with the pins of this file or,
/// without one, as `iwebs claim check` finds them.
pub fn spec_procedure(specification_path: &Path, pins_path: Option<&Path>) -> Result<Procedure> {
    let specification = read_specification(specification_path)?;
    contract_of(specification_path, &specification, pins_path)?.with_context(|| {
        format!(
            "a model {} names has no {CONTRACT_PINS_JSON}: import it again from the iwebs root, \
             where the stage projects and their Raster.lock files are",
            specification_path.display()
        )
    })
}

#[derive(Debug, Clone)]
pub struct ChallengePostOptions {
    pub specification_path: PathBuf,
    pub pins_path: Option<PathBuf>,
    /// The `challenge_bundle.json` that `challenge build` wrote.
    pub challenge_bundle_path: PathBuf,
    /// The disputed claim's `claim_submission.json`, in its claim directory.
    pub submission_path: PathBuf,
    /// The address the challenge is attributed to. With none, the zero
    /// address.
    pub challenger: Option<String>,
    pub da_dir: Option<PathBuf>,
    pub submitted_at: Option<u64>,
    /// The trace commitment to post, in place of the bundle's own: a copy of
    /// it that `test corrupt-challenge` changed.
    pub commitment_path: Option<PathBuf>,
    /// Where to write the submission. Default: beside the bundle.
    pub out_path: Option<PathBuf>,
}

pub struct ChallengePosted {
    pub report: ClaimCheckReport,
    pub submission_path: PathBuf,
}

pub struct ChallengeCheckOptions {
    pub specification_path: PathBuf,
    pub pins_path: Option<PathBuf>,
    /// The `challenge_submission.json` that `challenge post` wrote.
    pub challenge_path: PathBuf,
    /// The disputed claim's `claim_submission.json`, in its claim directory.
    pub submission_path: PathBuf,
    pub da_dir: Option<PathBuf>,
    pub submitted_at: Option<u64>,
}

/// Make the checks a contract would make of a challenge's submission as it
/// is. Nothing is posted and nothing is written.
pub fn check_challenge_submission(options: ChallengeCheckOptions) -> Result<ClaimCheckReport> {
    let procedure = spec_procedure(&options.specification_path, options.pins_path.as_deref())?;
    let submission: blockchain_rules::checks::ChallengeSubmission =
        read_json(&options.challenge_path)?;
    let layer = crate::posting::open_layer(options.da_dir.as_deref())?;
    let submitted_at = match options.submitted_at {
        Some(at) => at,
        None => crate::posting::now()?,
    };
    check_challenge(
        &procedure,
        &options.submission_path,
        &submission,
        layer.as_ref(),
        submitted_at,
    )
}

/// Post a challenge's trace body, write its submission beside the bundle,
/// and make the checks a contract would make when it arrives.
pub fn post_challenge(options: ChallengePostOptions) -> Result<ChallengePosted> {
    let procedure = spec_procedure(&options.specification_path, options.pins_path.as_deref())?;
    let claim: ClaimSubmission = read_json(&options.submission_path)?;
    let list = claim_list(&options.submission_path)?;
    let bundle = blockchain_rules::read_challenge_bundle(&options.challenge_bundle_path)?;
    let replay: blockchain_rules::ChallengeTrace = read_json(&bundle.challenge_trace_path)?;
    let stage = list
        .checkpoints
        .iter()
        .position(|checkpoint| checkpoint.stage == bundle.stage)
        .with_context(|| format!("the claim has no stage `{}`", bundle.stage))?;
    let layer = crate::posting::open_layer(options.da_dir.as_deref())?;
    let layer = layer.as_ref();
    let challenger = options
        .challenger
        .clone()
        .unwrap_or_else(|| blockchain_rules::checks::NO_ADDRESS.to_string());
    ensure!(
        blockchain_rules::checks::is_address(&challenger),
        "`{challenger}` is not an address: give `0x` and 40 lowercase hex digits"
    );
    let submission = challenge_submission(
        &claim.checkpoints_root,
        stage as u32,
        &list.checkpoints[stage],
        options
            .commitment_path
            .as_deref()
            .unwrap_or(&bundle.raster_commit_path),
        &replay.raster_output_sha256,
        &challenger,
        layer,
    )?;
    let submission_path = options.out_path.clone().unwrap_or_else(|| {
        options
            .challenge_bundle_path
            .with_file_name(blockchain_rules::checks::CHALLENGE_SUBMISSION_JSON)
    });
    procedure_definition::write_json(&submission_path, &submission)?;
    let submitted_at = match options.submitted_at {
        Some(at) => at,
        None => crate::posting::now()?,
    };
    let report = check_challenge(
        &procedure,
        &options.submission_path,
        &submission,
        layer,
        submitted_at,
    )?;
    Ok(ChallengePosted {
        report,
        submission_path,
    })
}

/// A challenge, attributed to `challenger`, of the checkpoint at `stage` of
/// the claim whose list has the root `checkpoints_root`: the Raster stage
/// challenge built from that checkpoint and the challenger's own run of the
/// stage (its trace commitment at `commit_path`, the output it ends in), with
/// the trace body posted.
pub fn challenge_submission(
    checkpoints_root: &str,
    stage: u32,
    checkpoint: &Checkpoint,
    commit_path: &Path,
    output_sha256: &str,
    challenger: &str,
    layer: &dyn crate::posting::PostingLayer,
) -> Result<blockchain_rules::checks::ChallengeSubmission> {
    let commitment = raster_prover::commitment_file::CommitmentFile::read_memory(commit_path)
        .with_context(|| format!("{} is not a Raster trace commitment", commit_path.display()))?;
    let challenge =
        blockchain_rules::checks::stage_challenge(checkpoint, commitment.header(), output_sha256)
            .context("the claim's checkpoint is not one a dispute can be built from")?;
    // The body a defender reads: the commitment in the encoding Raster's
    // admission decodes.
    let body = postcard::to_allocvec(&commitment.into_memory())
        .context("the trace commitment does not encode")?;
    Ok(blockchain_rules::checks::ChallengeSubmission {
        version: blockchain_rules::checks::CHALLENGE_SUBMISSION_VERSION,
        challenger: challenger.to_string(),
        checkpoints_root: checkpoints_root.to_string(),
        stage,
        challenge,
        availability: crate::posting::post_trace_body(layer, challenger, body)
            .context("posting the trace body")?,
    })
}

/// The checks a contract makes when a challenge arrives, before any replay:
/// it is about a checkpoint the claim lists, at a stage that names an
/// approved program, Raster admits it, and its trace body is posted for the
/// whole time a defender has to answer.
pub fn check_challenge(
    procedure: &Procedure,
    claim_submission_path: &Path,
    submission: &blockchain_rules::checks::ChallengeSubmission,
    layer: &dyn AvailabilityLayer,
    submitted_at: u64,
) -> Result<ClaimCheckReport> {
    let claim: ClaimSubmission = read_json(claim_submission_path)?;
    let dir = claim_dir(claim_submission_path)?;
    let list = claim_list(claim_submission_path)?;
    let list_path = dir.join(CHECKPOINTS_TXT);
    let hashes_bytes =
        fs::read(&list_path).with_context(|| format!("failed to read {}", list_path.display()))?;
    let mut lines = Vec::new();

    let hashes = (submission.checkpoints_root == claim.checkpoints_root)
        .then(|| CheckpointHashes::open(&claim, &hashes_bytes).ok())
        .flatten();
    lines.push(CheckLine {
        check: "challenge: of the claim's checkpoint list",
        rejection: hashes.is_none().then(|| {
            format!(
                "the challenge names checkpoint list {}, the claim stored {}",
                submission.checkpoints_root, claim.checkpoints_root
            )
        }),
    });
    let checkpoint = list.checkpoints.get(submission.stage as usize);
    let admission = match (&hashes, checkpoint) {
        (Some(hashes), Some(checkpoint)) => {
            Some(hashes.admit_stage_challenge(&procedure.programs, submission.stage, checkpoint))
        }
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
            blockchain_rules::checks::challenge_is_about(checkpoint, &submission.challenge)
        }))
        .then(|| String::from("the challenge's pipeline commitment is not this checkpoint's")),
    });
    lines.push(CheckLine {
        check: "challenge: Raster admits it",
        rejection: raster_prover::challenge::verify_challenge(&submission.challenge)
            .err()
            .map(|refusal| format!("{refusal:?}")),
    });
    lines.push(CheckLine {
        check:
            "challenge: trace body posted, proven and retrievable through the fault-proof period",
        rejection: blockchain_rules::availability::check_challenge_availability(
            layer,
            AvailabilityParameters::default(),
            &submission.availability,
            &submission.challenger,
            &submission.trace_header_digest(),
            submitted_at,
        )
        .err()
        .map(|rejection| rejection.to_string()),
    });
    Ok(ClaimCheckReport { lines })
}

/// A stage's output as JSON, read from its payload with its index. The
/// index has to be of the same value as the payload.
pub fn output_json(payload: &[u8], index: &[u8]) -> Result<serde_json::Value> {
    let artifact = raster_runtime::read_raster_artifact_from_bytes(
        payload,
        index,
        &raster_runtime::ReadLimits::unbounded(),
    )?;
    ensure!(
        artifact.roots_agree(),
        "the index is of another value than the payload"
    );
    crate::raster_json::value_json(&artifact.value)
}

pub fn claim_record(claim: &serde_json::Value) -> Result<SubmittedRecord> {
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

/// What a result stage of this kind states in its output.
pub fn result_in_output(
    result: procedure_definition::ResultKind,
    stated: &serde_json::Value,
) -> Result<ResultInOutput> {
    use procedure_definition::ResultKind;
    Ok(match result {
        ResultKind::OneOutput => ResultInOutput {
            inferences: vec![stated_inference(stated)?],
            consensus: None,
        },
        ResultKind::Agreement => {
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
