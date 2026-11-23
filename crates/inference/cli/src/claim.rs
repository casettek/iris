use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use direct_infer::{DirectInferenceConfig, DirectInferenceExecutor};
use staged_infer::chain_runner::StagedExecutionBackend;
use staged_infer::{
    CheckpointedInferenceConfig, CheckpointedInferenceExecutor, CheckpointedInferenceResult,
    ParityPolicy,
};

use inference_artifacts::contract::{ClaimSubmission, CLAIM_SUBMISSION_JSON};
use inference_artifacts::{
    file_sha256 as sha256_file, read_checkpoint_hashes, read_checkpoint_trace, read_json,
    read_run_spec, write_checkpoint_hashes, write_claim_artifacts_with_prepared_run, write_json,
    InferenceResult, InferenceRunSpec, ModelManifest, PreparedRun, CHECKPOINT_TRACE_JSON,
    PREPARED_RUN_JSON,
};
use serde::Serialize;
use sha2::{Digest, Sha256};

#[derive(Debug, Clone)]
pub struct ClaimBuildOptions {
    pub base_dir: PathBuf,
    pub run_spec_path: PathBuf,
    pub current_exe: PathBuf,
    pub staged_backend: StagedExecutionBackend,
    /// Debug-only: post these output ids instead of discovering them with direct
    /// inference. Used to build claims whose tokens are not the greedy
    /// continuation.
    pub claimed_token_ids: Option<Vec<u32>>,
    /// The output a direct run of this same spec already produced. Taken in
    /// place of running direct inference again, and held to every rule a
    /// discovered output is: the chain re-derives each token, and a claim it
    /// rejects is not written.
    pub direct_output_ids: Option<Vec<u32>>,
    /// The address the claim is attributed to, `0x` and 40 hex digits. With
    /// none, the claim names the zero address.
    pub claimer: Option<String>,
}

pub(crate) struct PreparedClaimRun {
    pub(crate) prepared_run: PreparedRun,
    pub(crate) manifest_path: PathBuf,
    /// The ids written into the run's `claimed_ids` externals: the first
    /// inference's, which are all of them in a pipeline of one. Kept for the
    /// claimer's own tooling only: the claim posts no copy of them, because its
    /// output is the chain's final record.
    pub(crate) claimed_token_ids: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClaimBuildResult {
    /// The pipeline the claim is of.
    pub pipeline: inference_artifacts::Pipeline,
    pub chain_dir: PathBuf,
    pub checkpoint_trace_path: PathBuf,
    pub checkpoint_hashes_path: PathBuf,
    pub claim_bundle_path: PathBuf,
    /// What a claimer would submit to a contract. Absent when the chain
    /// published no final record to submit.
    pub claim_submission_path: Option<PathBuf>,
    /// The first inference's result: the whole result of a pipeline of one.
    pub final_result: Option<InferenceResult>,
    /// What a consensus chain states besides.
    pub consensus: Option<inference_artifacts::ConsensusOutcome>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CorruptCheckpointSelection {
    Random { seed: Option<String> },
    Index(usize),
    Stage(String),
}

#[derive(Debug, Clone)]
pub struct CorruptCheckpointHashesOptions {
    pub hashes_path: PathBuf,
    pub trace_path: Option<PathBuf>,
    pub output_path: Option<PathBuf>,
    pub selection: CorruptCheckpointSelection,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CorruptCheckpointHashesResult {
    pub source_hashes_path: PathBuf,
    pub corrupted_hashes_path: PathBuf,
    pub corruption_manifest_path: PathBuf,
    pub checkpoint_index: usize,
    pub checkpoint_number: usize,
    pub stage: Option<String>,
    pub original_hash: String,
    pub corrupted_hash: String,
}

#[derive(Debug, Serialize)]
struct CorruptionManifest {
    version: u32,
    source_hashes_path: PathBuf,
    corrupted_hashes_path: PathBuf,
    checkpoint_index: usize,
    checkpoint_number: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    stage: Option<String>,
    original_hash: String,
    corrupted_hash: String,
}

pub fn build_claim(options: ClaimBuildOptions) -> Result<ClaimBuildResult> {
    let prepared = prepare_claim_run(&options)?;
    let executor = CheckpointedInferenceExecutor;
    let checkpointed = executor.run(CheckpointedInferenceConfig {
        base_dir: options.base_dir,
        manifest_path: prepared.manifest_path.clone(),
        current_exe: options.current_exe,
        staged_backend: options.staged_backend,
        parity_policy: ParityPolicy::Skip,
    })?;
    // A claimer applies the rule a challenger will: the claim is judged by
    // its own final record, and a claim of several inferences by each one's.
    let pipeline = prepared.prepared_run.pipeline;
    let mut rejection =
        crate::challenge::claimed_output_rejection(checkpointed.final_result.as_ref())?;
    if pipeline.inferences() > 1 && rejection.is_none() {
        let consensus = checkpointed
            .consensus
            .as_ref()
            .with_context(|| format!("the `{pipeline}` chain stated no result"))?;
        rejection = crate::challenge::claimed_output_rejection(Some(&consensus.second))?
            .map(|reason| format!("the second inference: {reason}"));
    }
    if let Some(reason) = rejection {
        if options.claimed_token_ids.is_none() {
            if options.direct_output_ids.is_some() {
                anyhow::bail!(
                    "{reason}; the ids given as this run's direct output are not what the chain \
                     derives, so they are not that output or direct and teacher-forced \
                     inference are not in parity"
                );
            }
            // The ids came from direct decode, so the two paths are not
            // bit-identical: an honest claimer would convict itself.
            anyhow::bail!("{reason}; direct and teacher-forced inference are not in parity");
        }
        eprintln!("warning: {reason}; writing the claim anyway because its ids were given (`iris test claim`)");
    }
    let result = write_claim_result(
        checkpointed,
        &prepared.manifest_path,
        &prepared.prepared_run,
    )?;
    // And the rules a contract will: the statement the chain bound is the one
    // recomputed from the pinned model and this run's parameters, every
    // checkpoint names an approved program, and the record is the last
    // stage's.
    //
    // The claim names its configuration and its claimer, and prepares what
    // it has to post: the inputs its run was given and its checkpoints. Posting is `iris claim post`'s, so that a post that fails
    // is tried again without the claim being built again.
    // Without pins there is no configuration to name, and nothing to check.
    if let Some(path) = result.claim_submission_path.as_ref() {
        let Some(configuration) = crate::contract::run_configuration(&prepared.prepared_run)?
        else {
            return Ok(result);
        };
        let mut submission: ClaimSubmission = inference_artifacts::read_json(path)?;
        submission.configuration = configuration.id();
        if let Some(claimer) = options.claimer {
            anyhow::ensure!(
                inference_artifacts::contract::is_address(&claimer),
                "`{claimer}` is not an address: give `0x` and 40 lowercase hex digits"
            );
            submission.claimer = claimer;
        }
        inference_artifacts::write_json(path, &submission)?;
        let posting = crate::availability::claim_blobs(
            &result.chain_dir,
            &submission.claimer,
            Some(crate::availability::posted_run(&prepared.prepared_run)?),
        )
        .context("preparing the claim's data")?;
        posting.write(&result.chain_dir)?;
        if let Some(reason) =
            crate::contract::claim_rejection(&configuration, path, posting.contents()?)?
        {
            if options.claimed_token_ids.is_none() {
                anyhow::bail!("a contract would not accept this claim: {reason}");
            }
            eprintln!(
                "warning: a contract would not accept this claim: {reason}; it was written \
                 anyway because its ids were given (`iris test claim`)"
            );
        }
    }
    Ok(result)
}

/// [`build_claim`], then its data posted: a claim as a contract receives it.
pub fn build_posted_claim(options: ClaimBuildOptions) -> Result<ClaimBuildResult> {
    let result = build_claim(options)?;
    if let Some(path) = result.claim_submission_path.as_ref() {
        // A claim with no configuration prepared nothing to post.
        if path
            .with_file_name(crate::availability::POSTING_OUTPUTS_JSON)
            .exists()
        {
            let posted = crate::contract::post_claim_submission(path, None, None)?;
            if let Some(reason) = posted.rejection {
                anyhow::bail!("a contract would not take what the claim posted: {reason}");
            }
        }
    }
    Ok(result)
}

pub fn corrupt_checkpoint_hashes(
    options: CorruptCheckpointHashesOptions,
) -> Result<CorruptCheckpointHashesResult> {
    let mut hashes = read_checkpoint_hashes(&options.hashes_path)?;
    if hashes.is_empty() {
        anyhow::bail!("cannot corrupt empty checkpoint hash file");
    }

    let trace_path = resolve_trace_path(&options.hashes_path, options.trace_path.as_deref());
    let trace = if trace_path.is_file() {
        Some(read_checkpoint_trace(&trace_path)?)
    } else {
        None
    };
    let checkpoint_index = select_checkpoint_index(&options.selection, &hashes, trace.as_ref())?;
    let checkpoint_number = checkpoint_index + 1;
    let stage = trace
        .as_ref()
        .and_then(|trace| trace.checkpoints.get(checkpoint_index))
        .map(|checkpoint| checkpoint.stage.clone());
    let original_hash = hashes[checkpoint_index].clone();
    let corrupted_hash = corrupt_hash(&original_hash);
    hashes[checkpoint_index] = corrupted_hash.clone();

    let corrupted_hashes_path = options
        .output_path
        .unwrap_or_else(|| default_corrupted_hashes_path(&options.hashes_path));
    write_checkpoint_hashes(&corrupted_hashes_path, &hashes)?;

    let corruption_manifest_path = corrupted_hashes_path.with_extension("corruption.json");
    write_json(
        &corruption_manifest_path,
        &CorruptionManifest {
            version: 1,
            source_hashes_path: options.hashes_path.clone(),
            corrupted_hashes_path: corrupted_hashes_path.clone(),
            checkpoint_index,
            checkpoint_number,
            stage: stage.clone(),
            original_hash: original_hash.clone(),
            corrupted_hash: corrupted_hash.clone(),
        },
    )?;

    Ok(CorruptCheckpointHashesResult {
        source_hashes_path: options.hashes_path,
        corrupted_hashes_path,
        corruption_manifest_path,
        checkpoint_index,
        checkpoint_number,
        stage,
        original_hash,
        corrupted_hash,
    })
}

fn write_claim_result(
    checkpointed: CheckpointedInferenceResult,
    manifest_path: &PathBuf,
    prepared_run: &PreparedRun,
) -> Result<ClaimBuildResult> {
    let prepared_run_path = prepared_run
        .run_manifest_path
        .as_ref()
        .and_then(|path| path.parent())
        .map(|dir| dir.join(PREPARED_RUN_JSON));
    let (checkpoint_trace_path, checkpoint_hashes_path, claim_bundle_path) =
        write_claim_artifacts_with_prepared_run(
            &checkpointed.chain_dir,
            manifest_path,
            prepared_run_path.as_deref(),
        )?;
    // The submission is stated over the checkpoint list written just above:
    // each inference's result, and what a consensus states of them.
    let results: Vec<&InferenceResult> = checkpointed
        .final_result
        .iter()
        .chain(
            checkpointed
                .consensus
                .iter()
                .map(|consensus| &consensus.second),
        )
        .collect();
    let consensus = checkpointed.consensus.as_ref().map(|consensus| {
        inference_artifacts::contract::SubmittedConsensus {
            same_input: consensus.same_input,
            agree: consensus.agree,
            code: consensus.code,
        }
    });
    let claim_submission_path = if results.len() == prepared_run.pipeline.inferences() {
        // The configuration and what was posted are filled in by
        // `build_claim`, which knows the contract.
        let submission = ClaimSubmission::from_run(
            prepared_run.pipeline,
            String::new(),
            &results,
            consensus,
            prepared_run.first()?.prompt.source.as_ref(),
            &checkpointed.chain_dir,
        )?;
        let path = checkpointed.chain_dir.join(CLAIM_SUBMISSION_JSON);
        inference_artifacts::write_json(&path, &submission)?;
        Some(path)
    } else {
        None
    };
    Ok(ClaimBuildResult {
        pipeline: prepared_run.pipeline,
        chain_dir: checkpointed.chain_dir,
        checkpoint_trace_path,
        checkpoint_hashes_path,
        claim_bundle_path,
        claim_submission_path,
        final_result: checkpointed.final_result,
        consensus: checkpointed.consensus,
    })
}

impl ClaimBuildOptions {
    pub fn from_current_dir(
        current_exe: PathBuf,
        staged_backend: StagedExecutionBackend,
        run_spec_path: PathBuf,
    ) -> Result<Self> {
        let base_dir = std::env::current_dir().context("failed to read current directory")?;
        Ok(Self {
            run_spec_path,
            base_dir,
            current_exe,
            staged_backend,
            claimed_token_ids: None,
            direct_output_ids: None,
            claimer: None,
        })
    }
}

pub(crate) fn prepare_claim_run(options: &ClaimBuildOptions) -> Result<PreparedClaimRun> {
    let run_spec_path = absolute(&options.base_dir, &options.run_spec_path);
    let run_spec = read_run_spec(&run_spec_path)?;
    prepare_run(options, run_spec_path, &run_spec, None, None, true)
}

/// Prepares a run on the inputs a claim posted, as a challenger of a posted
/// claim does: every file a stage is given from the run is written from the
/// claim's own value, so each stage runs on the inputs the claimer's did.
///
/// Nothing is tokenized and no output is discovered here. The stages check
/// what they are given as they run: marks that are not the tokenizer's, text
/// that is not the claimed text and ids that are not the greedy output each
/// end in a record that says so, which is another output than a claim that
/// stated a clean one.
pub(crate) fn prepare_posted_run(
    options: &ClaimBuildOptions,
    posted: &inference_artifacts::availability::PostedRun,
) -> Result<PreparedClaimRun> {
    let run_spec_path = absolute(&options.base_dir, &options.run_spec_path);
    let run_spec = read_run_spec(&run_spec_path)?;
    prepare_run(options, run_spec_path, &run_spec, None, Some(posted), false)
}

/// Prepares a run from a spec a claim already fixed, as a challenger does.
///
/// The ids are not held to the run's token budget here. A claim that breaks the
/// budget still has a chain to recompute, and is rejected on its final record.
///
/// `source` is the claim's source record with the source stage's inputs,
/// `forms` the forms its result stage compared the output with, and `ids` the
/// ids each of its inferences claimed, in order. A challenger reruns every
/// stage on what the claim committed to.
pub(crate) fn prepare_challenge_run(
    options: &ClaimBuildOptions,
    run_spec: &InferenceRunSpec,
    source: Option<FrozenSource<'_>>,
    forms: &[run_prep::source::Form],
    ids: &[Vec<u32>],
) -> Result<PreparedClaimRun> {
    let run_spec_path = absolute(&options.base_dir, &options.run_spec_path);
    prepare_run(
        options,
        run_spec_path,
        run_spec,
        Some(Frozen { source, forms, ids }),
        None,
        false,
    )
}

/// A claim's source as a challenger is given it: the record, and the files
/// the source stage ran on.
pub(crate) type FrozenSource<'a> = (
    &'a run_prep::source::SourceRecord,
    &'a inference_artifacts::PreparedSourceInputs,
);

/// What a claim fixed of its run, for a challenger to rebuild it under.
struct Frozen<'a> {
    source: Option<FrozenSource<'a>>,
    forms: &'a [run_prep::source::Form],
    ids: &'a [Vec<u32>],
}

/// One inference of a run being prepared: its model, and the prompt as that
/// model's tokenizer prepares it.
struct RunModel {
    manifest_path: PathBuf,
    manifest: ModelManifest,
    template_path: PathBuf,
    /// The model's part of a chain, as its import wrote it.
    part: String,
    tokenizer: serde_json::Value,
    prompt: inference_artifacts::PreparedPrompt,
}

/// The files written for one inference of a run.
struct WrittenInference {
    label: String,
    files: crate::run_inputs::InferenceRunFiles,
}

/// `given` is the inputs a claim posted, for a challenger of that claim:
/// they are the run's inputs, in place of the ones this would prepare.
fn prepare_run(
    options: &ClaimBuildOptions,
    run_spec_path: PathBuf,
    run_spec: &InferenceRunSpec,
    frozen: Option<Frozen<'_>>,
    given: Option<&inference_artifacts::availability::PostedRun>,
    check_claimed_length: bool,
) -> Result<PreparedClaimRun> {
    use inference_artifacts::availability::{PostedInference, PostedRun, PostedSourceInputs};
    let run_spec_dir = run_spec_path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("run spec has no parent directory"))?;
    let pipeline = run_spec.pipeline;
    // Ids given on the command line are one inference's.
    let given_ids = options
        .claimed_token_ids
        .as_ref()
        .or(options.direct_output_ids.as_ref());
    anyhow::ensure!(
        given_ids.is_none() || pipeline.inferences() == 1,
        "output ids can be given for a pipeline of one inference, and the `{pipeline}` pipeline \
         makes {}",
        pipeline.inferences()
    );
    if let Some(frozen) = &frozen {
        anyhow::ensure!(
            frozen.ids.len() == pipeline.inferences(),
            "the claim holds the ids of {} inference(s), and the `{pipeline}` pipeline makes {}",
            frozen.ids.len(),
            pipeline.inferences()
        );
    }

    // Each inference's model, and the run's prompt as its tokenizer prepares
    // it. The text, the source record and the forms are the same in all.
    let mut models = Vec::new();
    for manifest in run_spec.model_manifests() {
        let manifest_path = absolute(run_spec_dir, manifest);
        let manifest = ModelManifest::load_verified(&manifest_path)?;
        let (template_path, part) = manifest.verified_raster_template(&manifest_path)?;
        let model_manifest_dir = manifest_path
            .parent()
            .ok_or_else(|| anyhow::anyhow!("model manifest has no parent directory"))?;
        let tokenizer_path = absolute(model_manifest_dir, &manifest.bundle.tokenizer_path);
        let tokenizer: serde_json::Value = read_json(&tokenizer_path)?;
        let mut prompt = run_prep::prepare_prompt(&tokenizer, run_spec_dir, run_spec, &manifest)?;
        if let Some(frozen) = &frozen {
            if let Some((record, _)) = frozen.source {
                prompt.source = Some(record.clone());
            }
            prompt.forms = frozen.forms.to_vec();
        }
        models.push(RunModel {
            manifest_path,
            manifest,
            template_path,
            part,
            tokenizer,
            prompt,
        });
    }
    let first = models.first().context("the run names no model")?;

    let run_dir = options
        .base_dir
        .join("target")
        .join("iris")
        .join("runs")
        .join(run_id(
            &run_spec_path,
            &first.prompt.rendered_prompt,
            run_spec.tokens,
        ));
    let prompt_dir = run_dir.join("prompt");
    if let Some(given) = given {
        anyhow::ensure!(
            given.inferences.len() == pipeline.inferences(),
            "the claim posted the inputs of {} inference(s), and the `{pipeline}` pipeline makes {}",
            given.inferences.len(),
            pipeline.inferences()
        );
    }
    // The source stage's inputs: the claim's own for a challenger, or written
    // from the run's source. A run without a source has no such stage. The
    // values are kept beside the files: they are what a claim posts. A
    // challenger given only the files of a claim built here posts nothing.
    let frozen_source = frozen.as_ref().and_then(|frozen| frozen.source);
    let source_values = match (given, run_spec.source.as_ref()) {
        (Some(given), _) => given.source.clone(),
        (None, Some(source)) => {
            let loaded = run_prep::source::load_source(run_spec_dir, source)?;
            Some(PostedSourceInputs {
                signer: loaded.signer.clone(),
                recipe: loaded.recipe.recipe(),
                documents: loaded.documents(),
            })
        }
        (None, None) => None,
    };
    let source_inputs = match (&source_values, frozen_source) {
        (Some(source), _) => Some(crate::run_inputs::write_source(source, &prompt_dir)?),
        (None, Some((_, inputs))) if given.is_none() => Some(inputs.clone()),
        _ => None,
    };
    let postable = given.is_some() || source_values.is_some() == source_inputs.is_some();
    // The imports wrote each model's part of a chain. The pipeline makes the
    // chain's template from them.
    let parts: Vec<&str> = models.iter().map(|model| model.part.as_str()).collect();
    let template = pipeline.chain_template(&parts)?;
    let template_dir = first
        .template_path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("manifest template has no parent directory"))?;
    let form_values = match given {
        Some(given) => given.forms.clone(),
        None => source_core::ResultForms {
            forms: first.prompt.forms.to_vec().into(),
        },
    };
    let forms = run_prep::write_run_input(
        inference_artifacts::availability::RunInput::Forms(&form_values),
        &prompt_dir,
        "result_forms",
    )?;

    let mut written = Vec::new();
    let mut values = Vec::new();
    let mut claimed_ids = Vec::new();
    for (inference, model) in models.iter().enumerate() {
        let label = pipeline.label(inference);
        // A pipeline of one inference keeps its run files where they have
        // always been. Each inference of several has a directory of its own.
        let dir = if label.is_empty() {
            prompt_dir.clone()
        } else {
            prompt_dir.join(label.trim_end_matches('_'))
        };
        if let Some(given) = given {
            // The claim's own values: nothing is prepared.
            let posted = given.inferences[inference].clone();
            claimed_ids.push(posted.claimed.concat());
            written.push(WrittenInference {
                files: crate::run_inputs::write_inference(&posted, &dir)?,
                label,
            });
            values.push(posted);
            continue;
        }
        let ids = match (&frozen, given_ids) {
            (Some(frozen), _) => frozen.ids[inference].clone(),
            (None, Some(ids)) => ids.clone(),
            // Discovery is the only autoregression left: the claim posts what
            // direct decode produced and the chain checks it.
            (None, None) => {
                DirectInferenceExecutor
                    .run(DirectInferenceConfig {
                        base_dir: options.base_dir.clone(),
                        run_spec_path: run_spec_path.clone(),
                        inference,
                    })
                    .context("direct inference to discover the claimed output failed")?
                    .generated_token_ids
            }
        };
        if check_claimed_length {
            run_prep::validate_claimed_length(&ids, run_spec.tokens, &model.prompt.eos_token_ids)?;
        }
        let range_size = model.manifest.range_size;
        let bound = run_prep::bound_ranges(
            &model.prompt.resolved_prompt,
            run_spec.raw_prompt,
            range_size as usize,
            &run_prep::prompt_tokenizer(&model.tokenizer)?,
        )?;
        // A claim's ids are cut as `claimed_ranges` cuts them, which also
        // refuses a claim of no ids.
        run_prep::claimed_ranges(&ids, range_size as usize)?;
        let claimed: Vec<Vec<u32>> = ids
            .chunks(range_size as usize)
            .map(<[u32]>::to_vec)
            .collect();
        // The values a stage is given, which are what the claim posts. Each
        // file is written from its value, so the two cannot differ.
        let posted = PostedInference {
            config: inference_artifacts::availability::RunConfig {
                model_id: model.manifest.model_id.clone(),
                raw_prompt: run_spec.raw_prompt,
                max_new_tokens: run_spec.tokens,
                range_size,
            },
            schedule: inference_artifacts::availability::RunSchedule {
                prompt_ranges: u32::try_from(bound.ranges.len())
                    .context("prompt range count exceeds u32")?,
                claimed_ranges: u32::try_from(claimed.len())
                    .context("claimed range count exceeds u32")?,
            },
            claim: crate::run_inputs::recast(&bound.claim)?,
            text: bound
                .ranges
                .iter()
                .map(|range| crate::run_inputs::recast(&range.text))
                .collect::<Result<_>>()?,
            marks: bound
                .ranges
                .iter()
                .map(|range| range.marks.clone())
                .collect(),
            claimed,
        };
        written.push(WrittenInference {
            files: crate::run_inputs::write_inference(&posted, &dir)?,
            label,
        });
        values.push(posted);
        claimed_ids.push(ids);
    }
    let inferences: Vec<InferenceFiles<'_>> = written
        .iter()
        .map(|written| InferenceFiles {
            label: &written.label,
            text: &written.files.text,
            config: &written.files.config,
            schedule: &written.files.schedule,
            claim: &written.files.claim,
            marks: &written.files.marks,
            claimed: &written.files.claimed,
        })
        .collect();
    let run_manifest = render_chain(
        &template,
        template_dir,
        &inferences,
        source_inputs.as_ref(),
        &forms,
    )?;
    let manifest_path = run_dir.join("Raster.toml");
    fs::create_dir_all(&run_dir)
        .with_context(|| format!("failed to create {}", run_dir.display()))?;
    fs::write(&manifest_path, &run_manifest)
        .with_context(|| format!("failed to write {}", manifest_path.display()))?;
    let run_manifest_sha256 = format!("{:x}", Sha256::digest(run_manifest.as_bytes()));
    let prepared_run = PreparedRun {
        version: inference_artifacts::PREPARED_RUN_VERSION,
        pipeline,
        run_spec_path,
        inferences: models
            .into_iter()
            .map(|model| {
                Ok(inference_artifacts::PreparedInference {
                    model_manifest_sha256: sha256_file(&model.manifest_path)?,
                    model_manifest_path: model.manifest_path,
                    model_id: model.manifest.model_id,
                    prompt: model.prompt,
                })
            })
            .collect::<Result<Vec<_>>>()?,
        raw_prompt: run_spec.raw_prompt,
        tokens: run_spec.tokens,
        run_manifest_path: Some(manifest_path.clone()),
        run_manifest_sha256: Some(run_manifest_sha256),
        source_inputs,
    };
    write_json(&run_dir.join(PREPARED_RUN_JSON), &prepared_run)?;
    // The run's inputs as values, for the claim to post.
    if postable {
        crate::run_inputs::save(
            &PostedRun {
                source: source_values,
                forms: form_values,
                inferences: values,
            },
            &run_dir,
        )?;
    }

    Ok(PreparedClaimRun {
        prepared_run,
        manifest_path,
        claimed_token_ids: claimed_ids.into_iter().next().unwrap_or_default(),
    })
}

const RUN_PLACEHOLDER_DIR: &str = "target/iris/run-prompt-placeholder/";

/// Marks where the template names the last prompt range.
const LAST_PROMPT_RANGE: &str = "@LAST_PROMPT_RANGE@";

/// Marks where the template names the last claimed range.
const LAST_CLAIMED_RANGE: &str = "@LAST_CLAIMED_RANGE@";

/// The chain of a run whose text is taken as given: the template without its
/// source stage, and each inference's first bind given the constant its later
/// binds are.
pub(crate) fn without_source_stage(template: &str) -> Result<String> {
    let lines: Vec<&str> = template.lines().collect();
    let takes_record = |line: &str| line.trim() == "inputs.source = { from = \"source\" }";
    let mut out = Vec::new();
    let (mut dropped, mut rebound) = (0, 0);
    let mut index = 0;
    while index < lines.len() {
        let line = lines[index];
        let is_source_stage = line.trim() == "[[chain.stage]]"
            && lines
                .get(index + 1)
                .is_some_and(|next| next.trim() == "name = \"source\"");
        if is_source_stage {
            index += 1;
            while lines
                .get(index)
                .is_some_and(|next| !next.trim_start().starts_with('['))
            {
                index += 1;
            }
            dropped += 1;
            continue;
        }
        if takes_record(line) {
            // The constant is the one the same inference's later binds are
            // given: theirs is the next bind stage in the template.
            let constant = lines[index..]
                .iter()
                .map(|line| line.trim_start())
                .find(|line| line.starts_with("inputs.source = { external"))
                .context("chain template gives no later bind a constant source")?;
            out.push(constant.to_string());
            rebound += 1;
        } else {
            out.push(line.to_string());
        }
        index += 1;
    }
    anyhow::ensure!(
        dropped == 1 && rebound >= 1,
        "incompatible chain template: expected one source stage and a first bind that takes its record; found {dropped} and {rebound}"
    );
    out.push(String::new());
    Ok(out.join("\n"))
}

/// The run files of one inference, and the label its names carry in the
/// template: empty in a chain of one inference.
pub(crate) struct InferenceFiles<'a> {
    pub(crate) label: &'a str,
    /// One slice of the text per prompt range.
    pub(crate) text: &'a [run_prep::RunFile],
    pub(crate) config: &'a run_prep::RunFile,
    pub(crate) schedule: &'a run_prep::RunFile,
    pub(crate) claim: &'a run_prep::RunFile,
    /// One set of token marks per prompt range.
    pub(crate) marks: &'a [run_prep::RunFile],
    /// One set of claimed ids per claimed range.
    pub(crate) claimed: &'a [run_prep::RunFile],
}

/// The inference a stage, a repeat or a chain input belongs to, and its name
/// without the label. In a chain of one inference every name is that one's.
fn inference_name<'n>(
    name: &'n str,
    inferences: &[InferenceFiles<'_>],
) -> Option<(usize, &'n str)> {
    if let [only] = inferences {
        if only.label.is_empty() {
            return Some((0, name));
        }
    }
    inferences.iter().enumerate().find_map(|(index, files)| {
        name.strip_prefix(files.label)
            .filter(|_| !files.label.is_empty())
            .map(|rest| (index, rest))
    })
}

/// Fills the template of a chain of one inference for one run. See
/// [`render_chain`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn render_ranged_manifest(
    template: &str,
    template_dir: &Path,
    text: &[run_prep::RunFile],
    config: &run_prep::RunFile,
    schedule: &run_prep::RunFile,
    claim: &run_prep::RunFile,
    source: Option<&inference_artifacts::PreparedSourceInputs>,
    forms: &run_prep::RunFile,
    prompt: &[run_prep::RunFile],
    claimed: &[run_prep::RunFile],
) -> Result<String> {
    render_chain(
        template,
        template_dir,
        &[InferenceFiles {
            label: "",
            text,
            config,
            schedule,
            claim,
            marks: prompt,
            claimed,
        }],
        source,
        forms,
    )
}

/// Fills a chain template for one run. For each inference: the configuration,
/// the stage schedule and the text claim, the range counts, the last prompt
/// and claimed ranges' indexes, and each range's committed text, marks or
/// claimed ids. For the run: the source stage's inputs and the result stage's
/// forms. A run without a source gets a chain without the source stage.
pub(crate) fn render_chain(
    template: &str,
    template_dir: &Path,
    inferences: &[InferenceFiles<'_>],
    source: Option<&inference_artifacts::PreparedSourceInputs>,
    forms: &run_prep::RunFile,
) -> Result<String> {
    anyhow::ensure!(!inferences.is_empty(), "a run needs at least one inference");
    for files in inferences {
        anyhow::ensure!(
            !files.marks.is_empty() && !files.claimed.is_empty(),
            "a run needs at least one prompt range and one claimed range"
        );
        anyhow::ensure!(
            files.text.len() == files.marks.len(),
            "a run needs one slice of text for each prompt range"
        );
    }
    let template = inference_artifacts::resolve_raster_manifest_paths(template, template_dir)?;
    let template = match source {
        Some(_) => template,
        None => without_source_stage(&template)?,
    };
    let mut table = String::new();
    let mut name = String::new();
    let mut fan_index = String::new();
    // Where the table being read starts its index. Range 0 of the prompt and
    // of the checks is written out as plain stages, so those repeats and the
    // prompt ranges' layer fans start at 1 and run one time fewer.
    let mut start = 0_usize;
    // For each inference: prompt repeats, claimed repeats, prompt externals,
    // claimed externals, per-layer prompt fans, per-layer claimed fans, text
    // externals, configuration bindings, schedule bindings, text claim
    // bindings.
    let mut replaced = vec![[0_usize; 10]; inferences.len()];
    // For the run: the source stage's inputs, the result stage's forms.
    let (mut source_inputs, mut result_forms) = (0_usize, 0_usize);
    let mut out = Vec::new();
    for line in template.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('[') {
            table = trimmed.to_string();
            name.clear();
            fan_index.clear();
            start = 0;
        }
        if trimmed.starts_with("start =") {
            let value: toml::Value = toml::from_str(trimmed)?;
            start = value["start"]
                .as_integer()
                .and_then(|start| usize::try_from(start).ok())
                .context("invalid repeat start")?;
        }
        if trimmed.starts_with("name =") {
            let value: toml::Value = toml::from_str(trimmed)?;
            name = value["name"]
                .as_str()
                .context("invalid stage/repeat name")?
                .to_string();
        }
        if trimmed.starts_with("index =") {
            let value: toml::Value = toml::from_str(trimmed)?;
            fan_index = value["index"]
                .as_str()
                .context("invalid repeat index")?
                .to_string();
        }
        // The inference the stage or repeat being read belongs to.
        let named = inference_name(&name, inferences);
        // Every bind stage is given its inference's three statements: range
        // 0's, and the template of the ranges after it.
        if let Some((inference, base)) = named {
            if matches!(
                (table.as_str(), base),
                ("[[chain.stage]]", "prompt_bind_p0")
                    | ("[[chain.repeat.stage]]", "prompt_bind_p{j}")
            ) {
                let files = &inferences[inference];
                let binding = if trimmed.starts_with("inputs.config =") {
                    Some(("config", files.config, 7))
                } else if trimmed.starts_with("inputs.schedule =") {
                    Some(("schedule", files.schedule, 8))
                } else if trimmed.starts_with("inputs.claim =") {
                    Some(("claim", files.claim, 9))
                } else {
                    None
                };
                if let Some((param, file, slot)) = binding {
                    replaced[inference][slot] += 1;
                    out.push(
                        RunExternal::new(&file.path, &file.index_path, &file.commitment)
                            .binding(param),
                    );
                    continue;
                }
            }
        }
        if let (("[[chain.stage]]", "source"), Some(source)) =
            ((table.as_str(), name.as_str()), source)
        {
            let binding = [
                ("signer", &source.signer),
                ("recipe", &source.recipe),
                ("documents", &source.documents),
            ]
            .into_iter()
            .find(|(param, _)| trimmed.starts_with(&format!("inputs.{param} =")));
            if let Some((param, file)) = binding {
                source_inputs += 1;
                out.push(
                    RunExternal::new(&file.path, &file.index_path, &file.commitment).binding(param),
                );
                continue;
            }
        }
        if (table.as_str(), name.as_str()) == ("[[chain.stage]]", "result")
            && trimmed.starts_with("inputs.forms =")
        {
            result_forms += 1;
            out.push(
                RunExternal::new(&forms.path, &forms.index_path, &forms.commitment)
                    .binding("forms"),
            );
            continue;
        }
        // A chain input's table is named for the inference it belongs to.
        let input = table
            .strip_prefix("[chain.input.")
            .and_then(|input| input.strip_suffix(']'))
            .and_then(|input| inference_name(input, inferences));
        let files = input.and_then(|(inference, base)| {
            let files = &inferences[inference];
            match base {
                "prompt_text" => Some((inference, files.text, 6)),
                "prompt_marks" => Some((inference, files.marks, 2)),
                "claimed_ids" => Some((inference, files.claimed, 3)),
                _ => None,
            }
        });
        if let Some((inference, files, slot)) = files {
            if let Some(key) = ["index_path", "path"]
                .into_iter()
                .find(|key| trimmed.starts_with(&format!("{key} =")))
            {
                // Keep the indexed file name (`marks_{j}.rastered`), move it
                // from the placeholder directory to this run's.
                let value: toml::Value = toml::from_str(trimmed)?;
                let placeholder = value[key].as_str().context("invalid input path")?;
                anyhow::ensure!(
                    placeholder.contains(RUN_PLACEHOLDER_DIR),
                    "input path {placeholder} is not a run placeholder"
                );
                let file_name = Path::new(placeholder)
                    .file_name()
                    .context("input path has no file name")?;
                let dir = files[0]
                    .path
                    .parent()
                    .context("run file has no directory")?;
                out.push(format!("{key} = {:?}", dir.join(file_name)));
                continue;
            }
            if trimmed.starts_with("commitments =") {
                replaced[inference][slot] += 1;
                out.push(String::from("commitments = ["));
                out.extend(
                    files
                        .iter()
                        .enumerate()
                        .map(|(index, file)| format!("  {:?}, # {index}", file.commitment)),
                );
                out.push(String::from("]"));
                continue;
            }
        }
        if table == "[[chain.repeat]]" && trimmed.starts_with("count =") {
            let (inference, base) =
                named.with_context(|| format!("unexpected repeat `{name}` in chain template"))?;
            let files = &inferences[inference];
            let range_count = match base {
                "prompt" => Some((0, files.marks.len())),
                "claimed_embed" | "claimed" => Some((1, files.claimed.len())),
                // Layer blocks keep their model-fixed counts.
                "layers" | "shared_layers" => None,
                _ => anyhow::bail!("unexpected repeat `{name}` in chain template"),
            };
            if let Some((slot, count)) = range_count {
                replaced[inference][slot] += 1;
                out.push(format!("count = {}", iterations(count, start)?));
                continue;
            }
        }
        // A layer block's inner fans run once per prompt or claimed range.
        if table == "[[chain.repeat.stage]]" && trimmed.starts_with("count =") {
            if let Some((inference, _)) = named {
                let files = &inferences[inference];
                let range_count = match fan_index.as_str() {
                    "j" => Some((4, files.marks.len())),
                    "k" => Some((5, files.claimed.len())),
                    _ => None,
                };
                if let Some((slot, count)) = range_count {
                    replaced[inference][slot] += 1;
                    out.push(format!("count = {}", iterations(count, start)?));
                    continue;
                }
            }
        }
        // Where the template names an inference's last prompt or claimed
        // range, by that inference's label.
        let mut line = line.to_string();
        for files in inferences {
            let label = files.label;
            line = line
                .replace(
                    &format!("@{label}{}@", LAST_PROMPT_RANGE.trim_matches('@')),
                    &(files.marks.len() - 1).to_string(),
                )
                .replace(
                    &format!("@{label}{}@", LAST_CLAIMED_RANGE.trim_matches('@')),
                    &(files.claimed.len() - 1).to_string(),
                );
        }
        out.push(line);
    }
    let filled = replaced.iter().all(|replaced| {
        replaced[..4] == [1, 2, 1, 1]
            && replaced[4] != 0
            && replaced[4] == replaced[5]
            && replaced[6..10] == [1, 2, 2, 2]
    });
    if !filled || source_inputs != if source.is_some() { 3 } else { 0 } || result_forms != 1 {
        anyhow::bail!("incompatible chain template: expected of each inference [prompt repeats, claimed repeats, prompt commitments, claimed commitments] = [1, 2, 1, 1], matching per-layer range fans, one set of text commitments, and a configuration, a schedule and a text claim binding for range 0's bind and for the later ranges'; and of the run three inputs for the source stage when it has a source, and the result stage's forms; found {replaced:?}, {source_inputs} and {result_forms}");
    }
    out.push(String::new());
    Ok(out.join("\n"))
}

/// How many times a repeat over `ranges` ranges runs when its index starts
/// at `start`.
fn iterations(ranges: usize, start: usize) -> Result<usize> {
    ranges
        .checked_sub(start)
        .with_context(|| format!("a repeat starts at range {start} of {ranges}"))
}

/// One run-specific external: the files and commitment a template
/// placeholder is replaced with.
pub(crate) struct RunExternal<'a> {
    path: &'a Path,
    index_path: &'a Path,
    commitment: &'a str,
}

impl<'a> RunExternal<'a> {
    pub(crate) fn new(path: &'a Path, index_path: &'a Path, commitment: &'a str) -> Self {
        Self {
            path,
            index_path,
            commitment,
        }
    }

    pub(crate) fn binding(&self, param: &str) -> String {
        format!(
            "inputs.{param} = {{ external = {{ path = {:?}, index_path = {:?}, commitment = {:?} }} }}",
            self.path, self.index_path, self.commitment,
        )
    }
}

fn absolute(base: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        base.join(path)
    }
}

pub(crate) fn select_checkpoint_index(
    selection: &CorruptCheckpointSelection,
    hashes: &[String],
    trace: Option<&inference_artifacts::CheckpointTrace>,
) -> Result<usize> {
    match selection {
        CorruptCheckpointSelection::Index(index) => {
            if *index >= hashes.len() {
                anyhow::bail!(
                    "checkpoint index {index} is out of range for {} hashes",
                    hashes.len()
                );
            }
            Ok(*index)
        }
        CorruptCheckpointSelection::Stage(stage) => {
            let trace = trace.ok_or_else(|| {
                anyhow::anyhow!(
                    "stage selection requires a checkpoint trace; pass --trace or keep checkpoint_trace.json beside checkpoints.txt"
                )
            })?;
            let index = trace
                .checkpoints
                .iter()
                .position(|checkpoint| checkpoint.stage == *stage)
                .ok_or_else(|| {
                    anyhow::anyhow!("stage `{stage}` was not found in checkpoint trace")
                })?;
            if index >= hashes.len() {
                anyhow::bail!(
                    "stage `{stage}` is checkpoint {index}, but the hash file only has {} hashes",
                    hashes.len()
                );
            }
            Ok(index)
        }
        CorruptCheckpointSelection::Random { seed } => Ok(random_checkpoint_index(seed, hashes)),
    }
}

fn random_checkpoint_index(seed: &Option<String>, hashes: &[String]) -> usize {
    let seed_bytes = match seed {
        Some(seed) => seed.as_bytes().to_vec(),
        None => SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
            .to_string()
            .into_bytes(),
    };
    let digest = Sha256::digest(seed_bytes);
    let mut bytes = [0_u8; 8];
    bytes.copy_from_slice(&digest[..8]);
    (u64::from_be_bytes(bytes) as usize) % hashes.len()
}

pub(crate) fn corrupt_hash(hash: &str) -> String {
    let mut bytes = hash.as_bytes().to_vec();
    bytes[0] = if bytes[0] == b'0' { b'1' } else { b'0' };
    String::from_utf8(bytes).expect("checkpoints are valid hex")
}

fn resolve_trace_path(hashes_path: &Path, trace_path: Option<&Path>) -> PathBuf {
    trace_path
        .map(Path::to_path_buf)
        .or_else(|| {
            hashes_path
                .parent()
                .map(|parent| parent.join(CHECKPOINT_TRACE_JSON))
        })
        .unwrap_or_else(|| PathBuf::from(CHECKPOINT_TRACE_JSON))
}

fn default_corrupted_hashes_path(source: &Path) -> PathBuf {
    source.with_file_name("checkpoints.corrupt.txt")
}

fn run_id(run_spec_path: &Path, rendered_prompt: &str, tokens: u32) -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let digest = Sha256::digest(format!(
        "{}\0{}\0{}\0{}",
        run_spec_path.display(),
        rendered_prompt,
        tokens,
        nanos
    ));
    hex::encode(digest)[..16].to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use inference_artifacts::{CHECKPOINT_TRACE_JSON, CLAIM_BUNDLE_JSON};
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn options_from_current_dir_use_run_spec() {
        let saved = std::env::current_dir().unwrap();
        let base = temp_dir("claim-options");
        fs::create_dir_all(&base).unwrap();
        let expected_base = base.canonicalize().unwrap();
        std::env::set_current_dir(&base).unwrap();

        let options = ClaimBuildOptions::from_current_dir(
            PathBuf::from("iris"),
            StagedExecutionBackend::InProcess,
            PathBuf::from("inference.toml"),
        )
        .unwrap();

        std::env::set_current_dir(saved).unwrap();
        fs::remove_dir_all(&base).unwrap();

        assert_eq!(options.base_dir, expected_base);
        assert_eq!(options.run_spec_path, PathBuf::from("inference.toml"));
        assert_eq!(options.current_exe, PathBuf::from("iris"));
        assert_eq!(options.staged_backend, StagedExecutionBackend::InProcess);
    }

    #[test]
    fn preparation_uses_selected_model_and_rejects_stale_identity() {
        use crate::test_support::{model_fixture, run_spec};
        let base = temp_dir("model-selection");
        model_fixture(&base, "model-a");
        let model_path = model_fixture(&base, "model-b");
        fs::write(base.join("Raster.toml"), "stale model-a template").unwrap();
        let options = ClaimBuildOptions {
            base_dir: base.clone(),
            run_spec_path: run_spec(&base, "model-b"),
            current_exe: PathBuf::from("unused"),
            staged_backend: StagedExecutionBackend::InProcess,
            claimed_token_ids: Some(vec![0, 0, 0]),
            direct_output_ids: None,
            claimer: None,
        };
        let prepared = prepare_claim_run(&options).unwrap();
        let rendered = fs::read_to_string(&prepared.manifest_path).unwrap();
        assert!(rendered.contains("name = \"model-b\""));
        assert!(!rendered.contains("model-a"));
        assert_eq!(prepared.claimed_token_ids, [0, 0, 0]);
        // A prior direct run's output is taken without running direct again
        // (this fixture has no weights to run), and is held to the run's length.
        let given = |ids: &[u32]| ClaimBuildOptions {
            claimed_token_ids: None,
            direct_output_ids: Some(ids.to_vec()),
            claimer: None,
            ..options.clone()
        };
        assert_eq!(
            prepare_claim_run(&given(&[0, 0, 0]))
                .unwrap()
                .claimed_token_ids,
            [0, 0, 0]
        );
        assert!(prepare_claim_run(&given(&[0])).is_err());
        assert_eq!(
            fs::canonicalize(&prepared.prepared_run.inferences[0].model_manifest_path).unwrap(),
            fs::canonicalize(&model_path).unwrap()
        );

        for (file, label) in [
            ("model.detwgt", "model weights"),
            ("config.json", "model config"),
            ("tokenizer.json", "model tokenizer"),
            ("Raster.toml", "Raster template"),
        ] {
            let path = model_path.parent().unwrap().join(file);
            let original = fs::read(&path).unwrap();
            fs::write(&path, "replaced").unwrap();
            let error = prepare_claim_run(&options).err().unwrap();
            assert!(
                error
                    .to_string()
                    .contains(&format!("{label} hash mismatch")),
                "{error:#}"
            );
            fs::write(path, original).unwrap();
        }

        let mut model: ModelManifest = read_json(&model_path).unwrap();
        model.provenance = None;
        write_json(&model_path, &model).unwrap();
        let error = prepare_claim_run(&options).err().unwrap();
        assert!(error.to_string().contains("no Raster template provenance"));
        model.version = 99;
        write_json(&model_path, &model).unwrap();
        assert!(prepare_claim_run(&options)
            .err()
            .unwrap()
            .to_string()
            .contains("unsupported model manifest"));
        fs::remove_dir_all(base).unwrap();
    }

    #[test]
    fn claim_result_writes_existing_artifact_names() {
        let base = temp_dir("claim-result");
        fs::create_dir_all(&base).unwrap();
        write_stage(&base, "check_c0", "abc", b"output");
        let manifest_path = base.join("Raster.toml");
        fs::write(&manifest_path, "[chain]\nname = \"test\"\n").unwrap();

        let result = write_claim_result(
            CheckpointedInferenceResult {
                chain_dir: base.clone(),
                selected_stage_dir: None,
                final_result: None,
                consensus: None,
            },
            &manifest_path,
            &PreparedRun {
                version: inference_artifacts::PREPARED_RUN_VERSION,
                pipeline: inference_artifacts::Pipeline::Single,
                run_spec_path: base.join("inference.toml"),
                inferences: vec![inference_artifacts::PreparedInference {
                    model_manifest_path: base.join("model-artifacts/manifest.json"),
                    model_manifest_sha256: String::from("model-sha"),
                    model_id: String::from("model-id"),
                    prompt: inference_artifacts::PreparedPrompt {
                        resolved_prompt: String::from("hello"),
                        rendered_prompt: String::from("hello"),
                        initial_pieces: vec![inference_artifacts::PreparedPiece {
                            text: String::from("hello"),
                            segment: 0,
                        }],
                        eos_token_ids: Vec::new(),
                        source: None,
                        forms: Vec::new(),
                    },
                }],
                raw_prompt: false,
                tokens: 1,
                run_manifest_path: Some(manifest_path.clone()),
                run_manifest_sha256: Some(String::from("run-sha")),
                source_inputs: None,
            },
        )
        .unwrap();

        assert_eq!(result.chain_dir, base);
        assert_eq!(
            result.checkpoint_trace_path.file_name().unwrap(),
            CHECKPOINT_TRACE_JSON
        );
        assert_eq!(
            result.checkpoint_hashes_path.file_name().unwrap(),
            inference_artifacts::CHECKPOINT_HASHES_TXT
        );
        assert_eq!(
            result.claim_bundle_path.file_name().unwrap(),
            CLAIM_BUNDLE_JSON
        );
        assert_eq!(result.final_result, None);

        fs::remove_dir_all(result.chain_dir).unwrap();
    }

    #[test]
    fn corrupt_checkpoint_hashes_changes_selected_index() {
        let base = temp_dir("corrupt-index");
        fs::create_dir_all(&base).unwrap();
        let hashes_path = base.join("checkpoints.txt");
        let output_path = base.join("corrupted.txt");
        let hashes = vec![
            String::from("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"),
            String::from("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"),
        ];
        inference_artifacts::write_checkpoint_hashes(&hashes_path, &hashes).unwrap();

        let result = corrupt_checkpoint_hashes(CorruptCheckpointHashesOptions {
            hashes_path: hashes_path.clone(),
            trace_path: None,
            output_path: Some(output_path.clone()),
            selection: CorruptCheckpointSelection::Index(1),
        })
        .unwrap();

        let corrupted = inference_artifacts::read_checkpoint_hashes(&output_path).unwrap();
        assert_eq!(result.checkpoint_index, 1);
        assert_eq!(result.checkpoint_number, 2);
        assert_eq!(corrupted[0], hashes[0]);
        assert_ne!(corrupted[1], hashes[1]);
        assert_eq!(result.original_hash, hashes[1]);
        assert_eq!(result.corrupted_hash, corrupted[1]);
        assert!(result.corruption_manifest_path.is_file());

        fs::remove_dir_all(base).unwrap();
    }

    #[test]
    fn corrupt_checkpoint_hashes_can_select_stage_from_trace() {
        let base = temp_dir("corrupt-stage");
        fs::create_dir_all(&base).unwrap();
        let hashes_path = base.join("checkpoints.txt");
        let trace_path = base.join(CHECKPOINT_TRACE_JSON);
        let trace = inference_artifacts::CheckpointTrace {
            checkpoints: vec![
                inference_artifacts::Checkpoint {
                    stage: String::from("stage_a"),
                    program_commitment: String::from("program-a"),
                    input_sources: Default::default(),
                    input_commitment: String::from("input-a"),
                    output_commitment: String::from("output-a"),
                    output_sha256: String::from("sha-a"),
                },
                inference_artifacts::Checkpoint {
                    stage: String::from("stage_b"),
                    program_commitment: String::from("program-b"),
                    input_sources: Default::default(),
                    input_commitment: String::from("input-b"),
                    output_commitment: String::from("output-b"),
                    output_sha256: String::from("sha-b"),
                },
            ],
        };
        inference_artifacts::write_json(&trace_path, &trace).unwrap();
        let hashes = inference_artifacts::checkpoint_hashes(&trace).unwrap();
        inference_artifacts::write_checkpoint_hashes(&hashes_path, &hashes).unwrap();

        let result = corrupt_checkpoint_hashes(CorruptCheckpointHashesOptions {
            hashes_path: hashes_path.clone(),
            trace_path: None,
            output_path: Some(base.join("corrupted.txt")),
            selection: CorruptCheckpointSelection::Stage(String::from("stage_b")),
        })
        .unwrap();

        assert_eq!(result.checkpoint_index, 1);
        assert_eq!(result.checkpoint_number, 2);
        assert_eq!(result.stage, Some(String::from("stage_b")));

        fs::remove_dir_all(base).unwrap();
    }

    #[test]
    fn a_consensus_run_is_one_chain_of_two_inferences() {
        let base = temp_dir("claim-consensus");
        crate::test_support::model_fixture(&base, "alpha");
        crate::test_support::model_fixture(&base, "beta");
        let dir = base.join("specs");
        fs::create_dir_all(&dir).unwrap();
        let spec = dir.join("inference.toml");
        fs::write(
            &spec,
            "pipeline = \"consensus\"\nprompt = \"h\"\nraw_prompt = true\ntokens = 3\n\
             [[model]]\nmanifest = \"../models/alpha/manifest.json\"\n\
             [[model]]\nmanifest = \"../models/beta/manifest.json\"\n",
        )
        .unwrap();
        let options = |ids: Option<Vec<u32>>| ClaimBuildOptions {
            base_dir: base.clone(),
            run_spec_path: spec.clone(),
            current_exe: PathBuf::from("unused"),
            staged_backend: StagedExecutionBackend::InProcess,
            claimed_token_ids: ids,
            direct_output_ids: None,
            claimer: None,
        };
        let run_spec = read_run_spec(&spec).unwrap();
        // The fixture's ranges hold four tokens: the first inference claims
        // one range and the second two.
        let ids = [vec![0, 0, 0], vec![0, 0, 0, 0, 0]];
        let prepared = prepare_challenge_run(&options(None), &run_spec, None, &[], &ids).unwrap();

        let run = &prepared.prepared_run;
        assert_eq!(run.pipeline, inference_artifacts::Pipeline::Consensus);
        assert_eq!(run.inferences.len(), 2);
        assert_ne!(run.inferences[0].model_id, run.inferences[1].model_id);
        assert_eq!(
            run.inferences[0].prompt.resolved_prompt,
            run.inferences[1].prompt.resolved_prompt
        );
        assert_eq!(prepared.claimed_token_ids, ids[0]);

        let manifest: toml::Value =
            toml::from_str(&fs::read_to_string(&prepared.manifest_path).unwrap()).unwrap();
        let chain = &manifest["chain"];
        // Each inference has its own run files, in a directory of its own.
        for (label, claimed_ranges) in [("m0", 1), ("m1", 2)] {
            let claimed = &chain["input"][format!("{label}_claimed_ids").as_str()];
            assert_eq!(
                claimed["commitments"].as_array().unwrap().len(),
                claimed_ranges
            );
            let path = claimed["path"].as_str().unwrap();
            assert!(
                path.ends_with(&format!("prompt/{label}/claimed_{{k}}.rastered")),
                "{path}"
            );
            for input in ["prompt_text", "prompt_marks"] {
                let input = &chain["input"][format!("{label}_{input}").as_str()];
                assert_eq!(input["commitments"].as_array().unwrap().len(), 1);
            }
        }
        let stages = chain["stage"].as_array().unwrap();
        let stage = |name: &str| {
            stages
                .iter()
                .find(|stage| stage["name"].as_str() == Some(name))
                .unwrap_or_else(|| panic!("the chain has no stage `{name}`"))
        };
        // A text taken as given: no source stage, and each first bind is given
        // the constant of its own model.
        assert!(stages
            .iter()
            .all(|stage| stage["name"].as_str() != Some("source")));
        let config = |label: &str| {
            stage(&format!("{label}_prompt_bind_p0"))["inputs"]["config"]["external"]["commitment"]
                .as_str()
                .unwrap()
                .to_string()
        };
        for label in ["m0", "m1"] {
            let bind = stage(&format!("{label}_prompt_bind_p0"));
            assert_eq!(
                bind["inputs"]["source"]["external"]["commitment"].as_str(),
                Some("n0")
            );
            assert_eq!(
                bind["inputs"]["text"]["input"].as_str(),
                Some(format!("{label}_prompt_text_0").as_str())
            );
        }
        // Each bind is given its own model's configuration.
        assert_ne!(config("m0"), config("m1"));
        // The result stage reads each inference's last check.
        let result = stage("result");
        assert!(result["project"]
            .as_str()
            .unwrap()
            .ends_with("raster-stages/consensus-result"));
        assert_eq!(
            result["inputs"]["first"]["from"].as_str(),
            Some("m0_check_c0")
        );
        assert_eq!(
            result["inputs"]["second"]["from"].as_str(),
            Some("m1_check_c1")
        );
        assert!(result["inputs"]["forms"]["external"]["commitment"].is_str());
        // The repeat of the checks after the first runs for the second
        // inference only.
        let repeats = chain["repeat"].as_array().unwrap();
        let count = |name: &str| {
            repeats
                .iter()
                .find(|repeat| repeat["name"].as_str() == Some(name))
                .and_then(|repeat| repeat["count"].as_integer())
        };
        assert_eq!(count("m0_claimed"), Some(0));
        assert_eq!(count("m1_claimed"), Some(1));
        assert!(!fs::read_to_string(&prepared.manifest_path)
            .unwrap()
            .contains('@'));

        // Ids given on the command line are one inference's, and a claim
        // holds the ids of each.
        let error = prepare_challenge_run(&options(Some(vec![0])), &run_spec, None, &[], &ids)
            .err()
            .unwrap();
        assert!(format!("{error:#}").contains("one inference"), "{error:#}");
        assert!(prepare_challenge_run(&options(None), &run_spec, None, &[], &ids[..1]).is_err());
        fs::remove_dir_all(base).unwrap();
    }

    #[test]
    fn ranged_manifest_fills_counts_inputs_and_last_prompt_range() {
        let template = r#"[chain]
name = "iris"

[chain.input.prompt_marks]
index = "j"
path = "/repo/target/iris/run-prompt-placeholder/marks_{j}.rastered"
index_path = "/repo/target/iris/run-prompt-placeholder/marks_{j}.rindex"
commitments = []

[chain.input.prompt_text]
index = "j"
path = "/repo/target/iris/run-prompt-placeholder/text_{j}.rastered"
index_path = "/repo/target/iris/run-prompt-placeholder/text_{j}.rindex"
commitments = []

[chain.input.claimed_ids]
index = "k"
path = "/repo/target/iris/run-prompt-placeholder/claimed_{k}.rastered"
index_path = "/repo/target/iris/run-prompt-placeholder/claimed_{k}.rindex"
commitments = []

[[chain.stage]]
name = "source"
inputs.signer = { external = { path = "/repo/target/iris/run-prompt-placeholder/source_signer.rastered", index_path = "/repo/target/iris/run-prompt-placeholder/source_signer.rindex", commitment = "0" } }
inputs.recipe = { external = { path = "/repo/target/iris/run-prompt-placeholder/source_recipe.rastered", index_path = "/repo/target/iris/run-prompt-placeholder/source_recipe.rindex", commitment = "0" } }
inputs.documents = { external = { path = "/repo/target/iris/run-prompt-placeholder/source_documents.rastered", index_path = "/repo/target/iris/run-prompt-placeholder/source_documents.rindex", commitment = "0" } }

[[chain.stage]]
name = "prompt_bind_p0"
inputs.text = { input = "prompt_text_0" }
inputs.config = { external = { path = "/repo/target/iris/run-prompt-placeholder/config.rastered", index_path = "/repo/target/iris/run-prompt-placeholder/config.rindex", commitment = "0" } }
inputs.schedule = { external = { path = "/repo/target/iris/run-prompt-placeholder/schedule.rastered", index_path = "/repo/target/iris/run-prompt-placeholder/schedule.rindex", commitment = "0" } }
inputs.claim = { external = { path = "/repo/target/iris/run-prompt-placeholder/claim.rastered", index_path = "/repo/target/iris/run-prompt-placeholder/claim.rindex", commitment = "0" } }
inputs.source = { from = "source" }
inputs.previous = { external = { path = "/repo/model/empty_prompt.rastered", index_path = "/repo/model/empty_prompt.rindex", commitment = "e0" } }

[[chain.stage]]
name = "prompt_check_p0"
inputs.bound = { from = "prompt_bind_p0" }
inputs.previous = { external = { path = "/repo/model/empty_prompt.rastered", index_path = "/repo/model/empty_prompt.rindex", commitment = "e0" } }
inputs.marks = { input = "prompt_marks_0" }

[[chain.repeat]]
name = "prompt"
index = "j"
start = 1
count = 0

  [[chain.repeat.stage]]
name = "prompt_bind_p{j}"
inputs.text = { input = "prompt_text_{j}" }
inputs.config = { external = { path = "/repo/target/iris/run-prompt-placeholder/config.rastered", index_path = "/repo/target/iris/run-prompt-placeholder/config.rindex", commitment = "0" } }
inputs.schedule = { external = { path = "/repo/target/iris/run-prompt-placeholder/schedule.rastered", index_path = "/repo/target/iris/run-prompt-placeholder/schedule.rindex", commitment = "0" } }
inputs.claim = { external = { path = "/repo/target/iris/run-prompt-placeholder/claim.rastered", index_path = "/repo/target/iris/run-prompt-placeholder/claim.rindex", commitment = "0" } }
inputs.source = { external = { path = "/repo/model/prompt-bind/no_source.rastered", index_path = "/repo/model/prompt-bind/no_source.rindex", commitment = "n0" } }
inputs.previous = { from = "prompt_bind_p{j-1}", first = "prompt_bind_p0" }

  [[chain.repeat.stage]]
name = "prompt_check_p{j}"
inputs.bound = { from = "prompt_bind_p{j}" }
inputs.previous = { from = "prompt_check_p{j-1}", first = "prompt_check_p0" }
inputs.marks = { input = "prompt_marks_{j}" }

[[chain.repeat]]
name = "claimed_embed"
index = "k"
count = 0

  [[chain.repeat.stage]]
name = "embed_c{k}"
inputs.previous = { from = "embed_c{k-1}", first = "embed_p@LAST_PROMPT_RANGE@" }

[[chain.repeat]]
name = "layers"
index = "l"
start = 0
count = 4

  [[chain.repeat.stage]]
name = "aux_p{j}_l{l}"
index = "j"
count = 0

  [[chain.repeat.stage]]
name = "aux_c{k}_l{l}"
index = "k"
count = 0

  [[chain.repeat.stage]]
name = "range_p0_l{l}"

  [[chain.repeat.stage]]
name = "range_p{j}_l{l}"
index = "j"
start = 1
count = 0

  [[chain.repeat.stage]]
name = "range_c{k}_l{l}"
index = "k"
count = 0

[[chain.stage]]
name = "check_c0"

[[chain.repeat]]
name = "claimed"
index = "k"
start = 1
count = 0

[[chain.stage]]
name = "result"
inputs.check = { from = "check_c@LAST_CLAIMED_RANGE@" }
inputs.forms = { external = { path = "/repo/target/iris/run-prompt-placeholder/result_forms.rastered", index_path = "/repo/target/iris/run-prompt-placeholder/result_forms.rindex", commitment = "0" } }
"#;
        let file = |name: &str, commitment: &str| run_prep::RunFile {
            path: PathBuf::from(format!("/tmp/run/prompt/{name}.rastered")),
            index_path: PathBuf::from(format!("/tmp/run/prompt/{name}.rindex")),
            commitment: commitment.into(),
        };
        let text = [
            file("text_0", "t0"),
            file("text_1", "t1"),
            file("text_2", "t2"),
        ];
        let config = file("config", "g0");
        let schedule = file("schedule", "s0");
        let claim = file("claim", "m0");
        let forms = file("result_forms", "f0");
        let prompt = [
            file("marks_0", "a0"),
            file("marks_1", "a1"),
            file("marks_2", "a2"),
        ];
        let claimed = [file("claimed_0", "c0")];
        let render = |template: &str, claimed: &[run_prep::RunFile]| {
            render_ranged_manifest(
                template,
                Path::new("/repo"),
                &text,
                &config,
                &schedule,
                &claim,
                None,
                &forms,
                &prompt,
                claimed,
            )
        };
        let manifest = render(template, &claimed).unwrap();
        let parsed: toml::Value = toml::from_str(&manifest).unwrap();
        let inputs = &parsed["chain"]["input"];
        assert_eq!(
            inputs["prompt_marks"]["path"].as_str(),
            Some("/tmp/run/prompt/marks_{j}.rastered")
        );
        assert_eq!(
            inputs["claimed_ids"]["index_path"].as_str(),
            Some("/tmp/run/prompt/claimed_{k}.rindex")
        );
        assert_eq!(
            inputs["prompt_marks"]["commitments"],
            toml::Value::try_from(["a0", "a1", "a2"]).unwrap()
        );
        assert_eq!(
            inputs["claimed_ids"]["commitments"],
            toml::Value::try_from(["c0"]).unwrap()
        );
        let repeats = parsed["chain"]["repeat"].as_array().unwrap();
        let count = |value: &toml::Value| value.get("count").and_then(toml::Value::as_integer);
        // Range 0 of the prompt is plain stages, so the repeat runs twice.
        assert_eq!(count(&repeats[0]), Some(2));
        assert_eq!(count(&repeats[1]), Some(1));
        // The layer block keeps its count; its fans run once per range,
        // and the prompt ranges' layer fan once per range after the first.
        assert_eq!(count(&repeats[2]), Some(4));
        let fans = repeats[2]["stage"].as_array().unwrap();
        assert_eq!(count(&fans[0]), Some(3));
        assert_eq!(count(&fans[1]), Some(1));
        assert_eq!(count(&fans[2]), None);
        assert_eq!(count(&fans[3]), Some(2));
        assert_eq!(count(&fans[4]), Some(1));
        // One claimed range: its check is the plain stage and the repeat
        // of the checks after it does not run.
        assert_eq!(count(&repeats[3]), Some(0));
        // Both bind entries are given the run's statements.
        for commitment in ["g0", "s0", "m0"] {
            assert_eq!(
                manifest
                    .matches(&format!(r#"commitment = "{commitment}""#))
                    .count(),
                2
            );
        }
        assert!(manifest.contains(r#"first = "embed_p2""#));
        assert_eq!(
            inputs["prompt_text"]["path"].as_str(),
            Some("/tmp/run/prompt/text_{j}.rastered")
        );
        assert_eq!(
            inputs["prompt_text"]["commitments"],
            toml::Value::try_from(["t0", "t1", "t2"]).unwrap()
        );
        assert!(manifest.contains(r#"commitment = "g0""#));
        assert!(manifest.contains(r#"commitment = "s0""#));
        assert!(manifest.contains(r#"commitment = "m0""#));
        // The result stage reads the last check and the run's forms.
        assert!(manifest.contains(r#"inputs.check = { from = "check_c0" }"#));
        assert!(manifest.contains(r#"commitment = "f0""#));
        // A run without a source: no source stage, and the first bind is given
        // the constant.
        assert!(!manifest.contains(r#"name = "source""#));
        assert!(!manifest.contains(r#"from = "source""#));
        assert_eq!(manifest.matches(r#"commitment = "n0""#).count(), 2);

        let run_file = |name: &str, commitment: &str| inference_artifacts::RunFileRef {
            path: PathBuf::from(format!("/tmp/run/prompt/{name}.rastered")),
            index_path: PathBuf::from(format!("/tmp/run/prompt/{name}.rindex")),
            commitment: commitment.into(),
        };
        let source = inference_artifacts::PreparedSourceInputs {
            signer: run_file("source_signer", "k0"),
            recipe: run_file("source_recipe", "r0"),
            documents: run_file("source_documents", "d0"),
        };
        let sourced = render_ranged_manifest(
            template,
            Path::new("/repo"),
            &text,
            &config,
            &schedule,
            &claim,
            Some(&source),
            &forms,
            &prompt,
            &claimed,
        )
        .unwrap();
        assert!(sourced.contains(r#"name = "source""#));
        assert!(sourced.contains(r#"inputs.source = { from = "source" }"#));
        for commitment in ["k0", "r0", "d0"] {
            assert!(sourced.contains(&format!(r#"commitment = "{commitment}""#)));
        }
        assert_eq!(sourced.matches(r#"commitment = "n0""#).count(), 1);
        // Each range needs its own slice of the text.
        assert!(render_ranged_manifest(
            template,
            Path::new("/repo"),
            &text[..2],
            &config,
            &schedule,
            &claim,
            None,
            &forms,
            &prompt,
            &claimed,
        )
        .is_err());
        assert!(!manifest.contains("run-prompt-placeholder"));
        assert!(render(template, &[]).is_err());
        // A template for the removed cut-and-merge stages is not this chain's.
        let cuts = template.replace("prompt_marks", "prompt_cuts");
        assert!(render(&cuts, &claimed).is_err());
    }

    fn write_stage(base: &PathBuf, stage: &str, commitment: &str, output: &[u8]) {
        let stage_dir = base.join(stage);
        fs::create_dir_all(&stage_dir).unwrap();
        fs::write(stage_dir.join("input_manifest.json"), b"input-manifest").unwrap();
        fs::write(stage_dir.join("output.bin"), output).unwrap();
        fs::write(stage_dir.join("output.rindex"), b"index").unwrap();
        fs::write(
            stage_dir.join("output_manifest.json"),
            format!(
                r#"{{"output":{{"type":"sha256","encoding":"raster","commitment":"{commitment}"}}}}"#
            ),
        )
        .unwrap();
    }

    fn temp_dir(label: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("iris-cli-{label}-{}-{nanos}", std::process::id()))
    }
}
