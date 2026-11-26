use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use procedure_pipeline::pipeline_runner::NativeStageBackend;
use procedure_pipeline::{
    CheckpointedInferenceConfig, CheckpointedInferenceExecutor, CheckpointedInferenceResult,
    ParityPolicy,
};

use blockchain_rules::checks::{ClaimSubmission, CLAIM_SUBMISSION_JSON};
use blockchain_rules::write_claim_artifacts_with_prepared_run;
use procedure_definition::{
    read_case, read_checkpoint_hashes, read_checkpoint_list, read_specification,
    write_checkpoint_hashes, write_json, Case, InferenceResult, PreparedRun, Specification,
    CHECKPOINT_LIST_JSON, PREPARED_RUN_JSON,
};
use procedure_prepare::pipeline_run::Frozen;
pub use procedure_prepare::pipeline_run::{
    render_pipeline, render_ranged_manifest, without_source_stage, FrozenSource, InferenceFiles,
    PreparedPipelineRun, RunExternal,
};
use serde::Serialize;
use sha2::{Digest, Sha256};

#[derive(Debug, Clone)]
pub struct ClaimBuildOptions {
    pub base_dir: PathBuf,
    pub specification_path: PathBuf,
    /// The case the claim is about. None for a challenger rebuilding a
    /// claimer's prepared run, whose case is the prompt that run holds.
    pub case_path: Option<PathBuf>,
    pub current_exe: PathBuf,
    pub native_stage_backend: NativeStageBackend,
    /// Debug-only: post these output ids instead of discovering them with native
    /// inference. Used to build claims whose tokens are not the greedy
    /// continuation.
    pub claimed_token_ids: Option<Vec<u32>>,
    /// The output a native run of this same specification and case already produced. Taken in
    /// place of running the native run again, and held to every rule a
    /// discovered output is: the pipeline re-derives each token, and a claim it
    /// rejects is not written.
    pub native_output_ids: Option<Vec<u32>>,
    /// The address the claim is attributed to, `0x` and 40 hex digits. With
    /// none, the claim names the zero address.
    pub claimer: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClaimBuildResult {
    /// The procedure kind the claim is of.
    pub procedure_kind: procedure_definition::ProcedureKind,
    pub pipeline_dir: PathBuf,
    pub checkpoint_list_path: PathBuf,
    pub checkpoint_hashes_path: PathBuf,
    pub claim_bundle_path: PathBuf,
    /// What a claimer would submit to a contract. Absent when the pipeline
    /// published no final record to submit.
    pub claim_submission_path: Option<PathBuf>,
    /// Each inference's result, in the order the procedure kind runs them.
    /// Empty when the pipeline published no final record.
    pub results: Vec<InferenceResult>,
    /// What a consensus pipeline states of its two inferences.
    pub consensus: Option<procedure_definition::ConsensusOutcome>,
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
    pub list_path: Option<PathBuf>,
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
        native_stage_backend: options.native_stage_backend,
        parity_policy: ParityPolicy::Skip,
    })?;
    // A claimer applies the rule a challenger will: the claim is judged by
    // its own final record, and a claim of several inferences by each one's.
    let procedure_kind = prepared.prepared_run.procedure_kind;
    let rejection = crate::challenge::claimed_outputs_rejection(
        procedure_kind,
        &checkpointed.results,
        checkpointed.consensus,
    )
    .with_context(|| format!("the `{procedure_kind}` pipeline did not state its result"))?;
    if let Some(reason) = rejection {
        if options.claimed_token_ids.is_none() {
            if options.native_output_ids.is_some() {
                anyhow::bail!(
                    "{reason}; the ids given as this run's native output are not what the pipeline \
                     derives, so they are not that output or the native run and teacher-forced \
                     inference are not in parity"
                );
            }
            // The ids came from native decode, so the two paths are not
            // bit-identical: an honest claimer would convict itself.
            anyhow::bail!("{reason}; native and teacher-forced inference are not in parity");
        }
        eprintln!("warning: {reason}; writing the claim anyway because its ids were given (`iwebs test claim`)");
    }
    let result = write_claim_result(
        checkpointed,
        &prepared.manifest_path,
        &prepared.prepared_run,
    )?;
    // And the rules a contract will: the statement the pipeline bound is the one
    // recomputed from the pinned model and this run's parameters, every
    // checkpoint names an approved program, and the record is the last
    // stage's.
    //
    // The claim names its procedure and its claimer, and prepares what
    // it has to post: the inputs its run was given and its checkpoints. Posting is `iwebs claim post`'s, so that a post that fails
    // is tried again without the claim being built again.
    // Without pins there is no procedure to name, and nothing to check.
    if let Some(path) = result.claim_submission_path.as_ref() {
        let Some(procedure) = crate::claim_check::run_procedure(&prepared.prepared_run)? else {
            return Ok(result);
        };
        let mut submission: ClaimSubmission = procedure_definition::read_json(path)?;
        submission.procedure = procedure.id();
        if let Some(claimer) = options.claimer {
            anyhow::ensure!(
                blockchain_rules::checks::is_address(&claimer),
                "`{claimer}` is not an address: give `0x` and 40 lowercase hex digits"
            );
            submission.claimer = claimer;
        }
        procedure_definition::write_json(path, &submission)?;
        let posting = crate::posting::claim_blobs(
            &result.pipeline_dir,
            &submission.claimer,
            Some(crate::posting::posted_run(&prepared.prepared_run)?),
        )
        .context("preparing the claim's data")?;
        posting.write(&result.pipeline_dir)?;
        if let Some(reason) =
            crate::claim_check::claim_rejection(&procedure, path, posting.contents()?)?
        {
            if options.claimed_token_ids.is_none() {
                anyhow::bail!("a contract would not accept this claim: {reason}");
            }
            eprintln!(
                "warning: a contract would not accept this claim: {reason}; it was written \
                 anyway because its ids were given (`iwebs test claim`)"
            );
        }
    }
    Ok(result)
}

/// [`build_claim`], then its data posted: a claim as a contract receives it.
pub fn build_posted_claim(options: ClaimBuildOptions) -> Result<ClaimBuildResult> {
    let result = build_claim(options)?;
    if let Some(path) = result.claim_submission_path.as_ref() {
        // A claim with no procedure prepared nothing to post.
        if path
            .with_file_name(crate::posting::POSTING_OUTPUTS_JSON)
            .exists()
        {
            let posted = crate::claim_check::post_claim_submission(path, None, None)?;
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

    let list_path = resolve_list_path(&options.hashes_path, options.list_path.as_deref());
    let list = if list_path.is_file() {
        Some(read_checkpoint_list(&list_path)?)
    } else {
        None
    };
    let checkpoint_index = select_checkpoint_index(&options.selection, &hashes, list.as_ref())?;
    let checkpoint_number = checkpoint_index + 1;
    let stage = list
        .as_ref()
        .and_then(|list| list.checkpoints.get(checkpoint_index))
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
    let (checkpoint_list_path, checkpoint_hashes_path, claim_bundle_path) =
        write_claim_artifacts_with_prepared_run(
            &checkpointed.pipeline_dir,
            manifest_path,
            prepared_run_path.as_deref(),
        )?;
    // The submission is stated over the checkpoint list written just above:
    // each inference's result, and what a consensus states of them.
    let results: Vec<&InferenceResult> = checkpointed.results.iter().collect();
    let consensus = checkpointed.consensus.as_ref().map(|consensus| {
        blockchain_rules::checks::SubmittedConsensus {
            same_input: consensus.same_input,
            agree: consensus.agree,
            code: consensus.code,
        }
    });
    let claim_submission_path = if results.len() == prepared_run.procedure_kind.inferences() {
        // The procedure and what was posted are filled in by
        // `build_claim`, which knows the contract.
        let submission = ClaimSubmission::from_run(
            prepared_run.procedure_kind,
            String::new(),
            &results,
            consensus,
            prepared_run.first()?.prompt.source.as_ref(),
            // The record of each relay, which run preparation computed for
            // the inference it gives its prompt.
            prepared_run
                .inferences
                .iter()
                .enumerate()
                .filter(|(inference, _)| {
                    prepared_run
                        .procedure_kind
                        .relayed_from(*inference)
                        .is_some()
                })
                .map(|(inference, prepared)| {
                    prepared.prompt.source.clone().with_context(|| {
                        format!("the prepared run holds no relay record for inference {inference}")
                    })
                })
                .collect::<Result<_>>()?,
            &checkpointed.pipeline_dir,
        )?;
        let path = checkpointed.pipeline_dir.join(CLAIM_SUBMISSION_JSON);
        procedure_definition::write_json(&path, &submission)?;
        Some(path)
    } else {
        None
    };
    Ok(ClaimBuildResult {
        procedure_kind: prepared_run.procedure_kind,
        pipeline_dir: checkpointed.pipeline_dir,
        checkpoint_list_path,
        checkpoint_hashes_path,
        claim_bundle_path,
        claim_submission_path,
        results: checkpointed.results,
        consensus: checkpointed.consensus,
    })
}

impl ClaimBuildOptions {
    pub fn from_current_dir(
        current_exe: PathBuf,
        native_stage_backend: NativeStageBackend,
        specification_path: PathBuf,
        case_path: PathBuf,
    ) -> Result<Self> {
        let base_dir = std::env::current_dir().context("failed to read current directory")?;
        Ok(Self {
            specification_path,
            case_path: Some(case_path),
            base_dir,
            current_exe,
            native_stage_backend,
            claimed_token_ids: None,
            native_output_ids: None,
            claimer: None,
        })
    }
}

pub fn prepare_claim_run(options: &ClaimBuildOptions) -> Result<PreparedPipelineRun> {
    let specification_path = absolute(&options.base_dir, &options.specification_path);
    let specification = read_specification(&specification_path)?;
    let (case_path, case) = read_options_case(options)?;
    prepare_run(
        options,
        specification_path,
        &specification,
        (Some(case_path), &case),
        None,
        None,
        true,
    )
}

/// The case file the options name, read.
fn read_options_case(options: &ClaimBuildOptions) -> Result<(PathBuf, Case)> {
    let case_path = options
        .case_path
        .as_ref()
        .context("a run is prepared on a case, and none is named")?;
    let case_path = absolute(&options.base_dir, case_path);
    let case = read_case(&case_path)?;
    Ok((case_path, case))
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
pub fn prepare_posted_run(
    options: &ClaimBuildOptions,
    posted: &blockchain_rules::availability::PostedRun,
) -> Result<PreparedPipelineRun> {
    let specification_path = absolute(&options.base_dir, &options.specification_path);
    let specification = read_specification(&specification_path)?;
    let (case_path, case) = read_options_case(options)?;
    prepare_run(
        options,
        specification_path,
        &specification,
        (Some(case_path), &case),
        None,
        Some(posted),
        false,
    )
}

/// Prepares a run from a specification and a case a claim already fixed, as a
/// challenger does. The case is the claim's own prompt and no file.
///
/// The ids are not held to the run's token budget here. A claim that breaks the
/// budget still has a pipeline to recompute, and is rejected on its final record.
///
/// `source` is the claim's source record with the source stage's inputs,
/// `forms` the forms its result stage compared the output with, and `ids` the
/// ids each of its inferences claimed, in order. A challenger reruns every
/// stage on what the claim committed to.
pub fn prepare_challenge_run(
    options: &ClaimBuildOptions,
    specification: &Specification,
    case: &Case,
    source: Option<FrozenSource<'_>>,
    forms: &[procedure_prepare::source::Form],
    ids: &[Vec<u32>],
    prompts: &[procedure_definition::PreparedPrompt],
) -> Result<PreparedPipelineRun> {
    let specification_path = absolute(&options.base_dir, &options.specification_path);
    prepare_run(
        options,
        specification_path,
        specification,
        (None, case),
        Some(Frozen {
            source,
            forms,
            ids,
            prompts,
        }),
        None,
        false,
    )
}

/// Prepares the run, finding by the native run the output of any inference
/// whose ids no one gave: the only autoregression left, since the claim
/// posts what native decode produced and the pipeline checks it.
fn prepare_run(
    options: &ClaimBuildOptions,
    specification_path: PathBuf,
    specification: &Specification,
    (case_path, case): (Option<PathBuf>, &Case),
    frozen: Option<Frozen<'_>>,
    given: Option<&blockchain_rules::availability::PostedRun>,
    check_claimed_length: bool,
) -> Result<PreparedPipelineRun> {
    let discover =
        procedure_run::native_output(&options.base_dir, &specification_path, case_path.as_deref());
    procedure_prepare::pipeline_run::prepare_run(procedure_prepare::pipeline_run::PipelineRun {
        base_dir: &options.base_dir,
        specification_path: specification_path.clone(),
        specification,
        case_path: case_path.clone(),
        case,
        frozen,
        given,
        // Ids given on the command line are one inference's.
        given_ids: options
            .claimed_token_ids
            .as_ref()
            .or(options.native_output_ids.as_ref()),
        check_claimed_length,
        discover: &discover,
    })
}

fn absolute(base: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        base.join(path)
    }
}

pub fn select_checkpoint_index(
    selection: &CorruptCheckpointSelection,
    hashes: &[String],
    list: Option<&procedure_definition::CheckpointList>,
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
            let list = list.ok_or_else(|| {
                anyhow::anyhow!(
                    "stage selection requires a checkpoint list; pass --list or keep checkpoint_list.json beside checkpoints.txt"
                )
            })?;
            let index = list
                .checkpoints
                .iter()
                .position(|checkpoint| checkpoint.stage == *stage)
                .ok_or_else(|| {
                    anyhow::anyhow!("stage `{stage}` was not found in the checkpoint list")
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

pub fn corrupt_hash(hash: &str) -> String {
    let mut bytes = hash.as_bytes().to_vec();
    bytes[0] = if bytes[0] == b'0' { b'1' } else { b'0' };
    String::from_utf8(bytes).expect("checkpoints are valid hex")
}

fn resolve_list_path(hashes_path: &Path, list_path: Option<&Path>) -> PathBuf {
    list_path
        .map(Path::to_path_buf)
        .or_else(|| {
            hashes_path
                .parent()
                .map(|parent| parent.join(CHECKPOINT_LIST_JSON))
        })
        .unwrap_or_else(|| PathBuf::from(CHECKPOINT_LIST_JSON))
}

fn default_corrupted_hashes_path(source: &Path) -> PathBuf {
    source.with_file_name("checkpoints.corrupt.txt")
}

#[cfg(test)]
mod tests {
    use super::*;
    use blockchain_rules::CLAIM_BUNDLE_JSON;
    use inference_manifest::ModelManifest;
    use procedure_definition::read_json;
    use procedure_definition::CHECKPOINT_LIST_JSON;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn options_from_current_dir_use_the_specification_and_case() {
        let saved = std::env::current_dir().unwrap();
        let base = temp_dir("claim-options");
        fs::create_dir_all(&base).unwrap();
        let expected_base = base.canonicalize().unwrap();
        std::env::set_current_dir(&base).unwrap();

        let options = ClaimBuildOptions::from_current_dir(
            PathBuf::from("iwebs"),
            NativeStageBackend::InProcess,
            PathBuf::from("specification.toml"),
            PathBuf::from("case.toml"),
        )
        .unwrap();

        std::env::set_current_dir(saved).unwrap();
        fs::remove_dir_all(&base).unwrap();

        assert_eq!(options.base_dir, expected_base);
        assert_eq!(
            options.specification_path,
            PathBuf::from("specification.toml")
        );
        assert_eq!(options.case_path, Some(PathBuf::from("case.toml")));
        assert_eq!(options.current_exe, PathBuf::from("iwebs"));
        assert_eq!(options.native_stage_backend, NativeStageBackend::InProcess);
    }

    #[test]
    fn preparation_uses_selected_model_and_rejects_stale_identity() {
        use crate::test_support::{case_beside, model_fixture, specification};
        let base = temp_dir("model-selection");
        model_fixture(&base, "model-a");
        let model_path = model_fixture(&base, "model-b");
        fs::write(base.join("Raster.toml"), "stale model-a template").unwrap();
        let specification_path = specification(&base, "model-b");
        let options = ClaimBuildOptions {
            base_dir: base.clone(),
            case_path: Some(case_beside(&specification_path)),
            specification_path,
            current_exe: PathBuf::from("unused"),
            native_stage_backend: NativeStageBackend::InProcess,
            claimed_token_ids: Some(vec![0, 0, 0]),
            native_output_ids: None,
            claimer: None,
        };
        let prepared = prepare_claim_run(&options).unwrap();
        let rendered = fs::read_to_string(&prepared.manifest_path).unwrap();
        assert!(rendered.contains("name = \"model-b\""));
        assert!(!rendered.contains("model-a"));
        assert_eq!(prepared.claimed_token_ids, [0, 0, 0]);
        // A prior native run's output is taken without running native again
        // (this fixture has no weights to run), and is held to the run's length.
        let given = |ids: &[u32]| ClaimBuildOptions {
            claimed_token_ids: None,
            native_output_ids: Some(ids.to_vec()),
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
        fs::write(&manifest_path, "[pipeline]\nname = \"test\"\n").unwrap();

        let result = write_claim_result(
            CheckpointedInferenceResult {
                pipeline_dir: base.clone(),
                selected_stage_dir: None,
                results: Vec::new(),
                consensus: None,
            },
            &manifest_path,
            &PreparedRun {
                version: procedure_definition::PREPARED_RUN_VERSION,
                procedure_kind: procedure_definition::ProcedureKind::single(),
                specification_path: base.join("specification.toml"),
                case_path: Some(base.join("case.toml")),
                inferences: vec![procedure_definition::PreparedInference {
                    model_manifest_path: base.join("model-artifacts/manifest.json"),
                    model_manifest_sha256: String::from("model-sha"),
                    model_id: String::from("model-id"),
                    prompt: procedure_definition::PreparedPrompt {
                        resolved_prompt: String::from("hello"),
                        rendered_prompt: String::from("hello"),
                        initial_pieces: vec![procedure_definition::PreparedPiece {
                            text: String::from("hello"),
                            segment: 0,
                        }],
                        eos_token_ids: Vec::new(),
                        source: None,
                        forms: Vec::new(),
                    },
                }],
                raw_prompt: false,
                max_new_tokens: 1,
                run_manifest_path: Some(manifest_path.clone()),
                run_manifest_sha256: Some(String::from("run-sha")),
                source_inputs: None,
            },
        )
        .unwrap();

        assert_eq!(result.pipeline_dir, base);
        assert_eq!(
            result.checkpoint_list_path.file_name().unwrap(),
            CHECKPOINT_LIST_JSON
        );
        assert_eq!(
            result.checkpoint_hashes_path.file_name().unwrap(),
            procedure_definition::CHECKPOINT_HASHES_TXT
        );
        assert_eq!(
            result.claim_bundle_path.file_name().unwrap(),
            CLAIM_BUNDLE_JSON
        );
        assert!(result.results.is_empty());

        fs::remove_dir_all(result.pipeline_dir).unwrap();
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
        procedure_definition::write_checkpoint_hashes(&hashes_path, &hashes).unwrap();

        let result = corrupt_checkpoint_hashes(CorruptCheckpointHashesOptions {
            hashes_path: hashes_path.clone(),
            list_path: None,
            output_path: Some(output_path.clone()),
            selection: CorruptCheckpointSelection::Index(1),
        })
        .unwrap();

        let corrupted = procedure_definition::read_checkpoint_hashes(&output_path).unwrap();
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
    fn corrupt_checkpoint_hashes_can_select_stage_from_list() {
        let base = temp_dir("corrupt-stage");
        fs::create_dir_all(&base).unwrap();
        let hashes_path = base.join("checkpoints.txt");
        let list_path = base.join(CHECKPOINT_LIST_JSON);
        let list = procedure_definition::CheckpointList {
            checkpoints: vec![
                procedure_definition::Checkpoint {
                    stage: String::from("stage_a"),
                    program_commitment: String::from("program-a"),
                    input_sources: Default::default(),
                    input_commitment: String::from("input-a"),
                    output_commitment: String::from("output-a"),
                    output_sha256: String::from("sha-a"),
                },
                procedure_definition::Checkpoint {
                    stage: String::from("stage_b"),
                    program_commitment: String::from("program-b"),
                    input_sources: Default::default(),
                    input_commitment: String::from("input-b"),
                    output_commitment: String::from("output-b"),
                    output_sha256: String::from("sha-b"),
                },
            ],
        };
        procedure_definition::write_json(&list_path, &list).unwrap();
        let hashes = procedure_definition::checkpoint_hashes(&list).unwrap();
        procedure_definition::write_checkpoint_hashes(&hashes_path, &hashes).unwrap();

        let result = corrupt_checkpoint_hashes(CorruptCheckpointHashesOptions {
            hashes_path: hashes_path.clone(),
            list_path: None,
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
    fn a_consensus_run_is_one_pipeline_of_two_inferences() {
        let base = temp_dir("claim-consensus");
        crate::test_support::model_fixture(&base, "alpha");
        crate::test_support::model_fixture(&base, "beta");
        let dir = base.join("specs");
        fs::create_dir_all(&dir).unwrap();
        let specification = dir.join("specification.toml");
        fs::write(
            &specification,
            "procedure_kind = \"consensus\"\nraw_prompt = true\nmax_new_tokens = 3\n\
             [[model]]\nmanifest = \"../models/alpha/manifest.json\"\n\
             [[model]]\nmanifest = \"../models/beta/manifest.json\"\n",
        )
        .unwrap();
        // A challenger's case is the claim's prompt, and no file.
        let case = procedure_definition::Case::of_prompt(String::from("h"));
        let options = |ids: Option<Vec<u32>>| ClaimBuildOptions {
            base_dir: base.clone(),
            case_path: None,
            specification_path: specification.clone(),
            current_exe: PathBuf::from("unused"),
            native_stage_backend: NativeStageBackend::InProcess,
            claimed_token_ids: ids,
            native_output_ids: None,
            claimer: None,
        };
        let specification = read_specification(&specification).unwrap();
        // The fixture's ranges hold four tokens: the first inference claims
        // one range and the second two.
        let ids = [vec![0, 0, 0], vec![0, 0, 0, 0, 0]];
        let prepared =
            prepare_challenge_run(&options(None), &specification, &case, None, &[], &ids, &[])
                .unwrap();

        let run = &prepared.prepared_run;
        assert_eq!(
            run.procedure_kind,
            procedure_definition::ProcedureKind::named("consensus").unwrap()
        );
        assert_eq!(run.inferences.len(), 2);
        assert_ne!(run.inferences[0].model_id, run.inferences[1].model_id);
        assert_eq!(
            run.inferences[0].prompt.resolved_prompt,
            run.inferences[1].prompt.resolved_prompt
        );
        assert_eq!(prepared.claimed_token_ids, ids[0]);

        let manifest: toml::Value =
            toml::from_str(&fs::read_to_string(&prepared.manifest_path).unwrap()).unwrap();
        let pipeline = &manifest["pipeline"];
        // Each inference has its own run files, in a directory of its own.
        for (label, claimed_ranges) in [("m0", 1), ("m1", 2)] {
            let claimed = &pipeline["input"][format!("{label}_claimed_ids").as_str()];
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
                let input = &pipeline["input"][format!("{label}_{input}").as_str()];
                assert_eq!(input["commitments"].as_array().unwrap().len(), 1);
            }
        }
        let stages = pipeline["stage"].as_array().unwrap();
        let stage = |name: &str| {
            stages
                .iter()
                .find(|stage| stage["name"].as_str() == Some(name))
                .unwrap_or_else(|| panic!("the pipeline has no stage `{name}`"))
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
        // Each bind is given its own model's procedure.
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
        let repeats = pipeline["repeat"].as_array().unwrap();
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
        let error = prepare_challenge_run(
            &options(Some(vec![0])),
            &specification,
            &case,
            None,
            &[],
            &ids,
            &[],
        )
        .err()
        .unwrap();
        assert!(format!("{error:#}").contains("one inference"), "{error:#}");
        assert!(prepare_challenge_run(
            &options(None),
            &specification,
            &case,
            None,
            &[],
            &ids[..1],
            &[]
        )
        .is_err());
        fs::remove_dir_all(base).unwrap();
    }

    #[test]
    fn ranged_manifest_fills_counts_inputs_and_last_prompt_range() {
        let template = r#"[pipeline]
name = "iwebs"

[pipeline.input.prompt_marks]
index = "j"
path = "/repo/target/iwebs/run-prompt-placeholder/marks_{j}.rastered"
index_path = "/repo/target/iwebs/run-prompt-placeholder/marks_{j}.rindex"
commitments = []

[pipeline.input.prompt_text]
index = "j"
path = "/repo/target/iwebs/run-prompt-placeholder/text_{j}.rastered"
index_path = "/repo/target/iwebs/run-prompt-placeholder/text_{j}.rindex"
commitments = []

[pipeline.input.claimed_ids]
index = "k"
path = "/repo/target/iwebs/run-prompt-placeholder/claimed_{k}.rastered"
index_path = "/repo/target/iwebs/run-prompt-placeholder/claimed_{k}.rindex"
commitments = []

[[pipeline.stage]]
name = "source"
inputs.signer = { external = { path = "/repo/target/iwebs/run-prompt-placeholder/source_signer.rastered", index_path = "/repo/target/iwebs/run-prompt-placeholder/source_signer.rindex", commitment = "0" } }
inputs.recipe = { external = { path = "/repo/target/iwebs/run-prompt-placeholder/source_recipe.rastered", index_path = "/repo/target/iwebs/run-prompt-placeholder/source_recipe.rindex", commitment = "0" } }
inputs.documents = { external = { path = "/repo/target/iwebs/run-prompt-placeholder/source_documents.rastered", index_path = "/repo/target/iwebs/run-prompt-placeholder/source_documents.rindex", commitment = "0" } }

[[pipeline.stage]]
name = "prompt_bind_p0"
inputs.text = { input = "prompt_text_0" }
inputs.config = { external = { path = "/repo/target/iwebs/run-prompt-placeholder/config.rastered", index_path = "/repo/target/iwebs/run-prompt-placeholder/config.rindex", commitment = "0" } }
inputs.schedule = { external = { path = "/repo/target/iwebs/run-prompt-placeholder/schedule.rastered", index_path = "/repo/target/iwebs/run-prompt-placeholder/schedule.rindex", commitment = "0" } }
inputs.text_digest = { external = { path = "/repo/target/iwebs/run-prompt-placeholder/text_digest.rastered", index_path = "/repo/target/iwebs/run-prompt-placeholder/text_digest.rindex", commitment = "0" } }
inputs.source = { from = "source" }
inputs.previous = { external = { path = "/repo/model/empty_prompt.rastered", index_path = "/repo/model/empty_prompt.rindex", commitment = "e0" } }

[[pipeline.stage]]
name = "prompt_check_p0"
inputs.bound = { from = "prompt_bind_p0" }
inputs.previous = { external = { path = "/repo/model/empty_prompt.rastered", index_path = "/repo/model/empty_prompt.rindex", commitment = "e0" } }
inputs.marks = { input = "prompt_marks_0" }

[[pipeline.repeat]]
name = "prompt"
index = "j"
start = 1
count = 0

  [[pipeline.repeat.stage]]
name = "prompt_bind_p{j}"
inputs.text = { input = "prompt_text_{j}" }
inputs.config = { external = { path = "/repo/target/iwebs/run-prompt-placeholder/config.rastered", index_path = "/repo/target/iwebs/run-prompt-placeholder/config.rindex", commitment = "0" } }
inputs.schedule = { external = { path = "/repo/target/iwebs/run-prompt-placeholder/schedule.rastered", index_path = "/repo/target/iwebs/run-prompt-placeholder/schedule.rindex", commitment = "0" } }
inputs.text_digest = { external = { path = "/repo/target/iwebs/run-prompt-placeholder/text_digest.rastered", index_path = "/repo/target/iwebs/run-prompt-placeholder/text_digest.rindex", commitment = "0" } }
inputs.source = { external = { path = "/repo/model/prompt-bind/no_source.rastered", index_path = "/repo/model/prompt-bind/no_source.rindex", commitment = "n0" } }
inputs.previous = { from = "prompt_bind_p{j-1}", first = "prompt_bind_p0" }

  [[pipeline.repeat.stage]]
name = "prompt_check_p{j}"
inputs.bound = { from = "prompt_bind_p{j}" }
inputs.previous = { from = "prompt_check_p{j-1}", first = "prompt_check_p0" }
inputs.marks = { input = "prompt_marks_{j}" }

[[pipeline.repeat]]
name = "claimed_embed"
index = "k"
count = 0

  [[pipeline.repeat.stage]]
name = "embed_c{k}"
inputs.previous = { from = "embed_c{k-1}", first = "embed_p@LAST_PROMPT_RANGE@" }

[[pipeline.repeat]]
name = "layers"
index = "l"
start = 0
count = 4

  [[pipeline.repeat.stage]]
name = "aux_p{j}_l{l}"
index = "j"
count = 0

  [[pipeline.repeat.stage]]
name = "aux_c{k}_l{l}"
index = "k"
count = 0

  [[pipeline.repeat.stage]]
name = "range_p0_l{l}"

  [[pipeline.repeat.stage]]
name = "range_p{j}_l{l}"
index = "j"
start = 1
count = 0

  [[pipeline.repeat.stage]]
name = "range_c{k}_l{l}"
index = "k"
count = 0

[[pipeline.stage]]
name = "check_c0"

[[pipeline.repeat]]
name = "claimed"
index = "k"
start = 1
count = 0

[[pipeline.stage]]
name = "result"
inputs.check = { from = "check_c@LAST_CLAIMED_RANGE@" }
inputs.forms = { external = { path = "/repo/target/iwebs/run-prompt-placeholder/result_forms.rastered", index_path = "/repo/target/iwebs/run-prompt-placeholder/result_forms.rindex", commitment = "0" } }
"#;
        let file = |name: &str, commitment: &str| procedure_prepare::RunFile {
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
        let text_digest = file("text_digest", "m0");
        let forms = file("result_forms", "f0");
        let prompt = [
            file("marks_0", "a0"),
            file("marks_1", "a1"),
            file("marks_2", "a2"),
        ];
        let claimed = [file("claimed_0", "c0")];
        let render = |template: &str, claimed: &[procedure_prepare::RunFile]| {
            render_ranged_manifest(
                template,
                Path::new("/repo"),
                &text,
                &config,
                &schedule,
                &text_digest,
                None,
                &forms,
                &prompt,
                claimed,
            )
        };
        let manifest = render(template, &claimed).unwrap();
        let parsed: toml::Value = toml::from_str(&manifest).unwrap();
        let inputs = &parsed["pipeline"]["input"];
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
        let repeats = parsed["pipeline"]["repeat"].as_array().unwrap();
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

        let run_file = |name: &str, commitment: &str| procedure_definition::RunFileRef {
            path: PathBuf::from(format!("/tmp/run/prompt/{name}.rastered")),
            index_path: PathBuf::from(format!("/tmp/run/prompt/{name}.rindex")),
            commitment: commitment.into(),
        };
        let source = procedure_definition::PreparedSourceInputs {
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
            &text_digest,
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
            &text_digest,
            None,
            &forms,
            &prompt,
            &claimed,
        )
        .is_err());
        assert!(!manifest.contains("run-prompt-placeholder"));
        assert!(render(template, &[]).is_err());
        // A template for the removed cut-and-merge stages is not this pipeline's.
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
        std::env::temp_dir().join(format!("iwebs-cli-{label}-{}-{nanos}", std::process::id()))
    }
}
