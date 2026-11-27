//! A procedure run with no blockchain.
//!
//! A procedure finds an output or verifies a stated one, and a run is of one
//! procedure. [`run_native`] runs a generation procedure: every model's
//! output found natively. [`run_verification`] runs a verification
//! procedure: the output its case states is held to the model's own choice,
//! token by token, with a checkpoint per stage. Neither makes the other.
//! Whoever needs an output verified runs the first, states what it found in
//! the case, and runs the second.
//!
//! A run is of a procedure, named by its id: the id a contract registers is
//! the id two parties with no contract agree to.
//!
//! A claim is a verification run and what a blockchain needs besides.
//! `blockchain/host` takes from here the functions that give a pipeline
//! run's preparation its output.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use procedure_definition::divergence::locate_divergence;
use procedure_definition::registry::Procedure;
use procedure_definition::{
    build_checkpoint_list, read_case, read_checkpoint_list, read_specification, Case, CaseOutput,
    ConsensusOutcome, Divergence, InferenceResult, ProcedureKind, Specification,
};
use procedure_native::{absolute, InferRun, NativeInferenceConfig, NativeInferenceExecutor};
use procedure_pipeline::pipeline_runner::NativeStageBackend;
use procedure_pipeline::{
    CheckpointedInferenceConfig, CheckpointedInferenceExecutor, ParityPolicy,
};
use procedure_prepare::pipeline_run::{PipelineRun, PreparedPipelineRun};

pub mod record;

/// What a procedure is run from.
#[derive(Debug, Clone)]
pub struct RunOptions {
    /// Where the run's files are written, under `target/`. A relative
    /// specification or case path is taken from here.
    pub base_dir: PathBuf,
    pub specification_path: PathBuf,
    pub case_path: PathBuf,
    /// The binary the pipeline runner starts its workers from.
    pub current_exe: PathBuf,
    pub native_stage_backend: NativeStageBackend,
}

/// A pipeline run and the checkpoints it left.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckpointedRun {
    pub procedure_kind: ProcedureKind,
    /// The pipeline the run was made from: its `Raster.toml`.
    pub manifest_path: PathBuf,
    pub pipeline_dir: PathBuf,
    pub checkpoint_list_path: PathBuf,
    pub checkpoint_hashes_path: PathBuf,
    /// How many stages ran: one checkpoint each.
    pub checkpoints: usize,
    /// Each inference's result, in the order the procedure kind runs them.
    pub results: Vec<InferenceResult>,
    /// What a consensus pipeline states of its two inferences.
    pub consensus: Option<ConsensusOutcome>,
}

/// The procedure the options' specification describes, which its id names.
/// `Ok(None)` when a model it names was imported where the stage projects are
/// not locked: its stage programs then have no identity to pin.
pub fn procedure_of(options: &RunOptions) -> Result<Option<Procedure>> {
    let specification_path = absolute(&options.base_dir, &options.specification_path);
    let specification = read_specification(&specification_path)?;
    procedure_prepare::procedure::procedure_of(&specification_path, &specification, None)
}

/// The first stage of a pipeline run whose program the procedure does not
/// approve, if there is one. A run with such a stage is not a run of that
/// procedure: the stage was locked again after the model was imported.
pub fn first_unapproved(run: &CheckpointedRun, procedure: &Procedure) -> Result<Option<String>> {
    Ok(read_checkpoint_list(&run.checkpoint_list_path)?
        .checkpoints
        .into_iter()
        .find(|checkpoint| !procedure.programs.approves(&checkpoint.program_commitment))
        .map(|checkpoint| checkpoint.stage))
}

/// A run of a generation procedure: every model's output found natively,
/// and what the procedure states of them. No stage runs.
pub fn run_native(options: &RunOptions) -> Result<InferRun> {
    let specification_path = absolute(&options.base_dir, &options.specification_path);
    let procedure_kind = read_specification(&specification_path)?.procedure_kind;
    anyhow::ensure!(
        procedure_kind.generates(),
        "the `{procedure_kind}` procedure kind verifies a stated output, and finds none"
    );
    procedure_native::run_infer(
        specification_path,
        absolute(&options.base_dir, &options.case_path),
    )
}

/// A run of a verification procedure: the output the case states is held to
/// each model's own choice, stage by stage, and a checkpoint is left for each
/// stage. Nothing is generated.
pub fn run_verification(options: &RunOptions) -> Result<CheckpointedRun> {
    let specification_path = absolute(&options.base_dir, &options.specification_path);
    let specification = read_specification(&specification_path)?;
    let case_path = absolute(&options.base_dir, &options.case_path);
    let case = read_case(&case_path)?;
    let procedure_kind = specification.procedure_kind;
    anyhow::ensure!(
        !procedure_kind.generates(),
        "the `{procedure_kind}` procedure kind finds its output: it runs natively and has no \
         stages"
    );
    case.validate_for(&specification)?;
    anyhow::ensure!(
        !case.outputs.is_empty(),
        "the case states no output, and the `{procedure_kind}` procedure kind verifies one: \
         state each model's output in the case as an `[[output]]` table with its `ids`. A \
         generation procedure finds an output"
    );
    let stated = stated_output(
        &options.base_dir,
        &options.current_exe,
        options.native_stage_backend,
        &specification_path,
        &specification,
        Some(&case_path),
        &case,
    );
    let prepared = procedure_prepare::pipeline_run::prepare_run(PipelineRun {
        base_dir: &options.base_dir,
        specification_path: specification_path.clone(),
        specification: &specification,
        case_path: Some(case_path.clone()),
        case: &case,
        frozen: None,
        given: None,
        given_ids: None,
        check_claimed_length: true,
        discover: &stated,
    })?;
    execute(
        &options.base_dir,
        &options.current_exe,
        options.native_stage_backend,
        &prepared,
        procedure_kind,
    )
}

/// The function a pipeline run's preparation asks for the output of an
/// inference: the one the case states for it. Nothing is generated.
///
/// The ids are all a run needs of an output, with one exception. A later
/// model's prompt may hold an earlier one's output, and that prompt is made
/// of the earlier output's text and of the record its last check states. So
/// an output a later prompt holds is verified on its own first, as a run of
/// one model, and what that run states of it is what the prompt is made of.
/// Such an output is verified twice: there, and in the run itself.
pub fn stated_output<'a>(
    base_dir: &'a Path,
    current_exe: &'a Path,
    native_stage_backend: NativeStageBackend,
    specification_path: &'a Path,
    specification: &'a Specification,
    case_path: Option<&'a Path>,
    case: &'a Case,
) -> impl Fn(usize, Option<&InferenceResult>) -> Result<InferenceResult> + 'a {
    move |inference, _relayed| {
        let ids = case
            .outputs
            .get(inference)
            .with_context(|| format!("the case states no output for inference {inference}"))?
            .ids
            .clone();
        let of_ids = |ids: Vec<u32>| InferenceResult {
            generated_token_ids: ids,
            ..Default::default()
        };
        let procedure_kind = specification.procedure_kind;
        let held_later = (0..procedure_kind.inferences())
            .any(|later| procedure_kind.relayed_from(later) == Some(inference));
        if !held_later {
            return Ok(of_ids(ids));
        }
        anyhow::ensure!(
            procedure_kind.relayed_from(inference).is_none(),
            "inference {inference} of the `{procedure_kind}` procedure kind has its prompt from \
             an earlier output and gives its own to a later one, which a stated output does \
             not hold"
        );
        let mut alone = specification.clone();
        alone.procedure_kind = ProcedureKind::verify_single();
        alone.models = vec![specification
            .models
            .get(inference)
            .with_context(|| format!("the specification names no model {inference}"))?
            .clone()];
        alone.relay = None;
        let mut alone_case = case.clone();
        alone_case.outputs = vec![CaseOutput { ids: ids.clone() }];
        let stated = |_: usize, _: Option<&InferenceResult>| Ok(of_ids(ids.clone()));
        let prepared = procedure_prepare::pipeline_run::prepare_run(PipelineRun {
            base_dir,
            specification_path: specification_path.to_path_buf(),
            specification: &alone,
            case_path: case_path.map(Path::to_path_buf),
            case: &alone_case,
            frozen: None,
            given: None,
            given_ids: None,
            check_claimed_length: true,
            discover: &stated,
        })
        .with_context(|| {
            format!("preparing inference {inference} alone, whose output a later prompt holds")
        })?;
        execute(
            base_dir,
            current_exe,
            native_stage_backend,
            &prepared,
            alone.procedure_kind,
        )?
        .results
        .into_iter()
        .next()
        .with_context(|| format!("inference {inference}, run alone, stated no result"))
    }
}

/// Runs a prepared pipeline run stage by stage and writes its checkpoints.
pub(crate) fn execute(
    base_dir: &Path,
    current_exe: &Path,
    native_stage_backend: NativeStageBackend,
    prepared: &PreparedPipelineRun,
    procedure_kind: ProcedureKind,
) -> Result<CheckpointedRun> {
    let pipeline_run = CheckpointedInferenceExecutor.run(CheckpointedInferenceConfig {
        base_dir: base_dir.to_path_buf(),
        manifest_path: prepared.manifest_path.clone(),
        current_exe: current_exe.to_path_buf(),
        native_stage_backend,
        parity_policy: ParityPolicy::Skip,
    })?;
    let list = build_checkpoint_list(&pipeline_run.pipeline_dir, &prepared.manifest_path)?;
    let checkpoint_list_path = procedure_definition::checkpoint::write_checkpoint_list_artifact(
        &pipeline_run.pipeline_dir,
        &list,
    )?;
    let checkpoint_hashes_path =
        procedure_definition::checkpoint::write_checkpoint_hashes_artifact(
            &pipeline_run.pipeline_dir,
            &list,
        )?;
    anyhow::ensure!(
        !pipeline_run.results.is_empty(),
        "the pipeline run reached no final record"
    );
    Ok(CheckpointedRun {
        procedure_kind,
        manifest_path: prepared.manifest_path.clone(),
        pipeline_dir: pipeline_run.pipeline_dir,
        checkpoint_list_path,
        checkpoint_hashes_path,
        checkpoints: list.checkpoints.len(),
        results: pipeline_run.results,
        consensus: pipeline_run.consensus,
    })
}

/// The first field two results differ in, with each one's value. `Some(None)`
/// when they differ and no field of the first says where.
pub(crate) fn differing_field(
    first: &InferenceResult,
    second: &InferenceResult,
) -> Result<Option<Option<(String, serde_json::Value, serde_json::Value)>>> {
    if first == second {
        return Ok(None);
    }
    let first = serde_json::to_value(first)?;
    let second = serde_json::to_value(second)?;
    let fields = first
        .as_object()
        .context("an inference's result is not a record")?;
    for (field, first) in fields {
        let second = &second[field];
        if first != second {
            return Ok(Some(Some((field.clone(), first.clone(), second.clone()))));
        }
    }
    Ok(Some(None))
}

/// Where a run's checkpoints first differ from those of another run of the
/// same specification and case, if anywhere. `other_list_path` is that run's
/// checkpoint list.
pub fn first_difference(
    run: &CheckpointedRun,
    other_list_path: &Path,
) -> Result<Option<Divergence>> {
    let other = read_checkpoint_list(other_list_path)?;
    let own = read_checkpoint_list(&run.checkpoint_list_path)?;
    Ok(locate_divergence(
        other_list_path,
        &other,
        &run.checkpoint_list_path,
        &own,
    ))
}

/// The function a pipeline run's preparation asks for the output of an
/// inference whose ids no one gave: the native run of that one inference, on
/// the prompt an earlier inference's output makes when there is one.
pub fn native_output<'a>(
    base_dir: &'a Path,
    specification_path: &'a Path,
    case_path: Option<&'a Path>,
) -> impl Fn(usize, Option<&InferenceResult>) -> Result<InferenceResult> + 'a {
    move |inference, relayed| {
        NativeInferenceExecutor
            .run(NativeInferenceConfig {
                base_dir: base_dir.to_path_buf(),
                specification_path: specification_path.to_path_buf(),
                case_path: case_path
                    .context("the native run runs a case, and none is named")?
                    .to_path_buf(),
                inference,
                relayed: relayed.cloned(),
            })
            .context("the native run to discover the claimed output failed")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_difference_names_the_first_field_that_differs() {
        let stated = InferenceResult {
            generated_token_count: 2,
            generated_token_ids: vec![7, 9],
            generated_text: String::from("ciao"),
            ..Default::default()
        };
        assert_eq!(differing_field(&stated, &stated.clone()).unwrap(), None);

        let mut other = stated.clone();
        other.claim_mismatches = 1;
        let (field, first, second) = differing_field(&stated, &other).unwrap().unwrap().unwrap();
        assert_eq!(field, "claim_mismatches");
        assert_eq!(
            (first, second),
            (serde_json::json!(0), serde_json::json!(1))
        );
    }

    #[test]
    fn the_native_output_of_no_case_is_an_error() {
        let discover = native_output(Path::new("."), Path::new("specification.toml"), None);
        let error = discover(0, None).unwrap_err();
        assert!(format!("{error:#}").contains("none is named"));
    }
}
