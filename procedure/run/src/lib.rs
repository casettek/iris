//! A procedure run with no blockchain.
//!
//! The native run comes first, for the result as fast as possible. The
//! pipeline run follows on the output the native run generated: it verifies
//! every token of that output and leaves a checkpoint per stage. The two
//! results are then held equal.
//!
//! A claim is this and what a blockchain needs besides. `blockchain/host`
//! takes from here the function that asks the native run for an output.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use procedure_definition::divergence::locate_divergence;
use procedure_definition::{
    build_checkpoint_list, read_case, read_checkpoint_list, read_specification, ConsensusOutcome,
    Divergence, InferenceResult, ProcedureKind,
};
use procedure_native::{absolute, InferRun, NativeInferenceConfig, NativeInferenceExecutor};
use procedure_pipeline::pipeline_runner::NativeStageBackend;
use procedure_pipeline::{
    CheckpointedInferenceConfig, CheckpointedInferenceExecutor, ParityPolicy,
};
use procedure_prepare::pipeline_run::PipelineRun;

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

/// A procedure run both ways.
pub struct ProcedureRun {
    pub native_run: InferRun,
    pub pipeline_run: CheckpointedRun,
}

/// Runs a procedure: [`run_native`], then [`run_pipeline`] on its output.
pub fn run(options: &RunOptions) -> Result<ProcedureRun> {
    let native_run = run_native(options)?;
    let pipeline_run = run_pipeline(options, &native_run)?;
    Ok(ProcedureRun {
        native_run,
        pipeline_run,
    })
}

/// The native run: every inference of the procedure generated, and what the
/// procedure states of them.
pub fn run_native(options: &RunOptions) -> Result<InferRun> {
    procedure_native::run_infer(
        absolute(&options.base_dir, &options.specification_path),
        absolute(&options.base_dir, &options.case_path),
    )
}

/// The pipeline run on the output a native run generated: prepared, run stage
/// by stage, and its checkpoints written. It fails unless the pipeline run
/// states the result the native run did.
pub fn run_pipeline(options: &RunOptions, native_run: &InferRun) -> Result<CheckpointedRun> {
    let specification_path = absolute(&options.base_dir, &options.specification_path);
    let specification = read_specification(&specification_path)?;
    let case_path = absolute(&options.base_dir, &options.case_path);
    let case = read_case(&case_path)?;
    let procedure_kind = specification.procedure_kind;
    anyhow::ensure!(
        native_run.procedure_kind == procedure_kind
            && native_run.reports.len() == procedure_kind.inferences(),
        "the native run is of {} inference(s) of the `{}` procedure kind, and the specification \
         names the `{procedure_kind}` procedure kind",
        native_run.reports.len(),
        native_run.procedure_kind
    );
    let generated = |inference: usize, _: Option<&InferenceResult>| -> Result<InferenceResult> {
        Ok(native_run
            .reports
            .get(inference)
            .with_context(|| format!("the native run made no inference {inference}"))?
            .result
            .clone())
    };
    let prepared = procedure_prepare::pipeline_run::prepare_run(PipelineRun {
        base_dir: &options.base_dir,
        specification_path,
        specification: &specification,
        case_path: Some(case_path),
        case: &case,
        frozen: None,
        given: None,
        given_ids: None,
        check_claimed_length: true,
        discover: &generated,
    })?;
    let pipeline_run = CheckpointedInferenceExecutor.run(CheckpointedInferenceConfig {
        base_dir: options.base_dir.clone(),
        manifest_path: prepared.manifest_path.clone(),
        current_exe: options.current_exe.clone(),
        native_stage_backend: options.native_stage_backend,
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

    let stated = pipeline_run.results;
    anyhow::ensure!(
        !stated.is_empty(),
        "the pipeline run reached no final record"
    );
    anyhow::ensure!(
        stated.len() == native_run.reports.len(),
        "the pipeline run states {} result(s), and the native run made {} inference(s)",
        stated.len(),
        native_run.reports.len()
    );
    for (inference, (report, pipeline_run)) in native_run.reports.iter().zip(&stated).enumerate() {
        if let Some(difference) = result_difference(&report.result, pipeline_run)? {
            anyhow::bail!(
                "inference {inference}: {difference}; the native run and the pipeline run are not in \
                 parity"
            );
        }
    }
    anyhow::ensure!(
        native_run.consensus == pipeline_run.consensus,
        "the pipeline run states another consensus than the native run: {:?} and {:?}",
        pipeline_run.consensus,
        native_run.consensus
    );
    Ok(CheckpointedRun {
        procedure_kind,
        manifest_path: prepared.manifest_path,
        pipeline_dir: pipeline_run.pipeline_dir,
        checkpoint_list_path,
        checkpoint_hashes_path,
        checkpoints: list.checkpoints.len(),
        results: stated,
        consensus: pipeline_run.consensus,
    })
}

/// The first field of an inference's result that a pipeline run states
/// otherwise than the native run did, if there is one.
fn result_difference(
    native_run: &InferenceResult,
    pipeline_run: &InferenceResult,
) -> Result<Option<String>> {
    if native_run == pipeline_run {
        return Ok(None);
    }
    let native_run = serde_json::to_value(native_run)?;
    let pipeline_run = serde_json::to_value(pipeline_run)?;
    let fields = native_run
        .as_object()
        .context("an inference's result is not a record")?;
    for (field, native_run) in fields {
        let pipeline_run = &pipeline_run[field];
        if native_run != pipeline_run {
            return Ok(Some(format!(
                "the pipeline run states `{field}` as {pipeline_run}, and the native run as {native_run}"
            )));
        }
    }
    Ok(Some(String::from(
        "the pipeline run's result is not the native run's",
    )))
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
        let native_run = InferenceResult {
            generated_token_count: 2,
            generated_token_ids: vec![7, 9],
            generated_text: String::from("ciao"),
            ..Default::default()
        };
        assert_eq!(
            result_difference(&native_run, &native_run.clone()).unwrap(),
            None
        );

        let mut pipeline_run = native_run.clone();
        pipeline_run.claim_mismatches = 1;
        let difference = result_difference(&native_run, &pipeline_run)
            .unwrap()
            .unwrap();
        assert!(
            difference.contains("`claim_mismatches` as 1") && difference.ends_with("as 0"),
            "{difference}"
        );
    }

    #[test]
    fn the_native_output_of_no_case_is_an_error() {
        let discover = native_output(Path::new("."), Path::new("specification.toml"), None);
        let error = discover(0, None).unwrap_err();
        assert!(format!("{error:#}").contains("none is named"));
    }
}
