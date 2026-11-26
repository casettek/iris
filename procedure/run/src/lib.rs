//! A procedure run with no blockchain.
//!
//! The direct run comes first, for the result as fast as possible. The
//! staged run follows on the output the direct run generated: it verifies
//! every token of that output and leaves a checkpoint per stage. The two
//! results are then held equal.
//!
//! A claim is this and what a blockchain needs besides. `blockchain/host`
//! takes from here the function that asks the direct run for an output.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use procedure_definition::divergence::locate_divergence;
use procedure_definition::{
    build_checkpoint_list, read_case, read_checkpoint_list, read_specification, ConsensusOutcome,
    Divergence, InferenceResult, ProcedureKind,
};
use procedure_native::{absolute, DirectInferenceConfig, DirectInferenceExecutor, InferRun};
use procedure_pipeline::pipeline_runner::StagedExecutionBackend;
use procedure_pipeline::{
    CheckpointedInferenceConfig, CheckpointedInferenceExecutor, ParityPolicy,
};
use procedure_prepare::staged::StagedRun;

/// What a procedure is run from.
#[derive(Debug, Clone)]
pub struct RunOptions {
    /// Where the run's files are written, under `target/`. A relative
    /// specification or case path is taken from here.
    pub base_dir: PathBuf,
    pub specification_path: PathBuf,
    pub case_path: PathBuf,
    /// The binary the staged runner starts its workers from.
    pub current_exe: PathBuf,
    pub staged_backend: StagedExecutionBackend,
}

/// A staged run and the checkpoints it left.
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
    pub direct: InferRun,
    pub staged: CheckpointedRun,
}

/// Runs a procedure: [`run_direct`], then [`run_staged`] on its output.
pub fn run(options: &RunOptions) -> Result<ProcedureRun> {
    let direct = run_direct(options)?;
    let staged = run_staged(options, &direct)?;
    Ok(ProcedureRun { direct, staged })
}

/// The direct run: every inference of the procedure generated, and what the
/// procedure states of them.
pub fn run_direct(options: &RunOptions) -> Result<InferRun> {
    procedure_native::run_infer(
        absolute(&options.base_dir, &options.specification_path),
        absolute(&options.base_dir, &options.case_path),
    )
}

/// The staged run on the output a direct run generated: prepared, run stage
/// by stage, and its checkpoints written. It fails unless the staged run
/// states the result the direct run did.
pub fn run_staged(options: &RunOptions, direct: &InferRun) -> Result<CheckpointedRun> {
    let specification_path = absolute(&options.base_dir, &options.specification_path);
    let specification = read_specification(&specification_path)?;
    let case_path = absolute(&options.base_dir, &options.case_path);
    let case = read_case(&case_path)?;
    let procedure_kind = specification.procedure_kind;
    anyhow::ensure!(
        direct.procedure_kind == procedure_kind
            && direct.reports.len() == procedure_kind.inferences(),
        "the direct run is of {} inference(s) of the `{}` procedure kind, and the specification \
         names the `{procedure_kind}` procedure kind",
        direct.reports.len(),
        direct.procedure_kind
    );
    let generated = |inference: usize, _: Option<&InferenceResult>| -> Result<InferenceResult> {
        Ok(direct
            .reports
            .get(inference)
            .with_context(|| format!("the direct run made no inference {inference}"))?
            .result
            .clone())
    };
    let prepared = procedure_prepare::staged::prepare_run(StagedRun {
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
    let staged = CheckpointedInferenceExecutor.run(CheckpointedInferenceConfig {
        base_dir: options.base_dir.clone(),
        manifest_path: prepared.manifest_path.clone(),
        current_exe: options.current_exe.clone(),
        staged_backend: options.staged_backend,
        parity_policy: ParityPolicy::Skip,
    })?;
    let list = build_checkpoint_list(&staged.pipeline_dir, &prepared.manifest_path)?;
    let checkpoint_list_path = procedure_definition::checkpoint::write_checkpoint_list_artifact(
        &staged.pipeline_dir,
        &list,
    )?;
    let checkpoint_hashes_path =
        procedure_definition::checkpoint::write_checkpoint_hashes_artifact(
            &staged.pipeline_dir,
            &list,
        )?;

    let stated = staged.results;
    anyhow::ensure!(!stated.is_empty(), "the staged run reached no final record");
    anyhow::ensure!(
        stated.len() == direct.reports.len(),
        "the staged run states {} result(s), and the direct run made {} inference(s)",
        stated.len(),
        direct.reports.len()
    );
    for (inference, (report, staged)) in direct.reports.iter().zip(&stated).enumerate() {
        if let Some(difference) = result_difference(&report.result, staged)? {
            anyhow::bail!(
                "inference {inference}: {difference}; direct and staged inference are not in \
                 parity"
            );
        }
    }
    anyhow::ensure!(
        direct.consensus == staged.consensus,
        "the staged run states another consensus than the direct run: {:?} and {:?}",
        staged.consensus,
        direct.consensus
    );
    Ok(CheckpointedRun {
        procedure_kind,
        manifest_path: prepared.manifest_path,
        pipeline_dir: staged.pipeline_dir,
        checkpoint_list_path,
        checkpoint_hashes_path,
        checkpoints: list.checkpoints.len(),
        results: stated,
        consensus: staged.consensus,
    })
}

/// The first field of an inference's result that a staged run states
/// otherwise than the direct run did, if there is one.
fn result_difference(direct: &InferenceResult, staged: &InferenceResult) -> Result<Option<String>> {
    if direct == staged {
        return Ok(None);
    }
    let direct = serde_json::to_value(direct)?;
    let staged = serde_json::to_value(staged)?;
    let fields = direct
        .as_object()
        .context("an inference's result is not a record")?;
    for (field, direct) in fields {
        let staged = &staged[field];
        if direct != staged {
            return Ok(Some(format!(
                "the staged run states `{field}` as {staged}, and the direct run as {direct}"
            )));
        }
    }
    Ok(Some(String::from(
        "the staged run's result is not the direct run's",
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

/// The function a staged run's preparation asks for the output of an
/// inference whose ids no one gave: the direct run of that one inference, on
/// the prompt an earlier inference's output makes when there is one.
pub fn direct_output<'a>(
    base_dir: &'a Path,
    specification_path: &'a Path,
    case_path: Option<&'a Path>,
) -> impl Fn(usize, Option<&InferenceResult>) -> Result<InferenceResult> + 'a {
    move |inference, relayed| {
        DirectInferenceExecutor
            .run(DirectInferenceConfig {
                base_dir: base_dir.to_path_buf(),
                specification_path: specification_path.to_path_buf(),
                case_path: case_path
                    .context("direct inference runs a case, and none is named")?
                    .to_path_buf(),
                inference,
                relayed: relayed.cloned(),
            })
            .context("direct inference to discover the claimed output failed")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_difference_names_the_first_field_that_differs() {
        let direct = InferenceResult {
            generated_token_count: 2,
            generated_token_ids: vec![7, 9],
            generated_text: String::from("ciao"),
            ..Default::default()
        };
        assert_eq!(result_difference(&direct, &direct.clone()).unwrap(), None);

        let mut staged = direct.clone();
        staged.claim_mismatches = 1;
        let difference = result_difference(&direct, &staged).unwrap().unwrap();
        assert!(
            difference.contains("`claim_mismatches` as 1") && difference.ends_with("as 0"),
            "{difference}"
        );
    }

    #[test]
    fn the_direct_output_of_no_case_is_an_error() {
        let discover = direct_output(Path::new("."), Path::new("specification.toml"), None);
        let error = discover(0, None).unwrap_err();
        assert!(format!("{error:#}").contains("none is named"));
    }
}
