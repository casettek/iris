use std::path::PathBuf;

use crate::{DirectInferenceConfig, DirectInferenceExecutor};
use anyhow::{Context, Result};
use procedure_definition::{ConsensusOutcome, ProcedureKind, ResultKind};
use staged_infer::InferenceRunReport;

/// A procedure kind run directly: one report per inference, in order, and what a
/// consensus states of the two.
pub struct InferRun {
    pub procedure_kind: ProcedureKind,
    pub reports: Vec<InferenceRunReport>,
    pub consensus: Option<ConsensusOutcome>,
}

pub fn run_infer(specification_path: PathBuf, case_path: PathBuf) -> Result<InferRun> {
    let mut config =
        DirectInferenceConfig::from_current_dir().context("failed to build infer configuration")?;
    config.specification_path = specification_path;
    config.case_path = case_path;
    let specification_path = crate::absolute(&config.base_dir, &config.specification_path);
    let specification = procedure_definition::read_specification(&specification_path).with_context(|| {
        format!(
            "failed to load specification {}; create specification.toml with a `[[model]]` table, raw_prompt and max_new_tokens",
            specification_path.display()
        )
    })?;
    let procedure_kind = specification.procedure_kind;
    // One inference after the other: each loads its own model.
    let mut reports = Vec::new();
    for inference in 0..procedure_kind.inferences() {
        config.inference = inference;
        reports.push(
            DirectInferenceExecutor
                .run_with_report(config.clone())
                .with_context(|| {
                    format!("inference {inference} of the `{procedure_kind}` procedure kind")
                })?,
        );
    }
    let consensus = match (procedure_kind.result(), reports.as_slice()) {
        (ResultKind::OneOutput, _) => None,
        (ResultKind::Agreement, [first, second]) => {
            let forms = specification.forms();
            Some(ConsensusOutcome::of(
                &forms,
                &first.result,
                second.result.clone(),
            ))
        }
        (ResultKind::Agreement, _) => anyhow::bail!("a consensus is of two inferences"),
    };
    Ok(InferRun {
        procedure_kind,
        reports,
        consensus,
    })
}
