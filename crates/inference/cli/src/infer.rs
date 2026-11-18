use std::path::PathBuf;

use anyhow::{Context, Result};
use direct_infer::{DirectInferenceConfig, DirectInferenceExecutor};
use inference_artifacts::{ConsensusOutcome, Pipeline};
use staged_infer::InferenceRunReport;

/// A pipeline run directly: one report per inference, in order, and what a
/// consensus states of the two.
pub struct InferRun {
    pub pipeline: Pipeline,
    pub reports: Vec<InferenceRunReport>,
    pub consensus: Option<ConsensusOutcome>,
}

pub fn run_infer(run_spec_path: PathBuf) -> Result<InferRun> {
    let mut config =
        DirectInferenceConfig::from_current_dir().context("failed to build infer configuration")?;
    config.run_spec_path = run_spec_path;
    let run_spec_path = crate::absolute(&config.base_dir, &config.run_spec_path);
    let run_spec = inference_artifacts::read_run_spec(&run_spec_path).with_context(|| {
        format!(
            "failed to load run spec {}; create inference.toml with model_manifest, prompt/prompt_file, raw_prompt, and tokens",
            run_spec_path.display()
        )
    })?;
    let pipeline = run_spec.pipeline;
    // One inference after the other: each loads its own model.
    let mut reports = Vec::new();
    for inference in 0..pipeline.inferences() {
        config.inference = inference;
        reports.push(
            DirectInferenceExecutor
                .run_with_report(config.clone())
                .with_context(|| format!("inference {inference} of the `{pipeline}` pipeline"))?,
        );
    }
    let consensus = match (pipeline, reports.as_slice()) {
        (Pipeline::Single, _) => None,
        (Pipeline::Consensus, [first, second]) => {
            let forms = crate::source::run_source_of(&config.run_spec_path)?
                .map(|source| source.recipe.forms())
                .unwrap_or_default();
            Some(ConsensusOutcome::of(
                &forms,
                &first.result,
                second.result.clone(),
            ))
        }
        (Pipeline::Consensus, _) => anyhow::bail!("a consensus is of two inferences"),
    };
    Ok(InferRun {
        pipeline,
        reports,
        consensus,
    })
}
