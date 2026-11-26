//! One inference of a procedure, run directly: the run's text and source
//! resolved from a specification and a case, one model's generation, and
//! what the result forms make of the output.

use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use direct_infer::model::DirectInferenceModel;
use direct_infer::{generate, generate_with_boundaries, DirectBoundaryRecord, GenerationInput};
use procedure_definition::{
    read_case, read_specification, InferStageTiming, InferenceResult, InferenceRunReport,
    InferenceTimings,
};

use crate::DirectInferenceConfig;

#[derive(Debug)]
pub struct DirectDiagnosticRunReport {
    pub report: InferenceRunReport,
    pub boundaries: Vec<DirectBoundaryRecord>,
}

#[derive(Debug, Default)]
pub struct DirectInferenceExecutor;

impl DirectInferenceExecutor {
    pub fn run(&self, config: DirectInferenceConfig) -> Result<InferenceResult> {
        Ok(self.run_with_report(config)?.result)
    }

    pub fn run_with_report(&self, config: DirectInferenceConfig) -> Result<InferenceRunReport> {
        Ok(self.run_inference(config, false)?.report)
    }

    pub fn run_with_diagnostics(
        &self,
        config: DirectInferenceConfig,
    ) -> Result<DirectDiagnosticRunReport> {
        self.run_inference(config, true)
    }

    fn run_inference(
        &self,
        config: DirectInferenceConfig,
        boundaries: bool,
    ) -> Result<DirectDiagnosticRunReport> {
        let infer_started = Instant::now();
        let specification_path = if config.specification_path.is_absolute() {
            config.specification_path.clone()
        } else {
            config.base_dir.join(&config.specification_path)
        };
        let specification = read_specification(&specification_path).with_context(|| {
            format!(
                "failed to load specification {}; create specification.toml with a `[[model]]` table, raw_prompt and max_new_tokens",
                specification_path.display()
            )
        })?;
        let specification_dir = specification_path
            .parent()
            .ok_or_else(|| anyhow::anyhow!("specification has no parent directory"))?;
        let case_path = if config.case_path.is_absolute() {
            config.case_path.clone()
        } else {
            config.base_dir.join(&config.case_path)
        };
        let case = read_case(&case_path).with_context(|| {
            format!(
                "failed to load case {}; create case.toml with a prompt, or the documents of the specification's source",
                case_path.display()
            )
        })?;
        let case_dir = case_path
            .parent()
            .ok_or_else(|| anyhow::anyhow!("case has no parent directory"))?;
        // One inference of the run: the same text, on this one of its models.
        let model_manifest = specification.model_manifest_of(config.inference)?;
        let model_manifest_path = if model_manifest.is_absolute() {
            model_manifest.to_path_buf()
        } else {
            specification_dir.join(model_manifest)
        };

        let load_started = Instant::now();
        let model = DirectInferenceModel::load(&model_manifest_path)
            .context("failed to load direct-infer model")?;
        let load_duration = load_started.elapsed();

        // The run's text, what it was built from and its forms, and the text
        // as this model's tokenizer first cuts it.
        let prepare_started = Instant::now();
        let prepared_prompt = run_prep::prepare_prompt(
            model.tokenizer(),
            &specification,
            case_dir,
            &case,
            model.manifest(),
            config.relayed.as_ref(),
        )?;
        // What the statement names as the text's source: zeros when the text
        // is taken as given.
        let source_sha256 = run_prep::source_id(prepared_prompt.source.as_ref());
        let input = GenerationInput {
            text: &prepared_prompt.resolved_prompt,
            pieces: &prepared_prompt.initial_pieces,
            source_sha256: &source_sha256,
            raw_prompt: specification.raw_prompt,
            max_new_tokens: specification.max_new_tokens,
        };
        let prepare_duration = prepare_started.elapsed();

        let (generation, boundaries) = if boundaries {
            generate_with_boundaries(&model, &input)?
        } else {
            (generate(&model, &input)?, Vec::new())
        };
        let output = generation.output;
        // What the chain's result stage states of this output, when it reads
        // this inference's last check. An inference it does not read has no
        // code, and no forms it was compared with.
        let (result_code, result_forms_sha256) = if specification
            .procedure_kind
            .result_input_of(config.inference)
            .is_some()
        {
            let matched =
                run_prep::source::run_forms(prepared_prompt.forms.iter(), &output.generated_text);
            (matched.code, run_prep::source::forms_id(&matched))
        } else {
            (0, String::new())
        };
        let result = InferenceResult {
            model_fault: output.model_fault,
            generated_token_count: output.generated_token_count,
            generated_token_ids: output.generated_token_ids.iter().copied().collect(),
            generated_token_ids_sha256: output.generated_token_ids_sha256,
            generated_text: output.generated_text,
            stop_reason: output.stop_reason,
            claim_mismatches: output.claim_mismatches,
            first_claim_mismatch_position: output.first_claim_mismatch_position,
            check_fault: output.check_fault,
            check_fault_position: output.check_fault_position,
            output_text_sha256: output.output_text_sha256,
            output_fault: output.output_fault,
            input_text_sha256: output.input_text_sha256,
            source_sha256: output.source_sha256,
            model_id: output.model_id,
            raw_prompt: output.raw_prompt,
            max_new_tokens: output.max_new_tokens,
            range_size: output.range_size,
            prompt_tokens: output.prompt_tokens,
            prompt_complete: output.prompt_complete,
            prompt_fault: output.prompt_fault,
            prompt_fault_position: output.prompt_fault_position,
            statement_sha256: procedure_definition::inference::lower_hex(
                output.claim.statement.as_slice(),
            ),
            valid: output.claim.valid,
            result_code,
            result_forms_sha256,
        };

        // The phases a direct run reports: the model's load, then the
        // generation's own, with preparing the prompt counted in the first.
        let mut stages = vec![phase_timing("load_direct_model", load_duration)];
        stages.extend(generation.phases.into_iter().map(|phase| {
            let prepared = if phase.phase == "prompt_prepare" {
                prepare_duration
            } else {
                Duration::ZERO
            };
            phase_timing(&phase.phase, phase.duration + prepared)
        }));
        Ok(DirectDiagnosticRunReport {
            report: InferenceRunReport {
                input_token_count: Some(generation.input_token_count),
                result,
                timings: InferenceTimings {
                    total_duration: infer_started.elapsed(),
                    stages,
                    aux_waves: Vec::new(),
                },
            },
            boundaries,
        })
    }
}

fn phase_timing(stage: &str, duration: Duration) -> InferStageTiming {
    InferStageTiming {
        stage: stage.to_string(),
        routine: String::from("direct-infer"),
        input_synthesis_duration: Duration::ZERO,
        input_load_duration: Duration::ZERO,
        kernel_duration: duration,
        encode_duration: Duration::ZERO,
        total_duration: duration,
    }
}
