use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use inference_artifacts::{
    read_run_spec, InferStageTiming, InferenceResult, InferenceRunReport, InferenceTimings,
};

use crate::diagnostics::{DirectDiagnosticRunReport, DirectTrace};
use crate::model::DirectInferenceModel;
use crate::view_kernels::{
    run_decode_embed_view, run_input_embedding_view, run_ple_prepare_view, run_prefill_range_view,
    score_prefill_logits_view,
};
use crate::DirectInferenceConfig;

#[derive(Debug, Default)]
pub struct DirectInferenceExecutor;

impl DirectInferenceExecutor {
    pub fn run(&self, config: DirectInferenceConfig) -> Result<InferenceResult> {
        Ok(self.run_with_report(config)?.result)
    }

    pub fn run_with_report(&self, config: DirectInferenceConfig) -> Result<InferenceRunReport> {
        let mut trace = DirectTrace::new(false)?;
        self.run_traced(config, &mut trace)
    }

    pub fn run_with_diagnostics(
        &self,
        config: DirectInferenceConfig,
    ) -> Result<DirectDiagnosticRunReport> {
        let mut trace = DirectTrace::new(true)?;
        let report = self.run_traced(config, &mut trace)?;
        Ok(DirectDiagnosticRunReport {
            report,
            boundaries: trace.records,
        })
    }

    fn run_traced(
        &self,
        config: DirectInferenceConfig,
        trace: &mut DirectTrace,
    ) -> Result<InferenceRunReport> {
        let infer_started = Instant::now();
        let mut timings = Vec::new();
        let run_spec_path = if config.run_spec_path.is_absolute() {
            config.run_spec_path.clone()
        } else {
            config.base_dir.join(&config.run_spec_path)
        };
        let run_spec = read_run_spec(&run_spec_path).with_context(|| {
            format!(
                "failed to load run spec {}; create inference.toml with model_manifest, prompt/prompt_file, raw_prompt, and tokens",
                run_spec_path.display()
            )
        })?;
        let run_spec_dir = run_spec_path
            .parent()
            .ok_or_else(|| anyhow::anyhow!("run spec has no parent directory"))?;
        let model_manifest_path = if run_spec.model_manifest.is_absolute() {
            run_spec.model_manifest.clone()
        } else {
            run_spec_dir.join(&run_spec.model_manifest)
        };

        let load_started = Instant::now();
        let model = DirectInferenceModel::load(&model_manifest_path)
            .context("failed to load direct-infer model")?;
        timings.push(phase_timing("load_direct_model", load_started.elapsed()));

        let prompt_started = Instant::now();
        let prepared_prompt = model.prepare_prompt(run_spec_dir, &run_spec)?;
        trace.configure(model.shape().num_hidden_layers, run_spec.tokens);
        let prompt = model.prompt_inputs(&prepared_prompt)?;
        // Direct inference is the independent numerical oracle; construct its
        // diagnostic statement/history from the actual prompt and selected IDs.
        let prompt_ranges = model.prompt_ranges(&prepared_prompt);
        let execution_context = |generated: Option<&[u32]>, claimed_tokens: usize| {
            let mut context = crate::view_kernels::bound_model_context(&model);
            context.bind_statement(
                &output_finalize::sha256_hex(prepared_prompt.resolved_prompt.as_bytes()),
                run_spec.raw_prompt,
                run_spec.tokens,
                model.manifest().range_size,
            );
            // The schedule the chain runs this output under, so the two final
            // records agree byte for byte.
            if let Some(prompt_ranges) = prompt_ranges {
                let claimed_ranges = claimed_tokens.div_ceil(model.manifest().range_size as usize);
                context.bind_schedule(model_identity::pipeline_core::StageSchedule {
                    prompt_ranges,
                    claimed_ranges: u32::try_from(claimed_ranges).unwrap_or(u32::MAX),
                });
            }
            if let Some(mut e) = context.execution_context() {
                for id in prompt.token_ids.iter() {
                    context.require_execution(e.end.append(*id).is_ok());
                }
                e.prompt = Some(e.end);
                e.phase = model_identity::pipeline_core::PREPARED;
                // The steps the chain's last prompt range and last check record.
                e.step = model_identity::pipeline_core::Step::tokens(
                    e.schedule.prompt_ranges.saturating_sub(1),
                );
                if let Some(ids) = generated {
                    e.before = e.end;
                    e.claimed = true;
                    for id in ids {
                        context.require_execution(e.end.append(*id).is_ok());
                    }
                    e.phase = model_identity::pipeline_core::CHECKED;
                    e.step = model_identity::pipeline_core::Step::check(
                        e.schedule.claimed_ranges.saturating_sub(1),
                    );
                    context.completed_layers = context.layers;
                    context.producer = model_identity::context::DECODE;
                }
                context.set_execution(&e);
            }
            context
        };
        trace.record("prompt_prepare", &prompt)?;
        timings.push(phase_timing("prompt_prepare", prompt_started.elapsed()));

        let embedding_started = Instant::now();
        let empty_or_embedding = run_input_embedding_view(&model, &prompt)?;
        trace.record("input_embedding", &empty_or_embedding)?;
        timings.push(phase_timing("input_embedding", embedding_started.elapsed()));

        let prefill_started = Instant::now();
        let (mut prior_layers, mut logits) = run_layers_and_logits(
            &model,
            empty_or_embedding.clone(),
            empty_or_embedding.clone(),
            None,
            &mut timings,
            "prefill",
            trace,
        )?;
        timings.push(phase_timing("prefill", prefill_started.elapsed()));

        let decode_started = Instant::now();
        let mut edge = inference_kernels::kernels::decode_init::run_decode_init_direct(
            inference_kernels::kernels::decode_init::DecodeInitDirectInputs,
        )?;
        trace.record("decode_init", &edge)?;
        for token_idx in 0..run_spec.tokens {
            edge = crate::generation::select_next_token(&logits, &edge)?;
            edge.model = execution_context(
                Some(edge.generated_token_ids.as_slice()),
                edge.generated_token_ids.len(),
            );
            trace.record(&format!("decode_select_t{token_idx}"), &edge)?;
            if prepared_prompt.eos_token_ids.contains(&edge.token_id) {
                // Keep EOS in the result, but never run another decode pass.
                trace.stop_at_eos(model.shape().num_hidden_layers, token_idx + 1);
                break;
            }
            if token_idx + 1 == run_spec.tokens {
                break;
            }
            let decode_seed = run_decode_embed_view(&model, &edge)?;
            trace.record(&format!("decode_embed_t{token_idx}"), &decode_seed)?;
            let (next_prior_layers, next_logits) = run_layers_and_logits(
                &model,
                decode_seed,
                empty_or_embedding.clone(),
                Some(&prior_layers),
                &mut timings,
                &format!("decode_t{token_idx}"),
                trace,
            )?;
            prior_layers = next_prior_layers;
            logits = next_logits;
        }
        timings.push(phase_timing("decode", decode_started.elapsed()));

        let finalize_started = Instant::now();
        let output_edge = output_finalize::input::DecodeEdge {
            model: edge.model.clone(),
            has_selected: edge.has_selected,
            decode_position: edge.decode_position,
            token_id: edge.token_id,
            value: edge.value,
            claim_mismatches: edge.claim_mismatches,
            first_claim_mismatch_position: edge.first_claim_mismatch_position,
            check_fault: edge.check_fault,
            check_fault_position: edge.check_fault_position,
            generated_token_ids: edge.generated_token_ids.clone(),
        };
        let decoder = model.decoder_table()?;
        let descriptor = raster::BytesPage::__from_parts(
            0,
            0,
            if model.manifest().model_commitments.is_empty() {
                Vec::new()
            } else {
                model
                    .manifest()
                    .descriptor()?
                    .encode()
                    .map_err(|f| anyhow::anyhow!("{f:?}"))?
            },
        );
        let witness =
            output_finalize::decoder_witness(&decoder).map_err(|f| anyhow::anyhow!("{f:?}"))?;
        let finalize_inputs =
            inference_kernels::kernels::output_finalize::OutputFinalizeDirectInputs {
                edge: &output_edge,
                decoder: &decoder,
                descriptor: &descriptor,
                witness: &witness,
                // The record the chain publishes for this run: a direct result
                // and an honest claim's final record are the same bytes.
                context: &output_finalize::prompt::PromptContext {
                    model: execution_context(None, output_edge.generated_token_ids.len()),
                    input_text_sha256: output_finalize::sha256_hex(
                        prepared_prompt.resolved_prompt.as_bytes(),
                    ),
                    model_id: model.manifest().model_id.clone(),
                    raw_prompt: run_spec.raw_prompt,
                    max_new_tokens: run_spec.tokens,
                    range_size: model.manifest().range_size,
                    next_piece: prepared_prompt.initial_pieces.len() as u64,
                    prompt_tokens: prompt.token_ids.len() as u32,
                    complete: true,
                    fault: 0,
                    fault_position: 0,
                },
            };
        let output = if model.manifest().model_commitments.is_empty() {
            inference_kernels::kernels::output_finalize::run_output_finalize_discovery(
                finalize_inputs.edge,
                finalize_inputs.decoder,
                finalize_inputs.context,
            )?
        } else {
            inference_kernels::kernels::output_finalize::run_output_finalize_direct(
                finalize_inputs,
            )?
        };

        trace.record("output_finalize", &output)?;
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
            model_id: output.model_id,
            raw_prompt: output.raw_prompt,
            max_new_tokens: output.max_new_tokens,
            range_size: output.range_size,
            prompt_tokens: output.prompt_tokens,
            prompt_complete: output.prompt_complete,
            prompt_fault: output.prompt_fault,
            prompt_fault_position: output.prompt_fault_position,
        };
        timings.push(phase_timing("output_finalize", finalize_started.elapsed()));

        trace.finish()?;
        Ok(InferenceRunReport {
            input_token_count: Some(prompt.token_ids.len()),
            result,
            timings: InferenceTimings {
                total_duration: infer_started.elapsed(),
                stages: timings,
                aux_waves: Vec::new(),
            },
        })
    }
}

fn run_layers_and_logits(
    model: &DirectInferenceModel,
    seed: prefill_range::input::ActivationSequence,
    empty_or_embedding: prefill_range::input::ActivationSequence,
    prior_layers: Option<&[prefill_range::input::ActivationSequence]>,
    timings: &mut Vec<InferStageTiming>,
    label: &str,
    trace: &mut DirectTrace,
) -> Result<(
    Vec<prefill_range::input::ActivationSequence>,
    crate::generation::PrefillLogits,
)> {
    let ple_source = seed.clone();
    let mut current = seed;
    let mut layer_outputs = Vec::with_capacity(model.shape().num_hidden_layers as usize);
    for layer_idx in 0..model.shape().num_hidden_layers as usize {
        let aux_started = Instant::now();
        let ple_layer = model.direct_ple_layer(layer_idx)?;
        let ple = run_ple_prepare_view(&ple_source, &ple_layer)?;
        trace.record(&aux_stage_name(label, layer_idx), &ple)?;
        timings.push(phase_timing(
            &format!("{label}_prefill_prepare_aux_l{layer_idx}"),
            aux_started.elapsed(),
        ));

        let range_started = Instant::now();
        let layer = model.direct_transformer_layer(layer_idx)?;
        let prior_kv = prior_layers
            .and_then(|layers| layers.get(layer_idx))
            .cloned()
            .unwrap_or_else(|| {
                let mut empty = empty_or_embedding.clone();
                empty.model = Default::default();
                empty
            });
        let donor_a_kv = donor_input(
            &layer_outputs,
            layer.params.donor_a_layer,
            &empty_or_embedding,
        );
        let donor_b_kv = donor_input(
            &layer_outputs,
            layer.params.donor_b_layer,
            &empty_or_embedding,
        );
        current =
            run_prefill_range_view(&current, &layer, &prior_kv, &donor_a_kv, &donor_b_kv, &ple)?;
        trace.record(&range_stage_name(label, layer_idx), &current)?;
        layer_outputs.push(current.clone());
        timings.push(phase_timing(
            &format!("{label}_prefill_range_l{layer_idx}"),
            range_started.elapsed(),
        ));
    }

    let finalize_started = Instant::now();
    let head = model.direct_final_head()?;
    let logits = score_prefill_logits_view(&current, &head)?;
    trace.record(&finalize_stage_name(label), &logits)?;
    timings.push(phase_timing(
        &format!("{label}_prefill_finalize"),
        finalize_started.elapsed(),
    ));
    Ok((layer_outputs, logits))
}

fn donor_input(
    layers: &[prefill_range::input::ActivationSequence],
    donor_layer: i32,
    fallback: &prefill_range::input::ActivationSequence,
) -> prefill_range::input::ActivationSequence {
    if donor_layer < 0 {
        return fallback.clone();
    }
    layers
        .get(donor_layer as usize)
        .cloned()
        .unwrap_or_else(|| fallback.clone())
}

fn aux_stage_name(label: &str, layer_idx: usize) -> String {
    if label == "prefill" {
        format!("prefill_prepare_aux_l{layer_idx}")
    } else if let Some(token_idx) = label.strip_prefix("decode_t") {
        format!("decode_aux_t{token_idx}_l{layer_idx}")
    } else {
        format!("{label}_prefill_prepare_aux_l{layer_idx}")
    }
}

fn range_stage_name(label: &str, layer_idx: usize) -> String {
    if label == "prefill" {
        format!("prefill_range_l{layer_idx}")
    } else if let Some(token_idx) = label.strip_prefix("decode_t") {
        format!("decode_range_t{token_idx}_l{layer_idx}")
    } else {
        format!("{label}_prefill_range_l{layer_idx}")
    }
}

fn finalize_stage_name(label: &str) -> String {
    if label == "prefill" {
        String::from("prefill_finalize")
    } else if let Some(token_idx) = label.strip_prefix("decode_t") {
        format!("decode_finalize_t{token_idx}")
    } else {
        format!("{label}_prefill_finalize")
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direct_infer_uses_direct_manifest_by_default() {
        let config = DirectInferenceConfig::from_current_dir().expect("config should build");
        assert_eq!(
            config.run_spec_path,
            config
                .base_dir
                .join(inference_artifacts::INFERENCE_RUN_SPEC_TOML)
        );
    }
}
