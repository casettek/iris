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
        trace.configure(
            model.shape().num_hidden_layers,
            run_spec.tokens,
            run_prep::tokenizer_repeat_count(prepared_prompt.initial_pieces.len())?,
        )?;
        let prompt = model.prompt_inputs(&prepared_prompt)?;
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
            edge = inference_kernels::kernels::decode_select_token::run_decode_select_token_direct(
                inference_kernels::kernels::decode_select_token::DecodeSelectTokenDirectInputs {
                    logits: &logits,
                    prior: &edge,
                },
            )?;
            trace.record(&format!("decode_select_t{token_idx}"), &edge)?;
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
            has_selected: edge.has_selected,
            decode_position: edge.decode_position,
            token_id: edge.token_id,
            value: edge.value,
            generated_token_ids: edge.generated_token_ids.clone(),
        };
        let decoder = model.decoder_table()?;
        let output = inference_kernels::kernels::output_finalize::run_output_finalize_direct(
            inference_kernels::kernels::output_finalize::OutputFinalizeDirectInputs {
                edge: &output_edge,
                decoder: &decoder,
            },
        )?;
        trace.record("output_finalize", &output)?;
        let result = InferenceResult {
            generated_token_count: output.generated_token_count,
            generated_token_ids: output.generated_token_ids.iter().copied().collect(),
            generated_token_ids_sha256: output.generated_token_ids_sha256,
            generated_text: output.generated_text,
            stop_reason: output.stop_reason,
        };
        timings.push(phase_timing("output_finalize", finalize_started.elapsed()));

        trace.finish()?;
        Ok(InferenceRunReport {
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
    decode_select_token::input::PrefillLogits,
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
            .unwrap_or_else(|| empty_or_embedding.clone());
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
