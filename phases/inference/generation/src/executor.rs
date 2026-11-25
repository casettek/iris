//! One model's generation: from a prompt's pieces to the output and the
//! final record a chain publishes for the same run, byte for byte.
//!
//! This is the fast path that finds an output. It reads no specification and
//! no case: whoever runs a procedure prepares the prompt and gives it here.

use std::time::{Duration, Instant};

use anyhow::Result;
use prompt_prep::PreparedPiece;
use sequence_check::input::GeneratedOutput;

use crate::diagnostics::{DirectBoundaryRecord, DirectTrace};
use crate::model::DirectInferenceModel;
use crate::view_kernels::{
    run_decode_embed_view, run_input_embedding_view, run_ple_prepare_view, run_prefill_range_view,
    score_prefill_logits_view,
};

/// What a generation is given besides its model.
#[derive(Debug, Clone, Copy)]
pub struct GenerationInput<'a> {
    /// The run's text, which its statement names by hash.
    pub text: &'a str,
    /// The text as the model's tokenizer first cuts it, in the model's
    /// template unless the prompt is raw.
    pub pieces: &'a [PreparedPiece],
    /// What the statement names as the text's source: 64 zeros for none.
    pub source_sha256: &'a str,
    pub raw_prompt: bool,
    pub max_new_tokens: u32,
}

/// One phase of a generation, and how long it took.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhaseTiming {
    pub phase: String,
    pub duration: Duration,
}

/// What a generation produced.
#[derive(Debug)]
pub struct Generation {
    pub output: GeneratedOutput,
    /// The model's input length after templating and tokenization.
    pub input_token_count: usize,
    pub phases: Vec<PhaseTiming>,
}

pub fn generate(model: &DirectInferenceModel, input: &GenerationInput<'_>) -> Result<Generation> {
    let mut trace = DirectTrace::new(false)?;
    generate_traced(model, input, &mut trace)
}

/// The same generation, with every boundary it crosses kept as a stage would
/// publish it.
pub fn generate_with_boundaries(
    model: &DirectInferenceModel,
    input: &GenerationInput<'_>,
) -> Result<(Generation, Vec<DirectBoundaryRecord>)> {
    let mut trace = DirectTrace::new(true)?;
    let generation = generate_traced(model, input, &mut trace)?;
    Ok((generation, trace.records))
}

fn generate_traced(
    model: &DirectInferenceModel,
    input: &GenerationInput<'_>,
    trace: &mut DirectTrace,
) -> Result<Generation> {
    let mut timings = Vec::new();
    let prompt_started = Instant::now();
    trace.configure(
        model.num_layers(),
        input.max_new_tokens,
        model.family().has_auxiliary_stages(),
    );
    let prompt = model.prompt_inputs(input.pieces)?;
    // Direct inference is the independent numerical oracle; construct its
    // diagnostic statement/history from the actual prompt and selected IDs.
    let prompt_ranges = model.prompt_ranges(input.pieces);
    // What the statement names as the text's source: zeros when the text
    // is taken as given.
    let source_sha256 = input.source_sha256.to_string();
    let execution_context = |generated: Option<&[u32]>, claimed_tokens: usize| {
        let mut context = crate::view_kernels::bound_model_context(model);
        context.bind_statement(
            &sequence_check::sha256_hex(input.text.as_bytes()),
            &source_sha256,
            input.raw_prompt,
            input.max_new_tokens,
            model.manifest().range_size,
        );
        // The schedule the chain runs this output under, so the two final
        // records agree byte for byte.
        if let Some(prompt_ranges) = prompt_ranges {
            let claimed_ranges = claimed_tokens.div_ceil(model.manifest().range_size as usize);
            context.bind_schedule(model_identity::statement_core::StageSchedule {
                prompt_ranges,
                claimed_ranges: u32::try_from(claimed_ranges).unwrap_or(u32::MAX),
            });
        }
        if let Some(mut e) = context.execution_context() {
            for id in prompt.token_ids.iter() {
                context.require_execution(e.end.append(*id).is_ok());
            }
            e.prompt = Some(e.end);
            e.phase = model_identity::statement_core::PREPARED;
            // The steps the chain's last prompt range and last check record.
            e.step = model_identity::statement_core::Step::tokens(
                e.schedule.prompt_ranges.saturating_sub(1),
            );
            if let Some(ids) = generated {
                e.before = e.end;
                e.claimed = true;
                for id in ids {
                    context.require_execution(e.end.append(*id).is_ok());
                }
                e.phase = model_identity::statement_core::CHECKED;
                e.step = model_identity::statement_core::Step::check(
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
    let empty_or_embedding = run_input_embedding_view(model, &prompt)?;
    trace.record("input_embedding", &empty_or_embedding)?;
    timings.push(phase_timing("input_embedding", embedding_started.elapsed()));

    let prefill_started = Instant::now();
    let (mut prior_layers, mut logits) = run_layers_and_logits(
        model,
        empty_or_embedding.clone(),
        empty_or_embedding.clone(),
        None,
        &mut timings,
        "prefill",
        trace,
    )?;
    timings.push(phase_timing("prefill", prefill_started.elapsed()));

    let decode_started = Instant::now();
    let mut edge = inference_kernels::kernels::origin::empty_decode_edge();
    trace.record("decode_init", &edge)?;
    for token_idx in 0..input.max_new_tokens {
        edge = crate::generation::select_next_token(&logits, &edge)?;
        edge.model = execution_context(
            Some(edge.generated_token_ids.as_slice()),
            edge.generated_token_ids.len(),
        );
        trace.record(&format!("decode_select_t{token_idx}"), &edge)?;
        if model.manifest().eos_token_ids.contains(&edge.token_id) {
            // Keep EOS in the result, but never run another decode pass.
            trace.stop_at_eos(model.num_layers(), token_idx + 1);
            break;
        }
        if token_idx + 1 == input.max_new_tokens {
            break;
        }
        let decode_seed = run_decode_embed_view(model, &edge)?;
        trace.record(&format!("decode_embed_t{token_idx}"), &decode_seed)?;
        let (next_prior_layers, next_logits) = run_layers_and_logits(
            model,
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
    let output_edge = edge.clone();
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
        sequence_check::decoder_witness(&decoder).map_err(|f| anyhow::anyhow!("{f:?}"))?;
    let finalize_inputs = inference_kernels::kernels::finalize::FinalizeInputs {
        decoder: &decoder,
        descriptor: &descriptor,
        witness: &witness,
        // The record the chain publishes for this run: a direct result
        // and an honest claim's final record are the same bytes.
        context: &sequence_check::prompt::PromptContext {
            model: execution_context(None, output_edge.generated_token_ids.len()),
            input_text_sha256: sequence_check::sha256_hex(input.text.as_bytes()),
            source_sha256: source_sha256.clone(),
            model_id: model.manifest().model_id.clone(),
            raw_prompt: input.raw_prompt,
            max_new_tokens: input.max_new_tokens,
            range_size: model.manifest().range_size,
            next_piece: input.pieces.len() as u64,
            prompt_tokens: prompt.token_ids.len() as u32,
            complete: true,
            fault: 0,
            fault_position: 0,
        },
    };
    let final_edge = if model.manifest().model_commitments.is_empty() {
        inference_kernels::kernels::finalize::run_output_finalize_discovery(
            &output_edge,
            finalize_inputs.decoder,
            finalize_inputs.context,
        )?
    } else {
        inference_kernels::kernels::finalize::run_output_finalize_direct(
            &output_edge,
            &finalize_inputs,
        )?
    };

    // The run's last boundary is its final edge with the output record.
    trace.record("output_finalize", &final_edge)?;
    timings.push(phase_timing("output_finalize", finalize_started.elapsed()));

    trace.finish()?;
    Ok(Generation {
        output: sequence_check::input::generated_output(&final_edge),
        input_token_count: prompt.token_ids.len(),
        phases: timings,
    })
}

fn run_layers_and_logits(
    model: &DirectInferenceModel,
    seed: prefill_range::input::ActivationSequence,
    empty_or_embedding: prefill_range::input::ActivationSequence,
    prior_layers: Option<&[prefill_range::input::ActivationSequence]>,
    timings: &mut Vec<PhaseTiming>,
    label: &str,
    trace: &mut DirectTrace,
) -> Result<(
    Vec<prefill_range::input::ActivationSequence>,
    crate::generation::PrefillLogits,
)> {
    if model.family() == model_manifest::ModelFamily::MistralDense {
        return run_mistral_layers_and_logits(model, seed, prior_layers, timings, label, trace);
    }
    let ple_source = seed.clone();
    let mut current = seed;
    let mut layer_outputs = Vec::with_capacity(model.num_layers() as usize);
    for layer_idx in 0..model.num_layers() as usize {
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

/// A `mistral-dense` pass: every layer attends to the whole prefix and takes
/// nothing but the layer before it and its own earlier keys.
fn run_mistral_layers_and_logits(
    model: &DirectInferenceModel,
    seed: prefill_range::input::ActivationSequence,
    prior_layers: Option<&[prefill_range::input::ActivationSequence]>,
    timings: &mut Vec<PhaseTiming>,
    label: &str,
    trace: &mut DirectTrace,
) -> Result<(
    Vec<prefill_range::input::ActivationSequence>,
    crate::generation::PrefillLogits,
)> {
    let mut current = seed;
    let mut layer_outputs = Vec::with_capacity(model.num_layers() as usize);
    for layer_idx in 0..model.num_layers() as usize {
        let range_started = Instant::now();
        let layer = model.direct_mistral_layer(layer_idx)?;
        let prior_keys = prior_layers
            .and_then(|layers| layers.get(layer_idx))
            .map_or(&[][..], |sequence| sequence.kv.as_slice());
        current = crate::view_kernels::run_mistral_range_view(&current, &layer, prior_keys)?;
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

fn phase_timing(phase: &str, duration: Duration) -> PhaseTiming {
    PhaseTiming {
        phase: phase.to_string(),
        duration,
    }
}
