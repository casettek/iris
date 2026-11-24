//! A `mistral-dense` model's logits against two references.
//!
//! The integer layer is not floating point, so it never agrees with a float
//! implementation to the last digit. This measures how far apart they are, on
//! one fixed token sequence so that a different choice at one step does not
//! hide every step after it.
//!
//! - **The same weights in `f64`** ([`float`]): the model the bundle commits
//!   to, computed with no rounding of activations. This is what the integer
//!   layer approximates, and the test holds it close.
//! - **Another implementation's logits**, read from a file. llama.cpp on the
//!   CPU rounds activations to eight bits inside every matrix product, so it
//!   is the noisier of the two; the comparison is reported and held loosely.
//!
//! `MISTRAL_MANIFEST` is an imported model's `manifest.json`.
//! `MISTRAL_REFERENCE` is a file of JSON lines, one per prompt:
//! `{"file": ..., "prompt_ids": [...], "steps": [{"id": ..., "top": [[id, logit], ...]}, ...]}`.
//! Step `k`'s `top` are the other implementation's highest logits after the
//! prompt and the first `k` output ids, highest first, and `id` is the token
//! it went on with.
//! `MISTRAL_FLOAT_MAX_TOKENS` (default 128) is the longest sequence the `f64`
//! pass is run for; it takes a few seconds per ten tokens.

use std::path::Path;

use direct_infer::detwgt::MmapDetwgt;
use direct_infer::generation::PrefillLogits;
use direct_infer::model::DirectInferenceModel;
use direct_infer::view_kernels::{run_mistral_range_view, score_prefill_logits_view};
use inference_artifacts::MistralShape;
use prefill_range::input::{pack_i32s, ActivationRow, ActivationSequence};
use raster::List;

/// The most an integer logit may differ from the same weights in `f64`, among
/// the `f64` pass's twenty highest.
const FLOAT_TOLERANCE: f64 = 0.05;
/// A step is a near tie when a reference's two best logits are closer than this.
const NEAR_TIE: f64 = 0.25;

/// The model in `f64`, from the bundle's own weights.
mod float {
    use super::*;

    pub struct Matrix {
        rows: usize,
        cols: usize,
        /// Exact: a block's half-precision scale times a signed byte fits `f32`.
        values: Vec<f32>,
    }

    pub fn matrix(weights: &MmapDetwgt, name: &str) -> Matrix {
        let view = weights.q8_matrix(name).unwrap();
        let (rows, cols) = (view.rows(), view.cols());
        let mut values = Vec::with_capacity(rows * cols);
        for row in 0..rows {
            for block in view
                .row(row)
                .unwrap()
                .chunks_exact(det_num::q8::BLOCK_BYTES)
            {
                let scale = det_num::q8::scale_q24(u16::from_le_bytes([block[0], block[1]]))
                    .expect("a finite scale") as f64
                    / f64::from(1_u32 << 24);
                values.extend(
                    block[2..]
                        .iter()
                        .map(|q| (scale * f64::from(*q as i8)) as f32),
                );
            }
        }
        Matrix { rows, cols, values }
    }

    /// `matrix · x` for every row of `xs`.
    pub fn apply(matrix: &Matrix, xs: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let threads = std::thread::available_parallelism().map_or(4, usize::from);
        let mut out = vec![vec![0.0; matrix.rows]; xs.len()];
        let chunk = xs.len().div_ceil(threads).max(1);
        std::thread::scope(|scope| {
            for (xs, out) in xs.chunks(chunk).zip(out.chunks_mut(chunk)) {
                scope.spawn(move || {
                    for (x, out) in xs.iter().zip(out) {
                        assert_eq!(x.len(), matrix.cols);
                        for (row, slot) in matrix.values.chunks_exact(matrix.cols).zip(out) {
                            *slot = row.iter().zip(x).map(|(w, x)| f64::from(*w) * x).sum();
                        }
                    }
                });
            }
        });
        out
    }

    fn rms_norm(x: &[f64], weight: &[f64], eps: f64) -> Vec<f64> {
        let mean = x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64;
        let scale = 1.0 / (mean + eps).sqrt();
        x.iter().zip(weight).map(|(v, w)| v * scale * w).collect()
    }

    fn q16(values: Vec<i32>) -> Vec<f64> {
        values
            .into_iter()
            .map(|v| f64::from(v) / 65_536.0)
            .collect()
    }

    fn rotate(row: &mut [f64], head_dim: usize, frequencies: &[f64], position: usize) {
        for head in row.chunks_exact_mut(head_dim) {
            for (pair, frequency) in head.chunks_exact_mut(2).zip(frequencies) {
                let (sin, cos) = (position as f64 * frequency).sin_cos();
                let (a, b) = (pair[0], pair[1]);
                pair[0] = a * cos - b * sin;
                pair[1] = a * sin + b * cos;
            }
        }
    }

    /// The final normed state of every position from `from` on.
    pub fn final_states(
        weights: &MmapDetwgt,
        shape: &MistralShape,
        ids: &[u32],
        from: usize,
    ) -> Vec<Vec<f64>> {
        let eps = shape.norm_eps as f64 / 4_294_967_296.0;
        let frequencies: Vec<f64> = shape
            .rope_inverse_frequencies
            .iter()
            .map(|bits| *bits as f64 / 4_294_967_296.0)
            .collect();
        let (heads, kv_heads, head_dim) = (
            shape.num_attention_heads as usize,
            shape.num_key_value_heads as usize,
            shape.head_dim as usize,
        );
        let group = heads / kv_heads;
        let embedding = weights
            .q8_matrix("model.language_model.embed_tokens.weight")
            .unwrap();
        let mut x: Vec<Vec<f64>> = ids
            .iter()
            .map(|id| {
                let mut row = Vec::with_capacity(embedding.cols());
                for block in embedding
                    .row(*id as usize)
                    .unwrap()
                    .chunks_exact(det_num::q8::BLOCK_BYTES)
                {
                    let scale = det_num::q8::scale_q24(u16::from_le_bytes([block[0], block[1]]))
                        .unwrap() as f64
                        / f64::from(1_u32 << 24);
                    row.extend(block[2..].iter().map(|q| scale * f64::from(*q as i8)));
                }
                row
            })
            .collect();
        for layer in 0..shape.num_hidden_layers {
            let at = |suffix: &str| format!("model.language_model.layers.{layer}.{suffix}");
            let norm = q16(weights.values(&at("input_layernorm.weight")).unwrap());
            let normed: Vec<Vec<f64>> = x.iter().map(|row| rms_norm(row, &norm, eps)).collect();
            let mut q = apply(&matrix(weights, &at("self_attn.q_proj.weight")), &normed);
            let mut k = apply(&matrix(weights, &at("self_attn.k_proj.weight")), &normed);
            let v = apply(&matrix(weights, &at("self_attn.v_proj.weight")), &normed);
            for (position, (q, k)) in q.iter_mut().zip(&mut k).enumerate() {
                rotate(q, head_dim, &frequencies, position);
                rotate(k, head_dim, &frequencies, position);
            }
            let contexts: Vec<Vec<f64>> = (0..ids.len())
                .map(|position| {
                    let mut context = vec![0.0; heads * head_dim];
                    for head in 0..heads {
                        let kv = (head / group) * head_dim..(head / group + 1) * head_dim;
                        let q_head = &q[position][head * head_dim..(head + 1) * head_dim];
                        let scores: Vec<f64> = k[..=position]
                            .iter()
                            .map(|key| {
                                q_head
                                    .iter()
                                    .zip(&key[kv.clone()])
                                    .map(|(a, b)| a * b)
                                    .sum::<f64>()
                                    / (head_dim as f64).sqrt()
                            })
                            .collect();
                        let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                        let exps: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
                        let total: f64 = exps.iter().sum();
                        for (weight, value) in exps.iter().zip(&v[..=position]) {
                            for (slot, lane) in context[head * head_dim..(head + 1) * head_dim]
                                .iter_mut()
                                .zip(&value[kv.clone()])
                            {
                                *slot += weight / total * lane;
                            }
                        }
                    }
                    context
                })
                .collect();
            let projected = apply(&matrix(weights, &at("self_attn.o_proj.weight")), &contexts);
            let attended: Vec<Vec<f64>> = x
                .iter()
                .zip(projected)
                .map(|(x, o)| x.iter().zip(o).map(|(a, b)| a + b).collect())
                .collect();
            let norm = q16(weights
                .values(&at("pre_feedforward_layernorm.weight"))
                .unwrap());
            let ff_in: Vec<Vec<f64>> = attended
                .iter()
                .map(|row| rms_norm(row, &norm, eps))
                .collect();
            let gate = apply(&matrix(weights, &at("mlp.gate_proj.weight")), &ff_in);
            let up = apply(&matrix(weights, &at("mlp.up_proj.weight")), &ff_in);
            let gated: Vec<Vec<f64>> = gate
                .iter()
                .zip(up)
                .map(|(gate, up)| {
                    gate.iter()
                        .zip(up)
                        .map(|(g, u)| g / (1.0 + (-g).exp()) * u)
                        .collect()
                })
                .collect();
            let down = apply(&matrix(weights, &at("mlp.down_proj.weight")), &gated);
            x = attended
                .iter()
                .zip(down)
                .map(|(a, d)| a.iter().zip(d).map(|(a, b)| a + b).collect())
                .collect();
        }
        let norm = q16(weights.values("model.language_model.norm.weight").unwrap());
        x[from..]
            .iter()
            .map(|row| rms_norm(row, &norm, eps))
            .collect()
    }
}

fn rows(model: &DirectInferenceModel, ids: &[u32], start_position: u32) -> ActivationSequence {
    ActivationSequence {
        model: Default::default(),
        rows: List::from(
            ids.iter()
                .map(|id| ActivationRow {
                    token_id: *id,
                    values: pack_i32s(&model.scaled_embedding_row(*id).unwrap()),
                })
                .collect::<Vec<_>>(),
        ),
        errors: List::new(),
        kv: List::new(),
        start_position,
    }
}

fn pass(
    model: &DirectInferenceModel,
    seed: ActivationSequence,
    prior: Option<&[ActivationSequence]>,
) -> (Vec<ActivationSequence>, PrefillLogits) {
    let mut current = seed;
    let mut outputs = Vec::new();
    for layer_idx in 0..model.num_layers() as usize {
        let layer = model.direct_mistral_layer(layer_idx).unwrap();
        let keys = prior.map_or(&[][..], |layers| layers[layer_idx].kv.as_slice());
        current = run_mistral_range_view(&current, &layer, keys).unwrap();
        outputs.push(current.clone());
    }
    let logits = score_prefill_logits_view(&current, &model.direct_final_head().unwrap()).unwrap();
    (outputs, logits)
}

fn best(logits: &[f64]) -> usize {
    let mut best = 0;
    for (id, value) in logits.iter().enumerate() {
        if *value > logits[best] {
            best = id;
        }
    }
    best
}

#[test]
#[ignore = "needs MISTRAL_MANIFEST and MISTRAL_REFERENCE"]
fn logits_follow_the_same_weights_in_floating_point() {
    let manifest_path = std::env::var("MISTRAL_MANIFEST").unwrap();
    let model = DirectInferenceModel::load(Path::new(&manifest_path)).unwrap();
    let shape = model.manifest.shape.mistral().unwrap().clone();
    let weights = MmapDetwgt::open(
        &Path::new(&manifest_path)
            .parent()
            .unwrap()
            .join(&model.manifest.bundle.model_detwgt_path),
    )
    .unwrap();
    let float_limit: usize = std::env::var("MISTRAL_FLOAT_MAX_TOKENS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(128);
    let reference = std::fs::read(std::env::var("MISTRAL_REFERENCE").unwrap()).unwrap();
    let mut failures = Vec::new();
    for line in String::from_utf8_lossy(&reference).lines() {
        let case: serde_json::Value = serde_json::from_str(line).unwrap();
        let name = case["file"].as_str().unwrap().rsplit('/').next().unwrap();
        let prompt: Vec<u32> = case["prompt_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|id| id.as_u64().unwrap() as u32)
            .collect();
        let steps = case["steps"].as_array().unwrap();
        let forced: Vec<u32> = steps
            .iter()
            .map(|s| s["id"].as_u64().unwrap() as u32)
            .collect();

        // The integer model's logits at every step, on the reference's sequence.
        let mut ours: Vec<Vec<f64>> = Vec::with_capacity(steps.len());
        let (mut layers, mut logits) = pass(&model, rows(&model, &prompt, 0), None);
        for index in 0..steps.len() {
            ours.push(
                logits
                    .logits
                    .iter()
                    .map(|entry| f64::from(entry.value) / 65_536.0)
                    .collect(),
            );
            if index + 1 == steps.len() {
                break;
            }
            let position = (prompt.len() + index) as u32;
            let next = pass(
                &model,
                rows(&model, &[forced[index]], position),
                Some(&layers),
            );
            layers = next.0;
            logits = next.1;
        }

        // Against the other implementation: its own best logits.
        let (mut agreed, mut worst, mut differing) = (0, 0.0_f64, Vec::new());
        for (index, step) in steps.iter().enumerate() {
            let top: Vec<(usize, f64)> = step["top"]
                .as_array()
                .unwrap()
                .iter()
                .map(|entry| {
                    (
                        entry[0].as_u64().unwrap() as usize,
                        entry[1].as_f64().unwrap(),
                    )
                })
                .collect();
            for (id, expected) in &top {
                worst = worst.max((ours[index][*id] - expected).abs());
            }
            if best(&ours[index]) == top[0].0 {
                agreed += 1;
            } else {
                differing.push(format!(
                    "step {index} (its margin {:.3})",
                    top[0].1 - top[1].1
                ));
            }
        }
        println!(
            "{name}: {} prompt tokens, {} steps. Other implementation: same token at {agreed}, \
             logits within {worst:.4}, differing at {differing:?}",
            prompt.len(),
            steps.len(),
        );

        // Against the same weights in f64.
        let sequence: Vec<u32> = prompt
            .iter()
            .chain(&forced[..forced.len() - 1])
            .copied()
            .collect();
        if sequence.len() > float_limit {
            println!(
                "{name}: {} tokens is past MISTRAL_FLOAT_MAX_TOKENS; no f64 pass",
                sequence.len()
            );
            continue;
        }
        let states = float::final_states(&weights, &shape, &sequence, prompt.len() - 1);
        let head = float::matrix(&weights, "model.language_model.embed_tokens.weight");
        let exact = float::apply(&head, &states);
        let (mut agreed, mut worst, mut sum, mut compared) = (0, 0.0_f64, 0.0_f64, 0_u32);
        for (index, (ours, exact)) in ours.iter().zip(&exact).enumerate() {
            let mut order: Vec<usize> = (0..exact.len()).collect();
            order.sort_by(|a, b| exact[*b].total_cmp(&exact[*a]));
            for id in &order[..20] {
                let error = ours[*id] - exact[*id];
                worst = worst.max(error.abs());
                sum += error;
                compared += 1;
            }
            let margin = exact[order[0]] - exact[order[1]];
            if best(ours) == order[0] {
                agreed += 1;
            } else if margin >= NEAR_TIE {
                failures.push(format!(
                    "{name} step {index}: chose {}, f64 chose {} by {margin:.3}",
                    best(ours),
                    order[0]
                ));
            } else {
                println!("{name} step {index}: differs from f64, whose margin is {margin:.4}");
            }
        }
        println!(
            "{name}: same weights in f64: same token at {agreed} of {}, logits within {worst:.4} \
             (mean {:+.5}) over each step's twenty highest",
            steps.len(),
            sum / f64::from(compared.max(1)),
        );
        if worst > FLOAT_TOLERANCE {
            failures.push(format!("{name}: a logit is {worst:.4} from f64"));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
