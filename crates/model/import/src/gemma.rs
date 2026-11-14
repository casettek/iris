//! The Gemma dense family: what the importer knows about its config, its
//! tensors and its per-layer inputs.
//!
//! Everything here is specific to models whose layers are Gemma's (sliding and
//! full attention with two RoPE bases, K/V sharing, a gated-GELU feed-forward
//! block, per-layer embedding inputs and a soft-capped head). The parent module
//! is the part every family shares: the command line, the artifact layout,
//! paging, commitments, the chain template and the manifest.
//!
//! A second family gets its own module beside this one, and a variant of
//! `inference_artifacts::ModelFamily`.

use super::*;

pub(super) fn as_f64(value: &serde_json::Value) -> Option<f64> {
    value.as_f64()
}

/// One key out of `rope_parameters.<attention_type>`, which is where Gemma 3n
/// puts its per-family RoPE settings.
pub(super) fn rope_param(text: &serde_json::Value, attention_type: &str, key: &str) -> Option<f64> {
    text.get("rope_parameters")?
        .get(attention_type)?
        .get(key)
        .and_then(as_f64)
}

/// The shapes every stage needs, read from `config.json`.
pub(super) struct Shape {
    pub(super) hidden: usize,
    pub(super) ffn: usize,
    pub(super) layers: usize,
    pub(super) heads: usize,
    pub(super) kv_heads: usize,
    pub(super) head_dim: usize,
    /// Head dim of the *full-attention* ("global") layers, which is not the
    /// same as `head_dim` on Gemma 3n: E2B declares `head_dim: 256` for the
    /// sliding layers and `global_head_dim: 512` for the full ones, and the
    /// weights follow — a full layer's `q_proj` is [heads · 512, hidden] and
    /// its `q_norm` has 512 values. Defaults to `head_dim` for models that
    /// declare no such split.
    pub(super) global_head_dim: usize,
    /// RoPE base per attention type, from `rope_parameters`. Gemma 3n gives the
    /// two families different bases — 10 000 local, 1 000 000 global — so a
    /// single `rope_theta` would silently detune half the layers.
    pub(super) rope_base_sliding: f64,
    pub(super) rope_base_full: f64,
    /// Fraction of a full-attention head that rotates (`0.25` here). Sliding
    /// layers rotate the whole head.
    pub(super) full_partial_rotary_factor: f64,
    /// How many trailing layers borrow their K/V from an earlier donor.
    pub(super) num_kv_shared_layers: usize,
    pub(super) vocab: usize,
    pub(super) ple_width: usize,
    pub(super) sliding_window: u32,
    pub(super) layer_types: Vec<String>,
    pub(super) norm_eps: i64,
}

impl Shape {
    pub(super) fn is_sliding(&self, idx: usize) -> bool {
        matches!(
            self.layer_types.get(idx).map(String::as_str),
            Some("sliding_attention")
        )
    }

    /// Head dim of layer `idx` — `global_head_dim` for a full-attention layer,
    /// `head_dim` for a sliding one.
    pub(super) fn head_dim_at(&self, idx: usize) -> usize {
        if self.is_sliding(idx) {
            self.head_dim
        } else {
            self.global_head_dim
        }
    }

    /// The layer whose K/V cache layer `idx` borrows, or `-1` if it computes
    /// its own.
    ///
    /// Mirrors `casettek/.../shared/model/gemma/io.rs:171` exactly: sharing
    /// starts at `num_hidden_layers - num_kv_shared_layers`, and a sharing
    /// layer's donor is the **last layer before that boundary with the same
    /// attention type**. For this model that resolves to layer 13 for every
    /// sliding layer and layer 14 for every full one — 20 layers in all, which
    /// is the count the config declares.
    pub(super) fn kv_donor_layer(&self, idx: usize) -> i32 {
        let first_shared = self.layers.saturating_sub(self.num_kv_shared_layers);
        if self.num_kv_shared_layers == 0 || idx < first_shared {
            return -1;
        }
        let Some(attention_type) = self.layer_types.get(idx) else {
            return -1;
        };
        self.layer_types[..first_shared]
            .iter()
            .rposition(|candidate| candidate == attention_type)
            .map(|donor| donor as i32)
            .unwrap_or(-1)
    }

    /// RoPE parameters for layer `idx`: `(base_bits, rotary_dim, freq_base_dim)`.
    ///
    /// `base_bits` is the base as `Acc` (Q32.32) bits. Both bases this model
    /// declares are exact integers, so the encoding is a plain shift — asserted
    /// against local `f32_to_acc` in `det-num/tests/equivalence.rs`.
    ///
    /// `freq_base_dim` is the layer's `head_dim`, which is *not* `rotary_dim`
    /// once the rotation is partial: a full-attention head is 512 wide and
    /// rotates its first 128 lanes against a ladder built over 512.
    pub(super) fn rope_at(&self, idx: usize) -> (i64, u32, u32) {
        let head_dim = self.head_dim_at(idx);
        let (base, rotary_dim) = if self.is_sliding(idx) {
            (self.rope_base_sliding, head_dim)
        } else {
            (
                self.rope_base_full,
                (head_dim as f64 * self.full_partial_rotary_factor) as usize,
            )
        };
        ((base as i64) << 32, rotary_dim as u32, head_dim as u32)
    }
}

impl Shape {
    pub(super) fn from_config(text: &serde_json::Value) -> Result<Self, Box<dyn Error>> {
        let usize_at = |key: &str| -> Result<usize, Box<dyn Error>> {
            text.get(key)
                .and_then(serde_json::Value::as_u64)
                .map(|value| value as usize)
                .ok_or_else(|| format!("config.json text_config has no {key}").into())
        };
        let layer_types = text
            .get("layer_types")
            .and_then(serde_json::Value::as_array)
            .map(|types| {
                types
                    .iter()
                    .filter_map(|value| value.as_str().map(str::to_string))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        // The MLP width is not in this config; it is the gate projection's rows.
        Ok(Self {
            hidden: usize_at("hidden_size")?,
            ffn: 0,
            layers: usize_at("num_hidden_layers")?,
            heads: usize_at("num_attention_heads")?,
            kv_heads: usize_at("num_key_value_heads")?,
            head_dim: usize_at("head_dim")?,
            global_head_dim: text
                .get("global_head_dim")
                .and_then(serde_json::Value::as_u64)
                .map(|value| value as usize)
                .unwrap_or(usize_at("head_dim")?),
            rope_base_sliding: rope_param(text, "sliding_attention", "rope_theta")
                .or_else(|| text.get("rope_local_base_freq").and_then(as_f64))
                .unwrap_or(10_000.0),
            rope_base_full: rope_param(text, "full_attention", "rope_theta")
                .or_else(|| text.get("rope_theta").and_then(as_f64))
                .unwrap_or(1_000_000.0),
            full_partial_rotary_factor: rope_param(text, "full_attention", "partial_rotary_factor")
                .unwrap_or(1.0),
            num_kv_shared_layers: text
                .get("num_kv_shared_layers")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0) as usize,
            vocab: usize_at("vocab_size")?,
            ple_width: usize_at("hidden_size_per_layer_input")?,
            sliding_window: text
                .get("sliding_window")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0) as u32,
            layer_types,
            // `Acc` (Q32.32) bits, not Q16.16: the canonical `rms_norm` takes its
            // epsilon in accumulator units. Encoded through local
            // `f32_to_acc` so it is the same value, not a near one — 1e-6 in
            // Q16.16 rounds to zero, which is how this was silently disabled.
            norm_eps: det_num::f32_to_acc(
                text.get("rms_norm_eps")
                    .and_then(serde_json::Value::as_f64)
                    .unwrap_or(1e-6) as f32,
            )
            .to_bits(),
        })
    }
}

// ---------------------------------------------------------------------------
// Stage 2 — input embedding
// ---------------------------------------------------------------------------

/// Every family's embedding table. `embedding_scale` is what a row is
/// multiplied by before layer 0, in Q16.16: Gemma's is `sqrt(hidden)`,
/// converted by local `f32_to_act` so the committed bits are the deterministic
/// path's exactly, and a family that does not scale passes one.
pub(super) fn write_embedding(
    weights: &detwgt::MmapDetwgt,
    hidden: usize,
    vocab: usize,
    embedding_scale: i32,
    layout: &ArtifactLayout,
    stages: &mut Vec<String>,
) -> Result<(), Box<dyn Error>> {
    let table = EmbeddingTable {
        hidden_size: hidden as u32,
        embedding_scale,
        values: mapped_weights::q8_rows(
            weights,
            "model.language_model.embed_tokens.weight",
            0,
            vocab,
        )?,
    };
    let commitment = write_external(&table, layout, "raster-stages/input-embedding", "embedding")?;
    let witness = model_identity::sparse::witness(table.values.pages.as_slice())
        .map_err(|f| format!("{f:?}"))?;
    let witness_commitment =
        write_external(&witness, layout, "raster-stages/input-embedding", "witness")?;
    // A prompt range embeds only the prompt. A claimed range binds the run's
    // claimed ids here instead.
    let no_claim = write_external(
        &ClaimedOutput::default(),
        layout,
        "raster-stages/input-embedding",
        "no_claim",
    )?;

    stages.push(format!(
        concat!(
            "[[chain.stage]]\n",
            "name = \"input_embedding\"\n",
            "project = \"raster-stages/input-embedding\"\n",
            "inputs.prompt = {{ from = \"prompt_check\" }}\n",
            "{}{}{}"
        ),
        external_line(
            "witness",
            layout,
            "raster-stages/input-embedding",
            "witness",
            &witness_commitment
        ),
        external_line(
            "claimed",
            layout,
            "raster-stages/input-embedding",
            "no_claim",
            &no_claim
        ),
        external_line(
            "embedding",
            layout,
            "raster-stages/input-embedding",
            "embedding",
            &commitment
        )
    ));
    Ok(())
}

// ---------------------------------------------------------------------------
// Stage 3 — per-layer embeddings
// ---------------------------------------------------------------------------

pub(super) fn write_ple_layers(
    weights: &detwgt::MmapDetwgt,
    shape: &Shape,
    layout: &ArtifactLayout,
    stages: &mut Vec<String>,
) -> Result<(), Box<dyn Error>> {
    let projection_norm =
        weights.values("model.language_model.per_layer_projection_norm.weight")?;

    for layer_idx in 0..shape.layers {
        let start = layer_idx * shape.ple_width;
        let end = start + shape.ple_width;

        // The packed [vocab, layers · width] table gives this layer its slice.
        let embeddings = mapped_weights::q8_columns(
            weights,
            "model.language_model.embed_tokens_per_layer.weight",
            start,
            end,
        )?;

        let layer = PleLayer {
            params: PleLayerParams {
                layer_idx: layer_idx as u32,
                hidden_size: shape.hidden as u32,
                ple_width: shape.ple_width as u32,
                // Derived, not stored in the artifact — the comment this
                // replaces claimed unit scaling was "faithful", but the model
                // loader computes real values and hardcoding 1.0 silently
                // dropped three scalings from every per-layer embedding
                // (`casettek/.../gemma/io.rs:1176-1179`).
                embedding_scale: det_act_bits((shape.ple_width as f32).sqrt()),
                projection_scalar: det_act_bits((shape.hidden as f32).powf(-0.5)),
                input_scale: det_act_bits(2f32.powf(-0.5)),
                norm_eps: shape.norm_eps,
                norm_weights: page_of_i32s(&projection_norm),
            },
            embeddings,
            projection: mapped_weights::rows(
                weights,
                "model.language_model.per_layer_model_projection.weight",
                start,
                end,
            )?,
        };

        let name = format!("layer{layer_idx}");
        let commitment =
            write_external(&layer, layout, "raster-stages/prefill-prepare-aux", &name)?;
        let witness = model_identity::sparse::witness(layer.embeddings.pages.as_slice())
            .map_err(|f| format!("{f:?}"))?;
        let witness_name = format!("witness{layer_idx}");
        let witness_commitment = write_external(
            &witness,
            layout,
            "raster-stages/prefill-prepare-aux",
            &witness_name,
        )?;
        stages.push(format!(
            concat!(
                "[[chain.stage]]\n",
                "name = \"prefill_prepare_aux_l{idx}\"\n",
                "project = \"raster-stages/prefill-prepare-aux\"\n",
                "inputs.embedded = {{ from = \"input_embedding\" }}\n",
                "{line}{witness_line}"
            ),
            witness_line = external_line(
                "witness",
                layout,
                "raster-stages/prefill-prepare-aux",
                &witness_name,
                &witness_commitment
            ),
            idx = layer_idx,
            line = external_line(
                "layer",
                layout,
                "raster-stages/prefill-prepare-aux",
                &name,
                &commitment
            )
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Stage 4 — transformer layers
// ---------------------------------------------------------------------------

pub(super) fn write_transformer_layers(
    weights: &detwgt::MmapDetwgt,
    shape: &Shape,
    layout: &ArtifactLayout,
    stages: &mut Vec<String>,
) -> Result<(), Box<dyn Error>> {
    // Report the two RoPE families once. These are the numbers that silently
    // decide every attention score, so they are worth seeing rather than
    // trusting — a wrong base or rotary width produces a plausible-looking run
    // and the wrong token.
    for (label, idx) in shape
        .layer_types
        .iter()
        .enumerate()
        .map(|(i, t)| (t.as_str(), i))
        .fold(Vec::new(), |mut seen: Vec<(&str, usize)>, (t, i)| {
            if !seen.iter().any(|(s, _)| *s == t) {
                seen.push((t, i));
            }
            seen
        })
    {
        let (base, rotary, freq) = shape.rope_at(idx);
        println!(
            "rope[{label}]: head_dim {} · rotary_dim {rotary} · freq_base_dim {freq} · base {} (Q32.32 bits {base})",
            shape.head_dim_at(idx),
            base >> 32,
        );
    }

    // And the KV-sharing map, for the same reason: a wrong donor is not a crash,
    // it is a layer attending over another layer's keys.
    let donors: Vec<(usize, i32)> = (0..shape.layers)
        .map(|idx| (idx, shape.kv_donor_layer(idx)))
        .filter(|(_, donor)| *donor >= 0)
        .collect();
    let mut by_donor: BTreeMap<i32, Vec<usize>> = BTreeMap::new();
    for (idx, donor) in &donors {
        by_donor.entry(*donor).or_default().push(*idx);
    }
    println!(
        "kv sharing: {} of {} layers borrow (config declares {})",
        donors.len(),
        shape.layers,
        shape.num_kv_shared_layers
    );
    for (donor, borrowers) in &by_donor {
        println!("  donor layer {donor} -> {borrowers:?}");
    }
    if by_donor.len() > 2 {
        return Err(format!(
            "this chain supports at most two KV donor candidates, model declares {}",
            by_donor.len()
        )
        .into());
    }
    let mut donor_candidates = by_donor.keys().copied();
    let donor_a_layer = donor_candidates.next().unwrap_or(-2);
    let donor_b_layer = donor_candidates.next().unwrap_or(-3);

    for layer_idx in 0..shape.layers {
        let at = |suffix: &str| format!("model.language_model.layers.{layer_idx}.{suffix}");
        let gate = weights.tensor(&at("mlp.gate_proj.weight"))?;
        let ffn = gate.dims[0] as usize;

        // Alternating sliding/full attention, as `layer_types` declares.
        let sliding = match shape.layer_types.get(layer_idx).map(String::as_str) {
            Some("sliding_attention") => shape.sliding_window,
            _ => 0,
        };
        // Head dim is per-layer here, not global (see `Shape::global_head_dim`).
        let head_dim = shape.head_dim_at(layer_idx);
        let (rope_base, rotary_dim, rope_freq_base_dim) = shape.rope_at(layer_idx);
        let kv_donor_layer = shape.kv_donor_layer(layer_idx);
        // `attention_k_eq_v` layers store no separate V projection: V *is* K.
        // tiny-gemma-dev emits v_proj only for its sliding layers.
        let v_name = if weights.tensor(&at("self_attn.v_proj.weight")).is_ok() {
            at("self_attn.v_proj.weight")
        } else {
            at("self_attn.k_proj.weight")
        };
        let layer_scalar = weights
            .slice(&at("layer_scalar"))
            .ok()
            .and_then(|slice| slice.value_at(0).ok())
            .unwrap_or(0);

        let layer = TransformerLayer {
            params: LayerParams {
                layer_idx: layer_idx as u32,
                hidden_size: shape.hidden as u32,
                ffn_size: ffn as u32,
                num_heads: shape.heads as u32,
                num_kv_heads: shape.kv_heads as u32,
                head_dim: head_dim as u32,
                sliding_window: sliding,
                attn_scale: inv_sqrt_q16(head_dim),
                layer_scalar,
                norm_eps: shape.norm_eps,
                rope_base,
                rotary_dim,
                rope_freq_base_dim,
                kv_donor_layer,
                donor_a_layer,
                donor_b_layer,
                norm_input: page_of_i32s(&weights.values(&at("input_layernorm.weight"))?),
                norm_post_attn: page_of_i32s(
                    &weights.values(&at("post_attention_layernorm.weight"))?,
                ),
                norm_pre_ffw: page_of_i32s(
                    &weights.values(&at("pre_feedforward_layernorm.weight"))?,
                ),
                norm_post_ffw: page_of_i32s(
                    &weights.values(&at("post_feedforward_layernorm.weight"))?,
                ),
                q_norm: page_of_i32s(&weights.values(&at("self_attn.q_norm.weight"))?),
                k_norm: page_of_i32s(&weights.values(&at("self_attn.k_norm.weight"))?),
                ple_width: shape.ple_width as u32,
                // Hidden-width, despite the name: it normalises *after*
                // `per_layer_projection` maps back up from `ple_width`.
                ple_post_norm: page_of_i32s(
                    &weights.values(&at("post_per_layer_input_norm.weight"))?,
                ),
            },
            w_q: mapped_weights::q8_tensor(weights, &at("self_attn.q_proj.weight"))?,
            w_k: mapped_weights::q8_tensor(weights, &at("self_attn.k_proj.weight"))?,
            w_v: mapped_weights::q8_tensor(weights, &v_name)?,
            w_o: mapped_weights::q8_tensor(weights, &at("self_attn.o_proj.weight"))?,
            w_gate: mapped_weights::q8_tensor(weights, &at("mlp.gate_proj.weight"))?,
            w_up: mapped_weights::q8_tensor(weights, &at("mlp.up_proj.weight"))?,
            ple_input_gate: mapped_weights::tensor(weights, &at("per_layer_input_gate.weight"))?,
            ple_layer_projection: mapped_weights::tensor(
                weights,
                &at("per_layer_projection.weight"),
            )?,
            w_down: mapped_weights::q8_tensor(weights, &at("mlp.down_proj.weight"))?,
        };

        let name = format!("layer{layer_idx}");
        let commitment = write_external(&layer, layout, "raster-stages/prefill-range", &name)?;
        let upstream = if layer_idx == 0 {
            "input_embedding".to_string()
        } else {
            format!("prefill_range_l{}", layer_idx - 1)
        };
        // Before the sharing boundary both candidates must be the empty input:
        // binding the eventual donor stages here would be a forward reference.
        // Sharing layers receive both candidates and committed
        // `kv_donor_layer` selects the exact one inside the tile.
        let first_shared = shape.layers.saturating_sub(shape.num_kv_shared_layers);
        let (donor_a, donor_b) = if layer_idx < first_shared {
            (
                String::from("input_embedding"),
                String::from("input_embedding"),
            )
        } else {
            (
                format!("prefill_range_l{donor_a_layer}"),
                format!("prefill_range_l{donor_b_layer}"),
            )
        };
        stages.push(format!(
            concat!(
                "[[chain.stage]]\n",
                "name = \"prefill_range_l{idx}\"\n",
                "project = \"raster-stages/prefill-range\"\n",
                "inputs.activations = {{ from = \"{upstream}\" }}\n",
                // Prefill inherits an empty cache. Bound to `input_embedding`
                // for every layer, the same stage the non-sharing layers use
                // for `donor_kv` — a decode stage is where this points at the
                // previous step instead.
                "inputs.prior_kv = {{ from = \"input_embedding\" }}\n",
                "inputs.donor_a_kv = {{ from = \"{donor_a}\" }}\n",
                "inputs.donor_b_kv = {{ from = \"{donor_b}\" }}\n",
                "inputs.ple = {{ from = \"prefill_prepare_aux_l{idx}\" }}\n",
                "{line}"
            ),
            idx = layer_idx,
            upstream = upstream,
            donor_a = donor_a,
            donor_b = donor_b,
            line = external_line(
                "layer",
                layout,
                "raster-stages/prefill-range",
                &name,
                &commitment,
            )
        ));
    }
    Ok(())
}

/// `1/sqrt(n)` in Q16.16, computed in integers so the fixture and the guest
/// agree on the constant.
pub(super) fn inv_sqrt_q16(n: usize) -> i32 {
    if n == 0 {
        return 0;
    }
    let root = (0..).find(|r| r * r >= n).unwrap_or(1).max(1);
    let exact = (n as f64).sqrt();
    let _ = root;
    f32_to_q16(1.0 / exact)
}

// ---------------------------------------------------------------------------
// Stage 5 — output head
// ---------------------------------------------------------------------------

/// Every family's head: the final norm and the projection to the vocabulary.
pub(super) fn write_head(
    weights: &detwgt::MmapDetwgt,
    hidden: usize,
    vocab: usize,
    norm_eps: i64,
    text: &serde_json::Value,
    layout: &ArtifactLayout,
    stages: &mut Vec<String>,
) -> Result<(), Box<dyn Error>> {
    let tied = text
        .get("tie_word_embeddings")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    let projection = if tied {
        "model.language_model.embed_tokens.weight"
    } else {
        "model.language_model.lm_head.weight"
    };
    let softcap = text
        .get("final_logit_softcapping")
        .and_then(serde_json::Value::as_f64)
        .map(f32_to_q16)
        .unwrap_or(0);

    let commitment = write_external(
        &FinalHead {
            params: FinalHeadParams {
                hidden_size: hidden as u32,
                norm_eps,
                softcap,
                norm_weights: page_of_i32s(&weights.values("model.language_model.norm.weight")?),
            },
            projection: mapped_weights::q8_rows(weights, projection, 0, vocab)?,
        },
        layout,
        "raster-stages/sequence-check",
        "head",
    )?;

    stages.push(format!(
        "[[chain.stage]]\nname = \"final_head\"\nproject = \"raster-stages/sequence-check\"\n{}",
        external_line(
            "head",
            layout,
            "raster-stages/sequence-check",
            "head",
            &commitment
        )
    ));
    Ok(())
}

// ---------------------------------------------------------------------------
// What a bundle of this family may contain
// ---------------------------------------------------------------------------

const LANGUAGE_MODEL: &str = "model.language_model.";

/// Language-model tensors outside the layers.
const MODEL_TENSORS: &[&str] = &[
    "embed_tokens.weight",
    "embed_tokens_per_layer.weight",
    "lm_head.weight",
    "norm.weight",
    "per_layer_model_projection.weight",
    "per_layer_projection_norm.weight",
];

/// Tensors of one layer, after `layers.{n}.`.
const LAYER_TENSORS: &[&str] = &[
    "input_layernorm.weight",
    "layer_scalar",
    "mlp.down_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "per_layer_input_gate.weight",
    "per_layer_projection.weight",
    "post_attention_layernorm.weight",
    "post_feedforward_layernorm.weight",
    "post_per_layer_input_norm.weight",
    "pre_feedforward_layernorm.weight",
    "self_attn.k_norm.weight",
    "self_attn.k_proj.weight",
    "self_attn.o_proj.weight",
    "self_attn.q_norm.weight",
    "self_attn.q_proj.weight",
    "self_attn.v_proj.weight",
];

/// Whether this family's import reads the tensor, or has no business with it.
///
/// Tensors outside the language model (vision and audio towers) are not this
/// chain's, and are passed over as they always were.
fn is_known_tensor(name: &str, layers: usize) -> bool {
    let Some(rest) = name.strip_prefix(LANGUAGE_MODEL) else {
        return true;
    };
    let Some(layer) = rest.strip_prefix("layers.") else {
        return MODEL_TENSORS.contains(&rest);
    };
    let Some((index, suffix)) = layer.split_once('.') else {
        return false;
    };
    index.parse::<usize>().is_ok_and(|index| index < layers) && LAYER_TENSORS.contains(&suffix)
}

/// Refuses a bundle with language-model weights this family does not read.
///
/// An import ignores what it does not ask for, so a model with more in it than
/// a dense Gemma layer (expert matrices, say) would otherwise import cleanly as
/// a different model: one that runs, and computes something else.
pub(super) fn require_known_tensors<'a>(
    names: impl Iterator<Item = &'a str>,
    layers: usize,
) -> Result<(), Box<dyn Error>> {
    let unknown: Vec<&str> = names
        .filter(|name| !is_known_tensor(name, layers))
        .collect();
    if unknown.is_empty() {
        return Ok(());
    }
    let shown = unknown
        .iter()
        .take(5)
        .copied()
        .collect::<Vec<_>>()
        .join(", ");
    let more = match unknown.len().saturating_sub(5) {
        0 => String::new(),
        more => format!(" and {more} more"),
    };
    Err(format!(
        "the bundle has language-model weights the {} import does not read: {shown}{more}; \
         it is another model family, or one this build does not import yet",
        inference_artifacts::ModelFamily::GemmaDense.name()
    )
    .into())
}

impl Shape {
    /// What the chain template needs of this model's layers.
    pub(super) fn chain_layers(&self) -> ChainLayers {
        let mut donors = (0..self.layers)
            .map(|layer| self.kv_donor_layer(layer))
            .filter(|donor| *donor >= 0)
            .collect::<Vec<_>>();
        donors.sort_unstable();
        donors.dedup();
        ChainLayers {
            layers: self.layers,
            project: "raster-stages/prefill-range",
            stage: "prefill_range",
            auxiliary: true,
            donor_inputs: true,
            num_kv_shared_layers: self.num_kv_shared_layers,
            donors,
        }
    }

    /// The shape as the model manifest records it, for the native runtime.
    pub(super) fn manifest_shape(
        &self,
        text_config: &serde_json::Value,
    ) -> inference_artifacts::DirectInferShape {
        inference_artifacts::DirectInferShape {
            hidden_size: self.hidden as u32,
            num_hidden_layers: self.layers as u32,
            num_attention_heads: self.heads as u32,
            num_key_value_heads: self.kv_heads as u32,
            head_dim: self.head_dim as u32,
            global_head_dim: self.global_head_dim as u32,
            vocab_size: self.vocab as u32,
            hidden_size_per_layer_input: self.ple_width as u32,
            sliding_window: self.sliding_window,
            layer_types: self.layer_types.clone(),
            num_kv_shared_layers: self.num_kv_shared_layers as u32,
            norm_eps: self.norm_eps,
            rope_base_sliding: (self.rope_base_sliding as i64) << 32,
            rope_base_full: (self.rope_base_full as i64) << 32,
            full_partial_rotary_factor_q16: f32_to_q16(self.full_partial_rotary_factor),
            embedding_scale: det_act_bits((self.hidden as f32).sqrt()),
            ple_embedding_scale: det_act_bits((self.ple_width as f32).sqrt()),
            ple_projection_scalar: det_act_bits((self.hidden as f32).powf(-0.5)),
            ple_input_scale: det_act_bits(2f32.powf(-0.5)),
            final_logit_softcap: text_config
                .get("final_logit_softcapping")
                .and_then(serde_json::Value::as_f64)
                .map(f32_to_q16)
                .unwrap_or(0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_bundle_may_hold_only_tensors_the_import_reads() {
        let layer = |index: usize, suffix: &str| format!("{LANGUAGE_MODEL}layers.{index}.{suffix}");
        let mut names: Vec<String> = MODEL_TENSORS
            .iter()
            .map(|name| format!("{LANGUAGE_MODEL}{name}"))
            .collect();
        for index in 0..2 {
            names.extend(LAYER_TENSORS.iter().map(|suffix| layer(index, suffix)));
        }
        // Another tower's weights are not this chain's.
        names.push(String::from("model.vision_tower.patch_embedding.weight"));
        let check = |names: &[String]| require_known_tensors(names.iter().map(String::as_str), 2);
        assert!(check(&names).is_ok());

        // Expert weights, a layer past the config's count, and a new top-level
        // tensor each stop the import, and the message names them.
        for extra in [
            layer(0, "mlp.experts.3.down_proj.weight"),
            layer(2, "mlp.gate_proj.weight"),
            format!("{LANGUAGE_MODEL}router.weight"),
            format!("{LANGUAGE_MODEL}layers.x"),
        ] {
            let mut with_extra = names.clone();
            with_extra.push(extra.clone());
            let error = check(&with_extra).unwrap_err().to_string();
            assert!(error.contains(&extra), "{error}");
            assert!(error.contains("gemma-dense"), "{error}");
        }
        let many: Vec<String> = (0..8)
            .map(|index| layer(0, &format!("new.{index}")))
            .collect();
        assert!(check(&many).unwrap_err().to_string().contains("and 3 more"));
    }
}
