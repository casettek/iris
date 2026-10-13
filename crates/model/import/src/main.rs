//! Import a Gemma model bundle into this chain's committed externals.
//!
//! ```sh
//! cargo run -p model-import -- \
//!   --model ../raster-inference/assets/tiny-gemma-dev
//! ```
//!
//! Reads `model.detwgt` (deterministic Q16.16 weights), `config.json` (shapes,
//! attention types, softcap) and `tokenizer.json` (vocabulary and merges), and
//! writes one `.rastered`/`.rindex` pair per stage external and saves a
//! model-specific `[chain]` template, stages expanded per layer.
//!
//! Every value crosses over unchanged: detwgt stores canonical Q16.16 bit
//! patterns, which is the representation the stages compute in. Nothing here
//! requantises.

mod direct_manifest;
mod externals;

use externals::*;
use std::collections::BTreeMap;
use std::error::Error;
use std::fs;
use std::path::{Component, Path, PathBuf};

/// Q16.16 one.
const ONE: i32 = 1 << 16;

const INITIAL_PIECES_PLACEHOLDER_COMMITMENT: &str =
    "0000000000000000000000000000000000000000000000000000000000000000";

fn main() {
    if let Err(error) = run_from_env() {
        eprintln!("model-import: {error}");
        std::process::exit(1);
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImportConfig {
    pub model_dir: PathBuf,
    /// Also write a copy of the complete chain template here.
    pub manifest: Option<PathBuf>,
    /// Root directory for reusable imported model artifacts.
    pub artifact_root: PathBuf,
    /// Stable model-scoped artifact directory name. Defaults to the model bundle
    /// directory name.
    pub model_id: Option<String>,
    /// Rewrite only the tokenizer externals, leaving the weight externals
    /// alone. The weights are ~9 GB and depend on nothing the tokenizer
    /// touches, so a prompt or vocabulary-layout change has no reason to
    /// re-emit them. Prints only the `prompt_prepare` stage.
    pub only_tokenizer: bool,
    /// Rewrite only the transformer-layer externals (`prefill-range`). Nothing
    /// else depends on `LayerParams`, so a per-layer shape correction has no
    /// reason to re-emit the ~11 GB of tokenizer, embedding, PLE and head
    /// artifacts. Prints only the `prefill_range_*` stages.
    pub only_layers: bool,
    /// Rewrite only the embedding external. Same reasoning as `--only-layers`:
    /// nothing else depends on `EmbeddingTable`.
    pub only_embedding: bool,
    /// Rewrite only the PLE layer externals (`prefill-prepare-aux`).
    pub only_ple: bool,
    /// Write only the host-side direct-infer manifest, without regenerating
    /// committed Raster stage externals.
    pub only_direct: bool,
    /// Leave stage-local input.json and input_manifest.json fixtures untouched.
    pub no_stage_fixtures: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImportResult {
    pub manifest_path: Option<PathBuf>,
    pub manifest_text: Option<String>,
    pub stage_count: usize,
}

pub fn run_from_env() -> Result<(), Box<dyn Error>> {
    run_from_args(std::env::args().skip(1))
}

pub fn run_from_args(args: impl IntoIterator<Item = String>) -> Result<(), Box<dyn Error>> {
    let args = parse_args_from(args)?;
    import_model(args).map(|_| ())
}

fn parse_args_from(args: impl IntoIterator<Item = String>) -> Result<ImportConfig, Box<dyn Error>> {
    let mut model_dir = None;
    let mut manifest = None;
    let mut artifact_root = PathBuf::from(inference_artifacts::MODEL_ARTIFACTS_DIR);
    let mut model_id = None;
    let mut only_tokenizer = false;
    let mut only_layers = false;
    let mut only_embedding = false;
    let mut only_ple = false;
    let mut only_direct = false;
    let mut no_stage_fixtures = false;
    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model" => model_dir = args.next().map(PathBuf::from),
            "--manifest" => manifest = args.next().map(PathBuf::from),
            "--artifact-root" => {
                artifact_root = args
                    .next()
                    .map(PathBuf::from)
                    .ok_or("--artifact-root <dir> is required")?
            }
            "--model-id" => {
                let raw = args.next().ok_or("--model-id <id> is required")?;
                model_id = Some(sanitize_model_id(&raw));
            }
            "--only-tokenizer" => only_tokenizer = true,
            "--only-layers" => only_layers = true,
            "--only-embedding" => only_embedding = true,
            "--only-ple" => only_ple = true,
            "--only-direct" => only_direct = true,
            "--no-stage-fixtures" => no_stage_fixtures = true,
            other => return Err(format!("unknown argument '{other}'").into()),
        }
    }
    if only_direct && (only_tokenizer || only_layers || only_embedding || only_ple) {
        return Err("--only-direct cannot be combined with other --only-* modes".into());
    }
    Ok(ImportConfig {
        model_dir: model_dir.ok_or("--model <bundle-dir> is required")?,
        manifest,
        artifact_root,
        model_id,
        only_tokenizer,
        only_layers,
        only_embedding,
        only_ple,
        only_direct,
        no_stage_fixtures,
    })
}

impl ImportConfig {
    pub fn model_id(&self) -> String {
        self.model_id
            .clone()
            .unwrap_or_else(|| default_model_id(&self.model_dir))
    }

    pub fn artifact_dir(&self) -> PathBuf {
        self.artifact_root.join(self.model_id())
    }
}

#[derive(Debug, Clone)]
struct ArtifactLayout {
    model_dir: PathBuf,
    no_stage_fixtures: bool,
}

impl ArtifactLayout {
    fn from_config(args: &ImportConfig) -> Self {
        Self {
            model_dir: args.artifact_dir(),
            no_stage_fixtures: args.no_stage_fixtures,
        }
    }

    fn external_path(&self, stage_dir: &str, name: &str, extension: &str) -> PathBuf {
        self.external_dir(stage_dir)
            .join(format!("{name}.{extension}"))
    }

    fn external_dir(&self, stage_dir: &str) -> PathBuf {
        self.model_dir
            .join("raster")
            .join(stage_artifact_dir_name(stage_dir))
    }
}

fn default_model_id(model_dir: &Path) -> String {
    let name = model_dir
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("model");
    sanitize_model_id(name)
}

fn sanitize_model_id(raw: &str) -> String {
    let mut out = String::new();
    let mut last_dash = false;
    for ch in raw.chars() {
        let mapped = if ch.is_ascii_alphanumeric() || matches!(ch, '_' | '.') {
            ch
        } else {
            '-'
        };
        if mapped == '-' {
            if !last_dash {
                out.push(mapped);
            }
            last_dash = true;
        } else {
            out.push(mapped);
            last_dash = false;
        }
    }
    let trimmed = out.trim_matches(&['-', '_', '.'][..]);
    if trimmed.is_empty() {
        String::from("model")
    } else {
        trimmed.to_string()
    }
}

fn stage_artifact_dir_name(stage_dir: &str) -> String {
    Path::new(stage_dir)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or(stage_dir)
        .to_string()
}

pub fn import_model(args: ImportConfig) -> Result<ImportResult, Box<dyn Error>> {
    let layout = ArtifactLayout::from_config(&args);
    let tokenizer: serde_json::Value =
        serde_json::from_slice(&fs::read(args.model_dir.join("tokenizer.json"))?)?;

    let eos_ids = load_eos_ids(&args.model_dir);

    if args.only_direct {
        let config: serde_json::Value =
            serde_json::from_slice(&fs::read(args.model_dir.join("config.json"))?)?;
        let text = config
            .get("text_config")
            .ok_or("config.json has no text_config")?;
        let shape = Shape::from_config(text)?;
        // A direct-only refresh cannot attest that existing Raster externals
        // were generated from this bundle. Only a full import binds a template.
        direct_manifest::write_model_manifest(&args, &eos_ids, &shape, text, None)?;
        return Ok(ImportResult {
            manifest_path: None,
            manifest_text: None,
            stage_count: 0,
        });
    }

    if args.only_tokenizer {
        let mut stages = Vec::new();
        let decoder_commitment = write_tokenizer(&tokenizer, &eos_ids, &layout, &mut stages)?;
        println!();
        println!("# ---- replaces tokenizer and decoder externals in the root Raster.toml ----");
        for stage in &stages {
            println!();
            print!("{stage}");
        }
        println!();
        println!(
            "{}",
            external_line(
                "decoder",
                &layout,
                "raster-stages/output-finalize",
                "decoder",
                &decoder_commitment
            )
        );
        direct_manifest::warn_partial_direct_manifest_not_refreshed(&args);
        return Ok(ImportResult {
            manifest_path: None,
            manifest_text: None,
            stage_count: stages.len(),
        });
    }

    let weights = detwgt::load(&args.model_dir.join("model.detwgt"))?;
    let config: serde_json::Value =
        serde_json::from_slice(&fs::read(args.model_dir.join("config.json"))?)?;
    let text = config
        .get("text_config")
        .ok_or("config.json has no text_config")?;

    if args.only_ple {
        let shape = Shape::from_config(text)?;
        let mut stages = Vec::new();
        write_ple_layers(&weights, &shape, &layout, &mut stages)?;
        println!();
        println!("# ---- replaces the prefill_prepare_aux_* stages in the root Raster.toml ----");
        for stage in &stages {
            println!();
            print!("{stage}");
        }
        direct_manifest::warn_partial_direct_manifest_not_refreshed(&args);
        return Ok(ImportResult {
            manifest_path: None,
            manifest_text: None,
            stage_count: stages.len(),
        });
    }

    if args.only_embedding {
        let shape = Shape::from_config(text)?;
        let mut stages = Vec::new();
        write_embedding(&weights, &shape, &layout, &mut stages)?;
        println!();
        println!("# ---- replaces the input_embedding stage in the root Raster.toml ----");
        for stage in &stages {
            println!();
            print!("{stage}");
        }
        direct_manifest::warn_partial_direct_manifest_not_refreshed(&args);
        return Ok(ImportResult {
            manifest_path: None,
            manifest_text: None,
            stage_count: stages.len(),
        });
    }

    if args.only_layers {
        let shape = Shape::from_config(text)?;
        let mut stages = Vec::new();
        write_transformer_layers(&weights, &shape, &layout, &mut stages)?;
        println!();
        println!("# ---- replaces the prefill_range_* stages in the root Raster.toml ----");
        for stage in &stages {
            println!();
            print!("{stage}");
        }
        direct_manifest::warn_partial_direct_manifest_not_refreshed(&args);
        return Ok(ImportResult {
            manifest_path: None,
            manifest_text: None,
            stage_count: stages.len(),
        });
    }

    let shape = Shape::from_config(text)?;
    println!(
        "model: hidden {} · {} layers · {} heads over {} kv heads · head_dim {} · ffn {} · vocab {} · ple {}",
        shape.hidden, shape.layers, shape.heads, shape.kv_heads, shape.head_dim, shape.ffn,
        shape.vocab, shape.ple_width
    );
    println!("tensors: {}", weights.names().count());
    println!();

    let mut stages = Vec::new();
    let decoder_commitment = write_tokenizer(&tokenizer, &eos_ids, &layout, &mut stages)?;
    write_embedding(&weights, &shape, &layout, &mut stages)?;
    write_ple_layers(&weights, &shape, &layout, &mut stages)?;
    write_transformer_layers(&weights, &shape, &layout, &mut stages)?;
    write_head(&weights, &shape, text, &layout, &mut stages)?;

    let mut manifest =
        String::from("[chain]\nname = \"raster-chain-inference\"\nversion = \"0.1.0\"\n");
    manifest.push_str(&indexed_model_inputs(&shape, &layout, &stages)?);
    for stage in &stages {
        manifest.push('\n');
        manifest.push_str(stage);
    }
    manifest.push_str(&generation_stages(
        &shape,
        &layout,
        &decoder_commitment,
        0,
        &stages,
    )?);
    let stage_count = stages.len();
    let (template_path, template) = save_model_template(&args, &manifest)?;
    direct_manifest::write_model_manifest(
        &args,
        &eos_ids,
        &shape,
        text,
        Some((&template_path, &template)),
    )?;
    Ok(ImportResult {
        manifest_path: Some(template_path),
        manifest_text: None,
        stage_count,
    })
}

fn save_model_template(
    args: &ImportConfig,
    manifest: &str,
) -> Result<(PathBuf, String), Box<dyn Error>> {
    // Generated paths start at the importing workspace. Resolve them before
    // saving under the model directory (or an optional custom --manifest path).
    let template =
        inference_artifacts::resolve_raster_manifest_paths(manifest, &std::env::current_dir()?)
            .map_err(|error| -> Box<dyn Error> { error.to_string().into() })?;
    let path = args.artifact_dir().join("Raster.toml");
    fs::create_dir_all(args.artifact_dir())?;
    fs::write(&path, &template)?;
    println!("wrote {}", path.display());
    if let Some(copy) = &args.manifest {
        if let Some(parent) = copy
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            fs::create_dir_all(parent)?;
        }
        fs::write(copy, &template)?;
        println!("wrote {}", copy.display());
    }
    Ok((path, template))
}

fn stage_external_commitment(
    stages: &[String],
    stage_name: &str,
    parameter: &str,
) -> Result<String, Box<dyn Error>> {
    let name = format!("name = \"{stage_name}\"");
    let prefix = format!("inputs.{parameter} = ");
    let line = stages
        .iter()
        .find(|stage| stage.lines().any(|line| line == name))
        .and_then(|stage| stage.lines().find(|line| line.starts_with(&prefix)))
        .ok_or_else(|| {
            format!("generated prefill manifest has no {stage_name}.{parameter} binding")
        })?;
    let marker = "commitment = \"";
    let start = line
        .find(marker)
        .ok_or_else(|| format!("{stage_name}.{parameter} is not an external binding"))?
        + marker.len();
    let end = line[start..]
        .find('"')
        .ok_or_else(|| format!("{stage_name}.{parameter} has an unterminated commitment"))?
        + start;
    Ok(line[start..end].to_string())
}

/// Declares model weights once so every decode iteration can bind them by its
/// static layer index without copying commitments into the repeat block.
fn indexed_model_inputs(
    shape: &Shape,
    layout: &ArtifactLayout,
    stages: &[String],
) -> Result<String, Box<dyn Error>> {
    let aux = (0..shape.layers)
        .map(|layer| {
            stage_external_commitment(stages, &format!("prefill_prepare_aux_l{layer}"), "layer")
        })
        .collect::<Result<Vec<_>, _>>()?;
    let transformer = (0..shape.layers)
        .map(|layer| stage_external_commitment(stages, &format!("prefill_range_l{layer}"), "layer"))
        .collect::<Result<Vec<_>, _>>()?;
    let render = |name: &str, dir: &str, commitments: &[String]| {
        let entries = commitments
            .iter()
            .enumerate()
            .map(|(index, commitment)| format!("  \"{commitment}\", # l{index}"))
            .collect::<Vec<_>>()
            .join("\n");
        let path = path_string(&layout.external_path(dir, "layer{l}", "rastered"));
        let index_path = path_string(&layout.external_path(dir, "layer{l}", "rindex"));
        format!(
            "\n[chain.input.{name}]\n\
             index = \"l\"\n\
             path = \"{path}\"\n\
             index_path = \"{index_path}\"\n\
             commitments = [\n{entries}\n]\n"
        )
    };
    Ok(format!(
        "{}{}",
        render("aux_layer", "raster-stages/prefill-prepare-aux", &aux),
        render(
            "transformer_layer",
            "raster-stages/prefill-range",
            &transformer
        )
    ))
}

/// Emits exactly `tokens` select+transition iterations as one static repeat.
/// Expansion still produces one auditable chain stage per program invocation.
fn generation_stages(
    shape: &Shape,
    layout: &ArtifactLayout,
    decoder_commitment: &str,
    tokens: u32,
    prefill_stages: &[String],
) -> Result<String, Box<dyn Error>> {
    let embedding = stage_external_commitment(prefill_stages, "input_embedding", "embedding")?;
    let head = stage_external_commitment(prefill_stages, "prefill_finalize", "head")?;
    let first_shared = shape.layers.saturating_sub(shape.num_kv_shared_layers);
    let mut donors = (0..shape.layers)
        .map(|layer| shape.kv_donor_layer(layer))
        .filter(|donor| *donor >= 0)
        .collect::<Vec<_>>();
    donors.sort_unstable();
    donors.dedup();
    if donors.len() > 2 {
        return Err(format!(
            "decode repeat supports at most two donor candidates, model declares {}",
            donors.len()
        )
        .into());
    }
    let donor_a = donors.first().copied();
    let donor_b = donors.get(1).copied();

    let mut out = String::from(
        "\n[[chain.stage]]\nname = \"decode_init\"\nproject = \"raster-stages/decode-init\"\n\
         \n[[chain.repeat]]\nname = \"decode\"\nindex = \"t\"\n",
    );
    out.push_str(&format!("count = {tokens}\n"));
    out.push_str(
        "\n  [chain.repeat.exports.edge]\n\
         stage = \"decode_select_t{t}\"\n\
         entry = \"decode_init\"\n\
         \n  [chain.repeat.exports.logits]\n\
         stage = \"decode_finalize_t{t}\"\n\
         entry = \"prefill_finalize\"\n\
         \n  [[chain.repeat.stage]]\n\
         name = \"decode_select_t{t}\"\n\
         project = \"raster-stages/decode-select-token\"\n\
         inputs.logits = { from = \"decode_finalize_t{t-1}\", first = \"prefill_finalize\" }\n\
         inputs.prior = { from = \"decode_select_t{t-1}\", first = \"decode_init\" }\n\
         \n  [[chain.repeat.stage]]\n\
         name = \"decode_embed_t{t}\"\n\
         project = \"raster-stages/decode-embed\"\n\
         inputs.selected = { from = \"decode_select_t{t}\" }\n",
    );
    out.push_str(&format!(
        "  inputs.embedding = {{ external = {{ path = {:?}, index_path = {:?}, commitment = \"{embedding}\" }} }}\n",
        path_string(&layout.external_path("raster-stages/input-embedding", "embedding", "rastered")),
        path_string(&layout.external_path("raster-stages/input-embedding", "embedding", "rindex")),
    ));
    out.push_str(&format!(
        "\n  [[chain.repeat.stage]]\n\
         name = \"decode_aux_t{{t}}_l{{l}}\"\n\
         index = \"l\"\n\
         count = {}\n\
         project = \"raster-stages/prefill-prepare-aux\"\n\
         inputs.embedded = {{ from = \"decode_embed_t{{t}}\" }}\n\
         inputs.layer = {{ input = \"aux_layer_{{l}}\" }}\n",
        shape.layers
    ));

    if first_shared > 0 {
        out.push_str(&format!(
            "\n  [[chain.repeat.stage]]\n\
             name = \"decode_range_t{{t}}_l{{l}}\"\n\
             index = \"l\"\n\
             start = 0\n\
             count = {first_shared}\n\
             project = \"raster-stages/prefill-range\"\n\
             inputs.activations = {{ from = \"decode_range_t{{t}}_l{{l-1}}\", first = \"decode_embed_t{{t}}\" }}\n\
             inputs.prior_kv = {{ from = \"decode_range_t{{t-1}}_l{{l}}\", first = \"prefill_range_l{{l}}\" }}\n\
             inputs.donor_a_kv = {{ from = \"input_embedding\" }}\n\
             inputs.donor_b_kv = {{ from = \"input_embedding\" }}\n\
             inputs.ple = {{ from = \"decode_aux_t{{t}}_l{{l}}\" }}\n\
             inputs.layer = {{ input = \"transformer_layer_{{l}}\" }}\n"
        ));
    }

    if shape.num_kv_shared_layers > 0 {
        let donor_a_from = donor_a
            .map(|layer| format!("decode_range_t{{t}}_l{layer}"))
            .unwrap_or_else(|| String::from("input_embedding"));
        let donor_b_from = donor_b
            .map(|layer| format!("decode_range_t{{t}}_l{layer}"))
            .unwrap_or_else(|| String::from("input_embedding"));
        out.push_str(&format!(
            "\n  [[chain.repeat.stage]]\n\
             name = \"decode_range_t{{t}}_l{{l}}\"\n\
             index = \"l\"\n\
             start = {first_shared}\n\
             count = {}\n\
             project = \"raster-stages/prefill-range\"\n\
             inputs.activations = {{ from = \"decode_range_t{{t}}_l{{l-1}}\", first = \"decode_range_t{{t}}_l{}\" }}\n\
             inputs.prior_kv = {{ from = \"decode_range_t{{t-1}}_l{{l}}\", first = \"prefill_range_l{{l}}\" }}\n\
             inputs.donor_a_kv = {{ from = \"{donor_a_from}\" }}\n\
             inputs.donor_b_kv = {{ from = \"{donor_b_from}\" }}\n\
             inputs.ple = {{ from = \"decode_aux_t{{t}}_l{{l}}\" }}\n\
             inputs.layer = {{ input = \"transformer_layer_{{l}}\" }}\n",
            shape.num_kv_shared_layers,
            first_shared.saturating_sub(1)
        ));
    }

    out.push_str(&format!(
        "\n  [[chain.repeat.stage]]\n\
         name = \"decode_finalize_t{{t}}\"\n\
         project = \"raster-stages/prefill-finalize\"\n\
         inputs.activations = {{ from = \"decode_range_t{{t}}_l{}\" }}\n\
         inputs.head = {{ external = {{ path = {:?}, index_path = {:?}, commitment = \"{head}\" }} }}\n\
         \n[[chain.stage]]\n\
         name = \"output_finalize\"\n\
         project = \"raster-stages/output-finalize\"\n\
         inputs.edge = {{ from = \"decode.edge\" }}\n\
         {}",
        shape.layers - 1,
        path_string(&layout.external_path("raster-stages/prefill-finalize", "head", "rastered")),
        path_string(&layout.external_path("raster-stages/prefill-finalize", "head", "rindex")),
        external_line(
            "decoder",
            layout,
            "raster-stages/output-finalize",
            "decoder",
            decoder_commitment
        )
    ));
    Ok(out)
}

/// Q16.16 bits of an `f32`, through the local deterministic conversion so the
/// committed scalar is the same value the deterministic path uses, not a
/// near one.
fn det_act_bits(value: f32) -> i32 {
    det_num::f32_to_act(value).to_bits()
}

fn as_f64(value: &serde_json::Value) -> Option<f64> {
    value.as_f64()
}

/// One key out of `rope_parameters.<attention_type>`, which is where Gemma 3n
/// puts its per-family RoPE settings.
fn rope_param(text: &serde_json::Value, attention_type: &str, key: &str) -> Option<f64> {
    text.get("rope_parameters")?
        .get(attention_type)?
        .get(key)
        .and_then(as_f64)
}

/// The shapes every stage needs, read from `config.json`.
struct Shape {
    hidden: usize,
    ffn: usize,
    layers: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    /// Head dim of the *full-attention* ("global") layers, which is not the
    /// same as `head_dim` on Gemma 3n: E2B declares `head_dim: 256` for the
    /// sliding layers and `global_head_dim: 512` for the full ones, and the
    /// weights follow — a full layer's `q_proj` is [heads · 512, hidden] and
    /// its `q_norm` has 512 values. Defaults to `head_dim` for models that
    /// declare no such split.
    global_head_dim: usize,
    /// RoPE base per attention type, from `rope_parameters`. Gemma 3n gives the
    /// two families different bases — 10 000 local, 1 000 000 global — so a
    /// single `rope_theta` would silently detune half the layers.
    rope_base_sliding: f64,
    rope_base_full: f64,
    /// Fraction of a full-attention head that rotates (`0.25` here). Sliding
    /// layers rotate the whole head.
    full_partial_rotary_factor: f64,
    /// How many trailing layers borrow their K/V from an earlier donor.
    num_kv_shared_layers: usize,
    vocab: usize,
    ple_width: usize,
    sliding_window: u32,
    layer_types: Vec<String>,
    norm_eps: i64,
}

impl Shape {
    fn is_sliding(&self, idx: usize) -> bool {
        matches!(
            self.layer_types.get(idx).map(String::as_str),
            Some("sliding_attention")
        )
    }

    /// Head dim of layer `idx` — `global_head_dim` for a full-attention layer,
    /// `head_dim` for a sliding one.
    fn head_dim_at(&self, idx: usize) -> usize {
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
    fn kv_donor_layer(&self, idx: usize) -> i32 {
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
    fn rope_at(&self, idx: usize) -> (i64, u32, u32) {
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
    fn from_config(text: &serde_json::Value) -> Result<Self, Box<dyn Error>> {
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

/// Converts a config float to Q16.16, the representation everything else uses.
fn f32_to_q16(value: f64) -> i32 {
    let scaled = value * ONE as f64;
    scaled.round().clamp(i32::MIN as f64, i32::MAX as f64) as i32
}

// ---------------------------------------------------------------------------
// Stage 1 — tokenizer + prompt pieces
// ---------------------------------------------------------------------------

/// The model's `eos_token_id` set, from `config.json`.
///
/// Gemma declares more than one — `<eos>`, `<turn|>` and friends — so this is a
/// set, and it is read from the bundle rather than hardcoded. A bundle without
/// a `config.json` (or without the key) yields an empty set: nothing is marked
/// terminal, which is exactly the behaviour before this existed.
fn load_eos_ids(model_dir: &Path) -> std::collections::BTreeSet<u32> {
    let Ok(bytes) = fs::read(model_dir.join("config.json")) else {
        return Default::default();
    };
    let Ok(config) = serde_json::from_slice::<serde_json::Value>(&bytes) else {
        return Default::default();
    };
    let declared = config.get("eos_token_id").or_else(|| {
        config
            .get("text_config")
            .and_then(|t| t.get("eos_token_id"))
    });
    match declared {
        Some(serde_json::Value::Number(id)) => {
            id.as_u64().map(|id| id as u32).into_iter().collect()
        }
        Some(serde_json::Value::Array(ids)) => ids
            .iter()
            .filter_map(serde_json::Value::as_u64)
            .map(|id| id as u32)
            .collect(),
        _ => Default::default(),
    }
}

fn write_tokenizer(
    tokenizer: &serde_json::Value,
    eos_ids: &std::collections::BTreeSet<u32>,
    layout: &ArtifactLayout,
    stages: &mut Vec<String>,
) -> Result<String, Box<dyn Error>> {
    // Reject unsupported profiles, invalid IDs and malformed merge tables
    // before replacing any existing tokenizer/decoder artifact.
    let prepared_tokenizer = run_prep::prompt_tokenizer(tokenizer)?;
    let model = tokenizer
        .get("model")
        .ok_or("tokenizer.json has no model")?;
    let vocab_map = run_prep::vocab_map(model)?;

    let mut vocab: Vec<TokenEntry> = vocab_map
        .iter()
        .map(|(token, id)| TokenEntry {
            token: token.clone(),
            id: *id,
        })
        .collect();
    vocab.sort_by_key(|entry| entry.id);

    let special_ids: std::collections::BTreeSet<u32> = tokenizer
        .get("added_tokens")
        .and_then(serde_json::Value::as_array)
        .into_iter()
        .flatten()
        .filter(|entry| {
            entry
                .get("special")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false)
        })
        .filter_map(|entry| entry.get("id").and_then(serde_json::Value::as_u64))
        .map(|id| id as u32)
        .collect();

    let max_token_id = vocab.iter().map(|entry| entry.id).max().unwrap_or(0);
    let mut decoder_tokens = vec![DecoderToken::default(); max_token_id.saturating_add(1) as usize];
    for entry in &vocab {
        decoder_tokens[entry.id as usize] = DecoderToken {
            token: entry.token.clone(),
            special: special_ids.contains(&entry.id),
            terminal: eos_ids.contains(&entry.id),
        };
    }
    let decoder_commitment = write_external(
        &DecoderTable {
            tokens: decoder_tokens.into(),
        },
        layout,
        "raster-stages/output-finalize",
        "decoder",
    )?;

    let tokenizer_commitment = write_external(
        &prepared_tokenizer,
        layout,
        "raster-stages/prompt-prepare",
        "tokenizer",
    )?;
    // Run preparation supplies the prompt-specific merged_pieces binding.
    write_stage_fixtures(
        "raster-stages/prompt-prepare",
        layout,
        &[("tokenizer", "tokenizer", &tokenizer_commitment)],
    )?;
    let tokenizer_line = external_line(
        "tokenizer",
        layout,
        "raster-stages/prompt-prepare",
        "tokenizer",
        &tokenizer_commitment,
    );
    let pieces_line = external_line(
        "initial_pieces",
        layout,
        "target/raster-inference/run-prompt-placeholder",
        "initial_pieces",
        INITIAL_PIECES_PLACEHOLDER_COMMITMENT,
    );
    stages.push(format!(
        r#"[[chain.stage]]
name = "prompt_merge_seed"
project = "raster-stages/prompt-merge"
{tokenizer_line}{pieces_line}
[[chain.repeat]]
name = "tokenize"
index = "b"
count = 0

[[chain.repeat.stage]]
name = "prompt_merge_b{{b}}"
project = "raster-stages/prompt-merge"
inputs.initial_pieces = {{ from = "prompt_merge_b{{b-1}}", first = "prompt_merge_seed" }}
{tokenizer_line}
[chain.repeat.exports.pieces]
stage = "prompt_merge_b{{b}}"
entry = "prompt_merge_seed"

[[chain.stage]]
name = "prompt_prepare"
project = "raster-stages/prompt-prepare"
inputs.merged_pieces = {{ from = "tokenize.pieces" }}
{tokenizer_line}"#
    ));
    Ok(decoder_commitment)
}

// ---------------------------------------------------------------------------
// Stage 2 — input embedding
// ---------------------------------------------------------------------------

fn write_embedding(
    weights: &detwgt::Artifact,
    shape: &Shape,
    layout: &ArtifactLayout,
    stages: &mut Vec<String>,
) -> Result<(), Box<dyn Error>> {
    let embed = weights.get("model.language_model.embed_tokens.weight")?;
    let mut values = Vec::with_capacity(shape.vocab * shape.hidden);
    for token_id in 0..shape.vocab {
        values.extend_from_slice(embed.row(token_id)?);
    }

    let commitment = write_external(
        &EmbeddingTable {
            hidden_size: shape.hidden as u32,
            // Gemma scales token embeddings by sqrt(hidden) before layer 0. Converted
            // by local `f32_to_act` so the committed bits match the
            // deterministic path exactly rather than approximately.
            embedding_scale: det_num::f32_to_act((shape.hidden as f32).sqrt()).to_bits(),
            values: paged_i32s(&values).map_err(|error| error.to_string())?,
        },
        layout,
        "raster-stages/input-embedding",
        "embedding",
    )?;

    stages.push(format!(
        concat!(
            "[[chain.stage]]\n",
            "name = \"input_embedding\"\n",
            "project = \"raster-stages/input-embedding\"\n",
            "inputs.prompt = {{ from = \"prompt_prepare\" }}\n",
            "{}"
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

fn write_ple_layers(
    weights: &detwgt::Artifact,
    shape: &Shape,
    layout: &ArtifactLayout,
    stages: &mut Vec<String>,
) -> Result<(), Box<dyn Error>> {
    let per_layer = weights.get("model.language_model.embed_tokens_per_layer.weight")?;
    let projection = weights.get("model.language_model.per_layer_model_projection.weight")?;
    let projection_norm = weights.get("model.language_model.per_layer_projection_norm.weight")?;

    for layer_idx in 0..shape.layers {
        let start = layer_idx * shape.ple_width;
        let end = start + shape.ple_width;

        // The packed [vocab, layers · width] table gives this layer its slice.
        let slice = per_layer.column_slice(start, end)?;

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
                norm_weights: page_of_i32s(&projection_norm.values),
            },
            embeddings: paged_i32s(&slice).map_err(|error| error.to_string())?,
            projection: paged_i32s(&projection.rows(start, end)?)
                .map_err(|error| error.to_string())?,
        };

        let name = format!("layer{layer_idx}");
        let commitment =
            write_external(&layer, layout, "raster-stages/prefill-prepare-aux", &name)?;
        stages.push(format!(
            concat!(
                "[[chain.stage]]\n",
                "name = \"prefill_prepare_aux_l{idx}\"\n",
                "project = \"raster-stages/prefill-prepare-aux\"\n",
                "inputs.embedded = {{ from = \"input_embedding\" }}\n",
                "{line}"
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

fn write_transformer_layers(
    weights: &detwgt::Artifact,
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
        let gate = weights.get(&at("mlp.gate_proj.weight"))?;
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
        let w_v = match weights.get(&at("self_attn.v_proj.weight")) {
            Ok(tensor) => &tensor.values,
            Err(_) => &weights.get(&at("self_attn.k_proj.weight"))?.values,
        };
        let layer_scalar = weights
            .get(&at("layer_scalar"))
            .ok()
            .and_then(|tensor| tensor.values.first().copied())
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
                norm_input: page_of_i32s(&weights.get(&at("input_layernorm.weight"))?.values),
                norm_post_attn: page_of_i32s(
                    &weights.get(&at("post_attention_layernorm.weight"))?.values,
                ),
                norm_pre_ffw: page_of_i32s(
                    &weights.get(&at("pre_feedforward_layernorm.weight"))?.values,
                ),
                norm_post_ffw: page_of_i32s(
                    &weights
                        .get(&at("post_feedforward_layernorm.weight"))?
                        .values,
                ),
                q_norm: page_of_i32s(&weights.get(&at("self_attn.q_norm.weight"))?.values),
                k_norm: page_of_i32s(&weights.get(&at("self_attn.k_norm.weight"))?.values),
                ple_width: shape.ple_width as u32,
                // Hidden-width, despite the name: it normalises *after*
                // `per_layer_projection` maps back up from `ple_width`.
                ple_post_norm: page_of_i32s(
                    &weights.get(&at("post_per_layer_input_norm.weight"))?.values,
                ),
            },
            w_q: paged_i32s(&weights.get(&at("self_attn.q_proj.weight"))?.values)
                .map_err(|error| error.to_string())?,
            w_k: paged_i32s(&weights.get(&at("self_attn.k_proj.weight"))?.values)
                .map_err(|error| error.to_string())?,
            w_v: paged_i32s(w_v).map_err(|error| error.to_string())?,
            w_o: paged_i32s(&weights.get(&at("self_attn.o_proj.weight"))?.values)
                .map_err(|error| error.to_string())?,
            w_gate: paged_i32s(&gate.values).map_err(|error| error.to_string())?,
            w_up: paged_i32s(&weights.get(&at("mlp.up_proj.weight"))?.values)
                .map_err(|error| error.to_string())?,
            ple_input_gate: paged_i32s(&weights.get(&at("per_layer_input_gate.weight"))?.values)
                .map_err(|error| error.to_string())?,
            ple_layer_projection: paged_i32s(
                &weights.get(&at("per_layer_projection.weight"))?.values,
            )
            .map_err(|error| error.to_string())?,
            w_down: paged_i32s(&weights.get(&at("mlp.down_proj.weight"))?.values)
                .map_err(|error| error.to_string())?,
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
fn inv_sqrt_q16(n: usize) -> i32 {
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

fn write_head(
    weights: &detwgt::Artifact,
    shape: &Shape,
    text: &serde_json::Value,
    layout: &ArtifactLayout,
    stages: &mut Vec<String>,
) -> Result<(), Box<dyn Error>> {
    let tied = text
        .get("tie_word_embeddings")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    let projection = if tied {
        weights.get("model.language_model.embed_tokens.weight")?
    } else {
        weights.get("model.language_model.lm_head.weight")?
    };
    let softcap = text
        .get("final_logit_softcapping")
        .and_then(serde_json::Value::as_f64)
        .map(f32_to_q16)
        .unwrap_or(0);

    let mut rows = Vec::with_capacity(shape.vocab * shape.hidden);
    for token_id in 0..shape.vocab {
        rows.extend_from_slice(projection.row(token_id)?);
    }

    let commitment = write_external(
        &FinalHead {
            params: FinalHeadParams {
                hidden_size: shape.hidden as u32,
                norm_eps: shape.norm_eps,
                softcap,
                norm_weights: page_of_i32s(
                    &weights.get("model.language_model.norm.weight")?.values,
                ),
            },
            projection: paged_i32s(&rows).map_err(|error| error.to_string())?,
        },
        layout,
        "raster-stages/prefill-finalize",
        "head",
    )?;

    stages.push(format!(
        concat!(
            "[[chain.stage]]\n",
            "name = \"prefill_finalize\"\n",
            "project = \"raster-stages/prefill-finalize\"\n",
            "inputs.activations = {{ from = \"prefill_range_l{last}\" }}\n",
            "{line}"
        ),
        last = shape.layers - 1,
        line = external_line(
            "head",
            layout,
            "raster-stages/prefill-finalize",
            "head",
            &commitment
        )
    ));
    Ok(())
}

// ---------------------------------------------------------------------------

fn write_external<T: serde::Serialize>(
    value: &T,
    layout: &ArtifactLayout,
    stage_dir: &str,
    name: &str,
) -> Result<String, Box<dyn Error>> {
    let dir = layout.external_dir(stage_dir);
    fs::create_dir_all(&dir)?;
    let commitment = raster::write_raster_files(
        value,
        &dir.join(format!("{name}.rastered")),
        &dir.join(format!("{name}.rindex")),
    )?;
    println!(
        "wrote {}  commitment = {commitment}",
        layout.external_path(stage_dir, name, "rastered").display()
    );
    Ok(commitment)
}

/// Writes a stage's `input.json` / `input_manifest.json` so it can also be run
/// on its own with `cargo raster run`.
fn write_stage_fixtures(
    stage_dir: &str,
    layout: &ArtifactLayout,
    params: &[(&str, &str, &String)],
) -> Result<(), Box<dyn Error>> {
    if layout.no_stage_fixtures {
        return Ok(());
    }
    let entries = |render: &dyn Fn(&str, &str, &str) -> String| -> String {
        params
            .iter()
            .map(|(param, name, commitment)| render(param, name, commitment))
            .collect::<Vec<_>>()
            .join(",\n")
    };
    let input_json = entries(&|param, name, _| {
        let data_path = layout.external_path(stage_dir, name, "rastered");
        let index_path = layout.external_path(stage_dir, name, "rindex");
        let stage_relative_data = path_from_dir(Path::new(stage_dir), &data_path);
        let stage_relative_index = path_from_dir(Path::new(stage_dir), &index_path);
        format!(
            "  \"{param}\": {{ \"path\": {:?}, \"index_path\": {:?}, \"load_preference\": \"read\" }}",
            path_string(&stage_relative_data),
            path_string(&stage_relative_index)
        )
    });
    let manifest_json = entries(&|param, _, commitment| {
        format!("  \"{param}\": {{ \"type\": \"sha256\", \"encoding\": \"raster\", \"commitment\": \"{commitment}\" }}")
    });
    let dir = Path::new(stage_dir);
    fs::write(dir.join("input.json"), format!("{{\n{input_json}\n}}\n"))?;
    fs::write(
        dir.join("input_manifest.json"),
        format!("{{\n{manifest_json}\n}}\n"),
    )?;
    println!("wrote {stage_dir}/input.json + input_manifest.json");
    Ok(())
}

fn external_line(
    param: &str,
    layout: &ArtifactLayout,
    stage_dir: &str,
    name: &str,
    commitment: &str,
) -> String {
    let (data_path, index_path) = if stage_dir.starts_with("raster-stages/") {
        (
            layout.external_path(stage_dir, name, "rastered"),
            layout.external_path(stage_dir, name, "rindex"),
        )
    } else {
        (
            Path::new(stage_dir).join(format!("{name}.rastered")),
            Path::new(stage_dir).join(format!("{name}.rindex")),
        )
    };
    format!(
        "inputs.{param} = {{ external = {{ path = {:?}, index_path = {:?}, commitment = \"{commitment}\" }} }}\n",
        path_string(&data_path),
        path_string(&index_path),
    )
}

fn path_string(path: &Path) -> String {
    path.to_string_lossy().replace('\\', "/")
}

fn path_from_dir(base_dir: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        return path.to_path_buf();
    }
    let mut relative = PathBuf::new();
    for component in base_dir.components() {
        if matches!(component, Component::Normal(_)) {
            relative.push("..");
        }
    }
    relative.join(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn import_saves_independent_templates_with_verified_provenance() {
        let base = std::env::temp_dir().join(format!(
            "model-import-templates-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mut args = parse_args_from(["--model", "unused"].map(String::from)).unwrap();
        args.artifact_root = base.join("artifacts");
        let text = serde_json::json!({
            "hidden_size": 2, "num_hidden_layers": 1, "num_attention_heads": 1,
            "num_key_value_heads": 1, "head_dim": 2, "vocab_size": 2, "hidden_size_per_layer_input": 1
        });
        let shape = Shape::from_config(&text).unwrap();
        let mut models = Vec::new();
        for name in ["model-a", "model-b"] {
            args.model_dir = base.join(name);
            fs::create_dir_all(&args.model_dir).unwrap();
            fs::write(args.model_dir.join("model.detwgt"), name).unwrap();
            fs::write(
                args.model_dir.join("config.json"),
                serde_json::to_vec(&serde_json::json!({"text_config": text})).unwrap(),
            )
            .unwrap();
            fs::write(args.model_dir.join("tokenizer.json"), "{}").unwrap();
            let manifest = format!("[chain]\nname = \"{name}\"\n[[chain.stage]]\nname = \"prompt_prepare\"\nproject = \"raster-stages/prompt-prepare\"\ninputs.tokenizer = {{ external = {{ path = \"data/{name}.rastered\", index_path = \"data/{name}.rindex\", commitment = \"abc\" }} }}\n");
            let (path, template) = save_model_template(&args, &manifest).unwrap();
            assert!(path.is_file()); // No --manifest is necessary.
            assert!(template.contains(&format!(
                "project = {:?}",
                std::env::current_dir()
                    .unwrap()
                    .join("raster-stages/prompt-prepare")
                    .to_string_lossy()
            )));
            assert!(template.contains(&format!(
                "index_path = {:?}",
                std::env::current_dir()
                    .unwrap()
                    .join(format!("data/{name}.rindex"))
                    .to_string_lossy()
            )));
            let model_path = direct_manifest::write_model_manifest(
                &args,
                &Default::default(),
                &shape,
                &text,
                Some((&path, &template)),
            )
            .unwrap();
            let model = inference_artifacts::ModelManifest::load_verified(&model_path).unwrap();
            assert_eq!(
                model.verified_raster_template(&model_path).unwrap().1,
                template
            );
            assert_eq!(
                model.provenance.as_ref().unwrap().raster_manifest_path,
                PathBuf::from("Raster.toml")
            );
            models.push((model_path, template));
            // The next model may overwrite this shared copy, but provenance
            // must continue pointing at each model's own template.
            args.manifest = Some(base.join("shared/Raster.toml"));
        }
        fs::write(base.join("shared/Raster.toml"), "overwritten").unwrap();
        for (path, template) in &models {
            let model = inference_artifacts::ModelManifest::load_verified(path).unwrap();
            assert_eq!(model.verified_raster_template(path).unwrap().1, *template);
        }
        args.only_direct = true;
        import_model(args).unwrap();
        let direct_only = inference_artifacts::ModelManifest::load_verified(&models[1].0).unwrap();
        assert!(direct_only.provenance.is_none());
        assert!(direct_only.verified_raster_template(&models[1].0).is_err());
        fs::remove_dir_all(base).unwrap();
    }

    #[test]
    fn parse_args_builds_import_config() {
        let config = parse_args_from(
            [
                "--model",
                "model-bundles/model",
                "--manifest",
                "Raster.toml",
                "--only-tokenizer",
                "--only-embedding",
            ]
            .map(String::from),
        )
        .unwrap();

        assert_eq!(
            config,
            ImportConfig {
                model_dir: PathBuf::from("model-bundles/model"),
                manifest: Some(PathBuf::from("Raster.toml")),
                artifact_root: PathBuf::from(inference_artifacts::MODEL_ARTIFACTS_DIR),
                model_id: None,
                only_tokenizer: true,
                only_layers: false,
                only_embedding: true,
                only_ple: false,
                only_direct: false,
                no_stage_fixtures: false,
            }
        );
    }

    #[test]
    fn parse_args_accepts_only_direct() {
        let config =
            parse_args_from(["--model", "model-bundles/model", "--only-direct"].map(String::from))
                .unwrap();

        assert!(config.only_direct);
        assert!(!config.only_tokenizer);
        assert!(!config.only_layers);
        assert!(!config.only_embedding);
        assert!(!config.only_ple);
    }

    #[test]
    fn no_stage_fixtures_preserves_existing_files() {
        let mut config = parse_args_from(
            ["--model", "model-bundles/model", "--no-stage-fixtures"].map(String::from),
        )
        .unwrap();
        assert!(config.no_stage_fixtures);
        let dir = std::env::temp_dir().join(format!("import-no-fixtures-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("input.json"), "existing input").unwrap();
        fs::write(dir.join("input_manifest.json"), "existing manifest").unwrap();
        write_stage_fixtures(
            dir.to_str().unwrap(),
            &ArtifactLayout::from_config(&config),
            &[],
        )
        .unwrap();
        assert_eq!(
            fs::read_to_string(dir.join("input.json")).unwrap(),
            "existing input"
        );
        assert_eq!(
            fs::read_to_string(dir.join("input_manifest.json")).unwrap(),
            "existing manifest"
        );
        config.no_stage_fixtures = false;
        write_stage_fixtures(
            dir.to_str().unwrap(),
            &ArtifactLayout::from_config(&config),
            &[],
        )
        .unwrap();
        assert_ne!(
            fs::read_to_string(dir.join("input.json")).unwrap(),
            "existing input"
        );
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn parse_args_accepts_artifact_layout_options() {
        let config = parse_args_from(
            [
                "--model",
                "model-bundles/google/gemma-4",
                "--artifact-root",
                "runtime/models",
                "--model-id",
                "google/gemma 4",
            ]
            .map(String::from),
        )
        .unwrap();

        assert_eq!(config.artifact_root, PathBuf::from("runtime/models"));
        assert_eq!(config.model_id.as_deref(), Some("google-gemma-4"));
        assert_eq!(
            config.artifact_dir(),
            PathBuf::from("runtime/models/google-gemma-4")
        );
    }

    #[test]
    fn external_lines_use_model_scoped_raster_paths() {
        let config = ImportConfig {
            model_dir: PathBuf::from("model-bundles/model"),
            manifest: None,
            artifact_root: PathBuf::from("runtime/model-artifacts"),
            model_id: Some(String::from("gemma")),
            only_tokenizer: false,
            only_layers: false,
            only_embedding: false,
            only_ple: false,
            only_direct: false,
            no_stage_fixtures: false,
        };
        let layout = ArtifactLayout::from_config(&config);

        let line = external_line(
            "embedding",
            &layout,
            "raster-stages/input-embedding",
            "embedding",
            "abc",
        );

        assert!(line.contains(
            r#"path = "runtime/model-artifacts/gemma/raster/input-embedding/embedding.rastered""#
        ));
        assert!(line.contains(
            r#"index_path = "runtime/model-artifacts/gemma/raster/input-embedding/embedding.rindex""#
        ));
    }

    #[test]
    fn run_specific_prompt_placeholder_stays_outside_model_artifacts() {
        let config = ImportConfig {
            model_dir: PathBuf::from("model-bundles/model"),
            manifest: None,
            artifact_root: PathBuf::from("runtime/model-artifacts"),
            model_id: Some(String::from("gemma")),
            only_tokenizer: false,
            only_layers: false,
            only_embedding: false,
            only_ple: false,
            only_direct: false,
            no_stage_fixtures: false,
        };
        let layout = ArtifactLayout::from_config(&config);

        let line = external_line(
            "initial_pieces",
            &layout,
            "target/raster-inference/run-prompt-placeholder",
            "initial_pieces",
            "abc",
        );

        assert!(line.contains(
            r#"path = "target/raster-inference/run-prompt-placeholder/initial_pieces.rastered""#
        ));
        assert!(!line.contains("runtime/model-artifacts"));
    }

    #[test]
    fn stage_fixture_paths_are_relative_to_stage_dir() {
        let config = ImportConfig {
            model_dir: PathBuf::from("model-bundles/model"),
            manifest: None,
            artifact_root: PathBuf::from("runtime/model-artifacts"),
            model_id: Some(String::from("gemma")),
            only_tokenizer: false,
            only_layers: false,
            only_embedding: false,
            only_ple: false,
            only_direct: false,
            no_stage_fixtures: false,
        };
        let layout = ArtifactLayout::from_config(&config);
        let path = path_from_dir(
            Path::new("raster-stages/input-embedding"),
            &layout.external_path("raster-stages/input-embedding", "embedding", "rastered"),
        );

        assert_eq!(
            path,
            PathBuf::from(
                "../../runtime/model-artifacts/gemma/raster/input-embedding/embedding.rastered"
            )
        );
    }

    #[test]
    fn parse_args_rejects_only_direct_with_other_partial_modes() {
        let error = parse_args_from(
            [
                "--model",
                "model-bundles/model",
                "--only-direct",
                "--only-tokenizer",
            ]
            .map(String::from),
        )
        .unwrap_err()
        .to_string();

        assert!(error.contains("--only-direct cannot be combined"));
    }

    #[test]
    fn parse_args_requires_model_dir() {
        let error = parse_args_from([String::from("--manifest"), String::from("Raster.toml")])
            .unwrap_err()
            .to_string();

        assert!(error.contains("--model <bundle-dir> is required"));
    }

    #[test]
    fn parse_args_rejects_import_time_prompt_flags() {
        let error = parse_args_from(
            [
                "--model",
                "model-bundles/model",
                "--prompt",
                "hello",
                "--tokens",
                "3",
            ]
            .map(String::from),
        )
        .unwrap_err()
        .to_string();

        assert!(error.contains("unknown argument '--prompt'"));
    }
}
