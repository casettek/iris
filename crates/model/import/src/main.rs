//! Import a model bundle into this chain's committed externals.
//!
//! This file is what every model family shares: the command line, the artifact
//! layout, the tokenizer externals, commitments, the chain template and the
//! manifest. What is particular to a family (its config, its tensors and how
//! they become stage externals) is in that family's module; today that is
//! [`gemma`] alone, and [`bundle_family`] refuses anything else.
//!
//! ```sh
//! cargo run -p model-import -- \
//!   --model ../iris/assets/tiny-gemma-dev
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
mod gemma;
mod mapped_weights;

use externals::*;
use gemma::*;
use inference_artifacts::ModelFamily;
use std::collections::BTreeMap;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

/// Q16.16 one.
const ONE: i32 = 1 << 16;

const INITIAL_PIECES_PLACEHOLDER_COMMITMENT: &str =
    "0000000000000000000000000000000000000000000000000000000000000000";

/// Run preparation replaces every external under this directory with the
/// run's own prompt pieces and, for teacher-forced runs, claimed output.
const RUN_PLACEHOLDER_DIR: &str = "target/iris/run-prompt-placeholder";

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
    /// re-emit them. Prints only the `prompt_bind` stage's tokenizer bindings.
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
    /// Accepted and ignored. Import once wrote input fixtures into a stage's
    /// directory and this kept them; it writes nothing there now.
    pub no_stage_fixtures: bool,
    /// Largest teacher-forced range, in prompt tokens or claimed ids. Every
    /// per-position stage of the teacher-forced chain covers at most this many
    /// positions, which bounds the cost of replaying any one of them.
    pub range_size: u32,
}

/// Default teacher-forced range size. On Gemma 4 E2B one transformer-layer
/// stage replays at roughly 0.4 s per row, so 512 rows is a few minutes.
pub const DEFAULT_RANGE_SIZE: u32 = 512;

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
    let mut range_size = DEFAULT_RANGE_SIZE;
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
            "--range-size" => {
                range_size = args
                    .next()
                    .ok_or("--range-size <tokens> is required")?
                    .parse()?;
                if range_size < 2 {
                    return Err("--range-size must be at least 2".into());
                }
            }
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
        range_size,
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
}

impl ArtifactLayout {
    fn from_config(args: &ImportConfig) -> Self {
        Self {
            model_dir: args.artifact_dir(),
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

/// The family a bundle's `config.json` declares.
///
/// A config says what it is in `model_type`, on the text model or on the whole
/// model. Bundles made for this chain before families existed say nothing, and
/// are Gemma's. Two families can share a `model_type` (a dense and a
/// mixture-of-experts Gemma both say `gemma4`), so this is the first check and
/// the family's own tensor inventory is the second.
fn bundle_family(config: &serde_json::Value) -> Result<ModelFamily, Box<dyn Error>> {
    let declared = [config.get("text_config"), Some(config)]
        .into_iter()
        .flatten()
        .find_map(|section| {
            section
                .get("model_type")
                .and_then(serde_json::Value::as_str)
        });
    match declared {
        None => Ok(ModelFamily::GemmaDense),
        Some(model_type) if model_type.starts_with("gemma") => Ok(ModelFamily::GemmaDense),
        Some(model_type) => Err(format!(
            "config.json declares model type `{model_type}`, which is not a family this build \
             imports (supported: {})",
            ModelFamily::ALL.map(ModelFamily::name).join(", ")
        )
        .into()),
    }
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
        let family = bundle_family(&config)?;
        let shape = Shape::from_config(text)?;
        // A direct-only refresh cannot attest that existing Raster externals
        // were generated from this bundle. Only a full import binds a template.
        direct_manifest::write_model_manifest(
            &args,
            &eos_ids,
            family,
            shape.manifest_shape(text),
            None,
            &[],
        )?;
        return Ok(ImportResult {
            manifest_path: None,
            manifest_text: None,
            stage_count: 0,
        });
    }

    if args.only_tokenizer {
        let mut stages = Vec::new();
        let (decoder_commitment, witness_commitment) =
            write_tokenizer(&tokenizer, &eos_ids, &layout, &mut stages)?;
        println!();
        println!("# ---- tokenizer and decoder externals ----");
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
                "raster-stages/sequence-check",
                "decoder",
                &decoder_commitment
            )
        );
        print!(
            "{}",
            external_line(
                "witness",
                &layout,
                "raster-stages/sequence-check",
                "witness",
                &witness_commitment
            )
        );
        direct_manifest::warn_partial_direct_manifest_not_refreshed(&args);
        return Ok(ImportResult {
            manifest_path: None,
            manifest_text: None,
            stage_count: stages.len(),
        });
    }

    // Keep the bundle file-backed instead of owning the entire file plus all
    // decoded tensors. Writers pack only the current external into pages.
    let weights = detwgt::MmapDetwgt::open(&args.model_dir.join("model.detwgt"))?;
    let config: serde_json::Value =
        serde_json::from_slice(&fs::read(args.model_dir.join("config.json"))?)?;
    let text = config
        .get("text_config")
        .ok_or("config.json has no text_config")?;
    let family = bundle_family(&config)?;
    match family {
        ModelFamily::GemmaDense => {
            require_known_tensors(weights.names(), Shape::from_config(text)?.layers)?
        }
    }

    if args.only_ple {
        let shape = Shape::from_config(text)?;
        let mut stages = Vec::new();
        write_ple_layers(&weights, &shape, &layout, &mut stages)?;
        println!();
        println!("# ---- PLE layer externals ----");
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
        println!("# ---- embedding external ----");
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
        println!("# ---- transformer layer externals ----");
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
    println!("weights: memory-mapped DETWGT");
    println!();

    let mut stages = Vec::new();
    let (decoder_commitment, witness_commitment) =
        write_tokenizer(&tokenizer, &eos_ids, &layout, &mut stages)?;
    write_embedding(&weights, &shape, &layout, &mut stages)?;
    write_ple_layers(&weights, &shape, &layout, &mut stages)?;
    write_transformer_layers(&weights, &shape, &layout, &mut stages)?;
    write_head(&weights, &shape, text, &layout, &mut stages)?;

    let commitments = model_commitments(&shape, &decoder_commitment, &stages)?;
    let descriptor = inference_artifacts::model_identity::descriptor_from_commitments(
        &commitments,
        shape.layers,
    )?;
    let descriptor =
        raster::BytesPage::__from_parts(0, 0, descriptor.encode().map_err(|f| format!("{f:?}"))?);
    let descriptor_commitment = write_external(
        &descriptor,
        &layout,
        "raster-stages/sequence-check",
        "descriptor",
    )?;
    let identity_inputs = format!(
        "{}{}",
        external_line(
            "descriptor",
            &layout,
            "raster-stages/sequence-check",
            "descriptor",
            &descriptor_commitment
        ),
        external_line(
            "witness",
            &layout,
            "raster-stages/sequence-check",
            "witness",
            &witness_commitment
        ),
    );
    let stage_count = stages.len();
    let (template_path, template) = save_model_template(
        &args,
        &chain_template(
            &shape,
            &layout,
            &decoder_commitment,
            &identity_inputs,
            &stages,
        )?,
    )?;
    direct_manifest::write_model_manifest(
        &args,
        &eos_ids,
        family,
        shape.manifest_shape(text),
        Some((&template_path, &template)),
        &commitments,
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

/// Every committed model input by role, in the order the model id hashes them.
///
/// These are the commitments a stage's `input_manifest.json` binds, so a claim
/// that ran other weights can be held to the model it names by comparing them.
fn model_commitments(
    shape: &Shape,
    decoder_commitment: &str,
    stages: &[String],
) -> Result<Vec<inference_artifacts::ModelCommitment>, Box<dyn Error>> {
    let entry = |role: String, commitment: String| inference_artifacts::ModelCommitment {
        role,
        commitment,
    };
    let mut entries = vec![
        entry(
            String::from("tokenizer"),
            stage_external_commitment(stages, "prompt_bind", "tokenizer")?,
        ),
        entry(String::from("decoder"), decoder_commitment.to_string()),
        entry(
            String::from("embedding"),
            stage_external_commitment(stages, "input_embedding", "embedding")?,
        ),
        entry(
            String::from("head"),
            stage_external_commitment(stages, "final_head", "head")?,
        ),
    ];
    for layer in 0..shape.layers {
        entries.push(entry(
            format!("aux_layer_{layer}"),
            stage_external_commitment(stages, &format!("prefill_prepare_aux_l{layer}"), "layer")?,
        ));
    }
    for layer in 0..shape.layers {
        entries.push(entry(
            format!("transformer_layer_{layer}"),
            stage_external_commitment(stages, &format!("prefill_range_l{layer}"), "layer")?,
        ));
    }
    Ok(entries)
}

/// The chain template: one pass over `prompt ‖ claimed output`, cut into ranges of at most `range_size`
/// positions so no stage's cost grows with the prompt.
///
/// Run preparation tokenizes the prompt and cuts the tokens into ranges of at
/// most `range_size`, and the text into one slice per range.
/// `prompt_bind_p{j}` hashes its slice and splits it, and `prompt_check_p{j}`
/// checks the range's tokens against the merge. The claimed ids are cut into
/// ranges of the same size.
/// Every prompt range is checked and embedded, every claimed range embedded; then each
/// layer runs across all ranges in order (`range_p{j}_l{l}`, then
/// `range_c{k}_l{l}`), and `check_c{k}` re-derives the greedy selection for
/// each claimed range. A layer's range takes its rows from the layer below and
/// its inherited K/V cache from its own previous range, exactly as a decode
/// step does.
///
/// Range 0 has nothing before it. Where a later range takes the range before,
/// its stages take an empty value, written here as a committed constant, and
/// check that it is empty. A repeat's `first` can only name a stage, so range
/// 0 is written out as plain stages and each repeat starts at range 1 with
/// range 0 as its `first`. No stage exists only to produce an empty value.
///
/// Counts, the last prompt range's index, the text, the configuration and the
/// per-range externals are filled in by run preparation. Wrong marks do not
/// stop a stage: the fault is recorded in the prompt context, which the last
/// range's `prompt_check` publishes for the whole prompt.
fn chain_template(
    shape: &Shape,
    layout: &ArtifactLayout,
    decoder_commitment: &str,
    identity_inputs: &str,
    prefill_stages: &[String],
) -> Result<String, Box<dyn Error>> {
    let descriptor_line = identity_inputs
        .lines()
        .find(|line| line.starts_with("inputs.descriptor ="))
        .ok_or("missing model descriptor binding")?;
    let tokenizer = stage_external_commitment(prefill_stages, "prompt_bind", "tokenizer")?;
    let embedding = stage_external_commitment(prefill_stages, "input_embedding", "embedding")?;
    let no_claim = stage_external_commitment(prefill_stages, "input_embedding", "claimed")?;
    let head = stage_external_commitment(prefill_stages, "final_head", "head")?;
    let no_prompt = no_prompt_commitment(layout)?;
    let tokenizer_witness = stage_external_commitment(prefill_stages, "prompt_bind", "witness")?;
    let tokenizer_line = external_line(
        "tokenizer",
        layout,
        "raster-stages/prompt-bind",
        "tokenizer",
        &tokenizer,
    ) + &external_line(
        "witness",
        layout,
        "raster-stages/prompt-bind",
        "witness",
        &tokenizer_witness,
    );
    let embedding_witness =
        stage_external_commitment(prefill_stages, "input_embedding", "witness")?;
    let embedding_line = external_line(
        "embedding",
        layout,
        "raster-stages/input-embedding",
        "embedding",
        &embedding,
    ) + &external_line(
        "witness",
        layout,
        "raster-stages/input-embedding",
        "witness",
        &embedding_witness,
    );
    let range_input = |name: &str, family: &str, index: &str| {
        let path = |extension: &str| {
            path_string(
                &Path::new(RUN_PLACEHOLDER_DIR).join(format!("{family}_{{{index}}}.{extension}")),
            )
        };
        format!(
            "\n[chain.input.{name}]\nindex = \"{index}\"\npath = {:?}\nindex_path = {:?}\ncommitments = []\n",
            path("rastered"),
            path("rindex"),
        )
    };
    let last_layer = shape.layers - 1;
    let first_shared = shape.layers.saturating_sub(shape.num_kv_shared_layers);
    let mut donors = (0..shape.layers)
        .map(|layer| shape.kv_donor_layer(layer))
        .filter(|donor| *donor >= 0)
        .collect::<Vec<_>>();
    donors.sort_unstable();
    donors.dedup();
    if donors.len() > 2 {
        return Err(format!(
            "ranges support at most two donor candidates, model declares {}",
            donors.len()
        )
        .into());
    }

    let mut out = String::from("[chain]\nname = \"iris\"\nversion = \"0.1.0\"\n");
    out.push_str(&indexed_model_inputs(shape, layout, prefill_stages)?);
    out.push_str(&range_input("prompt_text", "text", "j"));
    out.push_str(&range_input("prompt_marks", "marks", "j"));
    out.push_str(&range_input("claimed_ids", "claimed", "k"));
    // The empty values range 0 starts from.
    use inference_kernels::kernels::origin;
    let empty_sequence = write_external(
        &origin::empty_sequence(),
        layout,
        "raster-stages/input-embedding",
        "empty_sequence",
    )?;
    let empty_prompt = write_external(
        &origin::empty_prompt(),
        layout,
        "raster-stages/prompt-bind",
        "empty_prompt",
    )?;
    let empty_edge = write_external(
        &origin::empty_decode_edge(),
        layout,
        "raster-stages/sequence-check",
        "empty_edge",
    )?;
    let no_sequence = |param: &str| {
        external_line(
            param,
            layout,
            "raster-stages/input-embedding",
            "empty_sequence",
            &empty_sequence,
        )
    };
    let no_prompt_state = |param: &str| {
        external_line(
            param,
            layout,
            "raster-stages/prompt-bind",
            "empty_prompt",
            &empty_prompt,
        )
    };

    // For every range: bind its slice of the text and split it into pieces,
    // check its tokens against them, and embed it. The first range's bind
    // also binds the configuration, the stage schedule and what the run
    // claims of the whole text; later ones are given the same three and
    // read none of them.
    let run_external = |param: &str| {
        external_line(
            param,
            layout,
            RUN_PLACEHOLDER_DIR,
            param,
            INITIAL_PIECES_PLACEHOLDER_COMMITMENT,
        )
    };
    let no_claim_line = external_line(
        "claimed",
        layout,
        "raster-stages/input-embedding",
        "no_claim",
        &no_claim,
    );
    // `j` is the range as the names spell it: `0`, or `{j}` inside the repeat.
    // `previous` binds what the bind, the check and the embedding continue.
    let prompt_stages = |header: &str, j: &str, previous: [String; 3]| -> String {
        let [bind_previous, check_previous, embed_previous] = previous;
        format!(
            "\n{header}\n\
             name = \"prompt_bind_p{j}\"\n\
             project = \"raster-stages/prompt-bind\"\n\
             {descriptor_line}\n\
             {tokenizer_line}\
             inputs.text = {{ input = \"prompt_text_{j}\" }}\n\
             {}\
             {}\
             {}\
             {bind_previous}\
             \n{header}\n\
             name = \"prompt_check_p{j}\"\n\
             project = \"raster-stages/prompt-check\"\n\
             {tokenizer_line}\
             inputs.bound = {{ from = \"prompt_bind_p{j}\" }}\n\
             {check_previous}\
             inputs.marks = {{ input = \"prompt_marks_{j}\" }}\n\
             \n{header}\n\
             name = \"embed_p{j}\"\n\
             project = \"raster-stages/input-embedding\"\n\
             inputs.prompt = {{ from = \"prompt_check_p{j}\" }}\n\
             {no_claim_line}\
             {embed_previous}\
             {embedding_line}",
            run_external("config"),
            run_external("schedule"),
            run_external("claim"),
        )
    };
    out.push_str(&prompt_stages(
        "[[chain.stage]]",
        "0",
        [
            no_prompt_state("previous"),
            no_prompt_state("previous"),
            no_sequence("previous"),
        ],
    ));
    out.push_str("\n[[chain.repeat]]\nname = \"prompt\"\nindex = \"j\"\nstart = 1\ncount = 0\n");
    out.push_str(&prompt_stages(
        "  [[chain.repeat.stage]]",
        "{j}",
        ["prompt_bind", "prompt_check", "embed"].map(|stage| {
            format!("inputs.previous = {{ from = \"{stage}_p{{j-1}}\", first = \"{stage}_p0\" }}\n")
        }),
    ));
    out.push_str("\n[[chain.repeat]]\nname = \"claimed_embed\"\nindex = \"k\"\ncount = 0\n");
    out.push_str(&format!(
        "\n  [[chain.repeat.stage]]\n\
         name = \"embed_c{{k}}\"\n\
         project = \"raster-stages/input-embedding\"\n\
         {}\
         inputs.claimed = {{ input = \"claimed_ids_{{k}}\" }}\n\
         inputs.previous = {{ from = \"embed_c{{k-1}}\", first = \"embed_p{LAST_PROMPT_RANGE}\" }}\n\
         {embedding_line}",
        external_line("prompt", layout, "raster-stages/input-embedding", "no_prompt", &no_prompt),
    ));

    // Then each layer across every range, so a layer's weights are loaded
    // once per claim rather than once per range. Range `j` of layer `l`
    // needs layer `l - 1`'s range `j` and layer `l`'s range `j - 1`, both
    // already computed in this order. The inner fans' counts are the range
    // counts, filled in by run preparation. Prompt range 0 inherits no K/V
    // cache, so it is its own entry and the fan of the others starts at 1.
    //
    // `first_rows` and `donor` are given the range as the names spell it.
    let layer_block = |name: &str,
                       start: usize,
                       count: usize,
                       first_rows: &dyn Fn(&str, &str) -> String,
                       donor: &dyn Fn(&str, Option<&i32>, &str, &str) -> String|
     -> String {
        let mut text = format!(
            "\n[[chain.repeat]]\nname = \"{name}\"\nindex = \"l\"\nstart = {start}\ncount = {count}\n"
        );
        for (p, r) in [("p", "j"), ("c", "k")] {
            text.push_str(&format!(
                "\n  [[chain.repeat.stage]]\n\
                 name = \"aux_{p}{{{r}}}_l{{l}}\"\n\
                 index = \"{r}\"\n\
                 count = 0\n\
                 project = \"raster-stages/prefill-prepare-aux\"\n\
                 inputs.embedded = {{ from = \"embed_{p}{{{r}}}\" }}\n\
                 inputs.layer = {{ input = \"aux_layer_{{l}}\" }}\n\
                 inputs.witness = {{ input = \"aux_witness_{{l}}\" }}\n"
            ));
        }
        let prior_of_later_prompt_ranges = String::from(
            "inputs.prior_kv = { from = \"range_p{j-1}_l{l}\", first = \"range_p0_l{l}\" }\n",
        );
        let prior_of_claimed_ranges = format!(
            "inputs.prior_kv = {{ from = \"range_c{{k-1}}_l{{l}}\", first = \"range_p{LAST_PROMPT_RANGE}_l{{l}}\" }}\n"
        );
        for (p, range, fan, prior_kv) in [
            ("p", "0", "", no_sequence("prior_kv")),
            (
                "p",
                "{j}",
                "index = \"j\"\nstart = 1\ncount = 0\n",
                prior_of_later_prompt_ranges,
            ),
            (
                "c",
                "{k}",
                "index = \"k\"\ncount = 0\n",
                prior_of_claimed_ranges,
            ),
        ] {
            text.push_str(&format!(
                "\n  [[chain.repeat.stage]]\n\
                 name = \"range_{p}{range}_l{{l}}\"\n\
                 {fan}\
                 project = \"raster-stages/prefill-range\"\n\
                 inputs.activations = {{ from = \"range_{p}{range}_l{{l-1}}\", first = \"{}\" }}\n\
                 {prior_kv}\
                 {}\
                 {}\
                 inputs.ple = {{ from = \"aux_{p}{range}_l{{l}}\" }}\n\
                 inputs.layer = {{ input = \"transformer_layer_{{l}}\" }}\n",
                first_rows(p, range),
                donor("donor_a_kv", donors.first(), p, range),
                donor("donor_b_kv", donors.get(1), p, range),
            ));
        }
        text
    };
    if first_shared > 0 {
        out.push_str(&layer_block(
            "layers",
            0,
            first_shared,
            &|p, range| format!("embed_{p}{range}"),
            &|param, _, _, _| no_sequence(param),
        ));
    }
    if shape.num_kv_shared_layers > 0 {
        let last_own = first_shared.saturating_sub(1);
        out.push_str(&layer_block(
            "shared_layers",
            first_shared,
            shape.num_kv_shared_layers,
            &|p, range| format!("range_{p}{range}_l{last_own}"),
            &|param, donor, p, range| match donor {
                Some(layer) => {
                    format!("inputs.{param} = {{ from = \"range_{p}{range}_l{layer}\" }}\n")
                }
                None => no_sequence(param),
            },
        ));
    }

    // Each claimed range's check continues the verdict of the one before. The
    // first has none before it and takes the empty edge.
    let head_line = external_line(
        "head",
        layout,
        "raster-stages/sequence-check",
        "head",
        &head,
    );
    // A check also states the output record of the transcript so far, from
    // the decoder table, its membership paths and the model descriptor. The
    // last range's check is the chain's last stage, and its record the claim's.
    let decoder_line = external_line(
        "decoder",
        layout,
        "raster-stages/sequence-check",
        "decoder",
        decoder_commitment,
    );
    let check_stage = |header: &str, k: &str, previous: String, prior: String| -> String {
        format!(
            "\n{header}\n\
             name = \"check_c{k}\"\n\
             project = \"raster-stages/sequence-check\"\n\
             inputs.activations = {{ from = \"range_c{k}_l{last_layer}\" }}\n\
             {previous}\
             {prior}\
             inputs.prompt = {{ from = \"prompt_check_p{LAST_PROMPT_RANGE}\" }}\n\
             {head_line}\
             {decoder_line}\
             {identity_inputs}"
        )
    };
    out.push_str(&check_stage(
        "[[chain.stage]]",
        "0",
        format!("inputs.previous = {{ from = \"range_p{LAST_PROMPT_RANGE}_l{last_layer}\" }}\n"),
        external_line(
            "prior",
            layout,
            "raster-stages/sequence-check",
            "empty_edge",
            &empty_edge,
        ),
    ));
    out.push_str("\n[[chain.repeat]]\nname = \"claimed\"\nindex = \"k\"\nstart = 1\ncount = 0\n");
    out.push_str(&check_stage(
        "  [[chain.repeat.stage]]",
        "{k}",
        format!(
            "inputs.previous = {{ from = \"range_c{{k-1}}_l{last_layer}\", first = \"range_c0_l{last_layer}\" }}\n"
        ),
        String::from("inputs.prior = { from = \"check_c{k-1}\", first = \"check_c0\" }\n"),
    ));
    Ok(out)
}

/// Run preparation replaces this with the index of the last prompt range.
const LAST_PROMPT_RANGE: &str = "@LAST_PROMPT_RANGE@";

/// The empty prompt a claimed range embeds before its claimed ids.
fn no_prompt_commitment(layout: &ArtifactLayout) -> Result<String, Box<dyn Error>> {
    write_external(
        &PromptTokenization::default(),
        layout,
        "raster-stages/input-embedding",
        "no_prompt",
    )
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
    let aux_witness = (0..shape.layers)
        .map(|layer| {
            stage_external_commitment(stages, &format!("prefill_prepare_aux_l{layer}"), "witness")
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
        "{}{}{}",
        render(
            "aux_witness",
            "raster-stages/prefill-prepare-aux",
            &aux_witness
        )
        .replace("layer{l}", "witness{l}"),
        render("aux_layer", "raster-stages/prefill-prepare-aux", &aux),
        render(
            "transformer_layer",
            "raster-stages/prefill-range",
            &transformer
        )
    ))
}

/// Q16.16 bits of an `f32`, through the local deterministic conversion so the
/// committed scalar is the same value the deterministic path uses, not a
/// near one.
fn det_act_bits(value: f32) -> i32 {
    det_num::f32_to_act(value).to_bits()
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
) -> Result<(String, String), Box<dyn Error>> {
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
    let decoder_table = DecoderTable {
        tokens: decoder_tokens.into(),
    };
    let witness = sequence_check::decoder_witness(&decoder_table).map_err(|f| format!("{f:?}"))?;
    let witness_commitment =
        write_external(&witness, layout, "raster-stages/sequence-check", "witness")?;
    let decoder_commitment = write_external(
        &decoder_table,
        layout,
        "raster-stages/sequence-check",
        "decoder",
    )?;

    let tokenizer_commitment = write_external(
        &prepared_tokenizer,
        layout,
        "raster-stages/prompt-bind",
        "tokenizer",
    )?;
    let tokenizer_witness = prompt_check::input::tokenizer_witness(&prepared_tokenizer)
        .map_err(|f| format!("{f:?}"))?;
    let tokenizer_witness_commitment = write_external(
        &tokenizer_witness,
        layout,
        "raster-stages/prompt-bind",
        "witness",
    )?;
    let tokenizer_line = external_line(
        "tokenizer",
        layout,
        "raster-stages/prompt-bind",
        "tokenizer",
        &tokenizer_commitment,
    ) + &external_line(
        "witness",
        layout,
        "raster-stages/prompt-bind",
        "witness",
        &tokenizer_witness_commitment,
    );
    // The stages that take the tokenizer also take run-specific inputs, which
    // run preparation supplies.
    stages.push(format!(
        "[[chain.stage]]\nname = \"prompt_bind\"\nproject = \"raster-stages/prompt-bind\"\n{tokenizer_line}"
    ));
    Ok((decoder_commitment, witness_commitment))
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
            let manifest = format!("[chain]\nname = \"{name}\"\n[[chain.stage]]\nname = \"prompt_bind\"\nproject = \"raster-stages/prompt-bind\"\ninputs.tokenizer = {{ external = {{ path = \"data/{name}.rastered\", index_path = \"data/{name}.rindex\", commitment = \"abc\" }} }}\n");
            let (path, template) = save_model_template(&args, &manifest).unwrap();
            assert!(path.is_file()); // No --manifest is necessary.
            assert!(template.contains(&format!(
                "project = {:?}",
                std::env::current_dir()
                    .unwrap()
                    .join("raster-stages/prompt-bind")
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
                ModelFamily::GemmaDense,
                shape.manifest_shape(&text),
                Some((&path, &template)),
                &[
                    "tokenizer",
                    "decoder",
                    "embedding",
                    "head",
                    "aux_layer_0",
                    "transformer_layer_0",
                ]
                .into_iter()
                .enumerate()
                .map(|(i, role)| inference_artifacts::ModelCommitment {
                    role: role.into(),
                    commitment: format!("{:02x}", i).repeat(32),
                })
                .collect::<Vec<_>>(),
            )
            .unwrap();
            let model = inference_artifacts::ModelManifest::load_verified(&model_path).unwrap();
            let descriptor_bytes = fs::read(
                model_path
                    .parent()
                    .unwrap()
                    .join(inference_artifacts::model_identity::MODEL_DESCRIPTOR_BIN),
            )
            .unwrap();
            assert_eq!(
                descriptor_bytes,
                model.descriptor().unwrap().encode().unwrap()
            );
            // Version and two counts, then four global and two layer roots.
            assert_eq!(descriptor_bytes.len(), 12 + 6 * 32);

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
                range_size: DEFAULT_RANGE_SIZE,
            }
        );
    }

    #[test]
    fn a_bundle_is_imported_only_as_a_family_this_build_has() {
        let family = |config: serde_json::Value| bundle_family(&config);
        // Gemma's own configs, and the bundles made for this chain that say nothing.
        for config in [
            serde_json::json!({"model_type": "gemma4", "text_config": {"model_type": "gemma4_text"}}),
            serde_json::json!({"model_type": "gemma3n", "text_config": {}}),
            serde_json::json!({"text_config": {"hidden_size": 2}}),
        ] {
            assert_eq!(family(config).unwrap(), ModelFamily::GemmaDense);
        }
        // The text model's type decides when the two differ.
        for config in [
            serde_json::json!({"model_type": "qwen3"}),
            serde_json::json!({"model_type": "gemma4", "text_config": {"model_type": "qwen3"}}),
        ] {
            let error = family(config).unwrap_err().to_string();
            assert!(error.contains("`qwen3`"), "{error}");
            assert!(error.contains("gemma-dense"), "{error}");
        }
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
    fn the_stage_fixtures_flag_is_still_accepted() {
        let config = parse_args_from(
            ["--model", "model-bundles/model", "--no-stage-fixtures"].map(String::from),
        )
        .unwrap();
        assert!(config.no_stage_fixtures);
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
            range_size: DEFAULT_RANGE_SIZE,
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
            range_size: DEFAULT_RANGE_SIZE,
        };
        let layout = ArtifactLayout::from_config(&config);

        let line = external_line(
            "initial_pieces",
            &layout,
            "target/iris/run-prompt-placeholder",
            "initial_pieces",
            "abc",
        );

        assert!(
            line.contains(r#"path = "target/iris/run-prompt-placeholder/initial_pieces.rastered""#)
        );
        assert!(!line.contains("runtime/model-artifacts"));
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
