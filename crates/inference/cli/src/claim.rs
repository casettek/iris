use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use direct_infer::{DirectInferenceConfig, DirectInferenceExecutor};
use staged_infer::chain_runner::StagedExecutionBackend;
use staged_infer::{
    CheckpointedInferenceConfig, CheckpointedInferenceExecutor, CheckpointedInferenceResult,
    ParityPolicy,
};

use inference_artifacts::{
    file_sha256 as sha256_file, read_checkpoint_hashes, read_checkpoint_trace, read_json,
    read_run_spec, write_checkpoint_hashes, write_claim_artifacts_with_prepared_run, write_json,
    InferenceResult, InferenceRunSpec, ModelManifest, PreparedRun, CHECKPOINT_TRACE_JSON,
    PREPARED_RUN_JSON,
};
use serde::Serialize;
use sha2::{Digest, Sha256};

#[derive(Debug, Clone)]
pub struct ClaimBuildOptions {
    pub base_dir: PathBuf,
    pub run_spec_path: PathBuf,
    pub current_exe: PathBuf,
    pub staged_backend: StagedExecutionBackend,
    /// Debug-only: post these output ids instead of discovering them with direct
    /// inference. Used to build claims whose tokens are not the greedy
    /// continuation.
    pub claimed_token_ids: Option<Vec<u32>>,
}

pub(crate) struct PreparedClaimRun {
    pub(crate) prepared_run: PreparedRun,
    pub(crate) manifest_path: PathBuf,
    /// The ids written into the run's `claimed_ids` externals. Kept for the
    /// claimer's own tooling only: the claim posts no copy of them, because its
    /// output is the chain's final record.
    pub(crate) claimed_token_ids: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClaimBuildResult {
    pub chain_dir: PathBuf,
    pub checkpoint_trace_path: PathBuf,
    pub checkpoint_hashes_path: PathBuf,
    pub claim_bundle_path: PathBuf,
    pub final_result: Option<InferenceResult>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CorruptCheckpointSelection {
    Random { seed: Option<String> },
    Index(usize),
    Stage(String),
}

#[derive(Debug, Clone)]
pub struct CorruptCheckpointHashesOptions {
    pub hashes_path: PathBuf,
    pub trace_path: Option<PathBuf>,
    pub output_path: Option<PathBuf>,
    pub selection: CorruptCheckpointSelection,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CorruptCheckpointHashesResult {
    pub source_hashes_path: PathBuf,
    pub corrupted_hashes_path: PathBuf,
    pub corruption_manifest_path: PathBuf,
    pub checkpoint_index: usize,
    pub checkpoint_number: usize,
    pub stage: Option<String>,
    pub original_hash: String,
    pub corrupted_hash: String,
}

#[derive(Debug, Serialize)]
struct CorruptionManifest {
    version: u32,
    source_hashes_path: PathBuf,
    corrupted_hashes_path: PathBuf,
    checkpoint_index: usize,
    checkpoint_number: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    stage: Option<String>,
    original_hash: String,
    corrupted_hash: String,
}

pub fn build_claim(options: ClaimBuildOptions) -> Result<ClaimBuildResult> {
    let prepared = prepare_claim_run(&options)?;
    let executor = CheckpointedInferenceExecutor;
    let checkpointed = executor.run(CheckpointedInferenceConfig {
        base_dir: options.base_dir,
        manifest_path: prepared.manifest_path.clone(),
        current_exe: options.current_exe,
        staged_backend: options.staged_backend,
        parity_policy: ParityPolicy::Skip,
    })?;
    // A claimer applies the rule a challenger will: the claim is judged by
    // its own final record.
    if let Some(reason) =
        crate::challenge::claimed_output_rejection(checkpointed.final_result.as_ref())?
    {
        if options.claimed_token_ids.is_none() {
            // The ids came from direct decode, so the two paths are not
            // bit-identical: an honest claimer would convict itself.
            anyhow::bail!("{reason}; direct and teacher-forced inference are not in parity");
        }
        eprintln!("warning: {reason}; writing the claim anyway because --claimed-ids was given");
    }
    write_claim_result(
        checkpointed,
        &prepared.manifest_path,
        &prepared.prepared_run,
    )
}

pub fn corrupt_checkpoint_hashes(
    options: CorruptCheckpointHashesOptions,
) -> Result<CorruptCheckpointHashesResult> {
    let mut hashes = read_checkpoint_hashes(&options.hashes_path)?;
    if hashes.is_empty() {
        anyhow::bail!("cannot corrupt empty checkpoint hash file");
    }

    let trace_path = resolve_trace_path(&options.hashes_path, options.trace_path.as_deref());
    let trace = if trace_path.is_file() {
        Some(read_checkpoint_trace(&trace_path)?)
    } else {
        None
    };
    let checkpoint_index = select_checkpoint_index(&options.selection, &hashes, trace.as_ref())?;
    let checkpoint_number = checkpoint_index + 1;
    let stage = trace
        .as_ref()
        .and_then(|trace| trace.checkpoints.get(checkpoint_index))
        .map(|checkpoint| checkpoint.stage.clone());
    let original_hash = hashes[checkpoint_index].clone();
    let corrupted_hash = corrupt_hash(&original_hash);
    hashes[checkpoint_index] = corrupted_hash.clone();

    let corrupted_hashes_path = options
        .output_path
        .unwrap_or_else(|| default_corrupted_hashes_path(&options.hashes_path));
    write_checkpoint_hashes(&corrupted_hashes_path, &hashes)?;

    let corruption_manifest_path = corrupted_hashes_path.with_extension("corruption.json");
    write_json(
        &corruption_manifest_path,
        &CorruptionManifest {
            version: 1,
            source_hashes_path: options.hashes_path.clone(),
            corrupted_hashes_path: corrupted_hashes_path.clone(),
            checkpoint_index,
            checkpoint_number,
            stage: stage.clone(),
            original_hash: original_hash.clone(),
            corrupted_hash: corrupted_hash.clone(),
        },
    )?;

    Ok(CorruptCheckpointHashesResult {
        source_hashes_path: options.hashes_path,
        corrupted_hashes_path,
        corruption_manifest_path,
        checkpoint_index,
        checkpoint_number,
        stage,
        original_hash,
        corrupted_hash,
    })
}

fn write_claim_result(
    checkpointed: CheckpointedInferenceResult,
    manifest_path: &PathBuf,
    prepared_run: &PreparedRun,
) -> Result<ClaimBuildResult> {
    let prepared_run_path = prepared_run
        .run_manifest_path
        .as_ref()
        .and_then(|path| path.parent())
        .map(|dir| dir.join(PREPARED_RUN_JSON));
    let (checkpoint_trace_path, checkpoint_hashes_path, claim_bundle_path) =
        write_claim_artifacts_with_prepared_run(
            &checkpointed.chain_dir,
            manifest_path,
            prepared_run_path.as_deref(),
        )?;
    Ok(ClaimBuildResult {
        chain_dir: checkpointed.chain_dir,
        checkpoint_trace_path,
        checkpoint_hashes_path,
        claim_bundle_path,
        final_result: checkpointed.final_result,
    })
}

impl ClaimBuildOptions {
    pub fn from_current_dir(
        current_exe: PathBuf,
        staged_backend: StagedExecutionBackend,
        run_spec_path: PathBuf,
    ) -> Result<Self> {
        let base_dir = std::env::current_dir().context("failed to read current directory")?;
        Ok(Self {
            run_spec_path,
            base_dir,
            current_exe,
            staged_backend,
            claimed_token_ids: None,
        })
    }
}

pub(crate) fn prepare_claim_run(options: &ClaimBuildOptions) -> Result<PreparedClaimRun> {
    let run_spec_path = absolute(&options.base_dir, &options.run_spec_path);
    let run_spec = read_run_spec(&run_spec_path)?;
    prepare_run(options, run_spec_path, &run_spec, true)
}

/// Prepares a run from a spec a claim already fixed, as a challenger does.
///
/// The ids are not held to the run's token budget here. A claim that breaks the
/// budget still has a chain to recompute, and is rejected on its final record.
pub(crate) fn prepare_challenge_run(
    options: &ClaimBuildOptions,
    run_spec: &InferenceRunSpec,
) -> Result<PreparedClaimRun> {
    let run_spec_path = absolute(&options.base_dir, &options.run_spec_path);
    prepare_run(options, run_spec_path, run_spec, false)
}

fn prepare_run(
    options: &ClaimBuildOptions,
    run_spec_path: PathBuf,
    run_spec: &InferenceRunSpec,
    check_claimed_length: bool,
) -> Result<PreparedClaimRun> {
    let run_spec_dir = run_spec_path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("run spec has no parent directory"))?;
    let model_manifest_path = absolute(run_spec_dir, &run_spec.model_manifest);
    let model_manifest = ModelManifest::load_verified(&model_manifest_path)?;
    let (template_path, template) =
        model_manifest.verified_raster_template(&model_manifest_path)?;
    let model_manifest_dir = model_manifest_path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("model manifest has no parent directory"))?;
    let tokenizer_path = absolute(model_manifest_dir, &model_manifest.bundle.tokenizer_path);
    let tokenizer: serde_json::Value = read_json(&tokenizer_path)?;
    let prompt = run_prep::prepare_prompt(&tokenizer, run_spec_dir, run_spec, &model_manifest)?;

    let run_dir = options
        .base_dir
        .join("target")
        .join("iris")
        .join("runs")
        .join(run_id(
            &run_spec_path,
            &prompt.rendered_prompt,
            run_spec.tokens,
        ));
    let prompt_dir = run_dir.join("prompt");
    let ids = match &options.claimed_token_ids {
        Some(ids) => ids.clone(),
        // Discovery is the only autoregression left: the claim posts what
        // direct decode produced and the chain checks it.
        None => {
            DirectInferenceExecutor
                .run(DirectInferenceConfig {
                    base_dir: options.base_dir.clone(),
                    run_spec_path: run_spec_path.clone(),
                })
                .context("direct inference to discover the claimed output failed")?
                .generated_token_ids
        }
    };
    if check_claimed_length {
        run_prep::validate_claimed_length(&ids, run_spec.tokens, &prompt.eos_token_ids)?;
    }
    let range_size = model_manifest.range_size;
    let ranges = run_prep::marked_ranges(
        &prompt.initial_pieces,
        range_size as usize,
        &run_prep::prompt_tokenizer(&tokenizer)?,
    )?;
    let template_dir = template_path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("manifest template has no parent directory"))?;
    let mark_files = run_prep::write_range_marks(&ranges, &prompt_dir)?;
    let claimed_files = run_prep::write_claimed_ranges(&ids, range_size as usize, &prompt_dir)?;
    let merge_repeats = run_prep::CHECK_MERGE_BATCHES;
    let run_manifest = render_ranged_manifest(
        &template,
        template_dir,
        &run_prep::write_prompt_text(&prompt.resolved_prompt, &prompt_dir)?,
        &run_prep::write_run_config(
            &run_prep::RunConfig {
                model_id: model_manifest.model_id.clone(),
                raw_prompt: run_spec.raw_prompt,
                max_new_tokens: run_spec.tokens,
                range_size,
            },
            &prompt_dir,
        )?,
        &run_prep::write_run_schedule(
            mark_files.len(),
            merge_repeats,
            claimed_files.len(),
            &prompt_dir,
        )?,
        &mark_files,
        &claimed_files,
        merge_repeats,
    )?;
    let manifest_path = run_dir.join("Raster.toml");
    fs::create_dir_all(&run_dir)
        .with_context(|| format!("failed to create {}", run_dir.display()))?;
    fs::write(&manifest_path, &run_manifest)
        .with_context(|| format!("failed to write {}", manifest_path.display()))?;
    let run_manifest_sha256 = format!("{:x}", Sha256::digest(run_manifest.as_bytes()));
    let prepared_run = PreparedRun {
        version: inference_artifacts::PREPARED_RUN_VERSION,
        run_spec_path,
        model_manifest_path: model_manifest_path.clone(),
        model_manifest_sha256: sha256_file(&model_manifest_path)?,
        model_id: model_manifest.model_id.clone(),
        prompt,
        raw_prompt: run_spec.raw_prompt,
        tokens: run_spec.tokens,
        run_manifest_path: Some(manifest_path.clone()),
        run_manifest_sha256: Some(run_manifest_sha256),
    };
    write_json(&run_dir.join(PREPARED_RUN_JSON), &prepared_run)?;

    Ok(PreparedClaimRun {
        prepared_run,
        manifest_path,
        claimed_token_ids: ids,
    })
}

/// Where model templates point run-specific externals until a run fills them.
const RUN_PLACEHOLDER_DIR: &str = "target/iris/run-prompt-placeholder/";

/// Marks where the template names the last prompt range.
const LAST_PROMPT_RANGE: &str = "@LAST_PROMPT_RANGE@";

/// Marks where the template names each range's last tokenizer batch.
const LAST_MERGE_BATCH: &str = "@LAST_MERGE_BATCH@";

/// Fills the template for one run: the original text, the configuration
/// and the stage schedule, the range counts, the last prompt range's index, the
/// tokenizer batches after each range's seed, and each range's committed cut or
/// claimed ids.
pub(crate) fn render_ranged_manifest(
    template: &str,
    template_dir: &Path,
    text: &run_prep::RunFile,
    config: &run_prep::RunFile,
    schedule: &run_prep::RunFile,
    prompt: &[run_prep::RunFile],
    claimed: &[run_prep::RunFile],
    merge_repeats: u32,
) -> Result<String> {
    anyhow::ensure!(
        !prompt.is_empty() && !claimed.is_empty(),
        "a run needs at least one prompt range and one claimed range"
    );
    anyhow::ensure!(
        merge_repeats > 0,
        "a prompt range needs at least one tokenizer batch after its seed"
    );
    let template = inference_artifacts::resolve_raster_manifest_paths(template, template_dir)?;
    let last_prompt = (prompt.len() - 1).to_string();
    let last_merge = (merge_repeats - 1).to_string();
    let mut table = String::new();
    let mut name = String::new();
    let mut fan_index = String::new();
    // Prompt repeats, claimed repeats, prompt externals, claimed externals,
    // per-layer prompt fans, per-layer claimed fans, tokenizer batch fans,
    // text bindings, configuration bindings, schedule bindings.
    let mut replaced = [0_usize; 10];
    let mut out = Vec::new();
    for line in template.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('[') {
            table = trimmed.to_string();
            name.clear();
            fan_index.clear();
        }
        if trimmed.starts_with("name =") {
            let value: toml::Value = toml::from_str(trimmed)?;
            name = value["name"]
                .as_str()
                .context("invalid stage/repeat name")?
                .to_string();
        }
        if trimmed.starts_with("index =") {
            let value: toml::Value = toml::from_str(trimmed)?;
            fan_index = value["index"]
                .as_str()
                .context("invalid repeat index")?
                .to_string();
        }
        if table == "[[chain.stage]]" && name == "prompt_bind" {
            let binding = if trimmed.starts_with("inputs.text =") {
                Some(("text", text, 7))
            } else if trimmed.starts_with("inputs.config =") {
                Some(("config", config, 8))
            } else if trimmed.starts_with("inputs.schedule =") {
                Some(("schedule", schedule, 9))
            } else {
                None
            };
            if let Some((param, file, slot)) = binding {
                replaced[slot] += 1;
                out.push(
                    RunExternal::new(&file.path, &file.index_path, &file.commitment).binding(param),
                );
                continue;
            }
        }
        let files = match table.as_str() {
            // A template either checks each range against marks, or cuts it
            // and replays its merges.
            "[chain.input.prompt_marks]" | "[chain.input.prompt_cuts]" => Some((prompt, 2)),
            "[chain.input.claimed_ids]" => Some((claimed, 3)),
            _ => None,
        };
        if let Some((files, slot)) = files {
            if let Some(key) = ["index_path", "path"]
                .into_iter()
                .find(|key| trimmed.starts_with(&format!("{key} =")))
            {
                // Keep the indexed file name (`cut_{j}.rastered`), move it
                // from the placeholder directory to this run's.
                let value: toml::Value = toml::from_str(trimmed)?;
                let placeholder = value[key].as_str().context("invalid input path")?;
                anyhow::ensure!(
                    placeholder.contains(RUN_PLACEHOLDER_DIR),
                    "input path {placeholder} is not a run placeholder"
                );
                let file_name = Path::new(placeholder)
                    .file_name()
                    .context("input path has no file name")?;
                let dir = files[0]
                    .path
                    .parent()
                    .context("run file has no directory")?;
                out.push(format!("{key} = {:?}", dir.join(file_name)));
                continue;
            }
            if trimmed.starts_with("commitments =") {
                replaced[slot] += 1;
                out.push(String::from("commitments = ["));
                out.extend(
                    files
                        .iter()
                        .enumerate()
                        .map(|(index, file)| format!("  {:?}, # {index}", file.commitment)),
                );
                out.push(String::from("]"));
                continue;
            }
        }
        if table == "[[chain.repeat]]" && trimmed.starts_with("count =") {
            let range_count = match name.as_str() {
                "prompt" => Some((0, prompt.len())),
                "claimed_embed" | "claimed" => Some((1, claimed.len())),
                // Layer blocks keep their model-fixed counts.
                "layers" | "shared_layers" => None,
                other => anyhow::bail!("unexpected repeat `{other}` in chain template"),
            };
            if let Some((slot, count)) = range_count {
                replaced[slot] += 1;
                out.push(format!("count = {count}"));
                continue;
            }
        }
        // A layer block's inner fans run once per prompt or claimed range.
        if table == "[[chain.repeat.stage]]" && trimmed.starts_with("count =") {
            let range_count = match fan_index.as_str() {
                "j" => Some((4, prompt.len())),
                "k" => Some((5, claimed.len())),
                "b" => Some((6, merge_repeats as usize)),
                _ => None,
            };
            if let Some((slot, count)) = range_count {
                replaced[slot] += 1;
                out.push(format!("count = {count}"));
                continue;
            }
        }
        out.push(
            line.replace(LAST_PROMPT_RANGE, &last_prompt)
                .replace(LAST_MERGE_BATCH, &last_merge),
        );
    }
    if replaced[..4] != [1, 2, 1, 1]
        || replaced[4] == 0
        || replaced[4] != replaced[5]
        || replaced[6] > 1
        || replaced[7..] != [1, 1, 1]
    {
        anyhow::bail!("incompatible chain template: expected [prompt repeats, claimed repeats, prompt commitments, claimed commitments] = [1, 2, 1, 1], matching per-layer range fans, one tokenizer batch fan and one text, one configuration and one schedule binding; found {replaced:?}");
    }
    out.push(String::new());
    Ok(out.join("\n"))
}

/// One run-specific external: the files and commitment a template
/// placeholder is replaced with.
pub(crate) struct RunExternal<'a> {
    path: &'a Path,
    index_path: &'a Path,
    commitment: &'a str,
}

impl<'a> RunExternal<'a> {
    pub(crate) fn new(path: &'a Path, index_path: &'a Path, commitment: &'a str) -> Self {
        Self {
            path,
            index_path,
            commitment,
        }
    }

    pub(crate) fn binding(&self, param: &str) -> String {
        format!(
            "inputs.{param} = {{ external = {{ path = {:?}, index_path = {:?}, commitment = {:?} }} }}",
            self.path, self.index_path, self.commitment,
        )
    }
}

fn absolute(base: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        base.join(path)
    }
}

fn select_checkpoint_index(
    selection: &CorruptCheckpointSelection,
    hashes: &[String],
    trace: Option<&inference_artifacts::CheckpointTrace>,
) -> Result<usize> {
    match selection {
        CorruptCheckpointSelection::Index(index) => {
            if *index >= hashes.len() {
                anyhow::bail!(
                    "checkpoint index {index} is out of range for {} hashes",
                    hashes.len()
                );
            }
            Ok(*index)
        }
        CorruptCheckpointSelection::Stage(stage) => {
            let trace = trace.ok_or_else(|| {
                anyhow::anyhow!(
                    "stage selection requires a checkpoint trace; pass --trace or keep checkpoint_trace.json beside checkpoints.txt"
                )
            })?;
            let index = trace
                .checkpoints
                .iter()
                .position(|checkpoint| checkpoint.stage == *stage)
                .ok_or_else(|| {
                    anyhow::anyhow!("stage `{stage}` was not found in checkpoint trace")
                })?;
            if index >= hashes.len() {
                anyhow::bail!(
                    "stage `{stage}` is checkpoint {index}, but the hash file only has {} hashes",
                    hashes.len()
                );
            }
            Ok(index)
        }
        CorruptCheckpointSelection::Random { seed } => Ok(random_checkpoint_index(seed, hashes)),
    }
}

fn random_checkpoint_index(seed: &Option<String>, hashes: &[String]) -> usize {
    let seed_bytes = match seed {
        Some(seed) => seed.as_bytes().to_vec(),
        None => SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
            .to_string()
            .into_bytes(),
    };
    let digest = Sha256::digest(seed_bytes);
    let mut bytes = [0_u8; 8];
    bytes.copy_from_slice(&digest[..8]);
    (u64::from_be_bytes(bytes) as usize) % hashes.len()
}

fn corrupt_hash(hash: &str) -> String {
    let mut bytes = hash.as_bytes().to_vec();
    bytes[0] = if bytes[0] == b'0' { b'1' } else { b'0' };
    String::from_utf8(bytes).expect("checkpoints are valid hex")
}

fn resolve_trace_path(hashes_path: &Path, trace_path: Option<&Path>) -> PathBuf {
    trace_path
        .map(Path::to_path_buf)
        .or_else(|| {
            hashes_path
                .parent()
                .map(|parent| parent.join(CHECKPOINT_TRACE_JSON))
        })
        .unwrap_or_else(|| PathBuf::from(CHECKPOINT_TRACE_JSON))
}

fn default_corrupted_hashes_path(source: &Path) -> PathBuf {
    source.with_file_name("checkpoints.corrupt.txt")
}

fn run_id(run_spec_path: &Path, rendered_prompt: &str, tokens: u32) -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let digest = Sha256::digest(format!(
        "{}\0{}\0{}\0{}",
        run_spec_path.display(),
        rendered_prompt,
        tokens,
        nanos
    ));
    hex::encode(digest)[..16].to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use inference_artifacts::{CHECKPOINT_TRACE_JSON, CLAIM_BUNDLE_JSON};
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn options_from_current_dir_use_run_spec() {
        let saved = std::env::current_dir().unwrap();
        let base = temp_dir("claim-options");
        fs::create_dir_all(&base).unwrap();
        let expected_base = base.canonicalize().unwrap();
        std::env::set_current_dir(&base).unwrap();

        let options = ClaimBuildOptions::from_current_dir(
            PathBuf::from("iris"),
            StagedExecutionBackend::InProcess,
            PathBuf::from("inference.toml"),
        )
        .unwrap();

        std::env::set_current_dir(saved).unwrap();
        fs::remove_dir_all(&base).unwrap();

        assert_eq!(options.base_dir, expected_base);
        assert_eq!(options.run_spec_path, PathBuf::from("inference.toml"));
        assert_eq!(options.current_exe, PathBuf::from("iris"));
        assert_eq!(options.staged_backend, StagedExecutionBackend::InProcess);
    }

    #[test]
    fn preparation_uses_selected_model_and_rejects_stale_identity() {
        use crate::test_support::{model_fixture, run_spec};
        let base = temp_dir("model-selection");
        model_fixture(&base, "model-a");
        let model_path = model_fixture(&base, "model-b");
        fs::write(base.join("Raster.toml"), "stale model-a template").unwrap();
        let options = ClaimBuildOptions {
            base_dir: base.clone(),
            run_spec_path: run_spec(&base, "model-b"),
            current_exe: PathBuf::from("unused"),
            staged_backend: StagedExecutionBackend::InProcess,
            claimed_token_ids: Some(vec![0, 0, 0]),
        };
        let prepared = prepare_claim_run(&options).unwrap();
        let rendered = fs::read_to_string(&prepared.manifest_path).unwrap();
        assert!(rendered.contains("name = \"model-b\""));
        assert!(!rendered.contains("model-a"));
        assert_eq!(prepared.claimed_token_ids, [0, 0, 0]);
        assert_eq!(
            fs::canonicalize(&prepared.prepared_run.model_manifest_path).unwrap(),
            fs::canonicalize(&model_path).unwrap()
        );

        for (file, label) in [
            ("model.detwgt", "model weights"),
            ("config.json", "model config"),
            ("tokenizer.json", "model tokenizer"),
            ("Raster.toml", "Raster template"),
        ] {
            let path = model_path.parent().unwrap().join(file);
            let original = fs::read(&path).unwrap();
            fs::write(&path, "replaced").unwrap();
            let error = prepare_claim_run(&options).err().unwrap();
            assert!(
                error
                    .to_string()
                    .contains(&format!("{label} hash mismatch")),
                "{error:#}"
            );
            fs::write(path, original).unwrap();
        }

        let mut model: ModelManifest = read_json(&model_path).unwrap();
        model.provenance = None;
        write_json(&model_path, &model).unwrap();
        let error = prepare_claim_run(&options).err().unwrap();
        assert!(error.to_string().contains("no Raster template provenance"));
        model.version = 99;
        write_json(&model_path, &model).unwrap();
        assert!(prepare_claim_run(&options)
            .err()
            .unwrap()
            .to_string()
            .contains("unsupported model manifest"));
        fs::remove_dir_all(base).unwrap();
    }

    #[test]
    fn claim_result_writes_existing_artifact_names() {
        let base = temp_dir("claim-result");
        fs::create_dir_all(&base).unwrap();
        write_stage(&base, "output_finalize", "abc", b"output");
        let manifest_path = base.join("Raster.toml");
        fs::write(&manifest_path, "[chain]\nname = \"test\"\n").unwrap();

        let result = write_claim_result(
            CheckpointedInferenceResult {
                chain_dir: base.clone(),
                selected_stage_dir: None,
                final_result: None,
            },
            &manifest_path,
            &PreparedRun {
                version: inference_artifacts::PREPARED_RUN_VERSION,
                run_spec_path: base.join("inference.toml"),
                model_manifest_path: base.join("model-artifacts/manifest.json"),
                model_manifest_sha256: String::from("model-sha"),
                model_id: String::from("model-id"),
                raw_prompt: false,
                prompt: inference_artifacts::PreparedPrompt {
                    resolved_prompt: String::from("hello"),
                    rendered_prompt: String::from("hello"),
                    initial_pieces: vec![inference_artifacts::PreparedPiece {
                        text: String::from("hello"),
                        segment: 0,
                    }],
                    eos_token_ids: Vec::new(),
                },
                tokens: 1,
                run_manifest_path: Some(manifest_path.clone()),
                run_manifest_sha256: Some(String::from("run-sha")),
            },
        )
        .unwrap();

        assert_eq!(result.chain_dir, base);
        assert_eq!(
            result.checkpoint_trace_path.file_name().unwrap(),
            CHECKPOINT_TRACE_JSON
        );
        assert_eq!(
            result.checkpoint_hashes_path.file_name().unwrap(),
            inference_artifacts::CHECKPOINT_HASHES_TXT
        );
        assert_eq!(
            result.claim_bundle_path.file_name().unwrap(),
            CLAIM_BUNDLE_JSON
        );
        assert_eq!(result.final_result, None);

        fs::remove_dir_all(result.chain_dir).unwrap();
    }

    #[test]
    fn corrupt_checkpoint_hashes_changes_selected_index() {
        let base = temp_dir("corrupt-index");
        fs::create_dir_all(&base).unwrap();
        let hashes_path = base.join("checkpoints.txt");
        let output_path = base.join("corrupted.txt");
        let hashes = vec![
            String::from("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"),
            String::from("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"),
        ];
        inference_artifacts::write_checkpoint_hashes(&hashes_path, &hashes).unwrap();

        let result = corrupt_checkpoint_hashes(CorruptCheckpointHashesOptions {
            hashes_path: hashes_path.clone(),
            trace_path: None,
            output_path: Some(output_path.clone()),
            selection: CorruptCheckpointSelection::Index(1),
        })
        .unwrap();

        let corrupted = inference_artifacts::read_checkpoint_hashes(&output_path).unwrap();
        assert_eq!(result.checkpoint_index, 1);
        assert_eq!(result.checkpoint_number, 2);
        assert_eq!(corrupted[0], hashes[0]);
        assert_ne!(corrupted[1], hashes[1]);
        assert_eq!(result.original_hash, hashes[1]);
        assert_eq!(result.corrupted_hash, corrupted[1]);
        assert!(result.corruption_manifest_path.is_file());

        fs::remove_dir_all(base).unwrap();
    }

    #[test]
    fn corrupt_checkpoint_hashes_can_select_stage_from_trace() {
        let base = temp_dir("corrupt-stage");
        fs::create_dir_all(&base).unwrap();
        let hashes_path = base.join("checkpoints.txt");
        let trace_path = base.join(CHECKPOINT_TRACE_JSON);
        let trace = inference_artifacts::CheckpointTrace {
            checkpoints: vec![
                inference_artifacts::Checkpoint {
                    stage: String::from("stage_a"),
                    input_commitment: String::from("input-a"),
                    output_commitment: String::from("output-a"),
                    output_sha256: String::from("sha-a"),
                },
                inference_artifacts::Checkpoint {
                    stage: String::from("stage_b"),
                    input_commitment: String::from("input-b"),
                    output_commitment: String::from("output-b"),
                    output_sha256: String::from("sha-b"),
                },
            ],
        };
        inference_artifacts::write_json(&trace_path, &trace).unwrap();
        let hashes = inference_artifacts::checkpoint_hashes(&trace).unwrap();
        inference_artifacts::write_checkpoint_hashes(&hashes_path, &hashes).unwrap();

        let result = corrupt_checkpoint_hashes(CorruptCheckpointHashesOptions {
            hashes_path: hashes_path.clone(),
            trace_path: None,
            output_path: Some(base.join("corrupted.txt")),
            selection: CorruptCheckpointSelection::Stage(String::from("stage_b")),
        })
        .unwrap();

        assert_eq!(result.checkpoint_index, 1);
        assert_eq!(result.checkpoint_number, 2);
        assert_eq!(result.stage, Some(String::from("stage_b")));

        fs::remove_dir_all(base).unwrap();
    }

    #[test]
    fn ranged_manifest_fills_counts_inputs_and_last_prompt_range() {
        let template = r#"[chain]
name = "iris"

[chain.input.prompt_cuts]
index = "j"
path = "/repo/target/iris/run-prompt-placeholder/cut_{j}.rastered"
index_path = "/repo/target/iris/run-prompt-placeholder/cut_{j}.rindex"
commitments = []

[chain.input.claimed_ids]
index = "k"
path = "/repo/target/iris/run-prompt-placeholder/claimed_{k}.rastered"
index_path = "/repo/target/iris/run-prompt-placeholder/claimed_{k}.rindex"
commitments = []

[[chain.stage]]
name = "prompt_bind"
inputs.text = { external = { path = "/repo/target/iris/run-prompt-placeholder/text.rastered", index_path = "/repo/target/iris/run-prompt-placeholder/text.rindex", commitment = "0" } }
inputs.config = { external = { path = "/repo/target/iris/run-prompt-placeholder/config.rastered", index_path = "/repo/target/iris/run-prompt-placeholder/config.rindex", commitment = "0" } }
inputs.schedule = { external = { path = "/repo/target/iris/run-prompt-placeholder/schedule.rastered", index_path = "/repo/target/iris/run-prompt-placeholder/schedule.rindex", commitment = "0" } }

[[chain.repeat]]
name = "prompt"
index = "j"
count = 0

  [[chain.repeat.stage]]
name = "prompt_merge_p{j}_b{b}"
index = "b"
count = 0

  [[chain.repeat.stage]]
name = "prompt_prepare_p{j}"
inputs.merged_pieces = { from = "prompt_merge_p{j}_b@LAST_MERGE_BATCH@" }

[[chain.repeat]]
name = "claimed_embed"
index = "k"
count = 0

  [[chain.repeat.stage]]
name = "embed_c{k}"
inputs.previous = { from = "embed_c{k-1}", first = "embed_p@LAST_PROMPT_RANGE@" }

[[chain.repeat]]
name = "layers"
index = "l"
start = 0
count = 4

  [[chain.repeat.stage]]
name = "range_p{j}_l{l}"
index = "j"
count = 0

  [[chain.repeat.stage]]
name = "range_c{k}_l{l}"
index = "k"
count = 0

[[chain.repeat]]
name = "claimed"
index = "k"
count = 0
"#;
        let file = |name: &str, commitment: &str| run_prep::RunFile {
            path: PathBuf::from(format!("/tmp/run/prompt/{name}.rastered")),
            index_path: PathBuf::from(format!("/tmp/run/prompt/{name}.rindex")),
            commitment: commitment.into(),
        };
        let text = file("text", "t0");
        let config = file("config", "g0");
        let schedule = file("schedule", "s0");
        let prompt = [
            file("cut_0", "a0"),
            file("cut_1", "a1"),
            file("cut_2", "a2"),
        ];
        let claimed = [file("claimed_0", "c0")];
        let render = |claimed: &[run_prep::RunFile], merge_repeats| {
            render_ranged_manifest(
                template,
                Path::new("/repo"),
                &text,
                &config,
                &schedule,
                &prompt,
                claimed,
                merge_repeats,
            )
        };
        let manifest = render(&claimed, 5).unwrap();
        let parsed: toml::Value = toml::from_str(&manifest).unwrap();
        let inputs = &parsed["chain"]["input"];
        assert_eq!(
            inputs["prompt_cuts"]["path"].as_str(),
            Some("/tmp/run/prompt/cut_{j}.rastered")
        );
        assert_eq!(
            inputs["claimed_ids"]["index_path"].as_str(),
            Some("/tmp/run/prompt/claimed_{k}.rindex")
        );
        assert_eq!(
            inputs["prompt_cuts"]["commitments"],
            toml::Value::try_from(["a0", "a1", "a2"]).unwrap()
        );
        assert_eq!(
            inputs["claimed_ids"]["commitments"],
            toml::Value::try_from(["c0"]).unwrap()
        );
        let repeats = parsed["chain"]["repeat"].as_array().unwrap();
        let count = |value: &toml::Value| value["count"].as_integer();
        assert_eq!(count(&repeats[0]), Some(3));
        assert_eq!(count(&repeats[0]["stage"][0]), Some(5));
        assert!(manifest.contains(r#"from = "prompt_merge_p{j}_b4""#));
        assert_eq!(count(&repeats[1]), Some(1));
        // The layer block keeps its count; its fans run once per range.
        assert_eq!(count(&repeats[2]), Some(4));
        let fans = repeats[2]["stage"].as_array().unwrap();
        assert_eq!(count(&fans[0]), Some(3));
        assert_eq!(count(&fans[1]), Some(1));
        assert_eq!(count(&repeats[3]), Some(1));
        assert!(manifest.contains(r#"first = "embed_p2""#));
        assert!(manifest.contains(r#"commitment = "t0""#));
        assert!(manifest.contains(r#"commitment = "g0""#));
        assert!(manifest.contains(r#"commitment = "s0""#));
        assert!(!manifest.contains("run-prompt-placeholder"));
        assert!(render(&[], 5).is_err());
        assert!(render(&claimed, 0).is_err());
    }

    fn write_stage(base: &PathBuf, stage: &str, commitment: &str, output: &[u8]) {
        let stage_dir = base.join(stage);
        fs::create_dir_all(&stage_dir).unwrap();
        fs::write(stage_dir.join("input_manifest.json"), b"input-manifest").unwrap();
        fs::write(stage_dir.join("output.bin"), output).unwrap();
        fs::write(stage_dir.join("output.rindex"), b"index").unwrap();
        fs::write(
            stage_dir.join("output_manifest.json"),
            format!(
                r#"{{"output":{{"type":"sha256","encoding":"raster","commitment":"{commitment}"}}}}"#
            ),
        )
        .unwrap();
    }

    fn temp_dir(label: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("iris-cli-{label}-{}-{nanos}", std::process::id()))
    }
}
