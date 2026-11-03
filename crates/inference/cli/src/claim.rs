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
    ChainKind, InferenceResult, ModelManifest, PreparedRun, CHECKPOINT_TRACE_JSON,
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
    /// Debug-only: post these output ids in a teacher-forced claim instead of
    /// discovering them with direct inference. Used to build claims whose
    /// tokens are not the greedy continuation.
    pub claimed_token_ids: Option<Vec<u32>>,
}

pub(crate) struct PreparedClaimRun {
    pub(crate) prepared_run: PreparedRun,
    pub(crate) manifest_path: PathBuf,
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
    if let Some(claimed) = &prepared.prepared_run.claimed_token_ids {
        let checked = checkpointed
            .final_result
            .as_ref()
            .map(|result| result.generated_token_ids.as_slice())
            .context("teacher-forced chain produced no final output")?;
        if let Some(position) = first_disagreement(claimed, checked) {
            let message = format!(
                "teacher-forced selection disagrees with the claimed output at generated token {position}: claimed {:?}, greedy {:?}",
                claimed.get(position),
                checked.get(position)
            );
            if options.claimed_token_ids.is_none() {
                // The ids came from direct decode, so the two paths are not
                // bit-identical: an honest claimer would convict itself.
                anyhow::bail!("{message}; direct and teacher-forced inference are not in parity");
            }
            eprintln!(
                "warning: {message}; writing the claim anyway because --claimed-ids was given"
            );
        }
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

/// The first generated position where a teacher-forced claim's posted ids
/// and the chain's greedy selections differ, including a length difference.
pub(crate) fn first_disagreement(claimed: &[u32], checked: &[u32]) -> Option<usize> {
    claimed
        .iter()
        .zip(checked)
        .position(|(claimed, checked)| claimed != checked)
        .or_else(|| (claimed.len() != checked.len()).then(|| claimed.len().min(checked.len())))
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
    let run_spec_dir = run_spec_path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("run spec has no parent directory"))?;
    let model_manifest_path = absolute(run_spec_dir, &run_spec.model_manifest);
    let model_manifest = ModelManifest::load_verified(&model_manifest_path)?;
    let (template_path, template) =
        model_manifest.verified_chain_template(&model_manifest_path, run_spec.chain)?;
    let model_manifest_dir = model_manifest_path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("model manifest has no parent directory"))?;
    let tokenizer_path = absolute(model_manifest_dir, &model_manifest.bundle.tokenizer_path);
    let tokenizer: serde_json::Value = read_json(&tokenizer_path)?;
    let prompt = run_prep::prepare_prompt(&tokenizer, run_spec_dir, &run_spec, &model_manifest)?;

    let run_dir = options
        .base_dir
        .join("target")
        .join("iris")
        .join("runs")
        .join(run_id(
            &run_spec_path,
            &prompt.rendered_prompt,
            run_spec.tokens,
            run_spec.chain,
        ));
    let prompt_dir = run_dir.join("prompt");
    let (pieces_path, pieces_index_path, pieces_commitment) =
        run_prep::write_initial_pieces(&prompt, &prompt_dir)?;
    let claimed = match run_spec.chain {
        ChainKind::Autoregressive => {
            if options.claimed_token_ids.is_some() {
                anyhow::bail!("--claimed-ids requires a run spec with chain = \"teacher-forced\"");
            }
            None
        }
        ChainKind::TeacherForced => {
            let ids = match &options.claimed_token_ids {
                Some(ids) => ids.clone(),
                // Discovery is the only autoregression left: the claim posts
                // what direct decode produced and the chain checks it.
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
            run_prep::validate_claimed_length(&ids, run_spec.tokens, &prompt.eos_token_ids)?;
            let external = run_prep::write_claimed_output(&ids, &prompt_dir)?;
            Some((ids, external))
        }
    };

    let template_dir = template_path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("manifest template has no parent directory"))?;
    let run_manifest = render_run_manifest(
        &template,
        template_dir,
        &RunExternal::new(&pieces_path, &pieces_index_path, &pieces_commitment),
        match &claimed {
            Some((_, (path, index_path, commitment))) => {
                RunTokens::TeacherForced(RunExternal::new(path, index_path, commitment))
            }
            None => RunTokens::Autoregressive(run_spec.tokens),
        },
        run_prep::tokenizer_repeat_count(prompt.initial_pieces.len())?,
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
        prompt,
        tokens: run_spec.tokens,
        run_manifest_path: Some(manifest_path.clone()),
        run_manifest_sha256: Some(run_manifest_sha256),
        chain: run_spec.chain,
        claimed_token_ids: claimed.map(|(ids, _)| ids),
    };
    write_json(&run_dir.join(PREPARED_RUN_JSON), &prepared_run)?;

    Ok(PreparedClaimRun {
        prepared_run,
        manifest_path,
    })
}

/// Where model templates point run-specific externals until a run fills them.
const RUN_PLACEHOLDER_DIR: &str = "target/iris/run-prompt-placeholder/";

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

    fn binding(&self, param: &str) -> String {
        format!(
            "inputs.{param} = {{ external = {{ path = {:?}, index_path = {:?}, commitment = {:?} }} }}",
            self.path, self.index_path, self.commitment,
        )
    }
}

/// How a run fixes its generated tokens: a decode repeat count, or the
/// claimed ids a teacher-forced pass checks.
pub(crate) enum RunTokens<'a> {
    Autoregressive(u32),
    TeacherForced(RunExternal<'a>),
}

pub(crate) fn render_run_manifest(
    template: &str,
    template_dir: &Path,
    pieces: &RunExternal<'_>,
    tokens: RunTokens<'_>,
    tokenizer_repeats: u32,
) -> Result<String> {
    let pieces_line = pieces.binding("initial_pieces");
    let template = inference_artifacts::resolve_raster_manifest_paths(template, template_dir)?;
    let mut table = "";
    let mut name = String::new();
    // Seed binding, decode count, tokenize count, claimed bindings.
    let mut replaced = [0_usize; 4];
    let mut out = Vec::new();
    for line in template.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('[') {
            table = match trimmed {
                "[[chain.repeat]]" => "repeat",
                "[[chain.stage]]" => "stage",
                _ => "",
            };
            name.clear();
        }
        if trimmed.starts_with("name =") {
            let value: toml::Value = toml::from_str(trimmed)?;
            name = value["name"]
                .as_str()
                .context("invalid stage/repeat name")?
                .to_string();
        }
        if table == "stage"
            && name == "prompt_merge_seed"
            && trimmed.starts_with("inputs.initial_pieces =")
        {
            replaced[0] += 1;
            out.push(pieces_line.clone());
        } else if table == "repeat" && trimmed.starts_with("count =") && name == "tokenize" {
            replaced[2] += 1;
            out.push(format!("count = {tokenizer_repeats}"));
        } else if table == "repeat" && trimmed.starts_with("count =") && name == "decode" {
            let RunTokens::Autoregressive(count) = &tokens else {
                anyhow::bail!(
                    "teacher-forced run manifest cannot fill an autoregressive decode repeat"
                );
            };
            replaced[1] += 1;
            out.push(format!("count = {count}"));
        } else if table == "stage"
            && trimmed.starts_with("inputs.claimed =")
            && trimmed.contains(RUN_PLACEHOLDER_DIR)
        {
            // The autoregressive chain's empty claim is a model external and
            // stays; only the teacher-forced run placeholder is filled.
            let RunTokens::TeacherForced(claimed) = &tokens else {
                anyhow::bail!("autoregressive run manifest has a claimed-output binding");
            };
            replaced[3] += 1;
            out.push(claimed.binding("claimed"));
        } else {
            out.push(line.to_string());
        }
    }
    // The teacher-forced template binds the claim into both the embedding and
    // the check stage, and has no decode repeat.
    let expected = match tokens {
        RunTokens::Autoregressive(_) => [1, 1, 1, 0],
        RunTokens::TeacherForced(_) => [1, 0, 1, 2],
    };
    if replaced != expected {
        anyhow::bail!("incompatible manifest template: expected [seed binding, decode count, tokenize count, claimed bindings] = {expected:?}; found {replaced:?}");
    }
    out.push(String::new());
    Ok(out.join("\n"))
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

fn run_id(run_spec_path: &Path, rendered_prompt: &str, tokens: u32, chain: ChainKind) -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let digest = Sha256::digest(format!(
        "{}\0{}\0{}\0{:?}\0{}",
        run_spec_path.display(),
        rendered_prompt,
        tokens,
        chain,
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
            claimed_token_ids: None,
        };
        let prepared = prepare_claim_run(&options).unwrap();
        let rendered = fs::read_to_string(&prepared.manifest_path).unwrap();
        assert!(rendered.contains("name = \"model-b\""));
        assert!(rendered.contains("count = 3"));
        assert!(!rendered.contains("model-a"));
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
                chain: Default::default(),
                claimed_token_ids: None,
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
    fn run_manifest_replaces_prompt_artifact_and_decode_count() {
        let template = r#"[chain]
name = "test"

[chain.input.weights]
index = "l"
path = "runtime/weights/layer{l}.rastered"
index_path = "runtime/weights/layer{l}.rindex"
commitments = ["abc"]

[[chain.stage]]
name = "prompt_merge_seed"
project = "raster-stages/prompt-prepare"
inputs.tokenizer = { external = { path = "tokenizer.rastered", index_path = "tokenizer.rindex", commitment = "tok" } }
inputs.initial_pieces = { external = { path = "old.rastered", index_path = "old.rindex", commitment = "old" } }

[[chain.repeat]]
name = "tokenize"
index = "b"
count = 99
[[chain.repeat.stage]]
name = "prompt_merge_b{b}"
project = "raster-stages/prompt-merge"
inputs.initial_pieces = { from = "prompt_merge_b{b-1}", first = "prompt_merge_seed" }

[[chain.repeat]]
name = "decode"
index = "t"
count = 1

  [[chain.repeat.stage]]
name = "decode_select_t{t}"
project = "raster-stages/decode-select-token"
"#;

        let pieces = RunExternal::new(
            Path::new("/tmp/run/prompt/initial_pieces.rastered"),
            Path::new("/tmp/run/prompt/initial_pieces.rindex"),
            "abc",
        );
        let manifest = render_run_manifest(
            template,
            Path::new("/repo"),
            &pieces,
            RunTokens::Autoregressive(7),
            0,
        )
        .unwrap();

        assert!(manifest.contains("path = \"/tmp/run/prompt/initial_pieces.rastered\""));
        assert!(manifest.contains("commitment = \"abc\""));
        assert!(manifest.contains("count = 7"));
        assert!(manifest.contains("count = 0"));
        assert!(!manifest.contains("count = 99"));
        assert!(manifest.contains("project = \"/repo/raster-stages/prompt-prepare\""));
        assert!(manifest.contains("project = \"/repo/raster-stages/decode-select-token\""));
        assert!(manifest.contains("path = \"/repo/tokenizer.rastered\""));
        assert!(manifest.contains("index_path = \"/repo/tokenizer.rindex\""));
        assert!(manifest.contains("path = \"/repo/runtime/weights/layer{l}.rastered\""));
        assert!(manifest.contains("index_path = \"/repo/runtime/weights/layer{l}.rindex\""));
        assert!(!manifest.contains("old.rastered"));
        assert_eq!(manifest.matches("inputs.initial_pieces =").count(), 2);

        let claimed = RunExternal::new(
            Path::new("/tmp/run/prompt/claimed_output.rastered"),
            Path::new("/tmp/run/prompt/claimed_output.rindex"),
            "def",
        );
        assert!(render_run_manifest(
            template,
            Path::new("/repo"),
            &pieces,
            RunTokens::TeacherForced(claimed),
            0,
        )
        .unwrap_err()
        .to_string()
        .contains("cannot fill an autoregressive decode repeat"));
    }

    #[test]
    fn teacher_forced_manifest_binds_claim_into_join_and_check() {
        let template = r#"[chain]
name = "iris"

[[chain.stage]]
name = "prompt_merge_seed"
project = "raster-stages/prompt-merge"
inputs.initial_pieces = { external = { path = "target/iris/run-prompt-placeholder/initial_pieces.rastered", index_path = "target/iris/run-prompt-placeholder/initial_pieces.rindex", commitment = "0000" } }

[[chain.repeat]]
name = "tokenize"
index = "b"
count = 0

[[chain.stage]]
name = "input_embedding"
project = "raster-stages/input-embedding"
inputs.prompt = { from = "prompt_prepare" }
inputs.claimed = { external = { path = "target/iris/run-prompt-placeholder/claimed_output.rastered", index_path = "target/iris/run-prompt-placeholder/claimed_output.rindex", commitment = "0000" } }

[[chain.stage]]
name = "sequence_check"
project = "raster-stages/sequence-check"
inputs.claimed = { external = { path = "target/iris/run-prompt-placeholder/claimed_output.rastered", index_path = "target/iris/run-prompt-placeholder/claimed_output.rindex", commitment = "0000" } }
"#;
        let pieces = RunExternal::new(
            Path::new("/tmp/run/prompt/initial_pieces.rastered"),
            Path::new("/tmp/run/prompt/initial_pieces.rindex"),
            "abc",
        );
        let claimed = RunExternal::new(
            Path::new("/tmp/run/prompt/claimed_output.rastered"),
            Path::new("/tmp/run/prompt/claimed_output.rindex"),
            "def",
        );
        let manifest = render_run_manifest(
            template,
            Path::new("/repo"),
            &pieces,
            RunTokens::TeacherForced(claimed),
            3,
        )
        .unwrap();
        assert_eq!(
            manifest
                .matches("path = \"/tmp/run/prompt/claimed_output.rastered\"")
                .count(),
            2
        );
        assert_eq!(manifest.matches("commitment = \"def\"").count(), 2);
        assert!(manifest.contains("count = 3"));
        assert!(!manifest.contains("run-prompt-placeholder"));
        assert!(render_run_manifest(
            template,
            Path::new("/repo"),
            &pieces,
            RunTokens::Autoregressive(1),
            3,
        )
        .is_err());
    }

    #[test]
    fn first_disagreement_reports_value_and_length_differences() {
        assert_eq!(first_disagreement(&[1, 2, 3], &[1, 2, 3]), None);
        assert_eq!(first_disagreement(&[1, 9, 3], &[1, 2, 3]), Some(1));
        assert_eq!(first_disagreement(&[1, 2], &[1, 2, 3]), Some(2));
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
        std::env::temp_dir().join(format!(
            "iris-cli-{label}-{}-{nanos}",
            std::process::id()
        ))
    }
}
