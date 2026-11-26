use std::collections::BTreeMap;
use std::ffi::OsString;
use std::fs;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{bail, Context, Result};
use procedure_definition::{
    checkpoint_hash, read_checkpoint_from_stage_dir, Checkpoint, CheckpointList, Divergence,
    DivergenceReason, CHECKPOINT_LIST_JSON,
};
use raster_core::input::payload_structural_root;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::cache::{CachedInputs, CachedStageValue, StageOutputCache};
use crate::parity::{parity_dir, EXECUTION_TIMES_JSON};
use crate::segments::inference::weights::{RunWeights, WeightAllocation};
use crate::segments::{self, StageKind};
use crate::worker::{WorkerJob, WorkerPool};

#[derive(Debug)]
pub struct HybridRun {
    pub pipeline_dir: PathBuf,
    pub selected_stage_dir: Option<PathBuf>,
    pub final_output: Option<CachedStageValue>,
    /// The output of the last check the run reached of each inference, in
    /// the order the pipeline runs them. The result stage follows them.
    pub final_checks: Vec<CachedStageValue>,
    /// For each input the run's last stage took from an earlier stage, the
    /// inference that stage is of: which check a result stage read.
    pub final_reads: BTreeMap<String, usize>,
}

/// A native prefix ending at one exact expanded pipeline stage.
#[derive(Debug, Clone)]
pub struct StagePrefixRun {
    pub pipeline_dir: PathBuf,
    pub stage: String,
    pub project_dir: PathBuf,
    pub stage_dir: PathBuf,
    pub input_path: PathBuf,
    pub input_manifest_path: PathBuf,
    pub output_path: PathBuf,
    pub output_index_path: PathBuf,
    pub output_manifest_path: PathBuf,
}

#[derive(Debug, Clone)]
pub struct CheckpointHashVerifier {
    pub claimed_hashes_path: PathBuf,
    pub claimed_hashes: Vec<String>,
    /// For each checkpoint of the claim, the producers it points at. A run
    /// of the pipeline as a claim arranges it states these as the claim does:
    /// a pointer is the claim's own statement, which the link comparison
    /// judges, and a run reproduces only what a stage loads and makes. With
    /// none, each checkpoint's pointers are the ones its bindings give.
    pub claimed_sources: Option<Vec<BTreeMap<String, u32>>>,
}

/// A stage of a manifest with its repeats expanded: the stage as it runs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpandedStage {
    pub name: String,
    pub project: String,
    pub inputs: BTreeMap<String, ExpandedInput>,
}

/// What an expanded stage's input is bound to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExpandedInput {
    /// The output of the stage with this name.
    From(String),
    /// A file, with the commitment it is held to.
    External {
        path: String,
        index_path: Option<String>,
        commitment: String,
    },
}

/// The stages a manifest runs, in order, each with what its inputs are bound
/// to. This is the expansion a run executes.
pub fn expanded_stages(manifest_path: &Path) -> Result<Vec<ExpandedStage>> {
    Ok(read_manifest(manifest_path)?
        .pipeline
        .stage
        .into_iter()
        .map(|stage| ExpandedStage {
            name: stage.name,
            project: stage.project,
            inputs: stage
                .inputs
                .into_iter()
                .map(|(name, binding)| {
                    let binding = match binding {
                        InputBinding::From(from) => ExpandedInput::From(from),
                        InputBinding::External(external) => ExpandedInput::External {
                            path: external.path,
                            index_path: external.index_path,
                            commitment: external.commitment,
                        },
                    };
                    (name, binding)
                })
                .collect(),
        })
        .collect())
}

#[derive(Debug)]
pub struct CheckpointVerifiedRun {
    pub run: HybridRun,
    pub recomputed_list: CheckpointList,
    pub divergence: Option<Divergence>,
}

struct PipelineRunOutput {
    run: HybridRun,
    recomputed_list: Option<CheckpointList>,
    divergence: Option<Divergence>,
}

#[derive(Debug)]
pub(crate) struct Manifest {
    pub(crate) pipeline: PipelineSpec,
}

#[derive(Debug, Clone)]
pub(crate) struct PipelineSpec {
    pub(crate) stage: Vec<StageSpec>,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub(crate) struct StageSpec {
    pub(crate) name: String,
    pub(crate) project: String,
    #[serde(default)]
    pub(crate) inputs: BTreeMap<String, InputBinding>,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum InputBinding {
    External(ExternalRef),
    From(String),
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub(crate) struct ExternalRef {
    path: String,
    #[serde(default)]
    index_path: Option<String>,
    commitment: String,
}

#[derive(Debug, Deserialize)]
struct RasterTomlDoc {
    pipeline: PipelineTable,
}

#[derive(Debug, Deserialize)]
struct PipelineTable {
    #[serde(default, rename = "input")]
    inputs: BTreeMap<String, InputDecl>,
    #[serde(default, rename = "stage")]
    stages: Vec<toml::Spanned<PlainStageSpec>>,
    #[serde(default, rename = "repeat")]
    repeats: Vec<toml::Spanned<RepeatSpec>>,
}

/// A `[[pipeline.stage]]` as the manifest writes it. Its bindings are a repeat
/// stage's, read with no index in scope: a stage, an external, or a pipeline
/// input by name.
#[derive(Debug, Deserialize)]
struct PlainStageSpec {
    name: String,
    project: String,
    #[serde(default)]
    inputs: BTreeMap<String, RepeatBinding>,
}

#[derive(Debug)]
enum PipelineItem {
    Stage(PlainStageSpec),
    Repeat(RepeatSpec),
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum InputDecl {
    Indexed(IndexedInputDecl),
    Single(ExternalRef),
}

#[derive(Debug, Deserialize)]
struct IndexedInputDecl {
    index: String,
    path: String,
    #[serde(default)]
    index_path: Option<String>,
    commitments: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct RepeatSpec {
    name: String,
    index: String,
    #[serde(default)]
    start: u32,
    count: u32,
    #[serde(default, rename = "stage")]
    stages: Vec<RepeatStageSpec>,
    #[serde(default)]
    exports: BTreeMap<String, ExportDecl>,
}

#[derive(Debug, Deserialize)]
struct RepeatStageSpec {
    name: String,
    project: String,
    #[serde(default)]
    index: Option<String>,
    #[serde(default)]
    start: u32,
    #[serde(default)]
    count: Option<u32>,
    #[serde(default)]
    inputs: BTreeMap<String, RepeatBinding>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum RepeatBinding {
    From {
        from: String,
        #[serde(default)]
        first: Option<String>,
    },
    Input {
        input: String,
    },
    External {
        external: ExternalRef,
    },
}

#[derive(Debug, Deserialize)]
struct ExportDecl {
    stage: String,
    entry: String,
}

#[derive(Clone, Copy)]
struct TemplateIndex<'a> {
    name: &'a str,
    value: u32,
    start: u32,
}

enum RenderedTemplate {
    Text(String),
    Underflow,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StageDispatch {
    Native,
    RasterReference,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeStageBackend {
    InProcess,
    Subprocess,
}

#[derive(Debug)]
struct StageOutput {
    payload_commitment: Vec<u8>,
    structural_commitment: Vec<u8>,
}

struct NativeStageRun {
    timings: Option<segments::NativeTimings>,
    duration: Duration,
    output: Option<CachedStageValue>,
}

#[derive(Debug, Serialize)]
struct StagePerformance {
    name: String,
    worker: Option<usize>,
    timings: segments::NativeTimings,
    cache: inference_kernels::weight_budget::WeightCacheStats,
}

struct PipelineRunState {
    auxiliary_process_starts: usize,
    memory: crate::segments::inference::weights::MemorySampler,
    weights: RunWeights,
    allocation: WeightAllocation,
    workers: Option<WorkerPool>,
    performance: Vec<StagePerformance>,
    output_commitments: Vec<Option<Vec<u8>>>,
    execution_times: Vec<Option<Duration>>,
    aux_waves: Vec<AuxWaveExecutionTime>,
    output_cache: StageOutputCache,
    selected_stage_dir: Option<PathBuf>,
    last_output: Option<CachedStageValue>,
    /// The last check reached of each inference, by the inference's index.
    last_checks: std::collections::BTreeMap<usize, CachedStageValue>,
    /// The inference each piped input of the last stage run is of.
    last_reads: BTreeMap<String, usize>,
    checkpoint_verifier: Option<CheckpointVerifierState>,
}

struct AuxStageJob {
    idx: usize,
    name: String,
    kind: StageKind,
    input_json_path: PathBuf,
    input_manifest_path: PathBuf,
    stage_dir: PathBuf,
}

struct AuxStageResult {
    performance: Option<StagePerformance>,
    idx: usize,
    name: String,
    stage_dir: PathBuf,
    duration: Duration,
    output: StageOutput,
}

struct AuxStageBatch {
    results: Vec<AuxStageResult>,
    wall_duration: Duration,
    parallelism: usize,
}

impl PipelineRunState {
    fn has_divergence(&self) -> bool {
        self.checkpoint_verifier
            .as_ref()
            .and_then(|verifier| verifier.divergence.as_ref())
            .is_some()
    }

    fn divergence(&self) -> Option<&Divergence> {
        self.checkpoint_verifier
            .as_ref()
            .and_then(|verifier| verifier.divergence.as_ref())
    }
}

#[derive(Debug)]
struct CheckpointVerifierState {
    claimed_hashes_path: PathBuf,
    claimed_hashes: Vec<String>,
    claimed_sources: Option<Vec<BTreeMap<String, u32>>>,
    recomputed_checkpoints: Vec<Checkpoint>,
    divergence: Option<Divergence>,
}

impl CheckpointVerifierState {
    fn new(verifier: CheckpointHashVerifier) -> Self {
        Self {
            claimed_hashes_path: verifier.claimed_hashes_path,
            claimed_hashes: verifier.claimed_hashes,
            claimed_sources: verifier.claimed_sources,
            recomputed_checkpoints: Vec::new(),
            divergence: None,
        }
    }

    fn observe_stage(
        &mut self,
        checkpoint_index: usize,
        stage_name: &str,
        stage_dir: &Path,
        recomputed_list_path: &Path,
    ) -> Result<bool> {
        if self.divergence.is_some() {
            return Ok(true);
        }
        if checkpoint_index != self.recomputed_checkpoints.len() {
            bail!(
                "checkpoint verifier observed stage index {checkpoint_index} after {} checkpoints",
                self.recomputed_checkpoints.len()
            );
        }

        let mut checkpoint = read_checkpoint_from_stage_dir(stage_name, stage_dir)?;
        if let Some(sources) = self
            .claimed_sources
            .as_ref()
            .and_then(|sources| sources.get(checkpoint_index))
        {
            checkpoint.input_sources = sources.clone();
        }
        let recomputed_hash = checkpoint_hash(&checkpoint)?;
        let claimed_hash = self.claimed_hashes.get(checkpoint_index).cloned();
        let reason = match claimed_hash.as_ref() {
            Some(claimed_hash) if claimed_hash == &recomputed_hash => None,
            Some(_) => Some(DivergenceReason::CheckpointHash),
            None => Some(DivergenceReason::MissingClaimedHash),
        };
        self.recomputed_checkpoints.push(checkpoint.clone());

        if let Some(reason) = reason {
            self.divergence = Some(Divergence {
                version: 2,
                checkpoint_index,
                stage: stage_name.to_string(),
                reason,
                claimed_list_path: None,
                claimed_checkpoint_hashes_path: Some(self.claimed_hashes_path.clone()),
                recomputed_list_path: recomputed_list_path.to_path_buf(),
                claimed_hash,
                recomputed_hash: Some(recomputed_hash),
                claimed: None,
                recomputed: Some(checkpoint),
            });
        }
        Ok(self.divergence.is_some())
    }

    fn finish(&mut self, recomputed_list_path: &Path) {
        if self.divergence.is_some() {
            return;
        }
        let checkpoint_index = self.recomputed_checkpoints.len();
        if let Some(claimed_hash) = self.claimed_hashes.get(checkpoint_index).cloned() {
            self.divergence = Some(Divergence {
                version: 2,
                checkpoint_index,
                stage: format!("<checkpoint-{checkpoint_index}>"),
                reason: DivergenceReason::MissingRecomputedHash,
                claimed_list_path: None,
                claimed_checkpoint_hashes_path: Some(self.claimed_hashes_path.clone()),
                recomputed_list_path: recomputed_list_path.to_path_buf(),
                claimed_hash: Some(claimed_hash),
                recomputed_hash: None,
                claimed: None,
                recomputed: None,
            });
        }
    }

    fn list(&self) -> CheckpointList {
        CheckpointList {
            checkpoints: self.recomputed_checkpoints.clone(),
        }
    }
}

#[derive(Debug, Serialize)]
struct ExecutionTimesDocument {
    version: u32,
    stages: Vec<StageExecutionTime>,
    total_exec_duration_ns: u128,
    total_wall_duration_ns: u128,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    aux_waves: Vec<AuxWaveExecutionTime>,
}

#[derive(Debug, Serialize)]
struct StageExecutionTime {
    name: String,
    exec_duration_ns: u128,
}

#[derive(Clone, Debug, Serialize)]
struct AuxWaveExecutionTime {
    name: String,
    first_stage: String,
    last_stage: String,
    stage_count: usize,
    parallelism: usize,
    wall_duration_ns: u128,
    stage_duration_sum_ns: u128,
}

pub fn run(
    manifest_path: &Path,
    raster_stage: Option<&str>,
    current_exe: &Path,
    native_stage_backend: NativeStageBackend,
) -> Result<HybridRun> {
    Ok(run_inner(
        manifest_path,
        raster_stage,
        current_exe,
        native_stage_backend,
        None,
        None,
    )?
    .run)
}

pub fn run_until_hash_divergence(
    manifest_path: &Path,
    current_exe: &Path,
    native_stage_backend: NativeStageBackend,
    verifier: CheckpointHashVerifier,
) -> Result<CheckpointVerifiedRun> {
    let output = run_inner(
        manifest_path,
        None,
        current_exe,
        native_stage_backend,
        Some(verifier),
        None,
    )?;
    Ok(CheckpointVerifiedRun {
        run: output.run,
        recomputed_list: output
            .recomputed_list
            .expect("verified run always records recomputed checkpoints"),
        divergence: output.divergence,
    })
}

pub fn run_through_stage(
    manifest_path: &Path,
    stage: &str,
    current_exe: &Path,
    native_stage_backend: NativeStageBackend,
) -> Result<StagePrefixRun> {
    let manifest = read_manifest(manifest_path)?;
    validate_reference_stage(&manifest.pipeline.stage, stage)?;
    let selected = manifest
        .pipeline
        .stage
        .iter()
        .find(|s| s.name == stage)
        .unwrap();
    let project_dir = absolute(&std::env::current_dir()?, &selected.project);
    let run = run_inner(
        manifest_path,
        None,
        current_exe,
        native_stage_backend,
        None,
        Some(stage),
    )?
    .run;
    let stage_dir = run.pipeline_dir.join(stage);
    Ok(StagePrefixRun {
        pipeline_dir: run.pipeline_dir,
        stage: stage.to_string(),
        project_dir,
        input_path: stage_dir.join("input.json"),
        input_manifest_path: stage_dir.join("input_manifest.json"),
        output_path: stage_dir.join("output.bin"),
        output_index_path: stage_dir.join("output.rindex"),
        output_manifest_path: stage_dir.join("output_manifest.json"),
        stage_dir,
    })
}

fn truncate_through_stage(stages: &mut Vec<StageSpec>, stage: &str) -> Result<()> {
    validate_reference_stage(stages, stage)?;
    let index = stages.iter().position(|s| s.name == stage).unwrap();
    stages.truncate(index + 1);
    Ok(())
}

fn run_inner(
    manifest_path: &Path,
    raster_stage: Option<&str>,
    current_exe: &Path,
    native_stage_backend: NativeStageBackend,
    checkpoint_verifier: Option<CheckpointHashVerifier>,
    through_stage: Option<&str>,
) -> Result<PipelineRunOutput> {
    let ple_loading = crate::segments::inference::ple::Loading::from_env()?;
    let base_dir = std::env::current_dir().context("failed to read current directory")?;
    let mut manifest = read_manifest(manifest_path)?;
    validate_supported_stages(&manifest.pipeline.stage)?;
    // Slice the expanded pipeline before scheduling. This also bounds aux waves,
    // output slots and timing records without scheduling any later stages.
    build_stage_index(&manifest.pipeline.stage)?;
    if let Some(stage) = through_stage {
        truncate_through_stage(&mut manifest.pipeline.stage, stage)?;
    }
    if let Some(raster_stage) = raster_stage {
        validate_reference_stage(&manifest.pipeline.stage, raster_stage)?;
    }

    let pipeline_dir = create_pipeline_dir(&base_dir)?;
    println!(
        "pipeline run  {}  ({} stages)",
        pipeline_run_id_label(&pipeline_dir),
        manifest.pipeline.stage.len()
    );
    println!("  dir: {}", pipeline_dir.display());
    match raster_stage {
        Some(stage) => {
            println!(
                "  mode: unauthenticated pipeline runner (--no-auth; Raster reference: {stage})"
            );
        }
        None => println!("  mode: unauthenticated procedure-pipeline (--no-auth)"),
    }
    println!();

    let pipeline_started = Instant::now();
    let stage_index = build_stage_index(&manifest.pipeline.stage)?;
    let largest_wave = (0..manifest.pipeline.stage.len())
        .filter_map(|i| aux_wave_range(&manifest.pipeline.stage, i).map(|r| r.len()))
        .max()
        .unwrap_or(0);
    let allocation = if native_stage_backend == NativeStageBackend::InProcess {
        WeightAllocation::resolve(persistent_aux_parallelism(largest_wave))?
    } else {
        WeightAllocation::split(0, 0, "diagnostic subprocess backend")
    };
    eprintln!(
        "retained weight budget: {} bytes ({}); main={}, workers={} x {} bytes",
        allocation.total_bytes,
        allocation.source,
        allocation.main_bytes,
        allocation.worker_count,
        allocation.worker_bytes
    );
    let mut state = PipelineRunState {
        auxiliary_process_starts: 0,
        memory: crate::segments::inference::weights::MemorySampler::start(),
        weights: RunWeights::new(allocation.main_bytes),
        allocation,
        workers: None,
        performance: Vec::new(),
        output_commitments: vec![None; manifest.pipeline.stage.len()],
        execution_times: vec![None; manifest.pipeline.stage.len()],
        aux_waves: Vec::new(),
        output_cache: StageOutputCache::default(),
        selected_stage_dir: None,
        last_output: None,
        last_checks: std::collections::BTreeMap::new(),
        last_reads: BTreeMap::new(),
        checkpoint_verifier: checkpoint_verifier.map(CheckpointVerifierState::new),
    };
    let recomputed_list_path = pipeline_dir.join(CHECKPOINT_LIST_JSON);

    let mut idx = 0;
    while idx < manifest.pipeline.stage.len() {
        if let Some(aux_range) = aux_wave_range(&manifest.pipeline.stage, idx) {
            run_aux_wave(
                aux_range.clone(),
                &manifest.pipeline.stage,
                &base_dir,
                &pipeline_dir,
                &stage_index,
                current_exe,
                native_stage_backend,
                raster_stage,
                &recomputed_list_path,
                &mut state,
            )?;
            if state.has_divergence() {
                break;
            }
            idx = aux_range.end;
            continue;
        }

        run_one_stage(
            idx,
            &manifest.pipeline.stage,
            &base_dir,
            &pipeline_dir,
            &stage_index,
            current_exe,
            native_stage_backend,
            raster_stage,
            &recomputed_list_path,
            &mut state,
        )?;
        if state.has_divergence() {
            break;
        }
        idx += 1;
    }

    if let Some(pool) = state.workers.as_mut() {
        pool.shutdown()?;
    }
    fs::write(
        pipeline_dir.join("pipeline-performance.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "version": 1,
            "ple_loading": ple_loading.as_str(),
            "embedding_loading": "selective",
            "head_loading": "paged",
            "allocation": state.allocation,
            "auxiliary_process_starts": state.auxiliary_process_starts,
            "wall_duration_ns": pipeline_started.elapsed().as_nanos(),
            "main_cache": state.weights.stats(),
            "sampled_peak_rss_bytes": state.memory.peak(),
            "stages": state.performance,
        }))?,
    )?;
    let pipeline_wall_duration = pipeline_started.elapsed();
    if let Some(verifier) = state.checkpoint_verifier.as_mut() {
        verifier.finish(&recomputed_list_path);
    }
    if state.has_divergence() {
        if let Some(divergence) = state.divergence() {
            println!(
                "procedure-pipeline stopped at checkpoint index {} / stage {}/{} ({})",
                divergence.checkpoint_index,
                divergence.checkpoint_index + 1,
                manifest.pipeline.stage.len(),
                divergence.stage
            );
        }
    } else {
        write_execution_times(
            &pipeline_dir,
            &manifest.pipeline.stage,
            &state.execution_times,
            pipeline_wall_duration,
            &state.aux_waves,
        )?;
        print_native_timing_summary(
            &manifest.pipeline.stage,
            &state.execution_times,
            pipeline_wall_duration,
            &state.aux_waves,
        )?;
    }
    if raster_stage.is_some() && state.selected_stage_dir.is_none() {
        bail!("selected Raster reference stage did not run");
    }

    let recomputed_list = state
        .checkpoint_verifier
        .as_ref()
        .map(CheckpointVerifierState::list);
    let divergence = state
        .checkpoint_verifier
        .as_ref()
        .and_then(|verifier| verifier.divergence.clone());

    Ok(PipelineRunOutput {
        run: HybridRun {
            pipeline_dir,
            selected_stage_dir: state.selected_stage_dir,
            final_output: state.last_output,
            final_checks: state.last_checks.into_values().collect(),
            final_reads: state.last_reads,
        },
        recomputed_list,
        divergence,
    })
}

pub(crate) fn read_manifest(path: &Path) -> Result<Manifest> {
    let text =
        fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))?;
    text.parse::<toml::Value>()
        .context("failed to parse Raster.toml as TOML")?;
    let doc: RasterTomlDoc =
        toml::from_str(&text).context("failed to decode Raster.toml pipeline")?;
    Ok(Manifest {
        pipeline: expand_pipeline_table(doc.pipeline)?,
    })
}

fn expand_pipeline_table(table: PipelineTable) -> Result<PipelineSpec> {
    let inputs = flatten_input_decls(table.inputs)?;
    let mut stages = Vec::new();
    let mut exports = BTreeMap::new();

    for item in merge_pipeline_items(table.stages, table.repeats) {
        match item {
            PipelineItem::Stage(stage) => {
                let mut bound = BTreeMap::new();
                for (param, binding) in &stage.inputs {
                    bound.insert(
                        param.clone(),
                        expand_repeat_binding(binding, &inputs, &[])
                            .with_context(|| format!("stage `{}`", stage.name))?,
                    );
                }
                stages.push(StageSpec {
                    name: stage.name,
                    project: stage.project,
                    inputs: bound,
                });
            }
            PipelineItem::Repeat(repeat) => {
                expand_repeat(&repeat, &inputs, &mut stages, &mut exports)?
            }
        }
    }

    resolve_export_sources(&mut stages, &exports);
    Ok(PipelineSpec { stage: stages })
}

fn merge_pipeline_items(
    stages: Vec<toml::Spanned<PlainStageSpec>>,
    repeats: Vec<toml::Spanned<RepeatSpec>>,
) -> Vec<PipelineItem> {
    let mut items: Vec<(usize, PipelineItem)> = stages
        .into_iter()
        .map(|stage| (stage.span().start, PipelineItem::Stage(stage.into_inner())))
        .chain(repeats.into_iter().map(|repeat| {
            (
                repeat.span().start,
                PipelineItem::Repeat(repeat.into_inner()),
            )
        }))
        .collect();
    items.sort_by_key(|(offset, _)| *offset);
    items.into_iter().map(|(_, item)| item).collect()
}

fn flatten_input_decls(
    decls: BTreeMap<String, InputDecl>,
) -> Result<BTreeMap<String, ExternalRef>> {
    let mut inputs = BTreeMap::new();
    for (name, decl) in decls {
        match decl {
            InputDecl::Single(external) => {
                if inputs.insert(name.clone(), external).is_some() {
                    bail!("pipeline input `{name}` is declared more than once");
                }
            }
            InputDecl::Indexed(indexed) => {
                for (member, external) in indexed.flatten(&name)? {
                    if inputs.insert(member.clone(), external).is_some() {
                        bail!("pipeline input `{member}` is declared more than once");
                    }
                }
            }
        }
    }
    Ok(inputs)
}

impl IndexedInputDecl {
    fn flatten(self, family: &str) -> Result<Vec<(String, ExternalRef)>> {
        let placeholder = format!("{{{}}}", self.index);
        if !self.path.contains(&placeholder) {
            bail!(
                "[pipeline.input.{family}]: path '{}' does not mention '{placeholder}'",
                self.path
            );
        }

        Ok(self
            .commitments
            .into_iter()
            .enumerate()
            .map(|(idx, commitment)| {
                let index = idx.to_string();
                let render = |value: &str| value.replace(&placeholder, &index);
                (
                    format!("{family}_{idx}"),
                    ExternalRef {
                        path: render(&self.path),
                        index_path: self.index_path.as_deref().map(render),
                        commitment,
                    },
                )
            })
            .collect())
    }
}

fn expand_repeat(
    repeat: &RepeatSpec,
    inputs: &BTreeMap<String, ExternalRef>,
    stages: &mut Vec<StageSpec>,
    exports: &mut BTreeMap<String, String>,
) -> Result<()> {
    for outer in repeat.start..repeat.start + repeat.count {
        let outer_index = TemplateIndex {
            name: &repeat.index,
            value: outer,
            start: repeat.start,
        };
        for stage in &repeat.stages {
            if let (Some(inner_name), Some(inner_count)) = (&stage.index, stage.count) {
                for inner in stage.start..stage.start + inner_count {
                    let inner_index = TemplateIndex {
                        name: inner_name,
                        value: inner,
                        start: stage.start,
                    };
                    push_repeat_stage(stage, inputs, &[outer_index, inner_index], stages)?;
                }
            } else {
                push_repeat_stage(stage, inputs, &[outer_index], stages)?;
            }
        }
    }

    let export_index = TemplateIndex {
        name: &repeat.index,
        value: repeat.start + repeat.count.saturating_sub(1),
        start: repeat.start,
    };
    for (name, export) in &repeat.exports {
        let source = if repeat.count == 0 {
            export.entry.clone()
        } else {
            render_text(&export.stage, &[export_index])?
        };
        exports.insert(format!("{}.{}", repeat.name, name), source);
    }

    Ok(())
}

fn push_repeat_stage(
    template: &RepeatStageSpec,
    named_inputs: &BTreeMap<String, ExternalRef>,
    indexes: &[TemplateIndex<'_>],
    stages: &mut Vec<StageSpec>,
) -> Result<()> {
    let mut inputs = BTreeMap::new();
    for (param, binding) in &template.inputs {
        inputs.insert(
            param.clone(),
            expand_repeat_binding(binding, named_inputs, indexes)?,
        );
    }
    stages.push(StageSpec {
        name: render_text(&template.name, indexes)?,
        project: template.project.clone(),
        inputs,
    });
    Ok(())
}

fn expand_repeat_binding(
    binding: &RepeatBinding,
    named_inputs: &BTreeMap<String, ExternalRef>,
    indexes: &[TemplateIndex<'_>],
) -> Result<InputBinding> {
    match binding {
        RepeatBinding::From { from, first } => match render_template(from, indexes)? {
            RenderedTemplate::Text(source) => Ok(InputBinding::From(source)),
            RenderedTemplate::Underflow => {
                let first = first.as_ref().ok_or_else(|| {
                    anyhow::anyhow!(
                        "repeat binding `{from}` underflows but has no `first` fallback"
                    )
                })?;
                Ok(InputBinding::From(render_text(first, indexes)?))
            }
        },
        RepeatBinding::Input { input } => {
            let name = render_text(input, indexes)?;
            let external = named_inputs.get(&name).ok_or_else(|| {
                anyhow::anyhow!("repeat binding names unknown pipeline input `{name}`")
            })?;
            Ok(InputBinding::External(external.clone()))
        }
        RepeatBinding::External { external } => Ok(InputBinding::External(external.clone())),
    }
}

fn resolve_export_sources(stages: &mut [StageSpec], exports: &BTreeMap<String, String>) {
    for stage in stages {
        for binding in stage.inputs.values_mut() {
            if let InputBinding::From(source) = binding {
                if let Some(target) = exports.get(source) {
                    *source = target.clone();
                }
            }
        }
    }
}

fn render_text(template: &str, indexes: &[TemplateIndex<'_>]) -> Result<String> {
    match render_template(template, indexes)? {
        RenderedTemplate::Text(value) => Ok(value),
        RenderedTemplate::Underflow => {
            bail!("template `{template}` underflows outside a `first` binding")
        }
    }
}

fn render_template(template: &str, indexes: &[TemplateIndex<'_>]) -> Result<RenderedTemplate> {
    let mut rendered = String::new();
    let mut rest = template;
    while let Some(open) = rest.find('{') {
        rendered.push_str(&rest[..open]);
        let after_open = &rest[open + 1..];
        let close = after_open.find('}').ok_or_else(|| {
            anyhow::anyhow!("template `{template}` has an unterminated placeholder")
        })?;
        let placeholder = &after_open[..close];
        let (name, previous) = placeholder
            .strip_suffix("-1")
            .map(|name| (name, true))
            .unwrap_or((placeholder, false));
        let index = indexes
            .iter()
            .find(|index| index.name == name)
            .ok_or_else(|| {
                anyhow::anyhow!("template `{template}` references unknown index `{name}`")
            })?;
        if previous && index.value == index.start {
            return Ok(RenderedTemplate::Underflow);
        }
        let value = if previous {
            index.value - 1
        } else {
            index.value
        };
        rendered.push_str(&value.to_string());
        rest = &after_open[close + 1..];
    }
    rendered.push_str(rest);
    Ok(RenderedTemplate::Text(rendered))
}

pub(crate) fn validate_supported_stages(stages: &[StageSpec]) -> Result<()> {
    for stage in stages {
        StageKind::from_stage_spec(&stage.project, &stage.name)?;
    }
    Ok(())
}

pub(crate) fn build_stage_index(stages: &[StageSpec]) -> Result<BTreeMap<String, usize>> {
    let mut stage_index = BTreeMap::new();
    for (idx, stage) in stages.iter().enumerate() {
        if stage_index.insert(stage.name.clone(), idx).is_some() {
            bail!("pipeline contains duplicate stage name `{}`", stage.name);
        }
    }
    Ok(stage_index)
}

fn validate_reference_stage(stages: &[StageSpec], raster_stage: &str) -> Result<()> {
    let count = stages
        .iter()
        .filter(|stage| stage.name == raster_stage)
        .count();
    if count != 1 {
        bail!("expected exactly one `{raster_stage}` stage in Raster.toml, found {count}");
    }
    let stage = stages
        .iter()
        .find(|stage| stage.name == raster_stage)
        .expect("stage existence checked above");
    StageKind::from_stage_spec(&stage.project, &stage.name)?;
    Ok(())
}

fn dispatch_for_stage(stage: &StageSpec, raster_stage: Option<&str>) -> Result<StageDispatch> {
    StageKind::from_stage_spec(&stage.project, &stage.name)?;
    if raster_stage == Some(stage.name.as_str()) {
        Ok(StageDispatch::RasterReference)
    } else {
        Ok(StageDispatch::Native)
    }
}

fn run_one_stage(
    idx: usize,
    stages: &[StageSpec],
    base_dir: &Path,
    pipeline_dir: &Path,
    stage_index: &BTreeMap<String, usize>,
    current_exe: &Path,
    native_stage_backend: NativeStageBackend,
    raster_stage: Option<&str>,
    recomputed_list_path: &Path,
    state: &mut PipelineRunState,
) -> Result<()> {
    let stage = &stages[idx];
    println!(
        "▸ stage {}/{}  {}   ({})",
        idx + 1,
        stages.len(),
        stage.name,
        stage.project
    );

    let stage_dir = pipeline_dir.join(&stage.name);
    fs::create_dir_all(&stage_dir)
        .with_context(|| format!("failed to create {}", stage_dir.display()))?;

    let (input_json_path, input_manifest_path) = synthesize_inputs(
        stage,
        &stage_dir,
        base_dir,
        pipeline_dir,
        &state.output_commitments,
        stage_index,
    )?;

    let stage_run = run_prepared_stage(
        stage,
        base_dir,
        current_exe,
        native_stage_backend,
        raster_stage,
        &input_json_path,
        &input_manifest_path,
        &stage_dir,
        stage_index,
        state,
    )?;

    finish_stage(
        idx,
        stage,
        stage_index,
        stage_run,
        stage_dir,
        recomputed_list_path,
        state,
    )
}

fn run_prepared_stage(
    stage: &StageSpec,
    base_dir: &Path,
    current_exe: &Path,
    native_stage_backend: NativeStageBackend,
    raster_stage: Option<&str>,
    input_json_path: &Path,
    input_manifest_path: &Path,
    stage_dir: &Path,
    stage_index: &BTreeMap<String, usize>,
    state: &mut PipelineRunState,
) -> Result<NativeStageRun> {
    let dispatch = dispatch_for_stage(stage, raster_stage)?;
    match dispatch {
        StageDispatch::Native => {
            let kind = StageKind::from_stage_spec(&stage.project, &stage.name)?;
            println!("    procedure-pipeline {} …", kind.native_stage());
            let cached_inputs = cached_inputs_for_stage(
                stage,
                stage_index,
                &state.output_commitments,
                &state.output_cache,
            )?;
            run_native_stage(
                native_stage_backend,
                current_exe,
                &kind,
                &stage.name,
                input_json_path,
                input_manifest_path,
                stage_dir,
                &cached_inputs,
                &state.weights,
            )
        }
        StageDispatch::RasterReference => {
            let kind = StageKind::from_stage_spec(&stage.project, &stage.name)?;
            println!("    raster reference {} …", kind.native_stage());
            let duration = run_raster_stage(
                stage,
                base_dir,
                input_json_path,
                input_manifest_path,
                stage_dir,
            )?;
            println!("    procedure-pipeline parity check …");
            run_compare_stage(current_exe, stage_dir)?;
            state.selected_stage_dir = Some(stage_dir.to_path_buf());
            Ok(NativeStageRun {
                timings: None,
                duration,
                output: None,
            })
        }
    }
}

fn finish_stage(
    idx: usize,
    stage: &StageSpec,
    stage_index: &BTreeMap<String, usize>,
    stage_run: NativeStageRun,
    stage_dir: PathBuf,
    recomputed_list_path: &Path,
    state: &mut PipelineRunState,
) -> Result<()> {
    if let Some(timings) = stage_run.timings {
        state.performance.push(StagePerformance {
            name: stage.name.clone(),
            worker: None,
            timings,
            cache: state.weights.stats(),
        });
    }
    state.execution_times[idx] = Some(stage_run.duration);
    record_stage_program(stage, &stage_dir, stage_index)?;
    let output = collect_output(&stage_dir)?;
    if let Some(value) = stage_run.output {
        if matches!(value, CachedStageValue::DecodeEdge(_)) {
            // A check of a pipeline of several inferences is named for its own.
            let inference = procedure_definition::procedure_kind::labelled_name(&stage.name)
                .map_or(0, |(inference, _)| inference);
            state.last_checks.insert(inference, value.clone());
        }
        state.last_output = Some(value.clone());
        // A stage of a pipeline of several inferences is named for its own.
        state.last_reads = stage
            .inputs
            .iter()
            .filter_map(|(param, binding)| match binding {
                InputBinding::From(producer) => Some((
                    param.clone(),
                    procedure_definition::procedure_kind::labelled_name(producer)
                        .map_or(0, |(inference, _)| inference),
                )),
                _ => None,
            })
            .collect();
        output_cache_insert(
            &mut state.output_cache,
            &stage.name,
            &output.structural_commitment,
            value,
        );
    } else {
        state.last_output = None;
        state.last_reads.clear();
    }
    print_stage_output(&output);
    state.output_commitments[idx] = Some(output.structural_commitment);
    observe_checkpoint(idx, &stage.name, &stage_dir, recomputed_list_path, state)?;
    Ok(())
}

/// Names in the stage directory the program this stage stands for, and where
/// each of its piped inputs came from. The checkpoint commits to both.
///
/// The program is the one the stage's project has locked, which is what a
/// Raster replay of the stage runs; a project with no lock leaves the
/// checkpoint naming no program. A piped input's producer is named by its
/// position in the run, which is its position in the claim's checkpoint list.
fn record_stage_program(
    stage: &StageSpec,
    stage_dir: &Path,
    stage_index: &BTreeMap<String, usize>,
) -> Result<()> {
    let program =
        procedure_definition::checkpoint::locked_program_commitment(Path::new(&stage.project))
            .unwrap_or_default();
    let mut input_sources = BTreeMap::new();
    for (param, binding) in &stage.inputs {
        if let InputBinding::From(producer) = binding {
            let index = stage_index.get(producer).with_context(|| {
                format!(
                    "stage `{}` takes `{param}` from `{producer}`, which the pipeline does not run",
                    stage.name
                )
            })?;
            input_sources.insert(param.clone(), u32::try_from(*index)?);
        }
    }
    procedure_definition::write_stage_program(stage_dir, &program, &input_sources)
}

fn observe_checkpoint(
    idx: usize,
    stage_name: &str,
    stage_dir: &Path,
    recomputed_list_path: &Path,
    state: &mut PipelineRunState,
) -> Result<()> {
    if let Some(verifier) = state.checkpoint_verifier.as_mut() {
        verifier.observe_stage(idx, stage_name, stage_dir, recomputed_list_path)?;
    }
    Ok(())
}

fn output_cache_insert(
    output_cache: &mut StageOutputCache,
    stage_name: &str,
    structural_commitment: &[u8],
    value: CachedStageValue,
) {
    output_cache.insert(stage_name, structural_commitment.to_vec(), value);
}

fn aux_wave_range(stages: &[StageSpec], start: usize) -> Option<Range<usize>> {
    let stage = stages.get(start)?;
    if !is_prefill_prepare_aux_stage(stage) {
        return None;
    }
    let end = stages[start..]
        .iter()
        .position(|stage| !is_prefill_prepare_aux_stage(stage))
        .map(|offset| start + offset)
        .unwrap_or(stages.len());
    (end - start > 1).then_some(start..end)
}

fn is_prefill_prepare_aux_stage(stage: &StageSpec) -> bool {
    StageKind::from_stage_spec(&stage.project, &stage.name)
        .is_ok_and(|kind| kind.auxiliary_layer().is_some())
}

fn run_aux_wave(
    range: Range<usize>,
    stages: &[StageSpec],
    base_dir: &Path,
    pipeline_dir: &Path,
    stage_index: &BTreeMap<String, usize>,
    current_exe: &Path,
    native_stage_backend: NativeStageBackend,
    raster_stage: Option<&str>,
    recomputed_list_path: &Path,
    state: &mut PipelineRunState,
) -> Result<()> {
    println!(
        "▸ stages {}-{}  prefill_prepare_aux   ({} parallel stages)",
        range.start + 1,
        range.end,
        range.end - range.start
    );

    let mut jobs = Vec::new();
    let mut reference_idx = None;
    for idx in range.clone() {
        let stage = &stages[idx];
        ensure_stage_inputs_ready(stage, &state.output_commitments, stage_index)?;
        match dispatch_for_stage(stage, raster_stage)? {
            StageDispatch::RasterReference => {
                if reference_idx.replace(idx).is_some() {
                    bail!("aux wave cannot contain more than one Raster reference stage");
                }
            }
            StageDispatch::Native => jobs.push(prepare_aux_stage_job(
                idx,
                stages,
                base_dir,
                pipeline_dir,
                stage_index,
                state,
            )?),
        }
    }

    if let Some(idx) = reference_idx {
        run_one_stage(
            idx,
            stages,
            base_dir,
            pipeline_dir,
            stage_index,
            current_exe,
            native_stage_backend,
            raster_stage,
            recomputed_list_path,
            state,
        )?;
        if state.has_divergence() {
            return Ok(());
        }
    }

    let batch = if native_stage_backend == NativeStageBackend::InProcess {
        if state.workers.is_none() && !jobs.is_empty() {
            state.workers = Some(WorkerPool::new(
                current_exe,
                state.allocation.worker_count,
                state.allocation.worker_bytes,
            )?);
            state
                .memory
                .add_workers(state.workers.as_ref().unwrap().process_ids());
            state.auxiliary_process_starts += state.workers.as_ref().unwrap().len();
        }
        run_persistent_aux_jobs(state.workers.as_mut(), jobs)?
    } else {
        state.auxiliary_process_starts += jobs.len();
        run_aux_stage_jobs(current_exe, jobs)?
    };
    let stage_duration_sum = batch
        .results
        .iter()
        .map(|result| result.duration.as_nanos())
        .sum();
    if let (Some(first), Some(last)) = (batch.results.first(), batch.results.last()) {
        state.aux_waves.push(AuxWaveExecutionTime {
            name: String::from("prefill_prepare_aux"),
            first_stage: first.name.clone(),
            last_stage: last.name.clone(),
            stage_count: batch.results.len(),
            parallelism: batch.parallelism,
            wall_duration_ns: batch.wall_duration.as_nanos(),
            stage_duration_sum_ns: stage_duration_sum,
        });
    }
    for result in batch.results {
        if let Some(performance) = result.performance {
            state.performance.push(performance);
        }
        state.execution_times[result.idx] = Some(result.duration);
        print_stage_output(&result.output);
        state.output_commitments[result.idx] = Some(result.output.structural_commitment);
        record_stage_program(&stages[result.idx], &result.stage_dir, stage_index)?;
        observe_checkpoint(
            result.idx,
            &result.name,
            &result.stage_dir,
            recomputed_list_path,
            state,
        )?;
        if state.has_divergence() {
            return Ok(());
        }
    }

    for idx in range {
        if state.output_commitments[idx].is_none() {
            bail!(
                "aux wave did not produce output for stage `{}`",
                stages[idx].name
            );
        }
    }
    Ok(())
}

fn prepare_aux_stage_job(
    idx: usize,
    stages: &[StageSpec],
    base_dir: &Path,
    pipeline_dir: &Path,
    stage_index: &BTreeMap<String, usize>,
    state: &PipelineRunState,
) -> Result<AuxStageJob> {
    let stage = &stages[idx];
    println!(
        "  queued stage {}/{}  {}   ({})",
        idx + 1,
        stages.len(),
        stage.name,
        stage.project
    );
    let stage_dir = pipeline_dir.join(&stage.name);
    fs::create_dir_all(&stage_dir)
        .with_context(|| format!("failed to create {}", stage_dir.display()))?;
    let (input_json_path, input_manifest_path) = synthesize_inputs(
        stage,
        &stage_dir,
        base_dir,
        pipeline_dir,
        &state.output_commitments,
        stage_index,
    )?;
    Ok(AuxStageJob {
        idx,
        name: stage.name.clone(),
        kind: StageKind::from_stage_spec(&stage.project, &stage.name)?,
        input_json_path,
        input_manifest_path,
        stage_dir,
    })
}

fn run_persistent_aux_jobs(
    pool: Option<&mut WorkerPool>,
    jobs: Vec<AuxStageJob>,
) -> Result<AuxStageBatch> {
    let started = Instant::now();
    let Some(pool) = pool else {
        return Ok(AuxStageBatch {
            results: Vec::new(),
            parallelism: 0,
            wall_duration: started.elapsed(),
        });
    };
    let worker_count = pool.len();
    let mut metadata = BTreeMap::new();
    let requests = jobs
        .into_iter()
        .map(|job| {
            let Some(layer) = job.kind.auxiliary_layer() else {
                unreachable!()
            };
            metadata.insert(job.idx, (job.stage_dir.clone(), layer % worker_count));
            WorkerJob {
                id: job.idx,
                name: job.name,
                kind: job.kind,
                input: job.input_json_path,
                input_manifest: job.input_manifest_path,
                output_dir: job.stage_dir,
            }
        })
        .collect();
    let responses = pool.run(requests)?;
    let mut results = Vec::new();
    for response in responses {
        let (stage_dir, worker) = metadata
            .remove(&response.id)
            .context("unexpected worker job ID")?;
        let timings = response.timings.context("worker omitted timings")?;
        let output = collect_output(&stage_dir)?;
        results.push(AuxStageResult {
            idx: response.id,
            name: response.name.clone(),
            stage_dir,
            duration: timings.native_stage_duration,
            output,
            performance: Some(StagePerformance {
                name: response.name,
                worker: Some(worker),
                timings,
                cache: response.cache,
            }),
        });
    }
    if !metadata.is_empty() {
        bail!("auxiliary worker omitted job results");
    }
    Ok(AuxStageBatch {
        results,
        parallelism: worker_count,
        wall_duration: started.elapsed(),
    })
}

fn run_aux_stage_jobs(current_exe: &Path, jobs: Vec<AuxStageJob>) -> Result<AuxStageBatch> {
    if jobs.is_empty() {
        return Ok(AuxStageBatch {
            results: Vec::new(),
            wall_duration: Duration::default(),
            parallelism: 0,
        });
    }
    let parallelism = aux_parallelism(jobs.len());
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(parallelism)
        .build()
        .context("failed to build aux stage worker pool")?;
    let started = Instant::now();
    let mut results = pool.install(|| {
        jobs.into_par_iter()
            .map(|job| run_aux_stage_job(current_exe, job))
            .collect::<Result<Vec<_>>>()
    })?;
    let wall_duration = started.elapsed();
    results.sort_by_key(|result| result.idx);
    Ok(AuxStageBatch {
        results,
        wall_duration,
        parallelism,
    })
}

fn run_aux_stage_job(current_exe: &Path, job: AuxStageJob) -> Result<AuxStageResult> {
    let stage_run = run_native_stage_subprocess(
        current_exe,
        &job.kind,
        &job.input_json_path,
        &job.input_manifest_path,
        &job.stage_dir,
    )
    .with_context(|| format!("failed to run parallel aux stage `{}`", job.name))?;
    let output = collect_output(&job.stage_dir)
        .with_context(|| format!("failed to collect parallel aux stage `{}`", job.name))?;
    Ok(AuxStageResult {
        performance: stage_run.timings.map(|timings| StagePerformance {
            name: job.name.clone(),
            worker: None,
            timings,
            cache: Default::default(),
        }),
        idx: job.idx,
        name: job.name,
        stage_dir: job.stage_dir,
        duration: stage_run.duration,
        output,
    })
}

fn ensure_stage_inputs_ready(
    stage: &StageSpec,
    outputs: &[Option<Vec<u8>>],
    stage_index: &BTreeMap<String, usize>,
) -> Result<()> {
    for (param, binding) in &stage.inputs {
        let InputBinding::From(producer) = binding else {
            continue;
        };
        let producer_idx = *stage_index.get(producer).ok_or_else(|| {
            anyhow::anyhow!(
                "stage '{}': parameter '{param}' is fed from unknown stage '{producer}'",
                stage.name
            )
        })?;
        if outputs.get(producer_idx).and_then(Option::as_ref).is_none() {
            bail!(
                "stage '{}': parameter '{param}' is fed from '{producer}', which has not run",
                stage.name
            );
        }
    }
    Ok(())
}

fn aux_parallelism(job_count: usize) -> usize {
    let requested = std::env::var("IWEBS_PIPELINE_AUX_PARALLELISM")
        .ok()
        .and_then(|value| value.parse::<usize>().ok());
    let available = std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(4);
    aux_parallelism_from(job_count, requested, available)
}

fn persistent_aux_parallelism(job_count: usize) -> usize {
    let available = std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(4);
    aux_parallelism(job_count).min(available)
}

fn aux_parallelism_from(
    job_count: usize,
    requested: Option<usize>,
    available_parallelism: usize,
) -> usize {
    if job_count == 0 {
        return 0;
    }
    requested
        .filter(|value| *value > 0)
        .unwrap_or_else(|| available_parallelism.clamp(1, 4))
        .clamp(1, job_count)
}

pub(crate) fn cached_inputs_for_stage(
    stage: &StageSpec,
    stage_index: &BTreeMap<String, usize>,
    outputs: &[Option<Vec<u8>>],
    output_cache: &StageOutputCache,
) -> Result<CachedInputs> {
    let mut cached_inputs = CachedInputs::new();
    for (param, binding) in &stage.inputs {
        let InputBinding::From(producer) = binding else {
            continue;
        };
        let Some(producer_idx) = stage_index.get(producer).copied() else {
            continue;
        };
        let Some(expected_commitment) = outputs.get(producer_idx).and_then(Option::as_ref) else {
            continue;
        };
        if let Some(value) = output_cache.get(producer, expected_commitment)? {
            cached_inputs.insert(param.clone(), value);
        }
    }
    Ok(cached_inputs)
}

pub(crate) fn synthesize_inputs(
    stage: &StageSpec,
    stage_dir: &Path,
    base_dir: &Path,
    pipeline_dir: &Path,
    outputs: &[Option<Vec<u8>>],
    stage_index: &BTreeMap<String, usize>,
) -> Result<(PathBuf, PathBuf)> {
    let mut input_entries: Vec<(String, serde_json::Value)> = Vec::new();
    let mut manifest_entries: Vec<(String, serde_json::Value)> = Vec::new();

    for (param, binding) in &stage.inputs {
        let (path, index_path, commitment) = match binding {
            InputBinding::External(ext) => {
                let path = absolute(base_dir, &ext.path);
                let index_path = ext
                    .index_path
                    .as_ref()
                    .map(|path| absolute(base_dir, path))
                    .unwrap_or_else(|| path.with_extension("rindex"));
                (path, index_path, ext.commitment.clone())
            }
            InputBinding::From(producer) => {
                let producer_idx = *stage_index.get(producer).ok_or_else(|| {
                    anyhow::anyhow!(
                        "stage '{}': parameter '{param}' is fed from stage '{producer}', which has not run",
                        stage.name
                    )
                })?;
                let structural = outputs.get(producer_idx).and_then(Option::as_ref).ok_or_else(|| {
                    anyhow::anyhow!(
                        "stage '{}': parameter '{param}' is fed from stage '{producer}', which has not run",
                        stage.name
                    )
                })?;
                if structural.is_empty() {
                    bail!(
                        "stage '{}': parameter '{param}' is fed from '{producer}', which produced no output",
                        stage.name
                    );
                }
                let producer_dir = pipeline_dir.join(producer);
                (
                    producer_dir.join("output.bin"),
                    producer_dir.join("output.rindex"),
                    hex::encode(structural),
                )
            }
        };

        input_entries.push((
            param.clone(),
            serde_json::json!({
                "path": path.to_string_lossy(),
                "index_path": index_path.to_string_lossy(),
                "load_preference": load_preference_for_binding(binding),
            }),
        ));
        manifest_entries.push((
            param.clone(),
            serde_json::json!({ "type": "sha256", "encoding": "raster", "commitment": commitment }),
        ));
    }

    let input_json_path = stage_dir.join("input.json");
    let input_manifest_path = stage_dir.join("input_manifest.json");
    fs::write(
        &input_json_path,
        serde_json::to_vec_pretty(&serde_json::Value::Object(
            input_entries.into_iter().collect(),
        ))
        .context("failed to serialize input.json")?,
    )
    .with_context(|| format!("failed to write {}", input_json_path.display()))?;
    fs::write(
        &input_manifest_path,
        serde_json::to_vec_pretty(&serde_json::Value::Object(
            manifest_entries.into_iter().collect(),
        ))
        .context("failed to serialize input_manifest.json")?,
    )
    .with_context(|| format!("failed to write {}", input_manifest_path.display()))?;

    Ok((input_json_path, input_manifest_path))
}

fn load_preference_for_binding(binding: &InputBinding) -> &'static str {
    match binding {
        InputBinding::External(_) => "mmap",
        InputBinding::From(_) => "read",
    }
}

fn run_raster_stage(
    stage: &StageSpec,
    base_dir: &Path,
    input_json_path: &Path,
    input_manifest_path: &Path,
    stage_dir: &Path,
) -> Result<Duration> {
    let project_dir = base_dir.join(&stage.project);
    let manifest_path = project_dir.join("Cargo.toml");
    if !manifest_path.is_file() {
        bail!(
            "stage '{}' has no Cargo manifest at {}",
            stage.name,
            manifest_path.display()
        );
    }

    let mut command = Command::new("cargo");
    command
        .current_dir(&project_dir)
        .args(["run", "--release", "--manifest-path", "Cargo.toml", "--"])
        .arg("--input")
        .arg(input_json_path)
        .arg("--input-manifest")
        .arg(input_manifest_path)
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit());
    apply_stage_env(&mut command, stage_dir);

    run_timed_command(command, &format!("Raster stage '{}'", stage.name))
}

fn run_native_stage(
    backend: NativeStageBackend,
    current_exe: &Path,
    kind: &StageKind,
    stage_name: &str,
    input_json_path: &Path,
    input_manifest_path: &Path,
    stage_dir: &Path,
    cached_inputs: &CachedInputs,
    weights: &RunWeights,
) -> Result<NativeStageRun> {
    match backend {
        NativeStageBackend::InProcess => run_native_stage_in_process(
            kind,
            stage_name,
            input_json_path,
            input_manifest_path,
            stage_dir,
            cached_inputs,
            weights,
        ),
        NativeStageBackend::Subprocess => run_native_stage_subprocess(
            current_exe,
            kind,
            input_json_path,
            input_manifest_path,
            stage_dir,
        ),
    }
}

fn run_native_stage_subprocess(
    current_exe: &Path,
    kind: &StageKind,
    input_json_path: &Path,
    input_manifest_path: &Path,
    stage_dir: &Path,
) -> Result<NativeStageRun> {
    let mut command = Command::new(current_exe);
    command
        .arg("--run-stage")
        .arg(stage_dir)
        .arg("--input")
        .arg(input_json_path)
        .arg("--input-manifest")
        .arg(input_manifest_path)
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit());
    apply_stage_env(&mut command, stage_dir);

    let duration = run_timed_command(
        command,
        &format!("procedure-pipeline {} stage", kind.native_stage()),
    )?;
    let timing_path = stage_dir.join("pipeline-timings.json");
    let timings = if timing_path.is_file() {
        Some(serde_json::from_slice(&fs::read(timing_path)?)?)
    } else {
        None
    };
    Ok(NativeStageRun {
        timings,
        duration,
        output: None,
    })
}

fn run_native_stage_in_process(
    kind: &StageKind,
    stage_name: &str,
    input_json_path: &Path,
    input_manifest_path: &Path,
    stage_dir: &Path,
    cached_inputs: &CachedInputs,
    weights: &RunWeights,
) -> Result<NativeStageRun> {
    let _env = StageEnvGuard::apply(stage_dir);
    let started = Instant::now();
    let caches = weights.caches();
    let native_run = segments::run_and_publish_from_paths_with_caches(
        kind,
        input_json_path,
        input_manifest_path,
        cached_inputs,
        caches,
    )?;
    let duration = started.elapsed();
    println!(
        "procedure-pipeline {stage_name}: output {} structural={} stage={}",
        native_run.artifact.data_path.display(),
        native_run.artifact.commitment,
        format_duration(duration),
    );
    Ok(NativeStageRun {
        timings: Some(native_run.timings),
        duration,
        output: Some(native_run.output),
    })
}

fn run_compare_stage(current_exe: &Path, stage_dir: &Path) -> Result<()> {
    let prior_report = parity_dir(stage_dir).join("report.json");
    if let Err(error) = fs::remove_file(&prior_report) {
        if error.kind() != std::io::ErrorKind::NotFound {
            return Err(error)
                .with_context(|| format!("failed to remove {}", prior_report.display()));
        }
    }

    let mut command = Command::new(current_exe);
    command
        .arg("--compare-stage")
        .arg(stage_dir)
        .arg("--input")
        .arg(stage_dir.join("input.json"))
        .arg("--input-manifest")
        .arg(stage_dir.join("input_manifest.json"))
        .env(raster_runtime::auth::AUTH_ENV, "0")
        .env_remove(raster_runtime::TRACE_PATH_ENV)
        .env_remove(raster_runtime::tracing::COMMIT_PATH_ENV)
        .env_remove(raster_runtime::tracing::COMMIT_PROJECT_ENV)
        .env_remove(raster_runtime::tracing::COMMIT_WINDOW_ENV)
        .env_remove(raster_runtime::TRACE_FORMAT_ENV)
        .env_remove(raster_runtime::OUTPUT_DIR_ENV)
        .env_remove(raster_runtime::PROFILE_PATH_ENV)
        .env_remove(raster_runtime::PROFILE_STREAM_PATH_ENV)
        .env_remove(raster_runtime::PROFILE_RUN_ID_ENV)
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit());

    let status = command
        .status()
        .context("failed to start procedure-pipeline comparison child")?;
    if !status.success() {
        bail!(
            "procedure-pipeline parity check for {} failed ({status})",
            stage_dir.display()
        );
    }
    Ok(())
}

fn run_timed_command(mut command: Command, label: &str) -> Result<Duration> {
    let started = Instant::now();
    let status = command
        .status()
        .with_context(|| format!("failed to start {label}"))?;
    let duration = started.elapsed();
    if !status.success() {
        bail!("{label} exited unsuccessfully ({status})");
    }
    Ok(duration)
}

fn collect_output(stage_dir: &Path) -> Result<StageOutput> {
    let output_bin = stage_dir.join("output.bin");
    let bytes = fs::read(&output_bin)
        .with_context(|| format!("failed to read {}", output_bin.display()))?;
    let payload_commitment = Sha256::digest(&bytes).to_vec();
    let structural_commitment = payload_structural_root(&bytes)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "{} is not a well-formed Raster payload",
                output_bin.display()
            )
        })?
        .to_vec();

    let manifest_commitment = read_output_manifest_commitment(stage_dir)?;
    if manifest_commitment != hex::encode(&structural_commitment) {
        bail!(
            "{}: output_manifest commitment {manifest_commitment} disagrees with the recomputed structural root {}",
            stage_dir.display(),
            hex::encode(&structural_commitment)
        );
    }

    Ok(StageOutput {
        payload_commitment,
        structural_commitment,
    })
}

fn read_output_manifest_commitment(stage_dir: &Path) -> Result<String> {
    let path = stage_dir.join("output_manifest.json");
    let text =
        fs::read_to_string(&path).with_context(|| format!("failed to read {}", path.display()))?;
    let doc: serde_json::Value = serde_json::from_str(&text)
        .with_context(|| format!("failed to parse {}", path.display()))?;
    doc.get("output")
        .and_then(|value| value.get("commitment"))
        .and_then(serde_json::Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| anyhow::anyhow!("{} has no output.commitment", path.display()))
}

fn write_execution_times(
    pipeline_dir: &Path,
    stages: &[StageSpec],
    execution_times: &[Option<Duration>],
    total_wall_duration: Duration,
    aux_waves: &[AuxWaveExecutionTime],
) -> Result<()> {
    let path = pipeline_dir.join(EXECUTION_TIMES_JSON);
    if execution_times.len() != stages.len() {
        bail!(
            "execution timing slot count {} does not match stage count {}",
            execution_times.len(),
            stages.len()
        );
    }
    let mut total_exec_duration_ns = 0;
    let mut timing_stages = Vec::with_capacity(stages.len());
    for (idx, (stage, duration)) in stages.iter().zip(execution_times).enumerate() {
        let duration = (*duration).ok_or_else(|| {
            anyhow::anyhow!(
                "missing execution timing for stage {} `{}`",
                idx + 1,
                stage.name
            )
        })?;
        total_exec_duration_ns += duration.as_nanos();
        timing_stages.push(StageExecutionTime {
            name: stage.name.clone(),
            exec_duration_ns: duration.as_nanos(),
        });
    }
    let document = ExecutionTimesDocument {
        version: 2,
        stages: timing_stages,
        total_exec_duration_ns,
        total_wall_duration_ns: total_wall_duration.as_nanos(),
        aux_waves: aux_waves.to_vec(),
    };
    fs::write(
        &path,
        serde_json::to_vec_pretty(&document).context("failed to encode execution-times.json")?,
    )
    .with_context(|| format!("failed to write {}", path.display()))
}

fn print_native_timing_summary(
    stages: &[StageSpec],
    execution_times: &[Option<Duration>],
    total_wall_duration: Duration,
    _aux_waves: &[AuxWaveExecutionTime],
) -> Result<()> {
    let stage_duration_sum = execution_times
        .iter()
        .enumerate()
        .map(|(idx, duration)| {
            duration.ok_or_else(|| {
                anyhow::anyhow!(
                    "missing execution timing for stage {} `{}`",
                    idx + 1,
                    stages[idx].name
                )
            })
        })
        .try_fold(Duration::default(), |sum, duration| {
            duration.map(|duration| sum + duration)
        })?;

    println!("procedure-pipeline timing:");
    println!(
        "  elapsed wall time: {}",
        format_duration(total_wall_duration)
    );
    println!(
        "  stage duration sum: {}",
        format_duration(stage_duration_sum)
    );
    Ok(())
}

fn create_pipeline_dir(base_dir: &Path) -> Result<PathBuf> {
    let root = base_dir
        .join("target")
        .join("procedure-pipeline")
        .join("runs");
    fs::create_dir_all(&root).with_context(|| format!("failed to create {}", root.display()))?;
    let pipeline_dir = root.join(pipeline_run_id());
    fs::create_dir_all(&pipeline_dir)
        .with_context(|| format!("failed to create {}", pipeline_dir.display()))?;
    Ok(pipeline_dir)
}

fn pipeline_run_id() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("{nanos:020}-pid{}", std::process::id())
}

fn pipeline_run_id_label(pipeline_dir: &Path) -> String {
    pipeline_dir
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("?")
        .to_string()
}

fn absolute(base_dir: &Path, path: &str) -> PathBuf {
    let path = Path::new(path);
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        base_dir.join(path)
    }
}

fn apply_stage_env(command: &mut Command, stage_dir: &Path) {
    command
        .env(raster_runtime::auth::AUTH_ENV, "0")
        .env(raster_runtime::OUTPUT_DIR_ENV, stage_dir)
        .env_remove(raster_runtime::TRACE_PATH_ENV)
        .env_remove(raster_runtime::tracing::COMMIT_PATH_ENV)
        .env_remove(raster_runtime::tracing::COMMIT_PROJECT_ENV)
        .env_remove(raster_runtime::tracing::COMMIT_WINDOW_ENV)
        .env_remove(raster_runtime::TRACE_FORMAT_ENV)
        .env_remove(raster_runtime::PROFILE_PATH_ENV)
        .env_remove(raster_runtime::PROFILE_STREAM_PATH_ENV)
        .env_remove(raster_runtime::PROFILE_RUN_ID_ENV);
}

pub(crate) struct StageEnvGuard {
    saved: Vec<(&'static str, Option<OsString>)>,
}

impl StageEnvGuard {
    pub(crate) fn apply(stage_dir: &Path) -> Self {
        let guard = Self::capture(&[
            raster_runtime::auth::AUTH_ENV,
            raster_runtime::OUTPUT_DIR_ENV,
            raster_runtime::TRACE_PATH_ENV,
            raster_runtime::tracing::COMMIT_PATH_ENV,
            raster_runtime::tracing::COMMIT_PROJECT_ENV,
            raster_runtime::tracing::COMMIT_WINDOW_ENV,
            raster_runtime::TRACE_FORMAT_ENV,
            raster_runtime::PROFILE_PATH_ENV,
            raster_runtime::PROFILE_STREAM_PATH_ENV,
            raster_runtime::PROFILE_RUN_ID_ENV,
        ]);
        std::env::set_var(raster_runtime::auth::AUTH_ENV, "0");
        std::env::set_var(raster_runtime::OUTPUT_DIR_ENV, stage_dir);
        for name in [
            raster_runtime::TRACE_PATH_ENV,
            raster_runtime::tracing::COMMIT_PATH_ENV,
            raster_runtime::tracing::COMMIT_PROJECT_ENV,
            raster_runtime::tracing::COMMIT_WINDOW_ENV,
            raster_runtime::TRACE_FORMAT_ENV,
            raster_runtime::PROFILE_PATH_ENV,
            raster_runtime::PROFILE_STREAM_PATH_ENV,
            raster_runtime::PROFILE_RUN_ID_ENV,
        ] {
            std::env::remove_var(name);
        }
        guard
    }

    fn capture(names: &[&'static str]) -> Self {
        Self {
            saved: names
                .iter()
                .map(|name| (*name, std::env::var_os(name)))
                .collect(),
        }
    }
}

impl Drop for StageEnvGuard {
    fn drop(&mut self) {
        for (name, value) in self.saved.drain(..).rev() {
            match value {
                Some(value) => std::env::set_var(name, value),
                None => std::env::remove_var(name),
            }
        }
    }
}

fn short_hex(bytes: &[u8]) -> String {
    let full = hex::encode(bytes);
    full.chars().take(12).collect::<String>() + "..."
}

fn print_stage_output(output: &StageOutput) {
    println!(
        "    output.bin  payload={}  structural={}",
        short_hex(&output.payload_commitment),
        short_hex(&output.structural_commitment)
    );
    println!();
}

fn format_duration(duration: Duration) -> String {
    format_duration_ns(duration.as_nanos())
}

fn format_duration_ns(ns: u128) -> String {
    if ns < 1_000 {
        format!("{ns}ns")
    } else if ns < 1_000_000 {
        format!("{:.2}µs", ns as f64 / 1_000.0)
    } else if ns < 1_000_000_000 {
        format!("{:.2}ms", ns as f64 / 1_000_000.0)
    } else {
        format!("{:.2}s", ns as f64 / 1_000_000_000.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static NEXT_TEMP_DIR: AtomicUsize = AtomicUsize::new(0);

    fn test_base_dir() -> PathBuf {
        let id = NEXT_TEMP_DIR.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "procedure-pipeline-pipeline-runner-test-{}-{id}",
            std::process::id()
        ))
    }

    fn repo_root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(Path::parent)
            .unwrap()
            .to_path_buf()
    }

    fn fixture_manifest() -> Manifest {
        read_manifest(&repo_root().join("procedure/pipeline/testdata/pipeline.toml")).unwrap()
    }

    #[test]
    fn fixture_dispatches_one_reference_and_remaining_stages_native() {
        let manifest = fixture_manifest();
        let mut native_run = 0usize;
        let mut reference = 0usize;

        for stage in &manifest.pipeline.stage {
            match dispatch_for_stage(stage, Some("range_p1_l2")).unwrap() {
                StageDispatch::Native => native_run += 1,
                StageDispatch::RasterReference => reference += 1,
            }
        }

        assert_eq!(manifest.pipeline.stage.len(), 42);
        assert_eq!(native_run, 41);
        assert_eq!(reference, 1);
    }

    #[test]
    fn stage_prefix_resolves_exact_expanded_instance_and_bounds_aux_waves() {
        let manifest = fixture_manifest();
        for name in ["prompt_check_p1", "range_p1_l2", "aux_c0_l0"] {
            let mut stages = manifest.pipeline.stage.clone();
            let selected = stages.iter().position(|s| s.name == name).unwrap();
            truncate_through_stage(&mut stages, name).unwrap();
            assert_eq!(stages.len(), selected + 1);
            assert_eq!(stages.last().unwrap().name, name);
            if name == "aux_c0_l0" {
                let first = stages
                    .iter()
                    .position(is_prefill_prepare_aux_stage)
                    .unwrap();
                assert_eq!(aux_wave_range(&stages, first), Some(first..selected + 1));
            }
        }
        let mut stages = manifest.pipeline.stage;
        assert!(truncate_through_stage(&mut stages, "prefill-range").is_err());
        assert!(truncate_through_stage(&mut stages, "prompt_check_p9").is_err());
        stages.push(stages[0].clone());
        assert!(truncate_through_stage(&mut stages, "prompt_bind_p0").is_err());
    }

    #[test]
    fn prefix_timing_report_contains_only_executed_stages() {
        let mut manifest = fixture_manifest();
        truncate_through_stage(&mut manifest.pipeline.stage, "embed_p0").unwrap();
        let base = test_base_dir();
        fs::create_dir_all(&base).unwrap();
        let times = vec![Some(Duration::from_nanos(7)); manifest.pipeline.stage.len()];
        write_execution_times(
            &base,
            &manifest.pipeline.stage,
            &times,
            Duration::from_nanos(30),
            &[],
        )
        .unwrap();
        let doc: serde_json::Value =
            serde_json::from_slice(&fs::read(base.join(EXECUTION_TIMES_JSON)).unwrap()).unwrap();
        let names: Vec<_> = doc["stages"]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| s["name"].as_str().unwrap())
            .collect();
        assert_eq!(names, ["prompt_bind_p0", "prompt_check_p0", "embed_p0"]);
        assert_eq!(doc["total_exec_duration_ns"], 21);
        fs::remove_dir_all(base).unwrap();
    }

    #[test]
    fn fixture_dispatches_every_stage_native_without_reference() {
        let manifest = fixture_manifest();
        for stage in &manifest.pipeline.stage {
            assert!(matches!(
                dispatch_for_stage(stage, None).unwrap(),
                StageDispatch::Native
            ));
        }
    }

    #[test]
    fn fixture_expands_ranges_and_the_claimed_export() {
        let manifest = fixture_manifest();
        let index = build_stage_index(&manifest.pipeline.stage).unwrap();

        assert!(index.contains_key("prompt_check_p1"));
        assert!(!index.contains_key("prompt_check_p2"));
        // The pipeline starts at the first range's bind: no stage comes before
        // it. That range's bind and check are given the empty prompt state,
        // a committed constant, where each later one is given the range
        // before; a check is always given its own bind.
        assert_eq!(manifest.pipeline.stage[0].name, "prompt_bind_p0");
        let previous =
            |stage: &str| manifest.pipeline.stage[index[stage]].inputs["previous"].clone();
        let empty_prompt = previous("prompt_bind_p0");
        assert!(matches!(&empty_prompt, InputBinding::External(file)
            if file.path.ends_with("empty_prompt.rastered")));
        assert_eq!(previous("prompt_check_p0"), empty_prompt);
        assert_eq!(
            previous("prompt_bind_p1"),
            InputBinding::From("prompt_bind_p0".into())
        );
        assert_eq!(
            previous("prompt_check_p1"),
            InputBinding::From("prompt_check_p0".into())
        );
        for range in 0..2 {
            let checked = &manifest.pipeline.stage[index[&format!("prompt_check_p{range}")]];
            assert_eq!(
                checked.inputs["bound"],
                InputBinding::From(format!("prompt_bind_p{range}"))
            );
        }
        // The same for the sequence a range's embedding and layers continue,
        // and for the verdict a check continues.
        let empty_sequence = previous("embed_p0");
        assert!(matches!(&empty_sequence, InputBinding::External(file)
            if file.path.ends_with("empty_sequence.rastered")));
        assert_eq!(previous("embed_p1"), InputBinding::From("embed_p0".into()));
        for layer in 0..4 {
            let first = &manifest.pipeline.stage[index[&format!("range_p0_l{layer}")]];
            assert_eq!(first.inputs["prior_kv"], empty_sequence);
            let second = &manifest.pipeline.stage[index[&format!("range_p1_l{layer}")]];
            assert_eq!(
                second.inputs["prior_kv"],
                InputBinding::From(format!("range_p0_l{layer}"))
            );
        }
        let check = |range: usize| &manifest.pipeline.stage[index[&format!("check_c{range}")]];
        assert!(
            matches!(&check(0).inputs["prior"], InputBinding::External(file)
            if file.path.ends_with("empty_edge.rastered"))
        );
        assert_eq!(
            check(0).inputs["previous"],
            InputBinding::From("range_p1_l3".into())
        );
        assert_eq!(
            check(1).inputs["prior"],
            InputBinding::From("check_c0".into())
        );
        assert_eq!(
            check(1).inputs["previous"],
            InputBinding::From("range_c0_l3".into())
        );

        assert!(index.contains_key("embed_c1"));
        assert!(index.contains_key("aux_c1_l3"));
        assert!(index.contains_key("range_c1_l3"));
        let range = &manifest.pipeline.stage[index["range_c1_l0"]];
        assert_eq!(
            range.inputs["prior_kv"],
            InputBinding::From("range_c0_l0".into())
        );
        let first = &manifest.pipeline.stage[index["range_c0_l0"]];
        assert_eq!(
            first.inputs["prior_kv"],
            InputBinding::From("range_p1_l0".into())
        );

        // The pipeline ends at the last claimed range's check. Every check is
        // given the decoder inputs its output record is stated from.
        assert_eq!(manifest.pipeline.stage.last().unwrap().name, "check_c1");
        for range in 0..2 {
            for input in ["decoder", "descriptor", "witness"] {
                assert!(
                    matches!(check(range).inputs[input], InputBinding::External(_)),
                    "check_c{range} {input}"
                );
            }
        }
    }

    #[test]
    fn a_repeat_with_no_iterations_exports_its_entry() {
        let base = test_base_dir();
        fs::create_dir_all(&base).unwrap();
        let path = base.join("Raster.toml");
        fs::write(
            &path,
            r#"[pipeline]
name = "empty-repeat"

[[pipeline.stage]]
name = "check_c0"
project = "sequence-check"

[[pipeline.repeat]]
name = "claimed"
index = "k"
start = 1
count = 0

  [pipeline.repeat.exports.edge]
stage = "check_c{k}"
entry = "check_c0"

  [[pipeline.repeat.stage]]
name = "check_c{k}"
project = "sequence-check"

[[pipeline.stage]]
name = "embed_c9"
project = "input-embedding"
inputs.edge = { from = "claimed.edge" }
"#,
        )
        .unwrap();
        let manifest = read_manifest(&path).unwrap();
        let index = build_stage_index(&manifest.pipeline.stage).unwrap();

        // A repeat that does not run exports its entry: a stage bound to the
        // export reads the stage before the repeat. The IWEBS pipeline has no
        // such binding now; the mechanism is the manifest format's.
        assert!(!index.contains_key("check_c1"));
        let output = manifest.pipeline.stage.last().unwrap();
        assert_eq!(
            output.inputs.get("edge"),
            Some(&InputBinding::From(String::from("check_c0")))
        );
        fs::remove_dir_all(base).unwrap();
    }

    #[test]
    fn synthesized_inputs_cover_external_and_piped_bindings() {
        let base = test_base_dir();
        let pipeline_dir = base.join("run");
        let producer_dir = pipeline_dir.join("producer");
        let stage_dir = pipeline_dir.join("consumer");
        fs::create_dir_all(&producer_dir).unwrap();
        fs::create_dir_all(&stage_dir).unwrap();

        let stage = StageSpec {
            name: "consumer".into(),
            project: "consumer-project".into(),
            inputs: BTreeMap::from([
                (
                    "external_arg".into(),
                    InputBinding::External(ExternalRef {
                        path: "external/value.rastered".into(),
                        index_path: None,
                        commitment: "abc123".into(),
                    }),
                ),
                ("piped_arg".into(), InputBinding::From("producer".into())),
            ]),
        };
        let stage_index = BTreeMap::from([("producer".into(), 0usize)]);
        let outputs = vec![Some(vec![0xde, 0xad, 0xbe, 0xef])];

        let (input_json, input_manifest) = synthesize_inputs(
            &stage,
            &stage_dir,
            &base,
            &pipeline_dir,
            &outputs,
            &stage_index,
        )
        .unwrap();
        let input: serde_json::Value =
            serde_json::from_slice(&fs::read(input_json).unwrap()).unwrap();
        let manifest: serde_json::Value =
            serde_json::from_slice(&fs::read(input_manifest).unwrap()).unwrap();

        assert_eq!(
            input["external_arg"]["path"].as_str(),
            Some(
                base.join("external/value.rastered")
                    .to_string_lossy()
                    .as_ref()
            )
        );
        assert_eq!(
            input["external_arg"]["index_path"].as_str(),
            Some(
                base.join("external/value.rindex")
                    .to_string_lossy()
                    .as_ref()
            )
        );
        assert_eq!(
            input["piped_arg"]["path"].as_str(),
            Some(producer_dir.join("output.bin").to_string_lossy().as_ref())
        );
        assert_eq!(input["external_arg"]["load_preference"], "mmap");
        assert_eq!(input["piped_arg"]["load_preference"], "read");
        assert_eq!(manifest["external_arg"]["commitment"], "abc123");
        assert_eq!(manifest["piped_arg"]["commitment"], "deadbeef");

        fs::remove_dir_all(base).unwrap();
    }

    #[test]
    fn cached_inputs_follow_from_bindings_by_commitment() {
        let stage = StageSpec {
            name: "consumer".into(),
            project: "input-embedding".into(),
            inputs: BTreeMap::from([("prompt".into(), InputBinding::From("producer".into()))]),
        };
        let stage_index = BTreeMap::from([("producer".into(), 0usize)]);
        let outputs = vec![Some(vec![0xaa, 0xbb])];
        let mut output_cache = StageOutputCache::default();
        output_cache.insert(
            "producer",
            vec![0xaa, 0xbb],
            CachedStageValue::PromptTokenization(prompt_check::input::PromptTokenization {
                token_ids: raster::List::from(vec![1, 2]),
                ..Default::default()
            }),
        );

        let inputs =
            cached_inputs_for_stage(&stage, &stage_index, &outputs, &output_cache).unwrap();

        assert!(inputs.contains_key("prompt"));
    }

    #[test]
    fn fixture_identifies_prefill_prepare_aux_wave() {
        let manifest = fixture_manifest();
        let first = manifest
            .pipeline
            .stage
            .iter()
            .position(is_prefill_prepare_aux_stage)
            .unwrap();

        let range = aux_wave_range(&manifest.pipeline.stage, first).unwrap();

        // Each layer's PLE runs over every range, then its transformer does.
        assert_eq!(range, first..first + 4);
        assert!(aux_wave_range(&manifest.pipeline.stage, 0).is_none());
        assert!(aux_wave_range(&manifest.pipeline.stage, first + 4).is_none());
        for stage in &manifest.pipeline.stage[range] {
            assert!(is_prefill_prepare_aux_stage(stage));
        }
        assert_eq!(
            manifest.pipeline.stage[first].inputs.get("embedded"),
            Some(&InputBinding::From(String::from("embed_p0")))
        );
    }

    #[test]
    fn raster_reference_inside_aux_wave_dispatches_once() {
        let manifest = fixture_manifest();
        let first = manifest
            .pipeline
            .stage
            .iter()
            .position(is_prefill_prepare_aux_stage)
            .unwrap();
        let range = aux_wave_range(&manifest.pipeline.stage, first).unwrap();
        let mut native_run = 0usize;
        let mut reference = Vec::new();

        for stage in &manifest.pipeline.stage[range] {
            match dispatch_for_stage(stage, Some("aux_p1_l0")).unwrap() {
                StageDispatch::Native => native_run += 1,
                StageDispatch::RasterReference => reference.push(stage.name.as_str()),
            }
        }

        assert_eq!(native_run, 3);
        assert_eq!(reference, ["aux_p1_l0"]);
    }

    #[test]
    fn checkpoint_verifier_detects_hash_mismatch() {
        let base = test_base_dir();
        let stage_dir = write_checkpoint_stage(&base, "stage_a", "commit-a", b"output-a");
        let list_path = base.join(CHECKPOINT_LIST_JSON);
        let expected = read_checkpoint_from_stage_dir("stage_a", &stage_dir).unwrap();
        let expected_hash = checkpoint_hash(&expected).unwrap();
        let claimed_hash = corrupt_hash(&expected_hash);
        let mut verifier = CheckpointVerifierState::new(CheckpointHashVerifier {
            claimed_hashes_path: base.join("checkpoints.txt"),
            claimed_hashes: vec![claimed_hash.clone()],
            claimed_sources: None,
        });

        assert!(verifier
            .observe_stage(0, "stage_a", &stage_dir, &list_path)
            .unwrap());

        let list = verifier.list();
        let divergence = verifier.divergence.unwrap();
        assert_eq!(divergence.checkpoint_index, 0);
        assert_eq!(divergence.stage, "stage_a");
        assert_eq!(divergence.reason, DivergenceReason::CheckpointHash);
        assert_eq!(divergence.claimed_hash, Some(claimed_hash));
        assert_eq!(divergence.recomputed_hash, Some(expected_hash));
        assert_eq!(list.checkpoints, vec![expected]);

        fs::remove_dir_all(base).unwrap();
    }

    #[test]
    fn checkpoint_verifier_detects_missing_claimed_hash_immediately() {
        let base = test_base_dir();
        let stage_dir = write_checkpoint_stage(&base, "stage_a", "commit-a", b"output-a");
        let list_path = base.join(CHECKPOINT_LIST_JSON);
        let mut verifier = CheckpointVerifierState::new(CheckpointHashVerifier {
            claimed_hashes_path: base.join("checkpoints.txt"),
            claimed_hashes: Vec::new(),
            claimed_sources: None,
        });

        assert!(verifier
            .observe_stage(0, "stage_a", &stage_dir, &list_path)
            .unwrap());

        let divergence = verifier.divergence.unwrap();
        assert_eq!(divergence.checkpoint_index, 0);
        assert_eq!(divergence.reason, DivergenceReason::MissingClaimedHash);
        assert_eq!(divergence.claimed_hash, None);
        assert!(divergence.recomputed_hash.is_some());

        fs::remove_dir_all(base).unwrap();
    }

    #[test]
    fn checkpoint_verifier_detects_missing_recomputed_hash_at_end() {
        let base = test_base_dir();
        let stage_dir = write_checkpoint_stage(&base, "stage_a", "commit-a", b"output-a");
        let list_path = base.join(CHECKPOINT_LIST_JSON);
        let expected = read_checkpoint_from_stage_dir("stage_a", &stage_dir).unwrap();
        let expected_hash = checkpoint_hash(&expected).unwrap();
        let extra_hash =
            String::from("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
        let mut verifier = CheckpointVerifierState::new(CheckpointHashVerifier {
            claimed_hashes_path: base.join("checkpoints.txt"),
            claimed_hashes: vec![expected_hash, extra_hash.clone()],
            claimed_sources: None,
        });

        assert!(!verifier
            .observe_stage(0, "stage_a", &stage_dir, &list_path)
            .unwrap());
        verifier.finish(&list_path);

        let divergence = verifier.divergence.unwrap();
        assert_eq!(divergence.checkpoint_index, 1);
        assert_eq!(divergence.stage, "<checkpoint-1>");
        assert_eq!(divergence.reason, DivergenceReason::MissingRecomputedHash);
        assert_eq!(divergence.claimed_hash, Some(extra_hash));
        assert_eq!(divergence.recomputed_hash, None);

        fs::remove_dir_all(base).unwrap();
    }

    #[test]
    fn synthesized_inputs_require_manifest_indexed_producer_output() {
        let base = test_base_dir();
        let pipeline_dir = base.join("run");
        let stage_dir = pipeline_dir.join("consumer");
        fs::create_dir_all(&stage_dir).unwrap();

        let stage = StageSpec {
            name: "consumer".into(),
            project: "input-embedding".into(),
            inputs: BTreeMap::from([("prompt".into(), InputBinding::From("producer".into()))]),
        };
        let stage_index = BTreeMap::from([("producer".into(), 3usize)]);
        let outputs = vec![None, None, None, Some(vec![0xaa, 0xbb])];

        let (_input_json, input_manifest) = synthesize_inputs(
            &stage,
            &stage_dir,
            &base,
            &pipeline_dir,
            &outputs,
            &stage_index,
        )
        .unwrap();
        let manifest: serde_json::Value =
            serde_json::from_slice(&fs::read(input_manifest).unwrap()).unwrap();

        assert_eq!(manifest["prompt"]["commitment"], "aabb");

        fs::remove_dir_all(base).unwrap();
    }

    #[test]
    fn synthesized_inputs_reject_missing_producer_output() {
        let base = test_base_dir();
        let pipeline_dir = base.join("run");
        let stage_dir = pipeline_dir.join("consumer");
        fs::create_dir_all(&stage_dir).unwrap();

        let stage = StageSpec {
            name: "consumer".into(),
            project: "input-embedding".into(),
            inputs: BTreeMap::from([("prompt".into(), InputBinding::From("producer".into()))]),
        };
        let stage_index = BTreeMap::from([("producer".into(), 0usize)]);
        let outputs = vec![None];

        let error = synthesize_inputs(
            &stage,
            &stage_dir,
            &base,
            &pipeline_dir,
            &outputs,
            &stage_index,
        )
        .unwrap_err();

        assert!(error.to_string().contains("has not run"));

        fs::remove_dir_all(base).unwrap();
    }

    #[test]
    fn aux_parallelism_is_bounded_and_overridable() {
        assert_eq!(aux_parallelism_from(0, None, 8), 0);
        assert_eq!(aux_parallelism_from(35, None, 16), 4);
        assert_eq!(aux_parallelism_from(2, None, 16), 2);
        assert_eq!(aux_parallelism_from(35, Some(6), 16), 6);
        assert_eq!(aux_parallelism_from(35, Some(0), 16), 4);
        // The diagnostic subprocess backend retains its explicit override behavior.
        assert_eq!(aux_parallelism_from(35, Some(16), 2), 16);
    }

    fn write_checkpoint_stage(
        base: &Path,
        stage_name: &str,
        commitment: &str,
        output: &[u8],
    ) -> PathBuf {
        let stage_dir = base.join(stage_name);
        fs::create_dir_all(&stage_dir).unwrap();
        fs::write(stage_dir.join("input_manifest.json"), b"input-manifest").unwrap();
        fs::write(stage_dir.join("output.bin"), output).unwrap();
        fs::write(
            stage_dir.join("output_manifest.json"),
            format!(
                r#"{{"output":{{"type":"sha256","encoding":"raster","commitment":"{commitment}"}}}}"#
            ),
        )
        .unwrap();
        stage_dir
    }

    fn corrupt_hash(hash: &str) -> String {
        let mut bytes = hash.as_bytes().to_vec();
        bytes[0] = if bytes[0] == b'0' { b'1' } else { b'0' };
        String::from_utf8(bytes).unwrap()
    }
}

/// Resolve an expanded stage without executing inference.
pub fn stage_project(manifest_path: &Path, stage: &str) -> Result<PathBuf> {
    let manifest = read_manifest(manifest_path)?;
    let selected = manifest
        .pipeline
        .stage
        .iter()
        .find(|s| s.name == stage)
        .context("stage is absent from frozen manifest")?;
    Ok(absolute(&std::env::current_dir()?, &selected.project))
}
