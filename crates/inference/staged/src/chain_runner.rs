use std::collections::BTreeMap;
use std::ffi::OsString;
use std::fs;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{bail, Context, Result};
use inference_artifacts::{
    checkpoint_hash, read_checkpoint_from_stage_dir, Checkpoint, CheckpointTrace, Divergence,
    DivergenceReason, CHECKPOINT_TRACE_JSON,
};
use raster_core::input::payload_structural_root;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::cache::{CachedInputs, CachedStageValue, StageOutputCache};
use crate::parity::{parity_dir, EXECUTION_TIMES_JSON};
use crate::routines::{self, StageKind};
use crate::weights::{RunWeights, WeightAllocation};
use crate::worker::{WorkerJob, WorkerPool};

#[derive(Debug)]
pub struct HybridRun {
    pub chain_dir: PathBuf,
    pub selected_stage_dir: Option<PathBuf>,
    pub final_output: Option<CachedStageValue>,
}

/// A native prefix ending at one exact expanded chain stage.
#[derive(Debug, Clone)]
pub struct StagePrefixRun {
    pub chain_dir: PathBuf,
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
}

#[derive(Debug)]
pub struct CheckpointVerifiedRun {
    pub run: HybridRun,
    pub recomputed_trace: CheckpointTrace,
    pub divergence: Option<Divergence>,
}

struct ChainRunOutput {
    run: HybridRun,
    recomputed_trace: Option<CheckpointTrace>,
    divergence: Option<Divergence>,
}

#[derive(Debug)]
pub(crate) struct Manifest {
    pub(crate) chain: ChainSpec,
}

#[derive(Debug, Clone)]
pub(crate) struct ChainSpec {
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
    chain: ChainTable,
}

#[derive(Debug, Deserialize)]
struct ChainTable {
    #[serde(default, rename = "input")]
    inputs: BTreeMap<String, InputDecl>,
    #[serde(default, rename = "stage")]
    stages: Vec<toml::Spanned<StageSpec>>,
    #[serde(default, rename = "repeat")]
    repeats: Vec<toml::Spanned<RepeatSpec>>,
}

#[derive(Debug)]
enum ChainItem {
    Stage(StageSpec),
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
    StagedNative,
    RasterReference,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StagedExecutionBackend {
    InProcess,
    Subprocess,
}

#[derive(Debug)]
struct StageOutput {
    payload_commitment: Vec<u8>,
    structural_commitment: Vec<u8>,
}

struct DirectStageRun {
    timings: Option<routines::DirectTimings>,
    duration: Duration,
    output: Option<CachedStageValue>,
}

#[derive(Debug, Serialize)]
struct StagePerformance {
    name: String,
    worker: Option<usize>,
    timings: routines::DirectTimings,
    cache: inference_kernels::weight_budget::WeightCacheStats,
}

struct ChainRunState {
    auxiliary_process_starts: usize,
    memory: crate::weights::MemorySampler,
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

impl ChainRunState {
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
    recomputed_checkpoints: Vec<Checkpoint>,
    divergence: Option<Divergence>,
}

impl CheckpointVerifierState {
    fn new(verifier: CheckpointHashVerifier) -> Self {
        Self {
            claimed_hashes_path: verifier.claimed_hashes_path,
            claimed_hashes: verifier.claimed_hashes,
            recomputed_checkpoints: Vec::new(),
            divergence: None,
        }
    }

    fn observe_stage(
        &mut self,
        checkpoint_index: usize,
        stage_name: &str,
        stage_dir: &Path,
        recomputed_trace_path: &Path,
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

        let checkpoint = read_checkpoint_from_stage_dir(stage_name, stage_dir)?;
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
                version: 1,
                checkpoint_index,
                stage: stage_name.to_string(),
                reason,
                claimed_trace_path: None,
                claimed_checkpoint_hashes_path: Some(self.claimed_hashes_path.clone()),
                recomputed_trace_path: recomputed_trace_path.to_path_buf(),
                claimed_hash,
                recomputed_hash: Some(recomputed_hash),
                claimed: None,
                recomputed: Some(checkpoint),
            });
        }
        Ok(self.divergence.is_some())
    }

    fn finish(&mut self, recomputed_trace_path: &Path) {
        if self.divergence.is_some() {
            return;
        }
        let checkpoint_index = self.recomputed_checkpoints.len();
        if let Some(claimed_hash) = self.claimed_hashes.get(checkpoint_index).cloned() {
            self.divergence = Some(Divergence {
                version: 1,
                checkpoint_index,
                stage: format!("<checkpoint-{checkpoint_index}>"),
                reason: DivergenceReason::MissingRecomputedHash,
                claimed_trace_path: None,
                claimed_checkpoint_hashes_path: Some(self.claimed_hashes_path.clone()),
                recomputed_trace_path: recomputed_trace_path.to_path_buf(),
                claimed_hash: Some(claimed_hash),
                recomputed_hash: None,
                claimed: None,
                recomputed: None,
            });
        }
    }

    fn trace(&self) -> CheckpointTrace {
        CheckpointTrace {
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
    staged_backend: StagedExecutionBackend,
) -> Result<HybridRun> {
    Ok(run_inner(
        manifest_path,
        raster_stage,
        current_exe,
        staged_backend,
        None,
        None,
    )?
    .run)
}

pub fn run_until_hash_divergence(
    manifest_path: &Path,
    current_exe: &Path,
    staged_backend: StagedExecutionBackend,
    verifier: CheckpointHashVerifier,
) -> Result<CheckpointVerifiedRun> {
    let output = run_inner(
        manifest_path,
        None,
        current_exe,
        staged_backend,
        Some(verifier),
        None,
    )?;
    Ok(CheckpointVerifiedRun {
        run: output.run,
        recomputed_trace: output
            .recomputed_trace
            .expect("verified run always records recomputed checkpoints"),
        divergence: output.divergence,
    })
}

pub fn run_through_stage(
    manifest_path: &Path,
    stage: &str,
    current_exe: &Path,
    staged_backend: StagedExecutionBackend,
) -> Result<StagePrefixRun> {
    let manifest = read_manifest(manifest_path)?;
    validate_reference_stage(&manifest.chain.stage, stage)?;
    let selected = manifest
        .chain
        .stage
        .iter()
        .find(|s| s.name == stage)
        .unwrap();
    let project_dir = absolute(&std::env::current_dir()?, &selected.project);
    let run = run_inner(
        manifest_path,
        None,
        current_exe,
        staged_backend,
        None,
        Some(stage),
    )?
    .run;
    let stage_dir = run.chain_dir.join(stage);
    Ok(StagePrefixRun {
        chain_dir: run.chain_dir,
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
    staged_backend: StagedExecutionBackend,
    checkpoint_verifier: Option<CheckpointHashVerifier>,
    through_stage: Option<&str>,
) -> Result<ChainRunOutput> {
    let ple_loading = crate::ple::Loading::from_env()?;
    let base_dir = std::env::current_dir().context("failed to read current directory")?;
    let mut manifest = read_manifest(manifest_path)?;
    validate_supported_stages(&manifest.chain.stage)?;
    // Slice the expanded chain before scheduling. This also bounds aux waves,
    // output slots and timing records without scheduling any later stages.
    build_stage_index(&manifest.chain.stage)?;
    if let Some(stage) = through_stage {
        truncate_through_stage(&mut manifest.chain.stage, stage)?;
    }
    if let Some(raster_stage) = raster_stage {
        validate_reference_stage(&manifest.chain.stage, raster_stage)?;
    }

    let chain_dir = create_chain_dir(&base_dir)?;
    println!(
        "staged-infer chain run  {}  ({} stages)",
        chain_run_id_label(&chain_dir),
        manifest.chain.stage.len()
    );
    println!("  dir: {}", chain_dir.display());
    match raster_stage {
        Some(stage) => {
            println!("  mode: unauthenticated chain runner (--no-auth; Raster reference: {stage})");
        }
        None => println!("  mode: unauthenticated staged-infer (--no-auth)"),
    }
    println!();

    let chain_started = Instant::now();
    let stage_index = build_stage_index(&manifest.chain.stage)?;
    let largest_wave = (0..manifest.chain.stage.len())
        .filter_map(|i| aux_wave_range(&manifest.chain.stage, i).map(|r| r.len()))
        .max()
        .unwrap_or(0);
    let allocation = if staged_backend == StagedExecutionBackend::InProcess {
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
    let mut state = ChainRunState {
        auxiliary_process_starts: 0,
        memory: crate::weights::MemorySampler::start(),
        weights: RunWeights::new(allocation.main_bytes),
        allocation,
        workers: None,
        performance: Vec::new(),
        output_commitments: vec![None; manifest.chain.stage.len()],
        execution_times: vec![None; manifest.chain.stage.len()],
        aux_waves: Vec::new(),
        output_cache: StageOutputCache::default(),
        selected_stage_dir: None,
        last_output: None,
        checkpoint_verifier: checkpoint_verifier.map(CheckpointVerifierState::new),
    };
    let recomputed_trace_path = chain_dir.join(CHECKPOINT_TRACE_JSON);

    let mut idx = 0;
    while idx < manifest.chain.stage.len() {
        if let Some(aux_range) = aux_wave_range(&manifest.chain.stage, idx) {
            run_aux_wave(
                aux_range.clone(),
                &manifest.chain.stage,
                &base_dir,
                &chain_dir,
                &stage_index,
                current_exe,
                staged_backend,
                raster_stage,
                &recomputed_trace_path,
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
            &manifest.chain.stage,
            &base_dir,
            &chain_dir,
            &stage_index,
            current_exe,
            staged_backend,
            raster_stage,
            &recomputed_trace_path,
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
        chain_dir.join("staged-performance.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "version": 1,
            "ple_loading": ple_loading.as_str(),
            "embedding_loading": "selective",
            "head_loading": "paged",
            "allocation": state.allocation,
            "auxiliary_process_starts": state.auxiliary_process_starts,
            "wall_duration_ns": chain_started.elapsed().as_nanos(),
            "main_cache": state.weights.stats(),
            "sampled_peak_rss_bytes": state.memory.peak(),
            "stages": state.performance,
        }))?,
    )?;
    let chain_wall_duration = chain_started.elapsed();
    if let Some(verifier) = state.checkpoint_verifier.as_mut() {
        verifier.finish(&recomputed_trace_path);
    }
    if state.has_divergence() {
        if let Some(divergence) = state.divergence() {
            println!(
                "staged-infer stopped at checkpoint index {} / stage {}/{} ({})",
                divergence.checkpoint_index,
                divergence.checkpoint_index + 1,
                manifest.chain.stage.len(),
                divergence.stage
            );
        }
    } else {
        write_execution_times(
            &chain_dir,
            &manifest.chain.stage,
            &state.execution_times,
            chain_wall_duration,
            &state.aux_waves,
        )?;
        print_direct_timing_summary(
            &manifest.chain.stage,
            &state.execution_times,
            chain_wall_duration,
            &state.aux_waves,
        )?;
    }
    if raster_stage.is_some() && state.selected_stage_dir.is_none() {
        bail!("selected Raster reference stage did not run");
    }

    let recomputed_trace = state
        .checkpoint_verifier
        .as_ref()
        .map(CheckpointVerifierState::trace);
    let divergence = state
        .checkpoint_verifier
        .as_ref()
        .and_then(|verifier| verifier.divergence.clone());

    Ok(ChainRunOutput {
        run: HybridRun {
            chain_dir,
            selected_stage_dir: state.selected_stage_dir,
            final_output: state.last_output,
        },
        recomputed_trace,
        divergence,
    })
}

pub(crate) fn read_manifest(path: &Path) -> Result<Manifest> {
    let text =
        fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))?;
    text.parse::<toml::Value>()
        .context("failed to parse Raster.toml as TOML")?;
    let doc: RasterTomlDoc = toml::from_str(&text).context("failed to decode Raster.toml chain")?;
    Ok(Manifest {
        chain: expand_chain_table(doc.chain)?,
    })
}

fn expand_chain_table(table: ChainTable) -> Result<ChainSpec> {
    let inputs = flatten_input_decls(table.inputs)?;
    let mut stages = Vec::new();
    let mut exports = BTreeMap::new();

    for item in merge_chain_items(table.stages, table.repeats) {
        match item {
            ChainItem::Stage(stage) => stages.push(stage),
            ChainItem::Repeat(repeat) => {
                expand_repeat(&repeat, &inputs, &mut stages, &mut exports)?
            }
        }
    }

    resolve_export_sources(&mut stages, &exports);
    Ok(ChainSpec { stage: stages })
}

fn merge_chain_items(
    stages: Vec<toml::Spanned<StageSpec>>,
    repeats: Vec<toml::Spanned<RepeatSpec>>,
) -> Vec<ChainItem> {
    let mut items: Vec<(usize, ChainItem)> = stages
        .into_iter()
        .map(|stage| (stage.span().start, ChainItem::Stage(stage.into_inner())))
        .chain(
            repeats
                .into_iter()
                .map(|repeat| (repeat.span().start, ChainItem::Repeat(repeat.into_inner()))),
        )
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
                    bail!("chain input `{name}` is declared more than once");
                }
            }
            InputDecl::Indexed(indexed) => {
                for (member, external) in indexed.flatten(&name)? {
                    if inputs.insert(member.clone(), external).is_some() {
                        bail!("chain input `{member}` is declared more than once");
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
                "[chain.input.{family}]: path '{}' does not mention '{placeholder}'",
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
                anyhow::anyhow!("repeat binding names unknown chain input `{name}`")
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
            bail!("chain contains duplicate stage name `{}`", stage.name);
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
        Ok(StageDispatch::StagedNative)
    }
}

fn run_one_stage(
    idx: usize,
    stages: &[StageSpec],
    base_dir: &Path,
    chain_dir: &Path,
    stage_index: &BTreeMap<String, usize>,
    current_exe: &Path,
    staged_backend: StagedExecutionBackend,
    raster_stage: Option<&str>,
    recomputed_trace_path: &Path,
    state: &mut ChainRunState,
) -> Result<()> {
    let stage = &stages[idx];
    println!(
        "▸ stage {}/{}  {}   ({})",
        idx + 1,
        stages.len(),
        stage.name,
        stage.project
    );

    let stage_dir = chain_dir.join(&stage.name);
    fs::create_dir_all(&stage_dir)
        .with_context(|| format!("failed to create {}", stage_dir.display()))?;

    let (input_json_path, input_manifest_path) = synthesize_inputs(
        stage,
        &stage_dir,
        base_dir,
        chain_dir,
        &state.output_commitments,
        stage_index,
    )?;

    let stage_run = run_prepared_stage(
        stage,
        base_dir,
        current_exe,
        staged_backend,
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
        stage_run,
        stage_dir,
        recomputed_trace_path,
        state,
    )
}

fn run_prepared_stage(
    stage: &StageSpec,
    base_dir: &Path,
    current_exe: &Path,
    staged_backend: StagedExecutionBackend,
    raster_stage: Option<&str>,
    input_json_path: &Path,
    input_manifest_path: &Path,
    stage_dir: &Path,
    stage_index: &BTreeMap<String, usize>,
    state: &mut ChainRunState,
) -> Result<DirectStageRun> {
    let dispatch = dispatch_for_stage(stage, raster_stage)?;
    match dispatch {
        StageDispatch::StagedNative => {
            let kind = StageKind::from_stage_spec(&stage.project, &stage.name)?;
            println!("    staged-infer {} …", kind.routine());
            let cached_inputs = cached_inputs_for_stage(
                stage,
                stage_index,
                &state.output_commitments,
                &state.output_cache,
            )?;
            run_staged_infer_stage(
                staged_backend,
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
            println!("    raster reference {} …", kind.routine());
            let duration = run_raster_stage(
                stage,
                base_dir,
                input_json_path,
                input_manifest_path,
                stage_dir,
            )?;
            println!("    staged-infer parity check …");
            run_compare_stage(current_exe, stage_dir)?;
            state.selected_stage_dir = Some(stage_dir.to_path_buf());
            Ok(DirectStageRun {
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
    stage_run: DirectStageRun,
    stage_dir: PathBuf,
    recomputed_trace_path: &Path,
    state: &mut ChainRunState,
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
    let output = collect_output(&stage_dir)?;
    if let Some(value) = stage_run.output {
        state.last_output = Some(value.clone());
        output_cache_insert(
            &mut state.output_cache,
            &stage.name,
            &output.structural_commitment,
            value,
        );
    } else {
        state.last_output = None;
    }
    print_stage_output(&output);
    state.output_commitments[idx] = Some(output.structural_commitment);
    observe_checkpoint(idx, &stage.name, &stage_dir, recomputed_trace_path, state)?;
    Ok(())
}

fn observe_checkpoint(
    idx: usize,
    stage_name: &str,
    stage_dir: &Path,
    recomputed_trace_path: &Path,
    state: &mut ChainRunState,
) -> Result<()> {
    if let Some(verifier) = state.checkpoint_verifier.as_mut() {
        verifier.observe_stage(idx, stage_name, stage_dir, recomputed_trace_path)?;
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
    matches!(
        StageKind::from_stage_spec(&stage.project, &stage.name),
        Ok(StageKind::PrefillPrepareAux { .. })
    )
}

fn run_aux_wave(
    range: Range<usize>,
    stages: &[StageSpec],
    base_dir: &Path,
    chain_dir: &Path,
    stage_index: &BTreeMap<String, usize>,
    current_exe: &Path,
    staged_backend: StagedExecutionBackend,
    raster_stage: Option<&str>,
    recomputed_trace_path: &Path,
    state: &mut ChainRunState,
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
            StageDispatch::StagedNative => jobs.push(prepare_aux_stage_job(
                idx,
                stages,
                base_dir,
                chain_dir,
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
            chain_dir,
            stage_index,
            current_exe,
            staged_backend,
            raster_stage,
            recomputed_trace_path,
            state,
        )?;
        if state.has_divergence() {
            return Ok(());
        }
    }

    let batch = if staged_backend == StagedExecutionBackend::InProcess {
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
        observe_checkpoint(
            result.idx,
            &result.name,
            &result.stage_dir,
            recomputed_trace_path,
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
    chain_dir: &Path,
    stage_index: &BTreeMap<String, usize>,
    state: &ChainRunState,
) -> Result<AuxStageJob> {
    let stage = &stages[idx];
    println!(
        "  queued stage {}/{}  {}   ({})",
        idx + 1,
        stages.len(),
        stage.name,
        stage.project
    );
    let stage_dir = chain_dir.join(&stage.name);
    fs::create_dir_all(&stage_dir)
        .with_context(|| format!("failed to create {}", stage_dir.display()))?;
    let (input_json_path, input_manifest_path) = synthesize_inputs(
        stage,
        &stage_dir,
        base_dir,
        chain_dir,
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
            let layer = match job.kind {
                StageKind::PrefillPrepareAux { layer } => layer,
                _ => unreachable!(),
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
            duration: timings.direct_stage_duration,
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
    let stage_run = run_staged_infer_stage_subprocess(
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
    let requested = std::env::var("STAGED_INFER_AUX_PARALLELISM")
        .or_else(|_| std::env::var("DIRECT_NATIVE_AUX_PARALLELISM"))
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
    chain_dir: &Path,
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
                let producer_dir = chain_dir.join(producer);
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

fn run_staged_infer_stage(
    backend: StagedExecutionBackend,
    current_exe: &Path,
    kind: &StageKind,
    stage_name: &str,
    input_json_path: &Path,
    input_manifest_path: &Path,
    stage_dir: &Path,
    cached_inputs: &CachedInputs,
    weights: &RunWeights,
) -> Result<DirectStageRun> {
    match backend {
        StagedExecutionBackend::InProcess => run_staged_infer_stage_in_process(
            kind,
            stage_name,
            input_json_path,
            input_manifest_path,
            stage_dir,
            cached_inputs,
            weights,
        ),
        StagedExecutionBackend::Subprocess => run_staged_infer_stage_subprocess(
            current_exe,
            kind,
            input_json_path,
            input_manifest_path,
            stage_dir,
        ),
    }
}

fn run_staged_infer_stage_subprocess(
    current_exe: &Path,
    kind: &StageKind,
    input_json_path: &Path,
    input_manifest_path: &Path,
    stage_dir: &Path,
) -> Result<DirectStageRun> {
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

    let duration = run_timed_command(command, &format!("staged-infer {} stage", kind.routine()))?;
    let timing_path = stage_dir.join("staged-timings.json");
    let timings = if timing_path.is_file() {
        Some(serde_json::from_slice(&fs::read(timing_path)?)?)
    } else {
        None
    };
    Ok(DirectStageRun {
        timings,
        duration,
        output: None,
    })
}

fn run_staged_infer_stage_in_process(
    kind: &StageKind,
    stage_name: &str,
    input_json_path: &Path,
    input_manifest_path: &Path,
    stage_dir: &Path,
    cached_inputs: &CachedInputs,
    weights: &RunWeights,
) -> Result<DirectStageRun> {
    let _env = StageEnvGuard::apply(stage_dir);
    let started = Instant::now();
    let caches = weights.caches();
    let direct = routines::run_and_publish_from_paths_with_caches(
        kind,
        input_json_path,
        input_manifest_path,
        cached_inputs,
        caches,
    )?;
    let duration = started.elapsed();
    println!(
        "staged-infer {stage_name}: output {} structural={} stage={}",
        direct.artifact.data_path.display(),
        direct.artifact.commitment,
        format_duration(duration),
    );
    Ok(DirectStageRun {
        timings: Some(direct.timings),
        duration,
        output: Some(direct.output),
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
        .env_remove(raster_runtime::TRACE_FORMAT_ENV)
        .env_remove(raster_runtime::OUTPUT_DIR_ENV)
        .env_remove(raster_runtime::PROFILE_PATH_ENV)
        .env_remove(raster_runtime::PROFILE_STREAM_PATH_ENV)
        .env_remove(raster_runtime::PROFILE_RUN_ID_ENV)
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit());

    let status = command
        .status()
        .context("failed to start staged-infer comparison child")?;
    if !status.success() {
        bail!(
            "staged-infer parity check for {} failed ({status})",
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
    chain_dir: &Path,
    stages: &[StageSpec],
    execution_times: &[Option<Duration>],
    total_wall_duration: Duration,
    aux_waves: &[AuxWaveExecutionTime],
) -> Result<()> {
    let path = chain_dir.join(EXECUTION_TIMES_JSON);
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

fn print_direct_timing_summary(
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

    println!("staged-infer timing:");
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

fn create_chain_dir(base_dir: &Path) -> Result<PathBuf> {
    let root = base_dir.join("target").join("staged-infer").join("runs");
    fs::create_dir_all(&root).with_context(|| format!("failed to create {}", root.display()))?;
    let chain_dir = root.join(chain_run_id());
    fs::create_dir_all(&chain_dir)
        .with_context(|| format!("failed to create {}", chain_dir.display()))?;
    Ok(chain_dir)
}

fn chain_run_id() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("{nanos:020}-pid{}", std::process::id())
}

fn chain_run_id_label(chain_dir: &Path) -> String {
    chain_dir
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
            raster_runtime::TRACE_FORMAT_ENV,
            raster_runtime::PROFILE_PATH_ENV,
            raster_runtime::PROFILE_STREAM_PATH_ENV,
            raster_runtime::PROFILE_RUN_ID_ENV,
        ]);
        std::env::set_var(raster_runtime::auth::AUTH_ENV, "0");
        std::env::set_var(raster_runtime::OUTPUT_DIR_ENV, stage_dir);
        for name in [
            raster_runtime::TRACE_PATH_ENV,
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
            "staged-infer-chain-runner-test-{}-{id}",
            std::process::id()
        ))
    }

    fn repo_root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(Path::parent)
            .and_then(Path::parent)
            .unwrap()
            .to_path_buf()
    }

    fn real_manifest_with_repeats(decode: u32, tokenize: u32) -> Manifest {
        let text = fs::read_to_string(repo_root().join("Raster.toml")).unwrap();
        let mut doc: RasterTomlDoc = toml::from_str(&text).unwrap();
        for repeat in &mut doc.chain.repeats {
            let repeat = repeat.get_mut();
            repeat.count = match repeat.name.as_str() {
                "decode" => decode,
                "tokenize" => tokenize,
                _ => panic!("unexpected top-level repeat"),
            };
        }
        Manifest {
            chain: expand_chain_table(doc.chain).unwrap(),
        }
    }

    #[test]
    fn real_manifest_dispatches_one_reference_and_remaining_stages_native() {
        let manifest = real_manifest_with_repeats(2, 2);
        let mut direct = 0usize;
        let mut reference = 0usize;

        for stage in &manifest.chain.stage {
            match dispatch_for_stage(stage, Some("prefill_range_l13")).unwrap() {
                StageDispatch::StagedNative => direct += 1,
                StageDispatch::RasterReference => reference += 1,
            }
        }

        assert_eq!(manifest.chain.stage.len(), 224);
        assert_eq!(direct, 223);
        assert_eq!(reference, 1);
    }

    #[test]
    fn stage_prefix_resolves_exact_expanded_instance_and_bounds_aux_waves() {
        let manifest = real_manifest_with_repeats(2, 2);
        for name in [
            "prompt_merge_b1",
            "prefill_range_l13",
            "prefill_prepare_aux_l5",
        ] {
            let mut stages = manifest.chain.stage.clone();
            let selected = stages.iter().position(|s| s.name == name).unwrap();
            truncate_through_stage(&mut stages, name).unwrap();
            assert_eq!(stages.len(), selected + 1);
            assert_eq!(stages.last().unwrap().name, name);
            if name == "prefill_prepare_aux_l5" {
                let first = stages
                    .iter()
                    .position(is_prefill_prepare_aux_stage)
                    .unwrap();
                assert_eq!(aux_wave_range(&stages, first), Some(first..selected + 1));
            }
        }
        let mut stages = manifest.chain.stage;
        assert!(truncate_through_stage(&mut stages, "prefill-range").is_err());
        assert!(truncate_through_stage(&mut stages, "prompt_merge_b99").is_err());
        stages.push(stages[0].clone());
        assert!(truncate_through_stage(&mut stages, "prompt_merge_seed").is_err());
    }

    #[test]
    fn prefix_timing_report_contains_only_executed_stages() {
        let mut manifest = real_manifest_with_repeats(2, 2);
        truncate_through_stage(&mut manifest.chain.stage, "prompt_merge_b1").unwrap();
        let base = test_base_dir();
        fs::create_dir_all(&base).unwrap();
        let times = vec![Some(Duration::from_nanos(7)); manifest.chain.stage.len()];
        write_execution_times(
            &base,
            &manifest.chain.stage,
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
        assert_eq!(
            names,
            ["prompt_merge_seed", "prompt_merge_b0", "prompt_merge_b1"]
        );
        assert_eq!(doc["total_exec_duration_ns"], 21);
        fs::remove_dir_all(base).unwrap();
    }

    #[test]
    fn real_manifest_dispatches_every_stage_native_without_reference() {
        let manifest = real_manifest_with_repeats(2, 2);
        let mut direct = 0usize;
        let mut reference = 0usize;

        for stage in &manifest.chain.stage {
            match dispatch_for_stage(stage, None).unwrap() {
                StageDispatch::StagedNative => direct += 1,
                StageDispatch::RasterReference => reference += 1,
            }
        }

        assert_eq!(manifest.chain.stage.len(), 224);
        assert_eq!(direct, 224);
        assert_eq!(reference, 0);
    }

    #[test]
    fn real_manifest_expands_decode_repeat_and_export() {
        let manifest = real_manifest_with_repeats(2, 3);
        let index = build_stage_index(&manifest.chain.stage).unwrap();

        assert!(index.contains_key("prompt_merge_b2"));
        assert!(!index.contains_key("prompt_merge_b3"));
        let prepare = &manifest.chain.stage[index["prompt_prepare"]];
        assert_eq!(
            prepare.inputs["merged_pieces"],
            InputBinding::From("prompt_merge_b2".into())
        );

        assert!(index.contains_key("decode_select_t0"));
        assert!(index.contains_key("decode_embed_t1"));
        assert!(index.contains_key("decode_range_t1_l34"));

        let output = manifest.chain.stage.last().unwrap();
        assert_eq!(output.name, "output_finalize");
        assert_eq!(
            output.inputs.get("edge"),
            Some(&InputBinding::From(String::from("decode_select_t1")))
        );
    }

    #[test]
    fn prefill_only_manifest_resolves_zero_count_decode_export() {
        let manifest_path = repo_root().join("runtime/manifests/Raster.prefill-only.toml");
        let manifest = read_manifest(&manifest_path).unwrap();
        let index = build_stage_index(&manifest.chain.stage).unwrap();

        assert!(!index.contains_key("decode_select_t0"));
        assert!(!index.contains_key("prompt_merge_b0"));
        let prepare = &manifest.chain.stage[index["prompt_prepare"]];
        assert_eq!(
            prepare.inputs["merged_pieces"],
            InputBinding::From("prompt_merge_seed".into())
        );
        let output = manifest.chain.stage.last().unwrap();
        assert_eq!(output.name, "output_finalize");
        assert_eq!(
            output.inputs.get("edge"),
            Some(&InputBinding::From(String::from("decode_init")))
        );
    }

    #[test]
    fn synthesized_inputs_cover_external_and_chained_bindings() {
        let base = test_base_dir();
        let chain_dir = base.join("run");
        let producer_dir = chain_dir.join("producer");
        let stage_dir = chain_dir.join("consumer");
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
                ("chained_arg".into(), InputBinding::From("producer".into())),
            ]),
        };
        let stage_index = BTreeMap::from([("producer".into(), 0usize)]);
        let outputs = vec![Some(vec![0xde, 0xad, 0xbe, 0xef])];

        let (input_json, input_manifest) = synthesize_inputs(
            &stage,
            &stage_dir,
            &base,
            &chain_dir,
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
            input["chained_arg"]["path"].as_str(),
            Some(producer_dir.join("output.bin").to_string_lossy().as_ref())
        );
        assert_eq!(input["external_arg"]["load_preference"], "mmap");
        assert_eq!(input["chained_arg"]["load_preference"], "read");
        assert_eq!(manifest["external_arg"]["commitment"], "abc123");
        assert_eq!(manifest["chained_arg"]["commitment"], "deadbeef");

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
            CachedStageValue::PromptTokenization(prompt_prepare::input::PromptTokenization {
                token_ids: raster::List::from(vec![1, 2]),
            }),
        );

        let inputs =
            cached_inputs_for_stage(&stage, &stage_index, &outputs, &output_cache).unwrap();

        assert!(inputs.contains_key("prompt"));
    }

    #[test]
    fn real_manifest_identifies_prefill_prepare_aux_wave() {
        let manifest_path = repo_root().join("Raster.toml");
        let manifest = read_manifest(&manifest_path).unwrap();

        let range = aux_wave_range(&manifest.chain.stage, 3).unwrap();

        assert_eq!(range, 3..38);
        assert!(aux_wave_range(&manifest.chain.stage, 0).is_none());
        assert!(aux_wave_range(&manifest.chain.stage, 38).is_none());
        for stage in &manifest.chain.stage[range] {
            assert!(is_prefill_prepare_aux_stage(stage));
            assert_eq!(
                stage.inputs.get("embedded"),
                Some(&InputBinding::From(String::from("input_embedding")))
            );
        }
    }

    #[test]
    fn raster_reference_inside_aux_wave_dispatches_once() {
        let manifest_path = repo_root().join("Raster.toml");
        let manifest = read_manifest(&manifest_path).unwrap();
        let range = aux_wave_range(&manifest.chain.stage, 3).unwrap();
        let mut direct = 0usize;
        let mut reference = Vec::new();

        for stage in &manifest.chain.stage[range] {
            match dispatch_for_stage(stage, Some("prefill_prepare_aux_l5")).unwrap() {
                StageDispatch::StagedNative => direct += 1,
                StageDispatch::RasterReference => reference.push(stage.name.as_str()),
            }
        }

        assert_eq!(direct, 34);
        assert_eq!(reference, ["prefill_prepare_aux_l5"]);
    }

    #[test]
    fn checkpoint_verifier_detects_hash_mismatch() {
        let base = test_base_dir();
        let stage_dir = write_checkpoint_stage(&base, "stage_a", "commit-a", b"output-a");
        let trace_path = base.join(CHECKPOINT_TRACE_JSON);
        let expected = read_checkpoint_from_stage_dir("stage_a", &stage_dir).unwrap();
        let expected_hash = checkpoint_hash(&expected).unwrap();
        let claimed_hash = corrupt_hash(&expected_hash);
        let mut verifier = CheckpointVerifierState::new(CheckpointHashVerifier {
            claimed_hashes_path: base.join("checkpoints.txt"),
            claimed_hashes: vec![claimed_hash.clone()],
        });

        assert!(verifier
            .observe_stage(0, "stage_a", &stage_dir, &trace_path)
            .unwrap());

        let trace = verifier.trace();
        let divergence = verifier.divergence.unwrap();
        assert_eq!(divergence.checkpoint_index, 0);
        assert_eq!(divergence.stage, "stage_a");
        assert_eq!(divergence.reason, DivergenceReason::CheckpointHash);
        assert_eq!(divergence.claimed_hash, Some(claimed_hash));
        assert_eq!(divergence.recomputed_hash, Some(expected_hash));
        assert_eq!(trace.checkpoints, vec![expected]);

        fs::remove_dir_all(base).unwrap();
    }

    #[test]
    fn checkpoint_verifier_detects_missing_claimed_hash_immediately() {
        let base = test_base_dir();
        let stage_dir = write_checkpoint_stage(&base, "stage_a", "commit-a", b"output-a");
        let trace_path = base.join(CHECKPOINT_TRACE_JSON);
        let mut verifier = CheckpointVerifierState::new(CheckpointHashVerifier {
            claimed_hashes_path: base.join("checkpoints.txt"),
            claimed_hashes: Vec::new(),
        });

        assert!(verifier
            .observe_stage(0, "stage_a", &stage_dir, &trace_path)
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
        let trace_path = base.join(CHECKPOINT_TRACE_JSON);
        let expected = read_checkpoint_from_stage_dir("stage_a", &stage_dir).unwrap();
        let expected_hash = checkpoint_hash(&expected).unwrap();
        let extra_hash =
            String::from("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
        let mut verifier = CheckpointVerifierState::new(CheckpointHashVerifier {
            claimed_hashes_path: base.join("checkpoints.txt"),
            claimed_hashes: vec![expected_hash, extra_hash.clone()],
        });

        assert!(!verifier
            .observe_stage(0, "stage_a", &stage_dir, &trace_path)
            .unwrap());
        verifier.finish(&trace_path);

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
        let chain_dir = base.join("run");
        let stage_dir = chain_dir.join("consumer");
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
            &chain_dir,
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
        let chain_dir = base.join("run");
        let stage_dir = chain_dir.join("consumer");
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
            &chain_dir,
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
