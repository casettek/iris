use std::collections::BTreeMap;
use std::fs;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{bail, Context, Result};
use inference_artifacts::{
    InferAuxWaveTiming, InferStageTiming, InferenceResult, InferenceRunReport, InferenceTimings,
};
use output_finalize::input::GeneratedOutput;
use rayon::prelude::*;

use crate::cache::{CachedInputs, CachedStageValue, MaterializationCache, StageOutputCache};
use crate::chain_runner::{self, InputBinding, StageSpec};
use crate::routines::{self, StageKind};

#[derive(Debug, Clone)]
pub struct UncheckpointedInferenceConfig {
    pub base_dir: PathBuf,
    pub manifest_path: PathBuf,
}

#[derive(Debug, Default)]
pub struct UncheckpointedInferenceExecutor;

#[derive(Default)]
struct InferRunCache {
    stage_outputs: StageOutputCache,
    materializations: MaterializationCache,
    prefill_range_weights: inference_kernels::prefill_range::PrefillRangeWeightCache,
}

struct InferRunState {
    output_commitments: Vec<Option<Vec<u8>>>,
    cache: InferRunCache,
    last_output: Option<Arc<CachedStageValue>>,
    stages: Vec<InferStageTiming>,
    aux_waves: Vec<InferAuxWaveTiming>,
}

struct InferenceTempDir {
    path: PathBuf,
}

struct InferStageRun {
    structural_commitment: Vec<u8>,
    output: CachedStageValue,
    timing: InferStageTiming,
}

struct InferAuxStageJob<'a> {
    idx: usize,
    name: String,
    kind: StageKind,
    input_json_path: PathBuf,
    input_manifest_path: PathBuf,
    cached_inputs: CachedInputs,
    materialization_cache: &'a MaterializationCache,
    prefill_range_weights: &'a inference_kernels::prefill_range::PrefillRangeWeightCache,
    input_synthesis_duration: Duration,
}

struct InferAuxStageResult {
    idx: usize,
    run: InferStageRun,
}

struct InferAuxStageBatch {
    results: Vec<InferAuxStageResult>,
    wall_duration: Duration,
    parallelism: usize,
}

impl UncheckpointedInferenceExecutor {
    pub fn run(&self, config: UncheckpointedInferenceConfig) -> Result<InferenceResult> {
        Ok(self.run_with_report(config)?.result)
    }

    pub fn run_with_report(
        &self,
        config: UncheckpointedInferenceConfig,
    ) -> Result<InferenceRunReport> {
        ensure_manifest_exists(&config.manifest_path)?;
        let infer_started = Instant::now();
        let manifest = chain_runner::read_manifest(&config.manifest_path)?;
        chain_runner::validate_supported_stages(&manifest.chain.stage)?;
        let stage_index = chain_runner::build_stage_index(&manifest.chain.stage)?;
        let temp_dir = InferenceTempDir::create()?;

        let mut state = InferRunState {
            output_commitments: vec![None; manifest.chain.stage.len()],
            cache: InferRunCache::default(),
            last_output: None,
            stages: Vec::with_capacity(manifest.chain.stage.len()),
            aux_waves: Vec::new(),
        };

        let mut idx = 0;
        while idx < manifest.chain.stage.len() {
            if let Some(aux_range) = aux_wave_range(&manifest.chain.stage, idx) {
                run_aux_wave(
                    aux_range.clone(),
                    &manifest.chain.stage,
                    &config.base_dir,
                    temp_dir.path(),
                    &stage_index,
                    &mut state,
                )?;
                idx = aux_range.end;
                continue;
            }
            let stage = &manifest.chain.stage[idx];
            run_stage(
                idx,
                stage,
                &config.base_dir,
                temp_dir.path(),
                &stage_index,
                &mut state,
            )?;
            idx += 1;
        }

        let result = match state.last_output.as_deref() {
            Some(CachedStageValue::GeneratedOutput(output)) => {
                inference_result_from_generated(output.clone())
            }
            Some(other) => bail!(
                "infer expected final stage to produce generated output, got {:?}",
                other
            ),
            None => bail!("infer cannot run an empty chain"),
        };
        Ok(InferenceRunReport {
            result,
            timings: InferenceTimings {
                total_duration: infer_started.elapsed(),
                stages: state.stages,
                aux_waves: state.aux_waves,
            },
        })
    }
}

impl UncheckpointedInferenceConfig {
    pub fn from_current_dir() -> Result<Self> {
        let base_dir = std::env::current_dir().context("failed to read current directory")?;
        Ok(Self {
            manifest_path: base_dir.join("Raster.toml"),
            base_dir,
        })
    }
}

pub(crate) fn inference_result_from_generated(output: GeneratedOutput) -> InferenceResult {
    InferenceResult {
        generated_token_count: output.generated_token_count,
        generated_token_ids: output.generated_token_ids.iter().copied().collect(),
        generated_token_ids_sha256: output.generated_token_ids_sha256,
        generated_text: output.generated_text,
        stop_reason: output.stop_reason,
    }
}

fn run_stage(
    idx: usize,
    stage: &StageSpec,
    base_dir: &Path,
    staging_dir: &Path,
    stage_index: &std::collections::BTreeMap<String, usize>,
    state: &mut InferRunState,
) -> Result<()> {
    let cached_inputs = chain_runner::cached_inputs_for_stage(
        stage,
        stage_index,
        &state.output_commitments,
        &state.cache.stage_outputs,
    )?;
    let kind = StageKind::from_stage_spec(&stage.project, &stage.name)?;
    let stage_dir = staging_dir.join(&stage.name);
    let run = run_cached_stage(
        stage,
        kind,
        base_dir,
        staging_dir,
        &stage_dir,
        &state.output_commitments,
        stage_index,
        cached_inputs,
        &state.cache.materializations,
        &state.cache.prefill_range_weights,
    )?;

    store_stage_run(idx, stage, run, state);
    Ok(())
}

fn run_cached_stage(
    stage: &StageSpec,
    kind: StageKind,
    base_dir: &Path,
    staging_dir: &Path,
    stage_dir: &Path,
    output_commitments: &[Option<Vec<u8>>],
    stage_index: &BTreeMap<String, usize>,
    cached_inputs: CachedInputs,
    materialization_cache: &MaterializationCache,
    prefill_range_weights: &inference_kernels::prefill_range::PrefillRangeWeightCache,
) -> Result<InferStageRun> {
    fs::create_dir_all(stage_dir)
        .with_context(|| format!("failed to create {}", stage_dir.display()))?;

    let stage_started = Instant::now();
    let input_synthesis_started = Instant::now();
    let (input_json_path, input_manifest_path) = chain_runner::synthesize_inputs(
        stage,
        stage_dir,
        base_dir,
        staging_dir,
        output_commitments,
        stage_index,
    )?;
    let input_synthesis_duration = input_synthesis_started.elapsed();

    let direct = routines::run_cached_from_paths_with_caches(
        &kind,
        &input_json_path,
        &input_manifest_path,
        &cached_inputs,
        routines::RoutineRunCaches {
            materializations: Some(materialization_cache),
            prefill_range_weights: Some(prefill_range_weights),
        },
    )
    .with_context(|| format!("infer failed at stage `{}`", stage.name))?;
    let structural_commitment = hex::decode(&direct.encoded.structural_commitment)
        .with_context(|| format!("stage `{}` produced a non-hex commitment", stage.name))?;

    Ok(InferStageRun {
        structural_commitment,
        output: direct.output,
        timing: InferStageTiming {
            stage: stage.name.clone(),
            routine: kind.routine().to_string(),
            input_synthesis_duration,
            input_load_duration: direct.timings.input_load_duration,
            kernel_duration: direct.timings.kernel_duration,
            encode_duration: direct.timings.encode_write_duration,
            total_duration: stage_started.elapsed(),
        },
    })
}

fn store_stage_run(idx: usize, stage: &StageSpec, run: InferStageRun, state: &mut InferRunState) {
    let output = state.cache.stage_outputs.insert(
        stage.name.clone(),
        run.structural_commitment.clone(),
        run.output,
    );
    state.output_commitments[idx] = Some(run.structural_commitment);
    state.last_output = Some(output);
    state.stages.push(run.timing);
}

fn run_aux_wave(
    range: Range<usize>,
    stages: &[StageSpec],
    base_dir: &Path,
    staging_dir: &Path,
    stage_index: &BTreeMap<String, usize>,
    state: &mut InferRunState,
) -> Result<()> {
    let mut jobs = Vec::with_capacity(range.len());
    for idx in range.clone() {
        let stage = &stages[idx];
        ensure_stage_inputs_ready(stage, &state.output_commitments, stage_index)?;
        jobs.push(prepare_aux_stage_job(
            idx,
            stage,
            base_dir,
            staging_dir,
            stage_index,
            state,
        )?);
    }

    let batch = run_aux_stage_jobs(jobs)?;
    let stage_duration_sum = batch
        .results
        .iter()
        .fold(Duration::default(), |sum, result| {
            sum + result.run.timing.total_duration
        });
    if let (Some(first), Some(last)) = (batch.results.first(), batch.results.last()) {
        state.aux_waves.push(InferAuxWaveTiming {
            name: String::from("prefill_prepare_aux"),
            first_stage: first.run.timing.stage.clone(),
            last_stage: last.run.timing.stage.clone(),
            stage_count: batch.results.len(),
            parallelism: batch.parallelism,
            wall_duration: batch.wall_duration,
            stage_duration_sum,
        });
    }
    for result in batch.results {
        let stage = &stages[result.idx];
        store_stage_run(result.idx, stage, result.run, state);
    }

    for idx in range {
        if state.output_commitments[idx].is_none() {
            bail!(
                "infer aux wave did not produce output for stage `{}`",
                stages[idx].name
            );
        }
    }
    Ok(())
}

fn prepare_aux_stage_job<'a>(
    idx: usize,
    stage: &StageSpec,
    base_dir: &Path,
    staging_dir: &Path,
    stage_index: &BTreeMap<String, usize>,
    state: &'a InferRunState,
) -> Result<InferAuxStageJob<'a>> {
    let cached_inputs = chain_runner::cached_inputs_for_stage(
        stage,
        stage_index,
        &state.output_commitments,
        &state.cache.stage_outputs,
    )?;
    let kind = StageKind::from_stage_spec(&stage.project, &stage.name)?;
    let stage_dir = staging_dir.join(&stage.name);
    fs::create_dir_all(&stage_dir)
        .with_context(|| format!("failed to create {}", stage_dir.display()))?;

    let input_synthesis_started = Instant::now();
    let (input_json_path, input_manifest_path) = chain_runner::synthesize_inputs(
        stage,
        &stage_dir,
        base_dir,
        staging_dir,
        &state.output_commitments,
        stage_index,
    )?;
    let input_synthesis_duration = input_synthesis_started.elapsed();

    Ok(InferAuxStageJob {
        idx,
        name: stage.name.clone(),
        kind,
        input_json_path,
        input_manifest_path,
        cached_inputs,
        materialization_cache: &state.cache.materializations,
        prefill_range_weights: &state.cache.prefill_range_weights,
        input_synthesis_duration,
    })
}

fn run_aux_stage_jobs(jobs: Vec<InferAuxStageJob<'_>>) -> Result<InferAuxStageBatch> {
    if jobs.is_empty() {
        return Ok(InferAuxStageBatch {
            results: Vec::new(),
            wall_duration: Duration::default(),
            parallelism: 0,
        });
    }
    let parallelism = aux_parallelism(jobs.len());
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(parallelism)
        .build()
        .context("failed to build infer aux stage worker pool")?;
    let started = Instant::now();
    let mut results = pool.install(|| {
        jobs.into_par_iter()
            .map(run_aux_stage_job)
            .collect::<Result<Vec<_>>>()
    })?;
    let wall_duration = started.elapsed();
    results.sort_by_key(|result| result.idx);
    Ok(InferAuxStageBatch {
        results,
        wall_duration,
        parallelism,
    })
}

fn run_aux_stage_job(job: InferAuxStageJob<'_>) -> Result<InferAuxStageResult> {
    let stage_started = Instant::now();
    let direct = routines::run_cached_from_paths_with_caches(
        &job.kind,
        &job.input_json_path,
        &job.input_manifest_path,
        &job.cached_inputs,
        routines::RoutineRunCaches {
            materializations: Some(job.materialization_cache),
            prefill_range_weights: Some(job.prefill_range_weights),
        },
    )
    .with_context(|| format!("infer failed at parallel aux stage `{}`", job.name))?;
    let structural_commitment = hex::decode(&direct.encoded.structural_commitment)
        .with_context(|| format!("stage `{}` produced a non-hex commitment", job.name))?;
    Ok(InferAuxStageResult {
        idx: job.idx,
        run: InferStageRun {
            structural_commitment,
            output: direct.output,
            timing: InferStageTiming {
                stage: job.name,
                routine: job.kind.routine().to_string(),
                input_synthesis_duration: job.input_synthesis_duration,
                input_load_duration: direct.timings.input_load_duration,
                kernel_duration: direct.timings.kernel_duration,
                encode_duration: direct.timings.encode_write_duration,
                total_duration: job.input_synthesis_duration + stage_started.elapsed(),
            },
        },
    })
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

fn ensure_manifest_exists(path: &Path) -> Result<()> {
    if !path.is_file() {
        bail!(
            "infer requires an imported workspace with {}",
            path.display()
        );
    }
    Ok(())
}

impl InferenceTempDir {
    fn create() -> Result<Self> {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .context("system clock is before UNIX_EPOCH")?
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "raster-inference-infer-{}-{nanos}",
            std::process::id()
        ));
        fs::create_dir_all(&path)
            .with_context(|| format!("failed to create {}", path.display()))?;
        Ok(Self { path })
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for InferenceTempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    #[test]
    fn config_from_current_dir_points_at_root_manifest() {
        let cwd = std::env::current_dir().unwrap();
        let config = UncheckpointedInferenceConfig::from_current_dir().unwrap();

        assert_eq!(config.base_dir, cwd);
        assert_eq!(config.manifest_path, cwd.join("Raster.toml"));
    }

    #[test]
    fn missing_manifest_is_rejected_before_execution() {
        let base_dir =
            std::env::temp_dir().join(format!("staged-infer-infer-missing-{}", std::process::id()));
        fs::create_dir_all(&base_dir).unwrap();
        let manifest_path = base_dir.join("Raster.toml");

        let error = UncheckpointedInferenceExecutor
            .run(UncheckpointedInferenceConfig {
                base_dir: base_dir.clone(),
                manifest_path,
            })
            .unwrap_err();

        assert!(error.to_string().contains("imported workspace"));
        fs::remove_dir_all(base_dir).unwrap();
    }

    #[test]
    fn aux_wave_range_groups_adjacent_prefill_prepare_aux_stages() {
        let stages = vec![
            stage("input_embedding", "input-embedding"),
            stage("prefill_prepare_aux_l0", "prefill-prepare-aux"),
            stage("prefill_prepare_aux_l1", "prefill-prepare-aux"),
            stage("prefill_range_l0", "prefill-range"),
        ];

        assert_eq!(aux_wave_range(&stages, 0), None);
        assert_eq!(aux_wave_range(&stages, 1), Some(1..3));
    }

    #[test]
    fn aux_parallelism_honors_requested_limit_and_job_count() {
        assert_eq!(aux_parallelism_from(0, Some(4), 8), 0);
        assert_eq!(aux_parallelism_from(2, Some(4), 8), 2);
        assert_eq!(aux_parallelism_from(8, Some(4), 8), 4);
        assert_eq!(aux_parallelism_from(8, Some(0), 16), 4);
    }

    fn stage(name: &str, project: &str) -> StageSpec {
        StageSpec {
            name: name.to_string(),
            project: project.to_string(),
            inputs: BTreeMap::new(),
        }
    }
}
