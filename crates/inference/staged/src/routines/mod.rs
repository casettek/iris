use std::path::Path;
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use raster_runtime::OutputArtifact;
use serde::Serialize;

use crate::artifact_io::{encode_output, write_output, EncodedArtifact};
use crate::cache::{CachedInputs, CachedStageValue, MaterializationCache};

pub mod decode_init;
pub mod input_embedding;
pub mod output_finalize;
pub mod prefill_prepare_aux;
pub mod prefill_range;
pub mod prompt_bind;
pub mod prompt_check;
pub mod range_origin;
pub mod sequence_check;

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum StageKind {
    PromptBind,
    PromptCheck,
    InputEmbedding,
    PrefillPrepareAux { layer: usize },
    PrefillRange { layer: usize },
    DecodeInit,
    OutputFinalize,
    SequenceCheck,
    RangeOrigin,
}

#[derive(Debug)]
pub struct DirectCompareOutput {
    pub encoded: EncodedArtifact,
    pub timings: DirectTimings,
}

pub struct DirectPublishOutput {
    pub artifact: OutputArtifact,
    pub timings: DirectTimings,
    pub output: CachedStageValue,
}

#[derive(Debug)]
pub struct DirectCachedOutput {
    pub encoded: EncodedArtifact,
    pub output: CachedStageValue,
    pub timings: DirectTimings,
}

#[derive(Clone, Copy, Default)]
pub struct RoutineRunCaches<'a> {
    pub materializations: Option<&'a MaterializationCache>,
    pub prefill_range_weights:
        Option<&'a inference_kernels::prefill_range::PrefillRangeWeightCache>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct DirectTimings {
    pub input_load_duration: Duration,
    pub kernel_duration: Duration,
    pub encode_write_duration: Duration,
    pub direct_stage_duration: Duration,
}

impl StageKind {
    pub fn from_stage_spec(project: &str, name: &str) -> Result<Self> {
        match project_basename(project) {
            "prompt-bind" if name == "prompt_bind" => Ok(Self::PromptBind),
            "prompt-check" if name.starts_with("prompt_check_p") => Ok(Self::PromptCheck),
            "input-embedding" if name.starts_with("embed_") => Ok(Self::InputEmbedding),
            "prefill-prepare-aux" => Ok(Self::PrefillPrepareAux {
                layer: parse_trailing_layer_index(name)?,
            }),
            "prefill-range" => Ok(Self::PrefillRange {
                layer: parse_trailing_layer_index(name)?,
            }),
            "decode-init" if name == "decode_init" => Ok(Self::DecodeInit),
            "output-finalize" if name == "output_finalize" => Ok(Self::OutputFinalize),
            "sequence-check" if name == "sequence_check" || name.starts_with("check_") => {
                Ok(Self::SequenceCheck)
            }
            "range-origin" if name == "range_origin" => Ok(Self::RangeOrigin),
            _ => bail!("stage `{name}` with project `{project}` is not supported by staged-infer"),
        }
    }

    pub fn from_stage_dir(stage_dir: &Path) -> Result<Self> {
        let name = stage_dir
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| anyhow::anyhow!("selected stage has no valid UTF-8 directory name"))?;
        Self::from_stage_name(name)
    }

    pub fn from_stage_name(name: &str) -> Result<Self> {
        match name {
            "prompt_bind" => Ok(Self::PromptBind),
            _ if name.starts_with("prompt_check_p") => Ok(Self::PromptCheck),
            _ if name.starts_with("embed_") => Ok(Self::InputEmbedding),
            "decode_init" => Ok(Self::DecodeInit),
            "output_finalize" => Ok(Self::OutputFinalize),
            "sequence_check" => Ok(Self::SequenceCheck),
            _ if name.starts_with("check_") => Ok(Self::SequenceCheck),
            "range_origin" => Ok(Self::RangeOrigin),
            _ if name.starts_with("aux_") => Ok(Self::PrefillPrepareAux {
                layer: parse_trailing_layer_index(name)?,
            }),
            _ if name.starts_with("range_p") || name.starts_with("range_c") => {
                Ok(Self::PrefillRange {
                    layer: parse_trailing_layer_index(name)?,
                })
            }
            _ => bail!("stage `{name}` is not supported by staged-infer"),
        }
    }

    pub fn project(&self) -> &'static str {
        match self {
            Self::PromptBind => "prompt-bind",
            Self::PromptCheck => "prompt-check",
            Self::InputEmbedding => "input-embedding",
            Self::PrefillPrepareAux { .. } => "prefill-prepare-aux",
            Self::PrefillRange { .. } => "prefill-range",
            Self::DecodeInit => "decode-init",
            Self::OutputFinalize => "output-finalize",
            Self::SequenceCheck => "sequence-check",
            Self::RangeOrigin => "range-origin",
        }
    }

    pub fn routine(&self) -> &'static str {
        match self {
            Self::PromptBind => "prompt_bind",
            Self::PromptCheck => "prompt_check",
            Self::InputEmbedding => "input_embedding",
            Self::PrefillPrepareAux { .. } => "prefill_prepare_aux",
            Self::PrefillRange { .. } => "prefill_range",
            Self::DecodeInit => "decode_init",
            Self::OutputFinalize => "output_finalize",
            Self::SequenceCheck => "sequence_check",
            Self::RangeOrigin => "range_origin",
        }
    }

    pub fn instance(&self) -> Option<usize> {
        match self {
            Self::PrefillPrepareAux { layer } | Self::PrefillRange { layer } => Some(*layer),
            Self::PromptBind
            | Self::PromptCheck
            | Self::InputEmbedding
            | Self::DecodeInit
            | Self::OutputFinalize
            | Self::SequenceCheck
            | Self::RangeOrigin => None,
        }
    }
}

pub fn run_for_compare(kind: &StageKind) -> Result<DirectCompareOutput> {
    match kind {
        StageKind::PromptBind => compare(
            prompt_bind::load_inputs_from_args,
            prompt_bind::run_direct,
            "prompt_bind",
        ),
        StageKind::PromptCheck => compare(
            prompt_check::load_inputs_from_args,
            prompt_check::run_direct,
            "prompt_check",
        ),
        StageKind::InputEmbedding => compare(
            input_embedding::load_inputs_from_args,
            input_embedding::run_direct,
            "input_embedding",
        ),
        StageKind::PrefillPrepareAux { .. } => compare(
            prefill_prepare_aux::load_inputs_from_args,
            prefill_prepare_aux::run_direct,
            "prefill_prepare_aux",
        ),
        StageKind::PrefillRange { .. } => compare(
            prefill_range::load_inputs_from_args,
            prefill_range::run_direct,
            "prefill_range",
        ),
        StageKind::DecodeInit => compare(
            decode_init::load_inputs_from_args,
            decode_init::run_direct,
            "decode_init",
        ),
        StageKind::OutputFinalize => compare(
            output_finalize::load_inputs_from_args,
            output_finalize::run_direct,
            "output_finalize",
        ),
        StageKind::SequenceCheck => compare(
            sequence_check::load_inputs_from_args,
            sequence_check::run_direct,
            "sequence_check",
        ),
        StageKind::RangeOrigin => compare(
            range_origin::load_inputs_from_args,
            range_origin::run_direct,
            "range_origin",
        ),
    }
}

pub fn run_and_publish(kind: &StageKind) -> Result<DirectPublishOutput> {
    match kind {
        StageKind::PromptBind => publish(
            prompt_bind::load_inputs_from_args,
            prompt_bind::run_direct,
            "prompt_bind",
            CachedStageValue::BpePieces,
        ),
        StageKind::PromptCheck => publish(
            prompt_check::load_inputs_from_args,
            prompt_check::run_direct,
            "prompt_check",
            CachedStageValue::PromptTokenization,
        ),
        StageKind::InputEmbedding => publish(
            input_embedding::load_inputs_from_args,
            input_embedding::run_direct,
            "input_embedding",
            CachedStageValue::EmbeddedActivations,
        ),
        StageKind::PrefillPrepareAux { .. } => publish(
            prefill_prepare_aux::load_inputs_from_args,
            prefill_prepare_aux::run_direct,
            "prefill_prepare_aux",
            CachedStageValue::PleLayerInputs,
        ),
        StageKind::PrefillRange { .. } => publish(
            prefill_range::load_inputs_from_args,
            prefill_range::run_direct,
            "prefill_range",
            CachedStageValue::RangeActivations,
        ),
        StageKind::DecodeInit => publish(
            decode_init::load_inputs_from_args,
            decode_init::run_direct,
            "decode_init",
            CachedStageValue::DecodeEdge,
        ),
        StageKind::OutputFinalize => publish(
            output_finalize::load_inputs_from_args,
            output_finalize::run_direct,
            "output_finalize",
            CachedStageValue::GeneratedOutput,
        ),
        StageKind::SequenceCheck => publish(
            sequence_check::load_inputs_from_args,
            sequence_check::run_direct,
            "sequence_check",
            CachedStageValue::DecodeEdge,
        ),
        StageKind::RangeOrigin => publish(
            range_origin::load_inputs_from_args,
            range_origin::run_direct,
            "range_origin",
            CachedStageValue::from_range_origin,
        ),
    }
}

pub fn run_and_publish_from_paths(
    kind: &StageKind,
    input_path: &Path,
    input_manifest_path: &Path,
    cached_inputs: &CachedInputs,
) -> Result<DirectPublishOutput> {
    run_and_publish_from_paths_with_caches(
        kind,
        input_path,
        input_manifest_path,
        cached_inputs,
        RoutineRunCaches::default(),
    )
}

pub fn run_and_publish_from_paths_with_caches(
    kind: &StageKind,
    input_path: &Path,
    input_manifest_path: &Path,
    cached_inputs: &CachedInputs,
    caches: RoutineRunCaches<'_>,
) -> Result<DirectPublishOutput> {
    match kind {
        StageKind::PromptBind => publish(
            || {
                prompt_bind::load_inputs_from_paths_with_cache(
                    input_path,
                    input_manifest_path,
                    caches.materializations,
                )
            },
            prompt_bind::run_direct,
            "prompt_bind",
            CachedStageValue::BpePieces,
        ),
        StageKind::PromptCheck => publish(
            || {
                prompt_check::load_inputs_from_paths_with_cache(
                    input_path,
                    input_manifest_path,
                    cached_inputs,
                    caches.materializations,
                )
            },
            prompt_check::run_direct,
            "prompt_check",
            CachedStageValue::PromptTokenization,
        ),
        StageKind::InputEmbedding => publish(
            || {
                input_embedding::load_inputs_from_paths_with_cache(
                    input_path,
                    input_manifest_path,
                    cached_inputs,
                    caches.materializations,
                )
            },
            input_embedding::run_direct,
            "input_embedding",
            CachedStageValue::EmbeddedActivations,
        ),
        StageKind::PrefillPrepareAux { .. } => publish(
            || {
                prefill_prepare_aux::load_inputs_from_paths_with_cache(
                    input_path,
                    input_manifest_path,
                    cached_inputs,
                    caches.materializations,
                )
            },
            prefill_prepare_aux::run_direct,
            "prefill_prepare_aux",
            CachedStageValue::PleLayerInputs,
        ),
        StageKind::PrefillRange { .. } => publish(
            || {
                prefill_range::load_inputs_from_paths_with_cache(
                    input_path,
                    input_manifest_path,
                    cached_inputs,
                    caches.materializations,
                )
            },
            |inputs| {
                prefill_range::run_direct_with_weight_cache(inputs, caches.prefill_range_weights)
            },
            "prefill_range",
            CachedStageValue::RangeActivations,
        ),
        StageKind::DecodeInit => publish(
            || decode_init::load_inputs_from_paths(input_path, input_manifest_path, cached_inputs),
            decode_init::run_direct,
            "decode_init",
            CachedStageValue::DecodeEdge,
        ),
        StageKind::OutputFinalize => publish(
            || {
                output_finalize::load_inputs_from_paths(
                    input_path,
                    input_manifest_path,
                    cached_inputs,
                )
            },
            output_finalize::run_direct,
            "output_finalize",
            CachedStageValue::GeneratedOutput,
        ),
        StageKind::SequenceCheck => publish(
            || {
                sequence_check::load_inputs_from_paths_with_cache(
                    input_path,
                    input_manifest_path,
                    cached_inputs,
                    caches.materializations,
                )
            },
            sequence_check::run_direct,
            "sequence_check",
            CachedStageValue::DecodeEdge,
        ),
        StageKind::RangeOrigin => publish(
            || range_origin::load_inputs_from_paths(input_path, input_manifest_path, cached_inputs),
            range_origin::run_direct,
            "range_origin",
            CachedStageValue::from_range_origin,
        ),
    }
}

pub fn run_cached_from_paths(
    kind: &StageKind,
    input_path: &Path,
    input_manifest_path: &Path,
    cached_inputs: &CachedInputs,
) -> Result<DirectCachedOutput> {
    run_cached_from_paths_with_caches(
        kind,
        input_path,
        input_manifest_path,
        cached_inputs,
        RoutineRunCaches::default(),
    )
}

pub fn run_cached_from_paths_with_caches(
    kind: &StageKind,
    input_path: &Path,
    input_manifest_path: &Path,
    cached_inputs: &CachedInputs,
    caches: RoutineRunCaches<'_>,
) -> Result<DirectCachedOutput> {
    match kind {
        StageKind::PromptBind => cache(
            || {
                prompt_bind::load_inputs_from_paths_with_cache(
                    input_path,
                    input_manifest_path,
                    caches.materializations,
                )
            },
            prompt_bind::run_direct,
            "prompt_bind",
            CachedStageValue::BpePieces,
        ),
        StageKind::PromptCheck => cache(
            || {
                prompt_check::load_inputs_from_paths_with_cache(
                    input_path,
                    input_manifest_path,
                    cached_inputs,
                    caches.materializations,
                )
            },
            prompt_check::run_direct,
            "prompt_check",
            CachedStageValue::PromptTokenization,
        ),
        StageKind::InputEmbedding => cache(
            || {
                input_embedding::load_inputs_from_paths_with_cache(
                    input_path,
                    input_manifest_path,
                    cached_inputs,
                    caches.materializations,
                )
            },
            input_embedding::run_direct,
            "input_embedding",
            CachedStageValue::EmbeddedActivations,
        ),
        StageKind::PrefillPrepareAux { .. } => cache(
            || {
                prefill_prepare_aux::load_inputs_from_paths_with_cache(
                    input_path,
                    input_manifest_path,
                    cached_inputs,
                    caches.materializations,
                )
            },
            prefill_prepare_aux::run_direct,
            "prefill_prepare_aux",
            CachedStageValue::PleLayerInputs,
        ),
        StageKind::PrefillRange { .. } => cache(
            || {
                prefill_range::load_inputs_from_paths_with_cache(
                    input_path,
                    input_manifest_path,
                    cached_inputs,
                    caches.materializations,
                )
            },
            |inputs| {
                prefill_range::run_direct_with_weight_cache(inputs, caches.prefill_range_weights)
            },
            "prefill_range",
            CachedStageValue::RangeActivations,
        ),
        StageKind::DecodeInit => cache(
            || decode_init::load_inputs_from_paths(input_path, input_manifest_path, cached_inputs),
            decode_init::run_direct,
            "decode_init",
            CachedStageValue::DecodeEdge,
        ),
        StageKind::OutputFinalize => cache(
            || {
                output_finalize::load_inputs_from_paths(
                    input_path,
                    input_manifest_path,
                    cached_inputs,
                )
            },
            output_finalize::run_direct,
            "output_finalize",
            CachedStageValue::GeneratedOutput,
        ),
        StageKind::SequenceCheck => cache(
            || {
                sequence_check::load_inputs_from_paths_with_cache(
                    input_path,
                    input_manifest_path,
                    cached_inputs,
                    caches.materializations,
                )
            },
            sequence_check::run_direct,
            "sequence_check",
            CachedStageValue::DecodeEdge,
        ),
        StageKind::RangeOrigin => cache(
            || range_origin::load_inputs_from_paths(input_path, input_manifest_path, cached_inputs),
            range_origin::run_direct,
            "range_origin",
            CachedStageValue::from_range_origin,
        ),
    }
}

fn compare<I, O>(
    load: impl FnOnce() -> Result<I>,
    run: impl FnOnce(&I) -> Result<O>,
    label: &str,
) -> Result<DirectCompareOutput>
where
    O: Serialize,
{
    let direct_stage_started = Instant::now();
    let input_load_started = Instant::now();
    let inputs = load().with_context(|| format!("failed to load {label} stage inputs"))?;
    let input_load_duration = input_load_started.elapsed();

    let kernel_started = Instant::now();
    let output = run(&inputs).with_context(|| format!("staged-infer {label} execution failed"))?;
    let kernel_duration = kernel_started.elapsed();

    let encode_write_started = Instant::now();
    let encoded = encode_output(&output)
        .with_context(|| format!("failed to encode staged-infer {label} output"))?;
    let encode_write_duration = encode_write_started.elapsed();

    Ok(DirectCompareOutput {
        encoded,
        timings: DirectTimings {
            input_load_duration,
            kernel_duration,
            encode_write_duration,
            direct_stage_duration: direct_stage_started.elapsed(),
        },
    })
}

fn cache<I, O>(
    load: impl FnOnce() -> Result<I>,
    run: impl FnOnce(&I) -> Result<O>,
    label: &str,
    cache_value: impl FnOnce(O) -> CachedStageValue,
) -> Result<DirectCachedOutput>
where
    O: Serialize,
{
    let direct_stage_started = Instant::now();
    let input_load_started = Instant::now();
    let inputs = load().with_context(|| format!("failed to load {label} stage inputs"))?;
    let input_load_duration = input_load_started.elapsed();

    let kernel_started = Instant::now();
    let output = run(&inputs).with_context(|| format!("staged-infer {label} execution failed"))?;
    let kernel_duration = kernel_started.elapsed();

    let encode_write_started = Instant::now();
    let encoded = encode_output(&output)
        .with_context(|| format!("failed to encode staged-infer {label} output"))?;
    let encode_write_duration = encode_write_started.elapsed();
    let output = cache_value(output);

    Ok(DirectCachedOutput {
        encoded,
        output,
        timings: DirectTimings {
            input_load_duration,
            kernel_duration,
            encode_write_duration,
            direct_stage_duration: direct_stage_started.elapsed(),
        },
    })
}

fn publish<I, O>(
    load: impl FnOnce() -> Result<I>,
    run: impl FnOnce(&I) -> Result<O>,
    label: &str,
    cache_value: impl FnOnce(O) -> CachedStageValue,
) -> Result<DirectPublishOutput>
where
    O: Serialize,
{
    let direct_stage_started = Instant::now();
    let input_load_started = Instant::now();
    let inputs = load().with_context(|| format!("failed to load {label} stage inputs"))?;
    let input_load_duration = input_load_started.elapsed();

    let kernel_started = Instant::now();
    let output = run(&inputs).with_context(|| format!("staged-infer {label} execution failed"))?;
    let kernel_duration = kernel_started.elapsed();

    let encode_write_started = Instant::now();
    let artifact = write_output(&output)
        .with_context(|| format!("failed to write staged-infer {label} output"))?;
    let encode_write_duration = encode_write_started.elapsed();
    let output = cache_value(output);

    Ok(DirectPublishOutput {
        artifact,
        timings: DirectTimings {
            input_load_duration,
            kernel_duration,
            encode_write_duration,
            direct_stage_duration: direct_stage_started.elapsed(),
        },
        output,
    })
}

fn parse_trailing_layer_index(name: &str) -> Result<usize> {
    let raw = name
        .rsplit_once("_l")
        .map(|(_, layer)| layer)
        .ok_or_else(|| anyhow::anyhow!("stage `{name}` has no trailing `_lN` layer index"))?;
    raw.parse::<usize>()
        .with_context(|| format!("failed to parse layer index from `{name}`"))
}

fn project_basename(project: &str) -> &str {
    Path::new(project)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or(project)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stage_kind_accepts_nested_project_paths() {
        assert_eq!(
            StageKind::from_stage_spec("raster-stages/prefill-range", "range_p0_l13").unwrap(),
            StageKind::PrefillRange { layer: 13 }
        );
        assert_eq!(
            StageKind::from_stage_spec("raster-stages/sequence-check", "check_c1").unwrap(),
            StageKind::SequenceCheck
        );
        assert!(
            StageKind::from_stage_spec("raster-stages/decode-embed", "decode_embed_t0").is_err()
        );
    }
}
