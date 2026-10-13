use std::path::Path;
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use raster_runtime::OutputArtifact;
use serde::Serialize;

use crate::artifact_io::{encode_output, write_output, EncodedArtifact};
use crate::cache::{CachedInputs, CachedStageValue, MaterializationCache};

pub mod decode_embed;
pub mod decode_init;
pub mod decode_select_token;
pub mod input_embedding;
pub mod output_finalize;
pub mod prefill_finalize;
pub mod prefill_prepare_aux;
pub mod prefill_range;
pub mod prompt_merge;
pub mod prompt_prepare;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StageKind {
    PromptPrepare,
    PromptMerge,
    InputEmbedding,
    PrefillPrepareAux { layer: usize },
    PrefillRange { layer: usize },
    PrefillFinalize,
    DecodeInit,
    DecodeSelectToken,
    DecodeEmbed,
    OutputFinalize,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DirectTimings {
    pub input_load_duration: Duration,
    pub kernel_duration: Duration,
    pub encode_write_duration: Duration,
    pub direct_stage_duration: Duration,
}

impl StageKind {
    pub fn from_stage_spec(project: &str, name: &str) -> Result<Self> {
        match project_basename(project) {
            "prompt-merge" if name == "prompt_merge_seed" || name.starts_with("prompt_merge_b") => {
                Ok(Self::PromptMerge)
            }
            "prompt-prepare" if name == "prompt_prepare" => Ok(Self::PromptPrepare),
            "input-embedding" if name == "input_embedding" => Ok(Self::InputEmbedding),
            "prefill-prepare-aux" => Ok(Self::PrefillPrepareAux {
                layer: parse_trailing_layer_index(name)?,
            }),
            "prefill-range" => Ok(Self::PrefillRange {
                layer: parse_trailing_layer_index(name)?,
            }),
            "prefill-finalize"
                if name == "prefill_finalize" || name.starts_with("decode_finalize_t") =>
            {
                Ok(Self::PrefillFinalize)
            }
            "decode-init" if name == "decode_init" => Ok(Self::DecodeInit),
            "decode-select-token"
                if name == "decode_select_token" || name.starts_with("decode_select_t") =>
            {
                Ok(Self::DecodeSelectToken)
            }
            "decode-embed" if name == "decode_embed" || name.starts_with("decode_embed_t") => {
                Ok(Self::DecodeEmbed)
            }
            "output-finalize" if name == "output_finalize" => Ok(Self::OutputFinalize),
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
            "prompt_merge_seed" | "prompt_merge" => Ok(Self::PromptMerge),
            _ if name.starts_with("prompt_merge_b") => Ok(Self::PromptMerge),
            "prompt_prepare" => Ok(Self::PromptPrepare),
            "input_embedding" => Ok(Self::InputEmbedding),
            "prefill_finalize" => Ok(Self::PrefillFinalize),
            "decode_init" => Ok(Self::DecodeInit),
            "decode_select_token" => Ok(Self::DecodeSelectToken),
            "decode_embed" => Ok(Self::DecodeEmbed),
            "output_finalize" => Ok(Self::OutputFinalize),
            _ if name.starts_with("decode_select_t") => Ok(Self::DecodeSelectToken),
            _ if name.starts_with("decode_embed_t") => Ok(Self::DecodeEmbed),
            _ if name.starts_with("decode_finalize_t") => Ok(Self::PrefillFinalize),
            _ if name.starts_with("prefill_prepare_aux_l") => Ok(Self::PrefillPrepareAux {
                layer: parse_layer_index(name, "prefill_prepare_aux_l")?,
            }),
            _ if name.starts_with("prefill_range_l") => Ok(Self::PrefillRange {
                layer: parse_layer_index(name, "prefill_range_l")?,
            }),
            _ if name.starts_with("decode_aux_t") => Ok(Self::PrefillPrepareAux {
                layer: parse_trailing_layer_index(name)?,
            }),
            _ if name.starts_with("decode_range_t") => Ok(Self::PrefillRange {
                layer: parse_trailing_layer_index(name)?,
            }),
            _ => bail!("stage `{name}` is not supported by staged-infer"),
        }
    }

    pub fn project(&self) -> &'static str {
        match self {
            Self::PromptPrepare => "prompt-prepare",
            Self::PromptMerge => "prompt-merge",
            Self::InputEmbedding => "input-embedding",
            Self::PrefillPrepareAux { .. } => "prefill-prepare-aux",
            Self::PrefillRange { .. } => "prefill-range",
            Self::PrefillFinalize => "prefill-finalize",
            Self::DecodeInit => "decode-init",
            Self::DecodeSelectToken => "decode-select-token",
            Self::DecodeEmbed => "decode-embed",
            Self::OutputFinalize => "output-finalize",
        }
    }

    pub fn routine(&self) -> &'static str {
        match self {
            Self::PromptPrepare => "prompt_prepare",
            Self::PromptMerge => "prompt_merge",
            Self::InputEmbedding => "input_embedding",
            Self::PrefillPrepareAux { .. } => "prefill_prepare_aux",
            Self::PrefillRange { .. } => "prefill_range",
            Self::PrefillFinalize => "prefill_finalize",
            Self::DecodeInit => "decode_init",
            Self::DecodeSelectToken => "decode_select_token",
            Self::DecodeEmbed => "decode_embed",
            Self::OutputFinalize => "output_finalize",
        }
    }

    pub fn instance(&self) -> Option<usize> {
        match self {
            Self::PrefillPrepareAux { layer } | Self::PrefillRange { layer } => Some(*layer),
            Self::PromptPrepare
            | Self::PromptMerge
            | Self::InputEmbedding
            | Self::PrefillFinalize
            | Self::DecodeInit
            | Self::DecodeSelectToken
            | Self::DecodeEmbed
            | Self::OutputFinalize => None,
        }
    }
}

pub fn run_for_compare(kind: &StageKind) -> Result<DirectCompareOutput> {
    match kind {
        StageKind::PromptPrepare => compare(
            prompt_prepare::load_inputs_from_args,
            prompt_prepare::run_direct,
            "prompt_prepare",
        ),
        StageKind::PromptMerge => compare(
            prompt_merge::load_inputs_from_args,
            prompt_merge::run_direct,
            "prompt_merge",
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
        StageKind::PrefillFinalize => compare(
            prefill_finalize::load_inputs_from_args,
            prefill_finalize::run_direct,
            "prefill_finalize",
        ),
        StageKind::DecodeInit => compare(
            decode_init::load_inputs_from_args,
            decode_init::run_direct,
            "decode_init",
        ),
        StageKind::DecodeSelectToken => compare(
            decode_select_token::load_inputs_from_args,
            decode_select_token::run_direct,
            "decode_select_token",
        ),
        StageKind::DecodeEmbed => compare(
            decode_embed::load_inputs_from_args,
            decode_embed::run_direct,
            "decode_embed",
        ),
        StageKind::OutputFinalize => compare(
            output_finalize::load_inputs_from_args,
            output_finalize::run_direct,
            "output_finalize",
        ),
    }
}

pub fn run_and_publish(kind: &StageKind) -> Result<DirectPublishOutput> {
    match kind {
        StageKind::PromptPrepare => publish(
            prompt_prepare::load_inputs_from_args,
            prompt_prepare::run_direct,
            "prompt_prepare",
            CachedStageValue::PromptTokenization,
        ),
        StageKind::PromptMerge => publish(
            prompt_merge::load_inputs_from_args,
            prompt_merge::run_direct,
            "prompt_merge",
            CachedStageValue::BpePieces,
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
        StageKind::PrefillFinalize => publish(
            prefill_finalize::load_inputs_from_args,
            prefill_finalize::run_direct,
            "prefill_finalize",
            CachedStageValue::PrefillLogits,
        ),
        StageKind::DecodeInit => publish(
            decode_init::load_inputs_from_args,
            decode_init::run_direct,
            "decode_init",
            CachedStageValue::DecodeEdge,
        ),
        StageKind::DecodeSelectToken => publish(
            decode_select_token::load_inputs_from_args,
            decode_select_token::run_direct,
            "decode_select_token",
            CachedStageValue::DecodeEdge,
        ),
        StageKind::DecodeEmbed => publish(
            decode_embed::load_inputs_from_args,
            decode_embed::run_direct,
            "decode_embed",
            CachedStageValue::DecodeActivations,
        ),
        StageKind::OutputFinalize => publish(
            output_finalize::load_inputs_from_args,
            output_finalize::run_direct,
            "output_finalize",
            CachedStageValue::GeneratedOutput,
        ),
    }
}

pub fn run_and_publish_from_paths(
    kind: &StageKind,
    input_path: &Path,
    input_manifest_path: &Path,
    cached_inputs: &CachedInputs,
) -> Result<DirectPublishOutput> {
    match kind {
        StageKind::PromptPrepare => publish(
            || {
                prompt_prepare::load_inputs_from_paths(
                    input_path,
                    input_manifest_path,
                    cached_inputs,
                )
            },
            prompt_prepare::run_direct,
            "prompt_prepare",
            CachedStageValue::PromptTokenization,
        ),
        StageKind::PromptMerge => publish(
            || prompt_merge::load_inputs_from_paths(input_path, input_manifest_path, cached_inputs),
            prompt_merge::run_direct,
            "prompt_merge",
            CachedStageValue::BpePieces,
        ),
        StageKind::InputEmbedding => publish(
            || {
                input_embedding::load_inputs_from_paths(
                    input_path,
                    input_manifest_path,
                    cached_inputs,
                )
            },
            input_embedding::run_direct,
            "input_embedding",
            CachedStageValue::EmbeddedActivations,
        ),
        StageKind::PrefillPrepareAux { .. } => publish(
            || {
                prefill_prepare_aux::load_inputs_from_paths(
                    input_path,
                    input_manifest_path,
                    cached_inputs,
                )
            },
            prefill_prepare_aux::run_direct,
            "prefill_prepare_aux",
            CachedStageValue::PleLayerInputs,
        ),
        StageKind::PrefillRange { .. } => publish(
            || {
                prefill_range::load_inputs_from_paths(
                    input_path,
                    input_manifest_path,
                    cached_inputs,
                )
            },
            prefill_range::run_direct,
            "prefill_range",
            CachedStageValue::RangeActivations,
        ),
        StageKind::PrefillFinalize => publish(
            || {
                prefill_finalize::load_inputs_from_paths(
                    input_path,
                    input_manifest_path,
                    cached_inputs,
                )
            },
            prefill_finalize::run_direct,
            "prefill_finalize",
            CachedStageValue::PrefillLogits,
        ),
        StageKind::DecodeInit => publish(
            || decode_init::load_inputs_from_paths(input_path, input_manifest_path, cached_inputs),
            decode_init::run_direct,
            "decode_init",
            CachedStageValue::DecodeEdge,
        ),
        StageKind::DecodeSelectToken => publish(
            || {
                decode_select_token::load_inputs_from_paths(
                    input_path,
                    input_manifest_path,
                    cached_inputs,
                )
            },
            decode_select_token::run_direct,
            "decode_select_token",
            CachedStageValue::DecodeEdge,
        ),
        StageKind::DecodeEmbed => publish(
            || decode_embed::load_inputs_from_paths(input_path, input_manifest_path, cached_inputs),
            decode_embed::run_direct,
            "decode_embed",
            CachedStageValue::DecodeActivations,
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
        StageKind::PromptPrepare => cache(
            || {
                prompt_prepare::load_inputs_from_paths(
                    input_path,
                    input_manifest_path,
                    cached_inputs,
                )
            },
            prompt_prepare::run_direct,
            "prompt_prepare",
            CachedStageValue::PromptTokenization,
        ),
        StageKind::PromptMerge => cache(
            || prompt_merge::load_inputs_from_paths(input_path, input_manifest_path, cached_inputs),
            prompt_merge::run_direct,
            "prompt_merge",
            CachedStageValue::BpePieces,
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
        StageKind::PrefillFinalize => cache(
            || {
                prefill_finalize::load_inputs_from_paths_with_cache(
                    input_path,
                    input_manifest_path,
                    cached_inputs,
                    caches.materializations,
                )
            },
            prefill_finalize::run_direct,
            "prefill_finalize",
            CachedStageValue::PrefillLogits,
        ),
        StageKind::DecodeInit => cache(
            || decode_init::load_inputs_from_paths(input_path, input_manifest_path, cached_inputs),
            decode_init::run_direct,
            "decode_init",
            CachedStageValue::DecodeEdge,
        ),
        StageKind::DecodeSelectToken => cache(
            || {
                decode_select_token::load_inputs_from_paths(
                    input_path,
                    input_manifest_path,
                    cached_inputs,
                )
            },
            decode_select_token::run_direct,
            "decode_select_token",
            CachedStageValue::DecodeEdge,
        ),
        StageKind::DecodeEmbed => cache(
            || {
                decode_embed::load_inputs_from_paths_with_cache(
                    input_path,
                    input_manifest_path,
                    cached_inputs,
                    caches.materializations,
                )
            },
            decode_embed::run_direct,
            "decode_embed",
            CachedStageValue::DecodeActivations,
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

fn parse_layer_index(name: &str, prefix: &str) -> Result<usize> {
    let raw = name
        .strip_prefix(prefix)
        .expect("caller checked stage name prefix");
    raw.parse::<usize>()
        .with_context(|| format!("failed to parse layer index from `{name}`"))
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
            StageKind::from_stage_spec("raster-stages/prefill-range", "prefill_range_l13").unwrap(),
            StageKind::PrefillRange { layer: 13 }
        );
        assert_eq!(
            StageKind::from_stage_spec("raster-stages/decode-select-token", "decode_select_t1")
                .unwrap(),
            StageKind::DecodeSelectToken
        );
    }
}
