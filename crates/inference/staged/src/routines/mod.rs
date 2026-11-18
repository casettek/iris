use std::path::Path;
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use raster_runtime::OutputArtifact;
use serde::Serialize;

use crate::artifact_io::{encode_output, write_output, EncodedArtifact};
use crate::cache::{CachedInputs, CachedStageValue, MaterializationCache};

pub mod claim_result;
pub mod consensus_result;
pub mod input_embedding;
pub mod layer_mistral;
pub mod prefill_prepare_aux;
pub mod prefill_range;
pub mod prompt_bind;
pub mod prompt_check;
pub mod prompt_source;
pub mod sequence_check;

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum StageKind {
    /// What the prompt text was built from, for a run that has a source.
    PromptSource,
    PromptBind,
    PromptCheck,
    InputEmbedding,
    PrefillPrepareAux { layer: usize },
    PrefillRange { layer: usize },
    /// A `mistral-dense` layer over one range. Its stages are named as
    /// `PrefillRange`'s are: the project tells the two apart.
    LayerMistral { layer: usize },
    SequenceCheck,
    /// What the claim's output stands for: the last stage of a chain of one
    /// inference.
    ClaimResult,
    /// Whether two inferences agree: the last stage of a consensus chain. It
    /// is named as `ClaimResult`'s is: the project tells the two apart.
    ConsensusResult,
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
        // A stage of a pipeline with several inferences is named with its
        // inference's label in front.
        let name = inference_artifacts::pipeline::unlabelled(name);
        match project_basename(project) {
            "prompt-source" if name == "source" => Ok(Self::PromptSource),
            "prompt-bind" if name.starts_with("prompt_bind_p") => Ok(Self::PromptBind),
            "prompt-check" if name.starts_with("prompt_check_p") => Ok(Self::PromptCheck),
            "input-embedding" if name.starts_with("embed_") => Ok(Self::InputEmbedding),
            "prefill-prepare-aux" => Ok(Self::PrefillPrepareAux {
                layer: parse_trailing_layer_index(name)?,
            }),
            "prefill-range" => Ok(Self::PrefillRange {
                layer: parse_trailing_layer_index(name)?,
            }),
            "layer-mistral" => Ok(Self::LayerMistral {
                layer: parse_trailing_layer_index(name)?,
            }),
            "sequence-check" if name == "sequence_check" || name.starts_with("check_") => {
                Ok(Self::SequenceCheck)
            }
            "claim-result" if name == "result" => Ok(Self::ClaimResult),
            "consensus-result" if name == "result" => Ok(Self::ConsensusResult),
            _ => bail!("stage `{name}` with project `{project}` is not supported by staged-infer"),
        }
    }

    pub fn from_stage_dir(stage_dir: &Path) -> Result<Self> {
        let name = stage_dir
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| anyhow::anyhow!("selected stage has no valid UTF-8 directory name"))?;
        let kind = Self::from_stage_name(name)?;
        // Nor does a result stage's name say which pipeline's it is. Its
        // inputs do: only a consensus takes two checks.
        if kind == Self::ClaimResult {
            let inputs = std::fs::read(stage_dir.join("input.json"))
                .ok()
                .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).ok());
            if inputs.is_some_and(|inputs| inputs.get("second").is_some()) {
                return Ok(Self::ConsensusResult);
            }
        }
        // A layer stage's name does not say which family's program it runs.
        // Its inputs do: only `prefill-range` takes a per-layer input.
        if let Self::PrefillRange { layer } = kind {
            let inputs = std::fs::read(stage_dir.join("input.json"))
                .ok()
                .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).ok());
            // A directory whose inputs cannot be read is left to the name; the
            // stage's own loading reports what is wrong with them.
            if inputs.is_some_and(|inputs| inputs.is_object() && inputs.get("ple").is_none()) {
                return Ok(Self::LayerMistral { layer });
            }
        }
        Ok(kind)
    }

    pub fn from_stage_name(name: &str) -> Result<Self> {
        let name = inference_artifacts::pipeline::unlabelled(name);
        match name {
            "source" => Ok(Self::PromptSource),
            "result" => Ok(Self::ClaimResult),
            _ if name.starts_with("prompt_bind_p") => Ok(Self::PromptBind),
            _ if name.starts_with("prompt_check_p") => Ok(Self::PromptCheck),
            _ if name.starts_with("embed_") => Ok(Self::InputEmbedding),
            "sequence_check" => Ok(Self::SequenceCheck),
            _ if name.starts_with("check_") => Ok(Self::SequenceCheck),
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
            Self::PromptSource => "prompt-source",
            Self::PromptBind => "prompt-bind",
            Self::PromptCheck => "prompt-check",
            Self::InputEmbedding => "input-embedding",
            Self::PrefillPrepareAux { .. } => "prefill-prepare-aux",
            Self::PrefillRange { .. } => "prefill-range",
            Self::LayerMistral { .. } => "layer-mistral",
            Self::SequenceCheck => "sequence-check",
            Self::ClaimResult => "claim-result",
            Self::ConsensusResult => "consensus-result",
        }
    }

    pub fn routine(&self) -> &'static str {
        match self {
            Self::PromptSource => "prompt_source",
            Self::PromptBind => "prompt_bind",
            Self::PromptCheck => "prompt_check",
            Self::InputEmbedding => "input_embedding",
            Self::PrefillPrepareAux { .. } => "prefill_prepare_aux",
            Self::PrefillRange { .. } => "prefill_range",
            Self::LayerMistral { .. } => "layer_mistral",
            Self::SequenceCheck => "sequence_check",
            Self::ClaimResult => "claim_result",
            Self::ConsensusResult => "consensus_result",
        }
    }

    pub fn instance(&self) -> Option<usize> {
        match self {
            Self::PrefillPrepareAux { layer }
            | Self::PrefillRange { layer }
            | Self::LayerMistral { layer } => Some(*layer),
            Self::PromptSource
            | Self::PromptBind
            | Self::PromptCheck
            | Self::InputEmbedding
            | Self::SequenceCheck
            | Self::ClaimResult
            | Self::ConsensusResult => None,
        }
    }
}

pub fn run_for_compare(kind: &StageKind) -> Result<DirectCompareOutput> {
    match kind {
        StageKind::PromptSource => compare(
            prompt_source::load_inputs_from_args,
            prompt_source::run_direct,
            "prompt_source",
        ),
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
        StageKind::LayerMistral { .. } => compare(
            layer_mistral::load_inputs_from_args,
            layer_mistral::run_direct,
            "layer_mistral",
        ),
        StageKind::SequenceCheck => compare(
            sequence_check::load_inputs_from_args,
            sequence_check::run_direct,
            "sequence_check",
        ),
        StageKind::ClaimResult => compare(
            claim_result::load_inputs_from_args,
            claim_result::run_direct,
            "claim_result",
        ),
        StageKind::ConsensusResult => compare(
            consensus_result::load_inputs_from_args,
            consensus_result::run_direct,
            "consensus_result",
        ),
    }
}

pub fn run_and_publish(kind: &StageKind) -> Result<DirectPublishOutput> {
    match kind {
        StageKind::PromptSource => publish(
            prompt_source::load_inputs_from_args,
            prompt_source::run_direct,
            "prompt_source",
            CachedStageValue::SourceRecord,
        ),
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
        StageKind::LayerMistral { .. } => publish(
            layer_mistral::load_inputs_from_args,
            layer_mistral::run_direct,
            "layer_mistral",
            CachedStageValue::RangeActivations,
        ),
        StageKind::SequenceCheck => publish(
            sequence_check::load_inputs_from_args,
            sequence_check::run_direct,
            "sequence_check",
            CachedStageValue::DecodeEdge,
        ),
        StageKind::ClaimResult => publish(
            claim_result::load_inputs_from_args,
            claim_result::run_direct,
            "claim_result",
            CachedStageValue::ResultRecord,
        ),
        StageKind::ConsensusResult => publish(
            consensus_result::load_inputs_from_args,
            consensus_result::run_direct,
            "consensus_result",
            CachedStageValue::ConsensusRecord,
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
        StageKind::PromptSource => publish(
            || {
                prompt_source::load_inputs_from_paths_with_cache(
                    input_path,
                    input_manifest_path,
                    cached_inputs,
                    caches.materializations,
                )
            },
            prompt_source::run_direct,
            "prompt_source",
            CachedStageValue::SourceRecord,
        ),
        StageKind::PromptBind => publish(
            || {
                prompt_bind::load_inputs_from_paths_with_cache(
                    input_path,
                    input_manifest_path,
                    cached_inputs,
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
        StageKind::LayerMistral { .. } => publish(
            || {
                layer_mistral::load_inputs_from_paths_with_cache(
                    input_path,
                    input_manifest_path,
                    cached_inputs,
                    caches.materializations,
                )
            },
            layer_mistral::run_direct,
            "layer_mistral",
            CachedStageValue::RangeActivations,
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
        StageKind::ClaimResult => publish(
            || {
                claim_result::load_inputs_from_paths_with_cache(
                    input_path,
                    input_manifest_path,
                    cached_inputs,
                    caches.materializations,
                )
            },
            claim_result::run_direct,
            "claim_result",
            CachedStageValue::ResultRecord,
        ),
        StageKind::ConsensusResult => publish(
            || {
                consensus_result::load_inputs_from_paths_with_cache(
                    input_path,
                    input_manifest_path,
                    cached_inputs,
                    caches.materializations,
                )
            },
            consensus_result::run_direct,
            "consensus_result",
            CachedStageValue::ConsensusRecord,
        ),
    }
}

pub fn run_cached_from_paths_with_caches(
    kind: &StageKind,
    input_path: &Path,
    input_manifest_path: &Path,
    cached_inputs: &CachedInputs,
    caches: RoutineRunCaches<'_>,
) -> Result<DirectCachedOutput> {
    match kind {
        StageKind::PromptSource => cache(
            || {
                prompt_source::load_inputs_from_paths_with_cache(
                    input_path,
                    input_manifest_path,
                    cached_inputs,
                    caches.materializations,
                )
            },
            prompt_source::run_direct,
            "prompt_source",
            CachedStageValue::SourceRecord,
        ),
        StageKind::PromptBind => cache(
            || {
                prompt_bind::load_inputs_from_paths_with_cache(
                    input_path,
                    input_manifest_path,
                    cached_inputs,
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
        StageKind::LayerMistral { .. } => cache(
            || {
                layer_mistral::load_inputs_from_paths_with_cache(
                    input_path,
                    input_manifest_path,
                    cached_inputs,
                    caches.materializations,
                )
            },
            layer_mistral::run_direct,
            "layer_mistral",
            CachedStageValue::RangeActivations,
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
        StageKind::ClaimResult => cache(
            || {
                claim_result::load_inputs_from_paths_with_cache(
                    input_path,
                    input_manifest_path,
                    cached_inputs,
                    caches.materializations,
                )
            },
            claim_result::run_direct,
            "claim_result",
            CachedStageValue::ResultRecord,
        ),
        StageKind::ConsensusResult => cache(
            || {
                consensus_result::load_inputs_from_paths_with_cache(
                    input_path,
                    input_manifest_path,
                    cached_inputs,
                    caches.materializations,
                )
            },
            consensus_result::run_direct,
            "consensus_result",
            CachedStageValue::ConsensusRecord,
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
        // The same stages, as the second inference of a pipeline names them.
        assert_eq!(
            StageKind::from_stage_spec("raster-stages/sequence-check", "m1_check_c1").unwrap(),
            StageKind::SequenceCheck
        );
        assert_eq!(
            StageKind::from_stage_name("m1_range_p0_l13").unwrap(),
            StageKind::PrefillRange { layer: 13 }
        );
        assert_eq!(
            StageKind::from_stage_name("m0_prompt_bind_p2").unwrap(),
            StageKind::PromptBind
        );
        assert_eq!(
            StageKind::from_stage_spec("raster-stages/prompt-source", "source").unwrap(),
            StageKind::PromptSource
        );
        assert_eq!(
            StageKind::from_stage_name("source").unwrap(),
            StageKind::PromptSource
        );
        assert!(
            StageKind::from_stage_spec("raster-stages/decode-embed", "decode_embed_t0").is_err()
        );
    }
}
