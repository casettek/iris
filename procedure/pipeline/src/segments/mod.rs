use std::path::Path;
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use raster_runtime::OutputArtifact;
use serde::Serialize;

use crate::artifact_io::{encode_output, write_output, EncodedArtifact};
use crate::cache::{CachedInputs, CachedStageValue, MaterializationCache};

// The native stages, grouped into the pipeline's segments: the stages that
// share one purpose. A native stage is a stage's code as this host runs it,
// which must give what the stage's Raster program gives.
//
// Each segment's folder owns its stages: the list of them, how a pipeline
// names each, and each one's `NativeStage`. What is in this file and at the
// top of the crate is the runner, which is not any one segment's: it finds a
// stage's segment and runs the stage through the one interface.
//
// To add a stage: its module in its segment's folder with a `NativeStage`,
// and its entry in that segment's `Stage`. To add a segment: its folder with
// a `Stage` of its own, and its variant in `StageKind`.
pub mod inference;
pub mod result;
pub mod source;

// Each native stage under the name of its Raster program.
pub use inference::{
    input_embedding, layer_mistral, prefill_prepare_aux, prefill_range, prompt_bind, prompt_check,
    sequence_check,
};
pub use result::{claim_result, consensus_result};
pub use source::prompt_source;

/// A native stage: a stage's code as this host runs it.
pub trait NativeStage {
    type Inputs;
    type Output: Serialize;
    /// The stage program's project: its directory name.
    const PROJECT: &'static str;
    /// What a timing record and an error call the stage.
    const ROUTINE: &'static str;

    /// The inputs of the stage this process was started as.
    fn load_from_args() -> Result<Self::Inputs>;

    /// The inputs of a stage of a run: from its files, or from what an
    /// earlier stage of the run left in memory.
    fn load_from_paths(
        input: &Path,
        input_manifest: &Path,
        cached_inputs: &CachedInputs,
        materializations: Option<&MaterializationCache>,
    ) -> Result<Self::Inputs>;

    fn run(inputs: &Self::Inputs, caches: NativeStageCaches<'_>) -> Result<Self::Output>;

    /// The output as a later stage of the same run takes it.
    fn cached(output: Self::Output) -> CachedStageValue;
}

/// Something done with a native stage whichever stage it is. A segment's
/// `Stage` hands its stage's type to one.
pub trait StageVisitor {
    type Out;
    fn visit<S: NativeStage>(self) -> Self::Out;
}

/// A stage of a pipeline, by the segment it is of.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum StageKind {
    Source(source::Stage),
    Inference(inference::Stage),
    Result(result::Stage),
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
pub struct NativeStageCaches<'a> {
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
        // A stage of a procedure kind with several inferences is named with its
        // inference's label in front.
        let name = procedure_definition::procedure_kind::unlabelled(name);
        let program = project_basename(project);
        if let Some(stage) = source::Stage::from_spec(program, name)? {
            return Ok(Self::Source(stage));
        }
        if let Some(stage) = inference::Stage::from_spec(program, name)? {
            return Ok(Self::Inference(stage));
        }
        if let Some(stage) = result::Stage::from_spec(program, name)? {
            return Ok(Self::Result(stage));
        }
        bail!("stage `{name}` with project `{project}` is not supported by staged-infer")
    }

    /// The stage a stage directory holds. Where its name does not say which
    /// program it runs, its inputs do.
    pub fn from_stage_dir(stage_dir: &Path) -> Result<Self> {
        let name = stage_dir
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| anyhow::anyhow!("selected stage has no valid UTF-8 directory name"))?;
        Ok(match Self::from_stage_name(name)? {
            Self::Source(stage) => Self::Source(stage.told_by_inputs(stage_dir)),
            Self::Inference(stage) => Self::Inference(stage.told_by_inputs(stage_dir)),
            Self::Result(stage) => Self::Result(stage.told_by_inputs(stage_dir)),
        })
    }

    pub fn from_stage_name(name: &str) -> Result<Self> {
        let name = procedure_definition::procedure_kind::unlabelled(name);
        if let Some(stage) = source::Stage::from_name(name)? {
            return Ok(Self::Source(stage));
        }
        if let Some(stage) = result::Stage::from_name(name)? {
            return Ok(Self::Result(stage));
        }
        if let Some(stage) = inference::Stage::from_name(name)? {
            return Ok(Self::Inference(stage));
        }
        bail!("stage `{name}` is not supported by staged-infer")
    }

    fn visit<V: StageVisitor>(&self, visitor: V) -> V::Out {
        match self {
            Self::Source(stage) => stage.visit(visitor),
            Self::Inference(stage) => stage.visit(visitor),
            Self::Result(stage) => stage.visit(visitor),
        }
    }

    pub fn project(&self) -> &'static str {
        struct Project;
        impl StageVisitor for Project {
            type Out = &'static str;
            fn visit<S: NativeStage>(self) -> Self::Out {
                S::PROJECT
            }
        }
        self.visit(Project)
    }

    pub fn routine(&self) -> &'static str {
        struct Routine;
        impl StageVisitor for Routine {
            type Out = &'static str;
            fn visit<S: NativeStage>(self) -> Self::Out {
                S::ROUTINE
            }
        }
        self.visit(Routine)
    }

    pub fn instance(&self) -> Option<usize> {
        match self {
            Self::Source(stage) => stage.instance(),
            Self::Inference(stage) => stage.instance(),
            Self::Result(stage) => stage.instance(),
        }
    }

    /// The layer of an inference's auxiliary stage, which a run hands to its
    /// workers. `None` for every other stage.
    pub fn auxiliary_layer(&self) -> Option<usize> {
        match self {
            Self::Inference(inference::Stage::PrefillPrepareAux { layer }) => Some(*layer),
            _ => None,
        }
    }
}

/// A stage directory's inputs as it wrote them, or `None` when they cannot be
/// read.
fn stage_inputs(stage_dir: &Path) -> Option<serde_json::Value> {
    std::fs::read(stage_dir.join("input.json"))
        .ok()
        .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).ok())
}

/// The files of a stage of a run, and what earlier stages left in memory.
#[derive(Clone, Copy)]
struct FromPaths<'a> {
    input: &'a Path,
    input_manifest: &'a Path,
    cached_inputs: &'a CachedInputs,
}

fn load<S: NativeStage>(
    from: Option<FromPaths<'_>>,
    caches: NativeStageCaches<'_>,
) -> Result<S::Inputs> {
    match from {
        None => S::load_from_args(),
        Some(from) => S::load_from_paths(
            from.input,
            from.input_manifest,
            from.cached_inputs,
            caches.materializations,
        ),
    }
}

struct Compare;

impl StageVisitor for Compare {
    type Out = Result<DirectCompareOutput>;
    fn visit<S: NativeStage>(self) -> Self::Out {
        compare(
            S::load_from_args,
            |inputs| S::run(inputs, NativeStageCaches::default()),
            S::ROUTINE,
        )
    }
}

struct Publish<'a> {
    from: Option<FromPaths<'a>>,
    caches: NativeStageCaches<'a>,
}

impl StageVisitor for Publish<'_> {
    type Out = Result<DirectPublishOutput>;
    fn visit<S: NativeStage>(self) -> Self::Out {
        publish(
            || load::<S>(self.from, self.caches),
            |inputs| S::run(inputs, self.caches),
            S::ROUTINE,
            S::cached,
        )
    }
}

struct Cache<'a> {
    from: FromPaths<'a>,
    caches: NativeStageCaches<'a>,
}

impl StageVisitor for Cache<'_> {
    type Out = Result<DirectCachedOutput>;
    fn visit<S: NativeStage>(self) -> Self::Out {
        cache(
            || load::<S>(Some(self.from), self.caches),
            |inputs| S::run(inputs, self.caches),
            S::ROUTINE,
            S::cached,
        )
    }
}

pub fn run_for_compare(kind: &StageKind) -> Result<DirectCompareOutput> {
    kind.visit(Compare)
}

pub fn run_and_publish(kind: &StageKind) -> Result<DirectPublishOutput> {
    kind.visit(Publish {
        from: None,
        caches: NativeStageCaches::default(),
    })
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
        NativeStageCaches::default(),
    )
}

pub fn run_and_publish_from_paths_with_caches(
    kind: &StageKind,
    input_path: &Path,
    input_manifest_path: &Path,
    cached_inputs: &CachedInputs,
    caches: NativeStageCaches<'_>,
) -> Result<DirectPublishOutput> {
    kind.visit(Publish {
        from: Some(FromPaths {
            input: input_path,
            input_manifest: input_manifest_path,
            cached_inputs,
        }),
        caches,
    })
}

pub fn run_cached_from_paths_with_caches(
    kind: &StageKind,
    input_path: &Path,
    input_manifest_path: &Path,
    cached_inputs: &CachedInputs,
    caches: NativeStageCaches<'_>,
) -> Result<DirectCachedOutput> {
    kind.visit(Cache {
        from: FromPaths {
            input: input_path,
            input_manifest: input_manifest_path,
            cached_inputs,
        },
        caches,
    })
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
        let range = |layer| StageKind::Inference(inference::Stage::PrefillRange { layer });
        let check = StageKind::Inference(inference::Stage::SequenceCheck);
        let source = StageKind::Source(source::Stage::PromptSource);
        assert_eq!(
            StageKind::from_stage_spec("raster-stages/prefill-range", "range_p0_l13").unwrap(),
            range(13)
        );
        assert_eq!(
            StageKind::from_stage_spec("raster-stages/sequence-check", "check_c1").unwrap(),
            check
        );
        // The same stages, as the second inference of a procedure kind names them.
        assert_eq!(
            StageKind::from_stage_spec("raster-stages/sequence-check", "m1_check_c1").unwrap(),
            check
        );
        assert_eq!(
            StageKind::from_stage_name("m1_range_p0_l13").unwrap(),
            range(13)
        );
        assert_eq!(
            StageKind::from_stage_name("m0_prompt_bind_p2").unwrap(),
            StageKind::Inference(inference::Stage::PromptBind)
        );
        assert_eq!(
            StageKind::from_stage_spec("raster-stages/prompt-source", "source").unwrap(),
            source
        );
        assert_eq!(StageKind::from_stage_name("source").unwrap(), source);
        assert!(
            StageKind::from_stage_spec("raster-stages/decode-embed", "decode_embed_t0").is_err()
        );
    }

    /// Every stage, with the name a pipeline gives it.
    fn every_stage() -> Vec<(StageKind, &'static str)> {
        use inference::Stage as Inference;
        vec![
            (StageKind::Source(source::Stage::PromptSource), "source"),
            (StageKind::Source(source::Stage::PromptRelay), "relay"),
            (
                StageKind::Inference(Inference::PromptBind),
                "prompt_bind_p0",
            ),
            (
                StageKind::Inference(Inference::PromptCheck),
                "prompt_check_p0",
            ),
            (StageKind::Inference(Inference::InputEmbedding), "embed_p0"),
            (
                StageKind::Inference(Inference::PrefillPrepareAux { layer: 2 }),
                "aux_p0_l2",
            ),
            (
                StageKind::Inference(Inference::PrefillRange { layer: 2 }),
                "range_p0_l2",
            ),
            (
                StageKind::Inference(Inference::LayerMistral { layer: 2 }),
                "range_p0_l2",
            ),
            (StageKind::Inference(Inference::SequenceCheck), "check_c0"),
            (StageKind::Result(result::Stage::ClaimResult), "result"),
            (StageKind::Result(result::Stage::ConsensusResult), "result"),
        ]
    }

    #[test]
    fn a_stage_is_found_by_its_project_and_name_and_names_both() {
        let mut projects = Vec::new();
        let mut routines = Vec::new();
        for (kind, name) in every_stage() {
            // The project a stage names is the one a pipeline finds it by.
            let project = format!("raster-stages/{}", kind.project());
            assert_eq!(StageKind::from_stage_spec(&project, name).unwrap(), kind);
            // A routine is its project's name, as a timing record writes it.
            assert_eq!(kind.routine(), kind.project().replace('-', "_"));
            assert_eq!(kind.instance().is_some(), name.ends_with("_l2"));
            projects.push(kind.project());
            routines.push(kind.routine());
        }
        // No two stages are of one project.
        projects.sort_unstable();
        projects.dedup();
        routines.sort_unstable();
        routines.dedup();
        assert_eq!(projects.len(), every_stage().len());
        assert_eq!(routines.len(), every_stage().len());
        assert_eq!(
            projects,
            [
                "claim-result",
                "consensus-result",
                "input-embedding",
                "layer-mistral",
                "prefill-prepare-aux",
                "prefill-range",
                "prompt-bind",
                "prompt-check",
                "prompt-relay",
                "prompt-source",
                "sequence-check",
            ]
        );
    }

    #[test]
    fn a_name_alone_gives_the_stage_its_inputs_may_tell_apart() {
        // A layer's name is one for both families, and a result's for both
        // kinds: the name gives the first, and the directory's inputs say.
        assert_eq!(
            StageKind::from_stage_name("range_c1_l7").unwrap(),
            StageKind::Inference(inference::Stage::PrefillRange { layer: 7 })
        );
        assert_eq!(
            StageKind::from_stage_name("result").unwrap(),
            StageKind::Result(result::Stage::ClaimResult)
        );
        assert_eq!(
            StageKind::from_stage_name("aux_p3_l4")
                .unwrap()
                .auxiliary_layer(),
            Some(4)
        );
        assert_eq!(
            StageKind::from_stage_name("range_p3_l4")
                .unwrap()
                .auxiliary_layer(),
            None
        );
        assert!(StageKind::from_stage_name("decode_embed_t0").is_err());
        assert!(StageKind::from_stage_name("range_p3").is_err());

        let dir = std::env::temp_dir().join(format!(
            "iris-stage-kind-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        for (name, inputs, expected) in [
            (
                "result",
                r#"{"check":{},"forms":{}}"#,
                StageKind::Result(result::Stage::ClaimResult),
            ),
            (
                "result",
                r#"{"first":{},"second":{},"forms":{}}"#,
                StageKind::Result(result::Stage::ConsensusResult),
            ),
            (
                "range_p0_l5",
                r#"{"activations":{},"ple":{}}"#,
                StageKind::Inference(inference::Stage::PrefillRange { layer: 5 }),
            ),
            (
                "m1_range_p0_l5",
                r#"{"activations":{}}"#,
                StageKind::Inference(inference::Stage::LayerMistral { layer: 5 }),
            ),
        ] {
            let stage_dir = dir.join(name);
            std::fs::create_dir_all(&stage_dir).unwrap();
            std::fs::write(stage_dir.join("input.json"), inputs).unwrap();
            assert_eq!(StageKind::from_stage_dir(&stage_dir).unwrap(), expected);
        }
        // A directory with no inputs is left to its name.
        let bare = dir.join("range_p1_l5");
        std::fs::create_dir_all(&bare).unwrap();
        assert_eq!(
            StageKind::from_stage_dir(&bare).unwrap(),
            StageKind::Inference(inference::Stage::PrefillRange { layer: 5 })
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
