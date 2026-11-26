use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use raster_core::pipeline::PipelineCommitment;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::encoding::{parse_hash, root_from_path, tree_path, tree_root, Encoder};
use inference_manifest::io::{read_json, write_json};

pub const CHECKPOINT_LIST_JSON: &str = "checkpoint_list.json";
pub const CHECKPOINTS_TXT: &str = "checkpoints.txt";
pub const CHECKPOINT_HASHES_TXT: &str = CHECKPOINTS_TXT;
pub const EXECUTION_TIMES_JSON: &str = "execution-times.json";
/// Written into a stage directory by the native runner: the program identity
/// the stage project has locked, which is the program a Raster replay of the
/// stage runs.
pub const STAGE_PROGRAM_JSON: &str = "program.json";

/// The checkpoints of one checkpointed inference run, one per stage, in order.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(transparent)]
pub struct CheckpointList {
    pub checkpoints: Vec<Checkpoint>,
}

/// One stage boundary of a claim: which program ran, on which committed
/// inputs, to which output. These are the values a Raster replay of the stage
/// is bound to, so a challenger can start one at the claimer's own boundary.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct Checkpoint {
    pub stage: String,
    /// The stage program's identity from its project's `Raster.lock`, in
    /// lowercase hex. Empty when the run recorded none, which no contract
    /// accepts.
    #[serde(default)]
    pub program_commitment: String,
    /// SHA-256 of the stage's `input_manifest.json`.
    pub input_commitment: String,
    /// For each input the stage takes from an earlier stage: that stage's
    /// position in the claim's checkpoint list. A pointer, so the comparison
    /// with the producer's output is one lookup; the stages themselves check
    /// that it is the right producer.
    #[serde(default)]
    pub input_sources: BTreeMap<String, u32>,
    pub output_commitment: String,
    pub output_sha256: String,
}

impl Checkpoint {
    /// Whether two checkpoints state the same stage boundary: the stage, its
    /// input manifest hash and its two output commitments.
    ///
    /// These are what a stage directory states whoever wrote it. A directory
    /// Raster wrote records no program id and no producers, since a Raster run
    /// names both in its pipeline commitment, so a native checkpoint and a Raster
    /// replay of the same stage are compared on this and not as wholes.
    pub fn same_boundary(&self, other: &Self) -> bool {
        self.stage == other.stage
            && self.input_commitment == other.input_commitment
            && self.output_commitment == other.output_commitment
            && self.output_sha256 == other.output_sha256
    }
}

#[derive(Debug, Deserialize)]
struct OutputManifest {
    output: OutputManifestEntry,
}

#[derive(Debug, Deserialize)]
struct OutputManifestEntry {
    commitment: String,
}

#[derive(Debug, Deserialize)]
struct ExecutionTimesDocument {
    stages: Vec<StageExecutionTime>,
}

#[derive(Debug, Deserialize)]
struct StageExecutionTime {
    name: String,
}

pub fn read_checkpoint_list(path: &Path) -> Result<CheckpointList> {
    read_json(path)
}

pub fn read_checkpoint_hashes(path: &Path) -> Result<Vec<String>> {
    let input =
        fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))?;
    input
        .lines()
        .enumerate()
        .map(|(index, line)| validate_checkpoint_hash(line.trim(), index + 1))
        .collect()
}

pub fn write_checkpoint_list_artifact(
    pipeline_dir: &Path,
    list: &CheckpointList,
) -> Result<PathBuf> {
    let list_path = pipeline_dir.join(CHECKPOINT_LIST_JSON);
    write_json(&list_path, list)?;
    Ok(list_path)
}

pub fn write_checkpoint_hashes_artifact(
    pipeline_dir: &Path,
    list: &CheckpointList,
) -> Result<PathBuf> {
    let hashes_path = pipeline_dir.join(CHECKPOINTS_TXT);
    let mut output = checkpoint_hashes(list)?.join("\n");
    if !output.is_empty() {
        output.push('\n');
    }
    fs::write(&hashes_path, output)
        .with_context(|| format!("failed to write {}", hashes_path.display()))?;
    Ok(hashes_path)
}

pub fn write_checkpoint_hashes(path: &Path, hashes: &[String]) -> Result<()> {
    let mut output = hashes.join("\n");
    if !output.is_empty() {
        output.push('\n');
    }
    fs::write(path, output).with_context(|| format!("failed to write {}", path.display()))
}

pub fn checkpoint_hashes(list: &CheckpointList) -> Result<Vec<String>> {
    list.checkpoints.iter().map(checkpoint_hash).collect()
}

/// A checkpoint's hash: its fields in order, in the encoding a contract
/// recomputes (`crate::encoding`). The producers are in name order.
pub fn checkpoint_hash(checkpoint: &Checkpoint) -> Result<String> {
    let mut encoder = Encoder::new(b"iris/checkpoint/v1\0");
    encoder
        .text(&checkpoint.stage)
        .hash(&checkpoint.program_commitment)
        .hash(&checkpoint.input_commitment)
        .count(checkpoint.input_sources.len());
    for (input, producer) in &checkpoint.input_sources {
        encoder.text(input).number(*producer);
    }
    encoder
        .hash(&checkpoint.output_commitment)
        .hash(&checkpoint.output_sha256);
    Ok(hex::encode(encoder.finish()))
}

fn list_values(hashes: &[String]) -> Result<Vec<[u8; 32]>> {
    hashes
        .iter()
        .enumerate()
        .map(|(index, hash)| {
            parse_hash(hash)
                .with_context(|| format!("checkpoint hash {index} is not 32 bytes of hex: {hash}"))
        })
        .collect()
}

fn list_commitment(count: u32, tree_root: &[u8; 32]) -> String {
    let mut encoder = Encoder::new(b"iris/checkpoints/v1\0");
    encoder.number(count).hash(&hex::encode(tree_root));
    hex::encode(encoder.finish())
}

/// A claim's commitment to its checkpoint list: how many checkpoints it has
/// and the root of the tree over their hashes, in order. `hashes` are the
/// lines of `checkpoints.txt`.
pub fn checkpoints_root(hashes: &[String]) -> Result<String> {
    let values = list_values(hashes)?;
    let root = tree_root(&values).context("a claim has more checkpoints than its tree holds")?;
    Ok(list_commitment(values.len() as u32, &root))
}

/// The path that opens the checkpoint at `index` against
/// [`checkpoints_root`], in lowercase hex.
pub fn checkpoint_opening(hashes: &[String], index: usize) -> Result<Vec<String>> {
    let values = list_values(hashes)?;
    Ok(tree_path(&values, index)
        .with_context(|| format!("the list has no checkpoint {index}"))?
        .iter()
        .map(hex::encode)
        .collect())
}

/// Whether `checkpoint` is the one at `index` of the list of `count`
/// checkpoints that `root` commits to. This is what a contract computes to
/// open one checkpoint: its hash, and one hash per level of the tree.
pub fn verify_checkpoint_opening(
    root: &str,
    count: u32,
    index: u32,
    checkpoint: &Checkpoint,
    path: &[String],
) -> bool {
    let opened = || {
        if index >= count {
            return None;
        }
        let value = parse_hash(&checkpoint_hash(checkpoint).ok()?)?;
        let path = path
            .iter()
            .map(|sibling| parse_hash(sibling))
            .collect::<Option<Vec<_>>>()?;
        let tree_root = root_from_path(&value, index, &path)?;
        Some(list_commitment(count, &tree_root) == root)
    };
    opened().unwrap_or(false)
}

fn validate_checkpoint_hash(hash: &str, line_number: usize) -> Result<String> {
    if hash.len() != 64 || !hash.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        anyhow::bail!("invalid checkpoint hash on line {line_number}: {hash}");
    }
    Ok(hash.to_ascii_lowercase())
}

pub fn build_checkpoint_list(pipeline_dir: &Path, _manifest_path: &Path) -> Result<CheckpointList> {
    let stage_order = read_stage_order(pipeline_dir)?.map(StageOrderIndex::from);

    let mut checkpoints = Vec::new();
    for entry in fs::read_dir(pipeline_dir).with_context(|| {
        format!(
            "failed to read pipeline directory {}",
            pipeline_dir.display()
        )
    })? {
        let entry =
            entry.with_context(|| format!("failed to read entry in {}", pipeline_dir.display()))?;
        let stage_dir = entry.path();
        if !stage_dir.is_dir() {
            continue;
        }
        let Some(stage) = stage_dir.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        let output_manifest_path = stage_dir.join("output_manifest.json");
        if !output_manifest_path.is_file() {
            continue;
        }
        checkpoints.push(read_checkpoint_from_stage_dir(stage, &stage_dir)?);
    }
    checkpoints.sort_by(|left, right| match stage_order.as_ref() {
        Some(stages) => stages
            .order(&left.stage)
            .cmp(&stages.order(&right.stage))
            .then_with(|| left.stage.cmp(&right.stage)),
        None => left.stage.cmp(&right.stage),
    });

    Ok(CheckpointList { checkpoints })
}

pub fn read_checkpoint_from_stage_dir(stage: &str, stage_dir: &Path) -> Result<Checkpoint> {
    let input_manifest_path = stage_dir.join("input_manifest.json");
    let output_manifest_path = stage_dir.join("output_manifest.json");
    let output_path = stage_dir.join("output.bin");
    let input_manifest = fs::read(&input_manifest_path)
        .with_context(|| format!("failed to read {}", input_manifest_path.display()))?;
    let manifest: OutputManifest = serde_json::from_slice(
        &fs::read(&output_manifest_path)
            .with_context(|| format!("failed to read {}", output_manifest_path.display()))?,
    )
    .with_context(|| format!("failed to parse {}", output_manifest_path.display()))?;
    let output = fs::read(&output_path)
        .with_context(|| format!("failed to read {}", output_path.display()))?;
    let program = read_stage_program(stage_dir)?;
    Ok(Checkpoint {
        stage: stage.to_string(),
        program_commitment: program.program_commitment,
        input_commitment: format!("{:x}", Sha256::digest(&input_manifest)),
        input_sources: program.input_sources,
        output_commitment: manifest.output.commitment,
        output_sha256: format!("{:x}", Sha256::digest(&output)),
    })
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct StageProgramDocument {
    program_commitment: String,
    #[serde(default)]
    input_sources: BTreeMap<String, u32>,
}

/// Record in a stage directory the program identity the stage ran as, and for
/// each input it took from an earlier stage, that stage's position in the run.
pub fn write_stage_program(
    stage_dir: &Path,
    program_commitment: &str,
    input_sources: &BTreeMap<String, u32>,
) -> Result<()> {
    write_json(
        &stage_dir.join(STAGE_PROGRAM_JSON),
        &StageProgramDocument {
            program_commitment: program_commitment.to_string(),
            input_sources: input_sources.clone(),
        },
    )
}

/// What a stage directory records of its program and producers, or nothing
/// when it has no record: a Raster run directory names both in its pipeline
/// commitment.
fn read_stage_program(stage_dir: &Path) -> Result<StageProgramDocument> {
    let path = stage_dir.join(STAGE_PROGRAM_JSON);
    if !path.is_file() {
        return Ok(StageProgramDocument::default());
    }
    read_json(&path)
}

/// Read the run's recorded order independently of optional timing reports.
/// Current Raster writes an ordered pipeline commitment even in no-auth mode;
/// pipeline-native and older Raster runs record their order in execution times.
/// A malformed commitment is an error, never a reason to use another inventory.
pub fn read_stage_order(pipeline_dir: &Path) -> Result<Option<Vec<String>>> {
    let commitment_path = pipeline_dir.join("pipeline-commitment");
    if commitment_path.exists() {
        let bytes = fs::read(&commitment_path)
            .with_context(|| format!("failed to read {}", commitment_path.display()))?;
        let pipeline: PipelineCommitment = postcard::from_bytes(&bytes)
            .with_context(|| format!("failed to parse {}", commitment_path.display()))?;
        return Ok(Some(
            pipeline
                .stages
                .into_iter()
                .map(|stage| stage.name)
                .collect(),
        ));
    }
    let times_path = pipeline_dir.join(EXECUTION_TIMES_JSON);
    if times_path.exists() {
        let document: ExecutionTimesDocument = read_json(&times_path)?;
        return Ok(Some(
            document
                .stages
                .into_iter()
                .map(|stage| stage.name)
                .collect(),
        ));
    }
    Ok(None)
}

struct StageOrderIndex {
    order: BTreeMap<String, usize>,
}

impl StageOrderIndex {
    fn order(&self, stage: &str) -> usize {
        self.order.get(stage).copied().unwrap_or(usize::MAX)
    }
}

impl From<Vec<String>> for StageOrderIndex {
    fn from(stages: Vec<String>) -> Self {
        let mut order = BTreeMap::new();
        for (idx, stage) in stages.into_iter().enumerate() {
            order.insert(stage, idx);
        }
        Self { order }
    }
}

/// The program identity a stage project's `Raster.lock` records.
pub fn locked_program_commitment(stage_project: &Path) -> Result<String> {
    #[derive(serde::Deserialize)]
    struct Lock {
        program_commitment: String,
    }
    let path = stage_project.join("Raster.lock");
    let lock: Lock = serde_json::from_slice(
        &std::fs::read(&path).with_context(|| format!("failed to read {}", path.display()))?,
    )
    .with_context(|| format!("failed to parse {}", path.display()))?;
    anyhow::ensure!(
        common_identity::context::parse_id(&lock.program_commitment).is_some(),
        "{} does not record a 32-byte program commitment",
        path.display()
    );
    Ok(lock.program_commitment)
}
