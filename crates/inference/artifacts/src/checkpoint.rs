use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use raster_core::chain::ChainCommitment;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::io::{read_json, write_json};

pub const CHECKPOINT_TRACE_JSON: &str = "checkpoint_trace.json";
pub const CHECKPOINTS_TXT: &str = "checkpoints.txt";
pub const CHECKPOINT_HASHES_TXT: &str = CHECKPOINTS_TXT;
pub const EXECUTION_TIMES_JSON: &str = "execution-times.json";

/// Ordered routine-boundary checkpoints from one checkpointed inference run.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(transparent)]
pub struct CheckpointTrace {
    pub checkpoints: Vec<Checkpoint>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Checkpoint {
    pub stage: String,
    pub input_commitment: String,
    pub output_commitment: String,
    pub output_sha256: String,
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

pub fn read_checkpoint_trace(path: &Path) -> Result<CheckpointTrace> {
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

pub fn write_checkpoint_trace_artifact(
    chain_dir: &Path,
    trace: &CheckpointTrace,
) -> Result<PathBuf> {
    let trace_path = chain_dir.join(CHECKPOINT_TRACE_JSON);
    write_json(&trace_path, trace)?;
    Ok(trace_path)
}

pub fn write_checkpoint_hashes_artifact(
    chain_dir: &Path,
    trace: &CheckpointTrace,
) -> Result<PathBuf> {
    let hashes_path = chain_dir.join(CHECKPOINTS_TXT);
    let mut output = checkpoint_hashes(trace)?.join("\n");
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

pub fn checkpoint_hashes(trace: &CheckpointTrace) -> Result<Vec<String>> {
    trace.checkpoints.iter().map(checkpoint_hash).collect()
}

pub fn checkpoint_hash(checkpoint: &Checkpoint) -> Result<String> {
    let encoded =
        serde_json::to_vec(checkpoint).context("failed to encode checkpoint for hashing")?;
    Ok(format!("{:x}", Sha256::digest(encoded)))
}

fn validate_checkpoint_hash(hash: &str, line_number: usize) -> Result<String> {
    if hash.len() != 64 || !hash.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        anyhow::bail!("invalid checkpoint hash on line {line_number}: {hash}");
    }
    Ok(hash.to_ascii_lowercase())
}

pub fn build_checkpoint_trace(chain_dir: &Path, _manifest_path: &Path) -> Result<CheckpointTrace> {
    let stage_order = read_stage_order(chain_dir)?.map(StageOrderIndex::from);

    let mut checkpoints = Vec::new();
    for entry in fs::read_dir(chain_dir)
        .with_context(|| format!("failed to read chain directory {}", chain_dir.display()))?
    {
        let entry =
            entry.with_context(|| format!("failed to read entry in {}", chain_dir.display()))?;
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

    Ok(CheckpointTrace { checkpoints })
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
    Ok(Checkpoint {
        stage: stage.to_string(),
        input_commitment: format!("{:x}", Sha256::digest(&input_manifest)),
        output_commitment: manifest.output.commitment,
        output_sha256: format!("{:x}", Sha256::digest(&output)),
    })
}

/// Read the run's recorded order independently of optional timing reports.
/// Current Raster writes an ordered chain commitment even in no-auth mode;
/// staged-native and older Raster runs record their order in execution times.
/// A malformed commitment is an error, never a reason to use another inventory.
pub fn read_stage_order(chain_dir: &Path) -> Result<Option<Vec<String>>> {
    let commitment_path = chain_dir.join("chain-commitment");
    if commitment_path.exists() {
        let bytes = fs::read(&commitment_path)
            .with_context(|| format!("failed to read {}", commitment_path.display()))?;
        let chain: ChainCommitment = postcard::from_bytes(&bytes)
            .with_context(|| format!("failed to parse {}", commitment_path.display()))?;
        return Ok(Some(
            chain.stages.into_iter().map(|stage| stage.name).collect(),
        ));
    }
    let times_path = chain_dir.join(EXECUTION_TIMES_JSON);
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
