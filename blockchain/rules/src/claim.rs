use std::path::{Path, PathBuf};

use anyhow::Result;
use serde::{Deserialize, Serialize};

use model_manifest::io::{read_json, write_json};
use procedure_definition::checkpoint::{
    build_checkpoint_list, write_checkpoint_hashes_artifact, CHECKPOINT_LIST_JSON,
};

pub const CLAIM_BUNDLE_JSON: &str = "claim_bundle.json";

/// Contract-shaped summary of a claimer's claim.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ClaimBundle {
    pub version: u32,
    pub checkpoint_list_path: PathBuf,
    pub input: ClaimEndpoint,
    pub output: ClaimEndpoint,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prepared_run_path: Option<PathBuf>,
}

pub fn read_claim_bundle(path: &Path) -> Result<ClaimBundle> {
    read_json(path)
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ClaimEndpoint {
    pub stage: String,
    pub commitment: String,
}

pub fn write_claim_artifacts(
    chain_dir: &Path,
    manifest_path: &Path,
) -> Result<(PathBuf, PathBuf, PathBuf)> {
    write_claim_artifacts_with_prepared_run(chain_dir, manifest_path, None)
}

pub fn write_claim_artifacts_with_prepared_run(
    chain_dir: &Path,
    manifest_path: &Path,
    prepared_run_path: Option<&Path>,
) -> Result<(PathBuf, PathBuf, PathBuf)> {
    let list = build_checkpoint_list(chain_dir, manifest_path)?;
    let list_path = chain_dir.join(CHECKPOINT_LIST_JSON);
    write_json(&list_path, &list)?;
    let hashes_path = write_checkpoint_hashes_artifact(chain_dir, &list)?;

    let first = list.checkpoints.first().ok_or_else(|| {
        anyhow::anyhow!("cannot build a claim bundle from an empty checkpoint list")
    })?;
    let last = list.checkpoints.last().ok_or_else(|| {
        anyhow::anyhow!("cannot build a claim bundle from an empty checkpoint list")
    })?;
    let bundle = ClaimBundle {
        version: 2,
        checkpoint_list_path: PathBuf::from(CHECKPOINT_LIST_JSON),
        input: ClaimEndpoint {
            stage: first.stage.clone(),
            commitment: first.input_commitment.clone(),
        },
        output: ClaimEndpoint {
            stage: last.stage.clone(),
            commitment: last.output_commitment.clone(),
        },
        prepared_run_path: prepared_run_path.map(Path::to_path_buf),
    };
    let bundle_path = chain_dir.join(CLAIM_BUNDLE_JSON);
    write_json(&bundle_path, &bundle)?;
    Ok((list_path, hashes_path, bundle_path))
}
