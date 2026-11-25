//! Where two runs of one chain first differ: the stage, and what differs in
//! its checkpoint.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::checkpoint::{Checkpoint, CheckpointList};

pub const DIVERGENCE_JSON: &str = "divergence.json";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Divergence {
    pub version: u32,
    pub checkpoint_index: usize,
    pub stage: String,
    pub reason: DivergenceReason,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub claimed_list_path: Option<PathBuf>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub claimed_checkpoint_hashes_path: Option<PathBuf>,
    pub recomputed_list_path: PathBuf,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub claimed_hash: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub recomputed_hash: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub claimed: Option<Checkpoint>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub recomputed: Option<Checkpoint>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DivergenceReason {
    StageName,
    /// The claim names another program for the stage than the one locked here.
    ProgramCommitment,
    InputCommitment,
    OutputCommitment,
    OutputSha256,
    CheckpointHash,
    MissingClaimedHash,
    MissingRecomputedHash,
    MissingClaimedCheckpoint,
    MissingRecomputedCheckpoint,
}

/// Where two runs of one chain first differ, comparing their checkpoints in
/// order: the stage's name, its program, its inputs, then its output.
pub fn locate_divergence(
    claimed_list_path: &Path,
    claimed: &CheckpointList,
    recomputed_list_path: &Path,
    recomputed: &CheckpointList,
) -> Option<Divergence> {
    let checkpoint_count = claimed.checkpoints.len().max(recomputed.checkpoints.len());
    for index in 0..checkpoint_count {
        let claimed_checkpoint = claimed.checkpoints.get(index);
        let recomputed_checkpoint = recomputed.checkpoints.get(index);
        let Some(reason) = divergence_reason(claimed_checkpoint, recomputed_checkpoint) else {
            continue;
        };
        let stage = claimed_checkpoint
            .or(recomputed_checkpoint)
            .map(|checkpoint| checkpoint.stage.clone())
            .unwrap_or_else(|| String::from("<unknown>"));
        return Some(Divergence {
            version: 2,
            checkpoint_index: index,
            stage,
            reason,
            claimed_list_path: Some(claimed_list_path.to_path_buf()),
            claimed_checkpoint_hashes_path: None,
            recomputed_list_path: recomputed_list_path.to_path_buf(),
            claimed_hash: None,
            recomputed_hash: None,
            claimed: claimed_checkpoint.cloned(),
            recomputed: recomputed_checkpoint.cloned(),
        });
    }
    None
}

fn divergence_reason(
    claimed: Option<&Checkpoint>,
    recomputed: Option<&Checkpoint>,
) -> Option<DivergenceReason> {
    match (claimed, recomputed) {
        (None, None) => None,
        (None, Some(_)) => Some(DivergenceReason::MissingClaimedCheckpoint),
        (Some(_), None) => Some(DivergenceReason::MissingRecomputedCheckpoint),
        (Some(claimed), Some(recomputed)) if claimed.stage != recomputed.stage => {
            Some(DivergenceReason::StageName)
        }
        (Some(claimed), Some(recomputed))
            if claimed.program_commitment != recomputed.program_commitment =>
        {
            Some(DivergenceReason::ProgramCommitment)
        }
        (Some(claimed), Some(recomputed))
            if claimed.input_commitment != recomputed.input_commitment =>
        {
            Some(DivergenceReason::InputCommitment)
        }
        (Some(claimed), Some(recomputed))
            if claimed.output_commitment != recomputed.output_commitment =>
        {
            Some(DivergenceReason::OutputCommitment)
        }
        (Some(claimed), Some(recomputed)) if claimed.output_sha256 != recomputed.output_sha256 => {
            Some(DivergenceReason::OutputSha256)
        }
        _ => None,
    }
}
