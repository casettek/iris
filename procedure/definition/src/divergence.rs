//! Where two runs of one chain first differ: the stage, and what differs in
//! its checkpoint.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::checkpoint::Checkpoint;

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
