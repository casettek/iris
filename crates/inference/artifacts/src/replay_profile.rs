//! Local, measurement-only stage replay reports. These are not claim artifacts.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::{Checkpoint, PreparedRun};

pub const REPLAY_PROFILE_JSON: &str = "replay-profile.json";
pub const NATIVE_RANKING_JSON: &str = "native-ranking.json";

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReplayProfileStatus {
    Running,
    Complete,
    NoTiles,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfilePass {
    pub command: Vec<String>,
    pub stdout_path: PathBuf,
    pub stderr_path: PathBuf,
    pub run_dir: Option<PathBuf>,
    pub profile_path: Option<PathBuf>,
    pub command_succeeded: Option<bool>,
    pub output_matches_baseline: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NativeInvocation {
    pub invocation_index: u64,
    pub coordinates: Vec<i32>,
    pub total_duration_ns: u64,
    pub user_duration_ns: u64,
    pub raster_overhead_ns: u64,
    pub input_bytes: u64,
    pub output_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NativeTileProfile {
    pub tile_id: String,
    pub invocations: u64,
    pub total_duration_ns: u64,
    pub total_user_duration_ns: u64,
    pub total_raster_overhead_ns: u64,
    pub max_input_bytes: u64,
    pub max_output_bytes: u64,
    pub slowest_invocation: NativeInvocation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayTileMeasurement {
    pub tile_id: String,
    pub recorded_invocations: u64,
    pub profiled_invocations: u64,
    pub image_id: Option<String>,
    pub max_guest_cycles: Option<u64>,
    pub total_guest_cycles: u64,
    pub heaviest_exec_index: Option<u64>,
    pub heaviest_coordinates: Option<Vec<i32>>,
    #[serde(default)]
    pub transition_overhead: Option<ReplayTransitionOverhead>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayTransitionOverhead {
    pub guest_cycles: u64,
    pub sampled_exec_index: u64,
    pub sampled_coordinates: Vec<i32>,
    pub transition_image_id: String,
    pub context: String,
    pub context_version: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageReplayProfile {
    pub version: u32,
    pub kind: String,
    pub status: ReplayProfileStatus,
    pub error: Option<String>,
    /// Reaching Raster's invocation cap is successful, but not exhaustive.
    pub replay_complete: Option<bool>,
    pub replay_invocation_limit: Option<u64>,
    /// Guest-cycle measurements are the primary replay-profile result.
    pub replay_tiles: Vec<ReplayTileMeasurement>,
    pub prepared_run: PreparedRun,
    pub stage: String,
    pub project_dir: PathBuf,
    pub stage_dir: PathBuf,
    pub input_path: PathBuf,
    pub input_manifest_path: PathBuf,
    pub input_sha256: String,
    pub input_manifest_sha256: String,
    pub baseline: Checkpoint,
    pub top_n: usize,
    pub selection_rule: String,
    pub selected_tile_ids: Vec<String>,
    pub native: ProfilePass,
    pub replay: ProfilePass,
    /// Native selection timings are saved in a separate artifact.
    #[serde(default)]
    pub native_ranking_path: Option<PathBuf>,
    /// Retained during selection and accepted when reading version 1 reports.
    /// Version 2 writes these timings only to native-ranking.json.
    #[serde(default, skip_serializing)]
    pub native_tiles: Vec<NativeTileProfile>,
}
