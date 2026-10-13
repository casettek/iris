use std::time::Duration;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct InferenceResult {
    pub generated_token_count: u32,
    pub generated_token_ids: Vec<u32>,
    pub generated_token_ids_sha256: String,
    pub generated_text: String,
    pub stop_reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InferenceRunReport {
    pub result: InferenceResult,
    pub timings: InferenceTimings,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InferenceTimings {
    pub total_duration: Duration,
    pub stages: Vec<InferStageTiming>,
    pub aux_waves: Vec<InferAuxWaveTiming>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InferStageTiming {
    pub stage: String,
    pub routine: String,
    pub input_synthesis_duration: Duration,
    pub input_load_duration: Duration,
    pub kernel_duration: Duration,
    pub encode_duration: Duration,
    pub total_duration: Duration,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InferAuxWaveTiming {
    pub name: String,
    pub first_stage: String,
    pub last_stage: String,
    pub stage_count: usize,
    pub parallelism: usize,
    pub wall_duration: Duration,
    pub stage_duration_sum: Duration,
}
