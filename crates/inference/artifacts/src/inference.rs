use std::time::Duration;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct InferenceResult {
    /// Required, without a legacy default: missing verification is not success.
    pub model_fault: u32,
    pub generated_token_count: u32,
    pub generated_token_ids: Vec<u32>,
    pub generated_token_ids_sha256: String,
    pub generated_text: String,
    pub stop_reason: String,
    /// `output-finalize`'s claim verdict: how many generated positions were
    /// given a claimed token that is not the greedy selection there. Only a
    /// teacher-forced chain is given claimed tokens; a teacher-forced claim
    /// stands only if this is zero.
    #[serde(default)]
    pub claim_mismatches: u32,
    /// Absolute sequence position of the first disagreement, or zero.
    #[serde(default)]
    pub first_claim_mismatch_position: u32,
    /// The teacher-forced check's fault and its position, or zero.
    pub check_fault: u32,
    pub check_fault_position: u32,
    /// SHA-256 of the exact UTF-8 bytes of `generated_text`, as `output-finalize`
    /// computed it. Distinct from `generated_token_ids_sha256`.
    pub output_text_sha256: String,
    /// What the transcript breaks (token budget, EOS), or zero.
    pub output_fault: u32,
    /// SHA-256 of the exact text the chain consumed.
    pub input_text_sha256: String,
    /// The committed model the chain was configured with.
    pub model_id: String,
    pub raw_prompt: bool,
    pub max_new_tokens: u32,
    pub range_size: u32,
    pub prompt_tokens: u32,
    /// The prompt ranges covered the whole prepared prompt, in order.
    pub prompt_complete: bool,
    /// What failed in prompt preparation and where, or zero.
    pub prompt_fault: u32,
    pub prompt_fault_position: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InferenceRunReport {
    /// Actual model input length after chat templating and tokenization.
    pub input_token_count: Option<usize>,
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

#[cfg(test)]
mod model_fault_contract_tests {
    use super::*;
    #[test]
    fn legacy_result_without_model_verdict_is_rejected() {
        let mut json = serde_json::to_value(InferenceResult::default()).unwrap();
        json.as_object_mut().unwrap().remove("model_fault");
        assert!(serde_json::from_value::<InferenceResult>(json).is_err());
    }
}
