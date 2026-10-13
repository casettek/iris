#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DirectInferenceState {
    pub generated_token_ids: Vec<u32>,
}
