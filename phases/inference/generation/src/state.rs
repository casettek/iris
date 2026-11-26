#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct NativeInferenceState {
    pub generated_token_ids: Vec<u32>,
}
