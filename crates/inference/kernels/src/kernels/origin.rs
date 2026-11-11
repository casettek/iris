//! The empty values a chain starts from.
//!
//! The first range of a run has nothing before it: no sequence, no decode edge
//! and no prompt state. Each stage that would continue one is given the empty
//! value instead, and checks that it is empty. The import writes these as
//! committed constants (see `model-import`), so they are data and no stage
//! produces them.
use input_embedding::input::ActivationSequence;
use prompt_check::input::PromptState;
use raster::List;
use sequence_check::input::DecodeEdge;

/// No rows, no keys, nothing before position 0.
pub fn empty_sequence() -> ActivationSequence {
    ActivationSequence {
        model: Default::default(),
        rows: List::new(),
        errors: List::new(),
        kv: List::new(),
        start_position: 0,
    }
}

/// No token selected or checked yet, so none has disagreed.
pub fn empty_decode_edge() -> DecodeEdge {
    DecodeEdge {
        model: Default::default(),
        has_selected: false,
        decode_position: 0,
        token_id: 0,
        value: 0,
        claim_mismatches: 0,
        first_claim_mismatch_position: 0,
        check_fault: 0,
        check_fault_position: 0,
        generated_token_ids: List::new(),
    }
}

/// No pieces, no tokens, no context and no text bound yet.
pub fn empty_prompt() -> PromptState {
    PromptState::default()
}
