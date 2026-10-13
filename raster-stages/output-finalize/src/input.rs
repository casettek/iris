//! Boundary and work types for generated-token finalization.

extern crate alloc;

use alloc::string::String;
use raster::{BytesPage, List};
use serde::{Deserialize, Serialize};

/// MUST match the output of `decode-select-token` and `decode-init`.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct DecodeEdge {
    pub has_selected: bool,
    pub decode_position: u32,
    pub token_id: u32,
    pub value: i32,
    pub generated_token_ids: List<u32>,
}

/// One dense decoder-table row. Its list position is its token id.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct DecoderToken {
    pub token: String,
    pub special: bool,
    /// In the model's `eos_token_id` set. The chain's decode loop is statically
    /// expanded to a fixed repeat count, so it cannot stop early; this is what
    /// lets the finalized output report where the answer actually ended.
    pub terminal: bool,
}

/// Committed tokenizer data needed by output finalization.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct DecoderTable {
    pub tokens: List<DecoderToken>,
}

/// Bounded loop state. `pending_bytes` only holds one byte-fallback run. The
/// text and JSON fields grow with the requested generation bound; the chain's
/// static repeat count pins that bound.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct FinalizeState {
    pub count: u32,
    pub json: String,
    pub text: String,
    pub pending_bytes: BytesPage,
    /// Set by the first terminal token. Everything after it is still counted
    /// and committed — it was generated — but it is not the model's answer, so
    /// it stops contributing to `text`.
    pub stopped: bool,
}

/// Final authorized inference output.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct GeneratedOutput {
    pub generated_token_count: u32,
    pub generated_token_ids: List<u32>,
    pub generated_token_ids_sha256: String,
    pub generated_text: String,
    pub stop_reason: String,
}
