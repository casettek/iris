//! Boundary type shared with `decode-select-token` and `output-finalize`.

extern crate alloc;

use raster::List;
use serde::{Deserialize, Serialize};

/// Empty before generation; one selected token plus the accumulated transcript
/// after each decode selection.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct DecodeEdge {
    pub has_selected: bool,
    pub decode_position: u32,
    pub token_id: u32,
    pub value: i32,
    pub generated_token_ids: List<u32>,
}
