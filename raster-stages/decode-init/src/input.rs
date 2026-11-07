//! Boundary type shared with `sequence-check` and `output-finalize`.

pub use model_identity::ModelContext;
extern crate alloc;

use raster::List;
use serde::{Deserialize, Serialize};

/// The edge before the first claimed range: no selection and no transcript.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct DecodeEdge {
    pub model: ModelContext,
    pub has_selected: bool,
    pub decode_position: u32,
    pub token_id: u32,
    pub value: i32,
    /// The claim verdict, zero here: see `DecodeEdge` in `sequence-check`.
    pub claim_mismatches: u32,
    pub first_claim_mismatch_position: u32,
    /// Why a check could not stand behind this edge, zero here.
    pub check_fault: u32,
    /// Absolute sequence position of that fault, or zero when there is none.
    pub check_fault_position: u32,
    pub generated_token_ids: List<u32>,
}
