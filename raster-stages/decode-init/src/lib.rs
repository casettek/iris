//! Opens the storage-backed empty transcript used at the decode repeat edge.

#![no_std]

extern crate alloc;

use raster::prelude::*;

pub mod input;

use input::*;

/// Sets every scalar field on the empty edge. The token list stays empty until
/// the first `sequence-check` stage appends to it.
#[tile(kind = iter, description = "Initialize an empty decode transcript")]
pub fn initialize_decode_edge() -> Draft<DecodeEdge> {
    let mut output = Draft::<DecodeEdge>::new();
    output.model().set(ModelContext::default());
    output.has_selected().set(false);
    output.decode_position().set(0);
    output.token_id().set(0);
    output.value().set(0);
    // No claimed token has been checked, so none has disagreed. This is also
    // the verdict the first teacher-forced check accumulates onto.
    output.claim_mismatches().set(0);
    output.first_claim_mismatch_position().set(0);
    output.check_fault().set(0);
    output.check_fault_position().set(0);
    output
}
