//! Opens the storage-backed empty transcript used at the decode repeat edge.

#![no_std]

extern crate alloc;

use raster::prelude::*;

pub mod input;

use input::*;

/// Sets every scalar field on the empty edge. The token list stays empty until
/// the first `decode-select-token` stage appends to it.
#[tile(kind = iter, description = "Initialize an empty decode transcript")]
pub fn initialize_decode_edge(output: Draft<DecodeEdge>) -> Draft<DecodeEdge> {
    let mut output = output;
    output.has_selected().set(false);
    output.decode_position().set(0);
    output.token_id().set(0);
    output.value().set(0);
    output
}
