//! Decode initialization sequence.

use raster::prelude::*;

use decode_init::input::*;
use decode_init::*;

/// Produces the storage-backed fallback exported when the decode repeat count
/// is zero.
#[sequence]
fn main() -> DecodeEdge {
    call!(initialize_decode_edge)
}
