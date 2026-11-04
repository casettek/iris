//! `range-origin`: sequences.

use raster::prelude::*;

use range_origin::input::*;
use range_origin::*;

/// Publishes the empty sequence every range chain starts from.
#[sequence]
fn main() -> ActivationSequence {
    call!(empty_sequence)
}
