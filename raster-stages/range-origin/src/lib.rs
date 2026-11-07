//! `range-origin`: tiles.
//!
//! The empty activation sequence at position 0 that a ranged chain starts
//! from. A `[[chain.repeat]]` `first =` fallback must name a stage, so the first
//! range's embedding (its start position) and every layer's first range (its
//! inherited and donor K/V caches) bind here.
//!
//! `#![no_std]` so the same tiles compile into RISC0 replay guests.

#![no_std]

extern crate alloc;

use raster::prelude::*;

pub mod input;

use input::*;

/// No rows, no keys, nothing before position 0.
#[tile(kind = iter, description = "Open the empty sequence ranges start from")]
pub fn empty_sequence() -> Draft<ActivationSequence> {
    let mut output = Draft::<ActivationSequence>::new();
    output.model().set(ModelContext::default());
    output.start_position().set(0);
    output
}
