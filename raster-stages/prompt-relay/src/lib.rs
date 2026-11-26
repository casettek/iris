//! `prompt-relay`: tiles.
//!
//! States what a prompt that holds an earlier inference's output was built
//! from: hashes the recipe's text before the output, the output the earlier
//! inference's last check stated, and the recipe's text after it, and names
//! the claim record that check stated. The first `prompt-bind` of the later
//! inference holds the text the run gives it to this stage's record, as it
//! holds a run's text to the `prompt-source` record.
//!
//! The logic is in `relay`, shared with direct inference and the native
//! stage. A tile calls it, so it is inside the tile's image id.
//!
//! Nothing a run gives can stop a tile: an earlier inference whose record is
//! not valid, or a recipe part over its limit, sets a fault in the record.
//!
//! `#![no_std]` so the same tiles compile into RISC0 replay guests.

#![no_std]

extern crate alloc;

use alloc::string::String;
use raster::prelude::*;

pub mod check;
pub mod relay;
pub mod source;

use check::ClaimRecord;
use relay::*;
use source::SourceRecord;

// Each loop below starts from a value a tile returns, so its seed is a whole
// stored value: Raster ties a loop's first step to a seed only when it is one.

#[tile(kind = iter, description = "No part of the recipe hashed yet")]
pub fn relay_seed() -> RelayFold {
    begin_relay_fold()
}

/// Hashes one chunk of the recipe's text, before or after the output.
#[tile(kind = recur, description = "Hash one chunk of the recipe's text")]
pub fn fold_chunk(
    input: RecurInput<String>,
    state: RecurState<RelayFold>,
) -> RecurState<RelayFold> {
    let mut state = state;
    let chunk = input.into_value();
    *state = fold_part_chunk((*state).clone(), &chunk);
    state
}

#[tile(kind = iter, description = "Close the text before the output")]
pub fn end_before(fold: RelayFold) -> RelayFold {
    close_part(fold)
}

/// Adds the earlier output to the text, and names the claim record it is
/// the output of.
#[tile(kind = iter, description = "Hash the earlier output into the prompt text")]
pub fn take_output(fold: RelayFold, output: String, claim: ClaimRecord) -> RelayFold {
    fold_relayed(
        fold,
        &output,
        claim.statement.as_slice(),
        claim.output_text_sha256.as_slice(),
        claim.valid,
    )
}

/// Closes the text after the output and states the record.
#[tile(kind = iter, description = "State the relay's source record")]
pub fn close(fold: RelayFold) -> SourceRecord {
    close_relay(fold)
}
