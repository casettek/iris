//! `consensus-result`: tiles.
//!
//! The last stage of a consensus chain. It reads the outputs the two
//! inferences' last checks stated by the run's forms, and states each one's
//! code and whether the two agree, beside the claim record each check stated.
//!
//! The logic is in `consensus` and `forms`, shared with direct inference and
//! the native stage. A tile calls it, so it is inside the tile's image id.
//!
//! `#![no_std]` so the same tiles compile into RISC0 replay guests.

#![no_std]

extern crate alloc;

use alloc::string::String;
use raster::prelude::*;

pub mod consensus;
pub mod forms;
pub mod input;

use consensus::*;
use forms::Form;
use input::ClaimRecord;

// The loop starts from a value a tile returns, so its seed is a whole stored
// value: Raster ties a loop's first step to a seed only when it is one. The
// seed also holds what does not depend on the forms: whether the two outputs
// are one text, and whether the two inferences were given one input.
#[tile(kind = iter, description = "Compare the two outputs and the inputs they name")]
pub fn consensus_seed(
    first_output: String,
    second_output: String,
    first_input: String,
    first_source: String,
    second_input: String,
    second_source: String,
) -> ConsensusMatch {
    begin_consensus(
        &first_output,
        &second_output,
        same_input(&first_input, &first_source, &second_input, &second_source),
    )
}

/// Compares both outputs with one form, and chains the form into the hash of
/// the forms compared.
#[tile(kind = recur, description = "Compare both outputs with one form")]
pub fn match_pair(
    input: RecurInput<Form>,
    state: RecurState<ConsensusMatch>,
    first_output: String,
    second_output: String,
) -> RecurState<ConsensusMatch> {
    let mut state = state;
    let form = input.into_value();
    *state = match_both((*state).clone(), &form, &first_output, &second_output);
    state
}

/// States each inference's result and whether the two agree.
#[tile(kind = iter, description = "State the consensus record")]
pub fn state_consensus(
    matched: ConsensusMatch,
    first: ClaimRecord,
    second: ClaimRecord,
) -> ConsensusRecord {
    close_consensus(
        matched,
        Claimed {
            statement: first.statement.as_slice(),
            output_text_sha256: first.output_text_sha256.as_slice(),
            valid: first.valid,
        },
        Claimed {
            statement: second.statement.as_slice(),
            output_text_sha256: second.output_text_sha256.as_slice(),
            valid: second.valid,
        },
    )
}
