//! `claim-result`: tiles.
//!
//! The chain's last stage. It compares the output text the last check stated
//! with the run's forms, and states the code of the first form the text is,
//! beside the claim record the check stated. Today that is all it does: it is
//! where an output is turned into what a claim is used for.
//!
//! The logic is in `forms`, shared with direct inference and the native
//! stage. A tile calls it, so it is inside the tile's image id.
//!
//! `#![no_std]` so the same tiles compile into RISC0 replay guests.

#![no_std]

extern crate alloc;

use alloc::string::String;
use raster::prelude::*;

pub mod forms;
pub mod input;

use forms::*;
use input::ClaimRecord;

// The loop starts from a value a tile returns, so its seed is a whole stored
// value: Raster ties a loop's first step to a seed only when it is one.
#[tile(kind = iter, description = "No form compared yet")]
pub fn forms_seed() -> FormMatch {
    begin_form_match()
}

/// Compares the output with one form, and chains the form into the hash of
/// the forms compared.
#[tile(kind = recur, description = "Compare the output with one form")]
pub fn match_one(
    input: RecurInput<Form>,
    state: RecurState<FormMatch>,
    output: String,
) -> RecurState<FormMatch> {
    let mut state = state;
    let form = input.into_value();
    *state = match_form((*state).clone(), &form, &output);
    state
}

/// States the result beside the claim record the check stated.
#[tile(kind = iter, description = "State the result record")]
pub fn state_result(matched: FormMatch, claim: ClaimRecord) -> ResultRecord {
    close_result(
        matched,
        claim.statement.as_slice(),
        claim.output_text_sha256.as_slice(),
        claim.valid,
    )
}
