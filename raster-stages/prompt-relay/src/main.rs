//! `prompt-relay`: sequences, one stage per run of a procedure kind that
//! gives one inference's output to the next.
//!
//! ```text
//!   recipe.before ──recur──▶ text hash, recipe hash
//!   check.record.generated_text, check.claim ──▶ text hash, the claim record named
//!   recipe.after ──recur──▶ text hash, recipe hash ──▶ SourceRecord
//! ```

use prompt_relay::check::*;
use prompt_relay::relay::*;
use prompt_relay::source::SourceRecord;
use prompt_relay::*;
use raster::prelude::*;

/// Stage entrypoint. `check` is the last check of the inference whose output
/// is relayed, and `recipe` the text before and after it.
#[sequence]
fn main(check: DecodeEdge, recipe: RelayRecipe) -> SourceRecord {
    let before = select!(List<String>, recipe.clone().before);
    let after = select!(List<String>, recipe.after);
    let output = select!(String, check.clone().record.generated_text);
    let claim = select!(ClaimRecord, check.claim);

    let none = call!(relay_seed);
    let led = call_recur!(tile = fold_chunk, input = before, state = none, args = ());
    let led = call!(end_before, led);
    let relayed = call!(take_output, led, output, claim);
    let text = call_recur!(tile = fold_chunk, input = after, state = relayed, args = ());
    let record = call!(close, text);
    raster::println!("prompt relay → {:?}", record);
    record
}
