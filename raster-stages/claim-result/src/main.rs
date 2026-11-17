//! `claim-result`: sequences, one stage per claim, after the last check.
//!
//! ```text
//!   check.record.generated_text ─┐
//!   forms ──recur──▶ the first form the text is, and a hash over the forms
//!   check.claim ──▶ ResultRecord
//! ```

use claim_result::forms::*;
use claim_result::input::*;
use claim_result::*;
use raster::prelude::*;

/// Stage entrypoint. `check` is the last check's output, and `forms` the
/// texts the run's output is compared with.
#[sequence]
fn main(check: DecodeEdge, forms: ResultForms) -> ResultRecord {
    let output = select!(String, check.clone().record.generated_text);
    let claim = select!(ClaimRecord, check.claim);
    let listed = select!(List<Form>, forms.forms);
    let none = call!(forms_seed);
    let matched = call_recur!(
        tile = match_one,
        input = listed,
        state = none,
        args = (output,)
    );
    let record = call!(state_result, matched, claim);
    raster::println!("claim result → {:?}", record);
    record
}
