//! `consensus-result`: sequences, one stage per claim, after both
//! inferences' last checks.
//!
//! ```text
//!   first.record, second.record ──▶ one text? one input?
//!   forms ──recur──▶ the first form each text is, and a hash over the forms
//!   first.claim, second.claim ──▶ ConsensusRecord
//! ```

use consensus_result::consensus::*;
use consensus_result::forms::*;
use consensus_result::input::*;
use consensus_result::*;
use raster::prelude::*;

/// Stage entrypoint. `first` and `second` are the outputs of the two
/// inferences' last checks, and `forms` the texts the run's outputs are
/// compared with.
#[sequence]
fn main(first: DecodeEdge, second: DecodeEdge, forms: ResultForms) -> ConsensusRecord {
    let first_output = select!(String, first.clone().record.generated_text);
    let first_input = select!(String, first.clone().record.input_text_sha256);
    let first_source = select!(String, first.clone().record.source_sha256);
    let first_claim = select!(ClaimRecord, first.claim);
    let second_output = select!(String, second.clone().record.generated_text);
    let second_input = select!(String, second.clone().record.input_text_sha256);
    let second_source = select!(String, second.clone().record.source_sha256);
    let second_claim = select!(ClaimRecord, second.claim);
    let listed = select!(List<Form>, forms.forms);
    let compared = call!(
        consensus_seed,
        clone!(first_output),
        clone!(second_output),
        first_input,
        first_source,
        second_input,
        second_source
    );
    let matched = call_recur!(
        tile = match_pair,
        input = listed,
        state = compared,
        args = (first_output, second_output)
    );
    let record = call!(state_consensus, matched, first_claim, second_claim);
    raster::println!("consensus result → {:?}", record);
    record
}
