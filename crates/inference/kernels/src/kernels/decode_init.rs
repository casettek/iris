use anyhow::Result;
use raster::List;
use sequence_check::input::DecodeEdge;

pub struct DecodeInitDirectInputs;

pub fn run_decode_init_direct(_inputs: DecodeInitDirectInputs) -> Result<DecodeEdge> {
    Ok(DecodeEdge {
        has_selected: false,
        decode_position: 0,
        token_id: 0,
        value: 0,
        claim_mismatches: 0,
        first_claim_mismatch_position: 0,
        check_fault: 0,
        check_fault_position: 0,
        generated_token_ids: List::new(),
    })
}
