use anyhow::Result;
use decode_select_token::input::DecodeEdge;
use raster::List;

pub struct DecodeInitDirectInputs;

pub fn run_decode_init_direct(_inputs: DecodeInitDirectInputs) -> Result<DecodeEdge> {
    Ok(DecodeEdge {
        has_selected: false,
        decode_position: 0,
        token_id: 0,
        value: 0,
        generated_token_ids: List::new(),
    })
}
