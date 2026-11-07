use anyhow::Result;
use range_origin::input::ActivationSequence;
use raster::List;

/// The empty sequence at position 0 every ranged chain starts from.
pub fn run_range_origin_direct() -> Result<ActivationSequence> {
    Ok(ActivationSequence {
        model: Default::default(),
        rows: List::new(),
        errors: List::new(),
        kv: List::new(),
        start_position: 0,
    })
}
