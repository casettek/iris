//! Verify that one-page replays consume all rows with one iteration per page.
use prefill_range::input::*;
use prefill_range::*;
use raster::prelude::*;

#[path = "../../../tests/tile_chunk_support.rs"]
mod support;

#[sequence]
fn test_projection(weights: Bytes<196_608>, row: BytesPage, width: u32, rows: u32) -> ProjAccum {
    let pages = select!(List<BytesPage>, weights.pages);
    let acc = call!(zero_accum, rows);
    call_recur!(
        tile = mac_weight_page,
        input = pages,
        state = acc,
        args = (row, width)
    )
}

#[test]
fn one_page_tiles_cover_empty_short_full_and_odd_page_counts() {
    raster::init_with(support::Counter);
    let _scope = raster::__private::SequenceScopeGuard::enter("chunk_fixture");
    for rows in support::cases() {
        support::reset_count();
        let (weights, expected) = support::fixture(rows);
        let projected = raster::materialize_auth_return(__raster_sequence_auth_test_projection(
            weights,
            raster::into_auth_ref(pack_i32s(&vec![support::ONE; support::WIDTH as usize])),
            raster::into_auth_ref(support::WIDTH),
            raster::into_auth_ref(rows as u32),
        ));
        assert!(projected.error.is_empty(), "{}", projected.error);
        assert_eq!(unpack_i32s(&projected.values).unwrap(), expected);
        let pages = rows.div_ceil(support::ROWS_PER_PAGE);
        assert_eq!(support::iteration_count(), pages);
    }
}
