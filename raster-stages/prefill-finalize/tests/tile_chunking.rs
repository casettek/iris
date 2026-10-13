//! Verify append order and offsets across a full pair and a short final chunk.
use prefill_finalize::input::*;
use prefill_finalize::*;
use raster::prelude::*;

#[path = "../../../tests/tile_chunk_support.rs"]
mod support;

#[sequence]
fn test_logits(
    weights: Bytes<196_608>,
    normalized: NormalizedPosition,
    params: FinalHeadParams,
    position: FinalPosition,
) -> PrefillLogits {
    let pages = select!(List<BytesPage>, weights.pages);
    let draft = call!(begin_logits, new!(PrefillLogits), position, 0u32);
    call_recur!(
        tile = project_logit_page,
        input = pages,
        chunk = 2,
        output = draft,
        args = (normalized, params)
    )
}

#[test]
fn two_page_tiles_preserve_every_logit_and_its_token_id() {
    raster::init_with(support::Counter);
    let _scope = raster::__private::SequenceScopeGuard::enter("chunk_fixture");
    for rows in support::cases() {
        support::reset_count();
        let (weights, expected) = support::fixture(rows);
        let values = pack_i32s(&vec![support::ONE; support::WIDTH as usize]);
        let logits = raster::materialize_auth_return(__raster_sequence_auth_test_logits(
            weights,
            raster::into_auth_ref(NormalizedPosition {
                values: values.clone(),
            }),
            raster::into_auth_ref(FinalHeadParams {
                hidden_size: support::WIDTH,
                norm_eps: 0,
                softcap: 0,
                norm_weights: values.clone(),
            }),
            raster::into_auth_ref(FinalPosition {
                found: true,
                token_id: 0,
                token_count: 1,
                values,
            }),
        ));
        assert!(logits.errors.is_empty());
        assert_eq!(logits.decode_position, 1);
        assert_eq!(logits.logits.len(), rows);
        for (id, (logit, value)) in logits.logits.iter().zip(expected).enumerate() {
            assert_eq!(logit.token_id, id as u32);
            assert_eq!(logit.value, value);
        }
        let pages = rows.div_ceil(support::ROWS_PER_PAGE);
        assert_eq!(support::iteration_count(), pages.div_ceil(2));
    }
}
