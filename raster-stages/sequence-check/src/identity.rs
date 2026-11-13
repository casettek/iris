use crate::input::*;
use model_identity::stream;
use raster::prelude::*;
#[tile(kind=iter)]
pub fn identity_empty() -> BytesPage {
    stream::empty()
}
#[tile(kind=iter)]
pub fn identity_finish(state: BytesPage) -> BytesPage {
    stream::finish(&state)
}
#[tile(kind=recur)]
pub fn identity_page(
    input: RecurInput<BytesPage>,
    state: RecurState<BytesPage>,
) -> RecurState<BytesPage> {
    RecurState::new(stream::append(&state, &stream::digest(&input.into_value())))
}
#[tile(kind=iter)]
pub fn identity_hash_finalheadparams(value: FinalHeadParams) -> BytesPage {
    stream::digest(&value)
}
#[tile(kind=iter)]
pub fn identity_bytes(byte_len: u64, page_size: u64, pages: BytesPage) -> BytesPage {
    stream::structure(&[
        ("byte_len", stream::digest(&byte_len)),
        ("page_size", stream::digest(&page_size)),
        ("pages", pages),
    ])
}
#[tile(kind=iter)]
pub fn identity_component(params: BytesPage, projection: BytesPage) -> BytesPage {
    stream::structure(&[("params", params), ("projection", projection)])
}
#[tile(kind=iter)]
pub fn identity_verify(context: ModelContext, root: BytesPage) -> ModelContext {
    stream::verify(context, &root, model_identity::Role::Head)
}

#[tile(kind=iter)]
pub fn identity_zero_count() -> u64 {
    0
}
#[tile(kind=recur)]
pub fn identity_count_tokens(
    input: RecurInput<Block<u32>>,
    state: RecurState<u64>,
) -> RecurState<u64> {
    RecurState::new(*state + input.into_value().len() as u64)
}
#[tile(kind=iter)]
pub fn identity_prior_guard(
    mut model: ModelContext,
    prior: ModelContext,
    count: u64,
) -> ModelContext {
    model.require(!prior.is_origin() || count == 0);
    model
}
