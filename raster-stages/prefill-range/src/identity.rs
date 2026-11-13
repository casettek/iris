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
pub fn identity_hash_layerparams(value: LayerParams) -> BytesPage {
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
pub fn identity_component(
    params: BytesPage,
    w_q: BytesPage,
    w_k: BytesPage,
    w_v: BytesPage,
    w_o: BytesPage,
    w_gate: BytesPage,
    w_up: BytesPage,
    w_down: BytesPage,
    ple_input_gate: BytesPage,
    ple_layer_projection: BytesPage,
) -> BytesPage {
    stream::structure(&[
        ("params", params),
        ("w_q", w_q),
        ("w_k", w_k),
        ("w_v", w_v),
        ("w_o", w_o),
        ("w_gate", w_gate),
        ("w_up", w_up),
        ("w_down", w_down),
        ("ple_input_gate", ple_input_gate),
        ("ple_layer_projection", ple_layer_projection),
    ])
}
#[tile(kind=iter)]
pub fn identity_verify(context: ModelContext, root: BytesPage, layer: u32) -> ModelContext {
    stream::verify(context, &root, model_identity::Role::Transformer(layer))
}

#[tile(kind=iter)]
pub fn identity_zero_count() -> u64 {
    0
}
#[tile(kind=recur)]
pub fn identity_count_keys(
    input: RecurInput<Block<KeyRow>>,
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
    model.require(
        !(prior.is_origin() || prior.producer == model_identity::context::EMBEDDING) || count == 0,
    );
    model
}
