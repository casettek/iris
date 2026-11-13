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
pub fn identity_hash_plelayerparams(value: PleLayerParams) -> BytesPage {
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
    embeddings: BytesPage,
    projection: BytesPage,
) -> BytesPage {
    stream::structure(&[
        ("params", params),
        ("embeddings", embeddings),
        ("projection", projection),
    ])
}
#[tile(kind=iter)]
pub fn identity_verify(context: ModelContext, root: BytesPage, layer: u32) -> ModelContext {
    stream::verify(context, &root, model_identity::Role::Auxiliary(layer))
}

#[tile(kind=iter)]
pub fn identity_sparse_bytes(byte_len: u64, page_size: u64, header: SparseHeader) -> BytesPage {
    stream::structure(&[
        ("byte_len", stream::digest(&byte_len)),
        ("page_size", stream::digest(&page_size)),
        ("pages", header.root),
    ])
}
#[tile(kind=iter)]
pub fn identity_sparse_page(
    mut model: ModelContext,
    page: BytesPage,
    index: u64,
    header: SparseHeader,
    path: SparsePath,
) -> ModelContext {
    model.require(model_identity::sparse::verify(
        &header.root,
        header.count,
        index,
        &path,
        &stream::digest(&page),
    ));
    model
}
