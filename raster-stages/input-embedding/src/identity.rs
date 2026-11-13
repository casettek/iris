use crate::input::*;
use model_identity::stream;
use raster::prelude::*;
#[tile(kind=iter)]
pub fn identity_embedding_context(
    prompt: ModelContext,
    previous: ModelContext,
    root: BytesPage,
    previous_rows: RowCount,
    prompt_count: u64,
) -> ModelContext {
    let mut model = ModelContext::embedding(&prompt, &previous);
    model.require(!previous.is_origin() || previous_rows.count == 0);
    model.require(!prompt.is_origin() || prompt_count == 0);
    stream::verify(model, &root, model_identity::Role::Embedding)
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

/// Header verification binds actual geometry and scalars to the statement.
#[tile(kind=iter)]
pub fn identity_sparse_header(
    hidden_size: u32,
    embedding_scale: i32,
    byte_len: u64,
    page_size: u64,
    header: SparseHeader,
) -> BytesPage {
    let values = stream::structure(&[
        ("byte_len", stream::digest(&byte_len)),
        ("page_size", stream::digest(&page_size)),
        ("pages", header.root),
    ]);
    stream::structure(&[
        ("hidden_size", stream::digest(&hidden_size)),
        ("embedding_scale", stream::digest(&embedding_scale)),
        ("values", values),
    ])
}
#[tile(kind=iter)]
pub fn identity_sparse_page(
    model: ModelContext,
    page: BytesPage,
    index: u64,
    header: SparseHeader,
    path: SparsePath,
) -> ModelContext {
    let mut model = model;
    model.require(model_identity::sparse::verify(
        &header.root,
        header.count,
        index,
        &path,
        &stream::digest(&page),
    ));
    model
}
