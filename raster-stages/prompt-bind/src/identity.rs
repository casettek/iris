//! The tokenizer's commitment, rebuilt from the parts this stage reads, for
//! the model's descriptor to be held to.
use crate::input::*;
use model_identity::stream;
use raster::prelude::*;

#[tile(kind=iter)]
pub fn identity_hash_tokenizerprofile(value: TokenizerProfile) -> BytesPage {
    stream::digest(&value)
}
#[tile(kind=iter)]
pub fn identity_hash_u32(value: u32) -> BytesPage {
    stream::digest(&value)
}
#[tile(kind=iter)]
pub fn identity_component(
    vocab_bucket_count: BytesPage,
    merge_bucket_count: BytesPage,
    profile: BytesPage,
    vocab_buckets: BytesPage,
    merge_buckets: BytesPage,
) -> BytesPage {
    stream::structure(&[
        ("vocab_bucket_count", vocab_bucket_count),
        ("merge_bucket_count", merge_bucket_count),
        ("profile", profile),
        ("vocab_buckets", vocab_buckets),
        ("merge_buckets", merge_buckets),
    ])
}

#[tile(kind=iter)]
pub fn identity_sparse_root(header: SparseHeader) -> BytesPage {
    header.root
}
