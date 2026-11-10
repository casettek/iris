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
#[tile(kind=iter)]
pub fn identity_append(state: BytesPage, root: BytesPage) -> BytesPage {
    stream::append(&state, &root)
}
#[tile(kind=iter)]
pub fn identity_hash_tokenizerprofile(value: TokenizerProfile) -> BytesPage {
    stream::digest(&value)
}
#[tile(kind=iter)]
pub fn identity_hash_u32(value: u32) -> BytesPage {
    stream::digest(&value)
}
#[tile(kind=recur)]
pub fn identity_vocabbucket_entry(
    input: RecurInput<TokenEntry>,
    state: RecurState<BytesPage>,
) -> RecurState<BytesPage> {
    RecurState::new(stream::append(&state, &stream::digest(&input.into_value())))
}
#[tile(kind=iter)]
pub fn identity_vocabbucket_root(root: BytesPage) -> BytesPage {
    stream::structure(&[("entries", root)])
}
#[tile(kind=recur)]
pub fn identity_mergebucket_entry(
    input: RecurInput<BpeMerge>,
    state: RecurState<BytesPage>,
) -> RecurState<BytesPage> {
    RecurState::new(stream::append(&state, &stream::digest(&input.into_value())))
}
#[tile(kind=iter)]
pub fn identity_mergebucket_root(root: BytesPage) -> BytesPage {
    stream::structure(&[("rules", root)])
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
#[tile(kind=iter)]
pub fn identity_sparse_context(
    mut context: PromptContext,
    root: BytesPage,
    index: u32,
    header: SparseHeader,
    path: SparsePath,
) -> PromptContext {
    context.model.require(model_identity::sparse::verify(
        &header.root,
        header.count,
        u64::from(index),
        &path,
        &root,
    ));
    context
}
#[tile(kind=iter)]
pub fn identity_merge_scan(
    mut scan: MergeScan,
    root: BytesPage,
    index: u32,
    header: SparseHeader,
    path: SparsePath,
) -> MergeScan {
    scan.best.model_fault |= u32::from(!model_identity::sparse::verify(
        &header.root,
        header.count,
        u64::from(index),
        &path,
        &root,
    ));
    scan
}
#[tile(kind=iter)]
pub fn identity_round_fault(mut context: PromptContext, fault: u32) -> PromptContext {
    context.model.fault |= fault;
    context
}
#[tile(kind=iter)]
pub fn identity_zero_fault() -> u32 {
    0
}
#[tile(kind=iter)]
pub fn identity_open_round(best: MergeCandidate, prior: u32) -> Draft<MergeRound> {
    let mut output = Draft::<MergeRound>::new();
    output.model_fault().set(best.model_fault | prior);
    output
}

#[tile(kind=iter)]
pub fn identity_atom_query(atom: PromptAtom) -> VocabQuery {
    VocabQuery { piece: atom.text }
}
#[tile(kind=iter)]
pub fn identity_bound_context(
    fold: TextFold,
    config: RunConfig,
    descriptor: BytesPage,
    root: BytesPage,
) -> PromptContext {
    let mut context = open_prompt_context(&fold, &config);
    context.model = model_identity::ModelContext::bind(descriptor.as_slice(), &config.model_id);
    context.model = stream::verify(context.model, &root, model_identity::Role::Tokenizer);
    context.model.bind_statement(
        &context.input_text_sha256,
        context.raw_prompt,
        context.max_new_tokens,
        context.range_size,
    );
    context
}
#[tile(kind=iter)]
pub fn identity_open_bound(context: PromptContext) -> Draft<PromptState> {
    let mut output = Draft::<PromptState>::new();
    output.context().set(context);
    output
}
