use prompt_prepare::input::*;
use prompt_prepare::*;
use raster::prelude::*;

#[sequence(kind = recur)]
fn scan_pair(
    input: RecurSequenceInput<BpePiece>,
    state: RecurSequenceState<MergeScan>,
    merge_buckets: List<MergeBucket>,
    count: u32,
    header: SparseHeader,
    paths: List<SparsePath>,
) -> RecurSequenceState<MergeScan> {
    let step = call!(begin_ranked_step, state, input);
    let query = select!(MergeStep, step.clone().query);
    let index = call!(merge_bucket_index, query.clone(), count);
    let root = call_seq!(
        identity_mergebucket_contents,
        merge_buckets.clone(),
        index.clone()
    );
    let bucket = select!(MergeBucket, merge_buckets[index]);
    let proof = select!(SparsePath, paths[index]);
    let rules = select!(List<BpeMerge>, bucket.rules);
    let empty = call!(empty_merge_match);
    let hit = call_recur!(
        tile = scan_merge_rules,
        input = rules,
        state = empty,
        args = (query,)
    );
    let scan = call!(finish_ranked_step, step, hit);
    call!(identity_merge_scan, scan, root, index, header, proof)
}

#[sequence(kind = recur)]
fn resolve_piece(
    input: RecurSequenceInput<BpePiece>,
    output: RecurSequenceOutput<PromptState>,
    vocab_buckets: List<VocabBucket>,
    count: u32,
) -> RecurSequenceOutput<PromptState> {
    let query = call!(begin_vocab_lookup, input);
    let index = call!(vocab_bucket_index, query.clone(), count);
    let bucket = select!(VocabBucket, vocab_buckets[index]);
    let entries = select!(List<TokenEntry>, bucket.entries);
    let empty = call!(empty_vocab_match);
    let hit = call_recur!(
        tile = scan_vocab_chunk,
        input = entries,
        state = empty,
        args = (query,)
    );
    call!(append_token_id, output, hit)
}

#[sequence]
fn main(
    tokenizer: PromptTokenizer,
    witness: TokenizerWitness,
    merged_pieces: PromptState,
) -> PromptState {
    let identity = call_seq!(identity_root, tokenizer.clone(), witness.clone());
    let merge_count = select!(u32, tokenizer.clone().merge_bucket_count);
    let vocab_count = select!(u32, tokenizer.clone().vocab_bucket_count);
    let merges = select!(List<MergeBucket>, tokenizer.clone().merge_buckets);
    let vocab = select!(List<VocabBucket>, tokenizer.vocab_buckets);
    let context = select!(PromptContext, merged_pieces.clone().context);
    let context = call!(identity_prompt, context, identity);
    let pieces = select!(List<BpePiece>, merged_pieces.pieces);

    let merge_header = select!(SparseHeader, witness.clone().merges);
    let merge_paths = select!(List<SparsePath>, witness.clone().merge_paths);
    let empty = call!(empty_merge_scan);
    let scan = call_recur_seq!(
        sequence = scan_pair,
        input = pieces.clone(),
        state = empty,
        args = (merges, merge_count, merge_header, merge_paths)
    );
    let best = select!(MergeCandidate, scan.clone().best);
    let tokens = select!(u64, scan.position);
    let fault = select!(u32, best.clone().model_fault);
    let context = call!(identity_round_fault, context, fault);
    let context = call!(finish_tokens, context, best, tokens);

    let vocab_header = select!(SparseHeader, witness.clone().vocab);
    let vocab_paths = select!(List<SparsePath>, witness.vocab_paths);
    let context = call_recur_seq!(
        sequence = verify_piece,
        input = pieces.clone(),
        state = context,
        args = (
            vocab.clone(),
            vocab_count.clone(),
            vocab_header,
            vocab_paths
        )
    );
    let opened = call!(open_tokens, context);
    call_recur_seq!(
        sequence = resolve_piece,
        input = pieces,
        output = opened,
        args = (vocab, vocab_count)
    )
}

#[sequence]
fn identity_root(value: PromptTokenizer, witness: TokenizerWitness) -> BytesPage {
    let vocab_bucket_count = select!(u32, value.clone().vocab_bucket_count);
    let vocab_bucket_count = call!(identity_hash_u32, vocab_bucket_count);
    let merge_bucket_count = select!(u32, value.clone().merge_bucket_count);
    let merge_bucket_count = call!(identity_hash_u32, merge_bucket_count);
    let boundary_bucket_count = select!(u32, value.clone().boundary_bucket_count);
    let boundary_bucket_count = call!(identity_hash_u32, boundary_bucket_count);
    let profile = select!(TokenizerProfile, value.clone().profile);
    let profile = call!(identity_hash_tokenizerprofile, profile);
    let vocab_header = select!(SparseHeader, witness.clone().vocab);
    let vocab_buckets = call!(identity_sparse_root, vocab_header);
    let merge_header = select!(SparseHeader, witness.clone().merges);
    let merge_buckets = call!(identity_sparse_root, merge_header);
    let boundary_header = select!(SparseHeader, witness.boundaries);
    let boundary_buckets = call!(identity_sparse_root, boundary_header);
    call!(
        identity_component,
        vocab_bucket_count,
        merge_bucket_count,
        boundary_bucket_count,
        profile,
        vocab_buckets,
        merge_buckets,
        boundary_buckets
    )
}

#[sequence]
fn identity_vocabbucket_contents(buckets: List<VocabBucket>, index: u32) -> BytesPage {
    let input = select!(VocabBucket, buckets[index]);
    let entries = select!(List<TokenEntry>, input.entries);
    let seed = call!(identity_empty);
    let folded = call_recur!(
        tile = identity_vocabbucket_entry,
        input = entries,
        state = seed,
        args = ()
    );
    let root = call!(identity_finish, folded);
    let root = call!(identity_vocabbucket_root, root);
    root
}

#[sequence]
fn identity_mergebucket_contents(buckets: List<MergeBucket>, index: u32) -> BytesPage {
    let input = select!(MergeBucket, buckets[index]);
    let entries = select!(List<BpeMerge>, input.rules);
    let seed = call!(identity_empty);
    let folded = call_recur!(
        tile = identity_mergebucket_entry,
        input = entries,
        state = seed,
        args = ()
    );
    let root = call!(identity_finish, folded);
    let root = call!(identity_mergebucket_root, root);
    root
}

#[sequence]
fn identity_boundarybucket_contents(buckets: List<BoundaryBucket>, index: u32) -> BytesPage {
    let input = select!(BoundaryBucket, buckets[index]);
    let entries = select!(List<BoundaryPair>, input.pairs);
    let seed = call!(identity_empty);
    let folded = call_recur!(
        tile = identity_boundarybucket_entry,
        input = entries,
        state = seed,
        args = ()
    );
    let root = call!(identity_finish, folded);
    let root = call!(identity_boundarybucket_root, root);
    root
}

#[sequence(kind=recur)]
fn verify_piece(
    input: RecurSequenceInput<BpePiece>,
    state: RecurSequenceState<PromptContext>,
    buckets: List<VocabBucket>,
    count: u32,
    header: SparseHeader,
    paths: List<SparsePath>,
) -> RecurSequenceState<PromptContext> {
    let query = call!(begin_vocab_lookup, input);
    let index = call!(vocab_bucket_index, query, count);
    let root = call_seq!(identity_vocabbucket_contents, buckets, index.clone());
    let proof = select!(SparsePath, paths[index]);
    call!(identity_sparse_context, state, root, index, header, proof)
}
