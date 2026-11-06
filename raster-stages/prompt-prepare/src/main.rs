use prompt_prepare::input::*;
use prompt_prepare::*;
use raster::prelude::*;

#[sequence(kind = recur)]
fn scan_pair(
    input: RecurSequenceInput<BpePiece>,
    state: RecurSequenceState<MergeScan>,
    merge_buckets: List<MergeBucket>,
    count: u32,
) -> RecurSequenceState<MergeScan> {
    let step = call!(begin_ranked_step, state, input);
    let query = select!(MergeStep, step.clone().query);
    let index = call!(merge_bucket_index, query.clone(), count);
    let bucket = select!(MergeBucket, merge_buckets[index]);
    let rules = select!(List<BpeMerge>, bucket.rules);
    let empty = call!(empty_merge_match);
    let hit = call_recur!(
        tile = scan_merge_rules,
        input = rules,
        state = empty,
        args = (query,)
    );
    call!(finish_ranked_step, step, hit)
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
fn main(tokenizer: PromptTokenizer, merged_pieces: PromptState) -> PromptState {
    let merge_count = select!(u32, tokenizer.clone().merge_bucket_count);
    let vocab_count = select!(u32, tokenizer.clone().vocab_bucket_count);
    let merges = select!(List<MergeBucket>, tokenizer.clone().merge_buckets);
    let vocab = select!(List<VocabBucket>, tokenizer.vocab_buckets);
    let context = select!(PromptContext, merged_pieces.clone().context);
    let pieces = select!(List<BpePiece>, merged_pieces.pieces);

    let empty = call!(empty_merge_scan);
    let scan = call_recur_seq!(
        sequence = scan_pair,
        input = pieces.clone(),
        state = empty,
        args = (merges, merge_count)
    );
    let best = select!(MergeCandidate, scan.clone().best);
    let tokens = select!(u64, scan.position);
    let context = call!(finish_tokens, context, best, tokens);

    let opened = call!(open_tokens, context);
    call_recur_seq!(
        sequence = resolve_piece,
        input = pieces,
        output = opened,
        args = (vocab, vocab_count)
    )
}
