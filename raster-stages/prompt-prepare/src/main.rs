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

#[sequence]
fn find_best(
    pieces: List<BpePiece>,
    merge_buckets: List<MergeBucket>,
    count: u32,
) -> MergeCandidate {
    let empty = call!(empty_merge_scan);
    let scan = call_recur_seq!(
        sequence = scan_pair,
        input = pieces,
        state = empty,
        args = (merge_buckets, count)
    );
    select!(MergeCandidate, scan.best)
}

#[sequence(kind = recur)]
fn resolve_piece(
    input: RecurSequenceInput<BpePiece>,
    output: RecurSequenceOutput<PromptTokenization>,
    vocab_buckets: List<VocabBucket>,
    count: u32,
) -> RecurSequenceOutput<PromptTokenization> {
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
fn main(tokenizer: PromptTokenizer, merged_pieces: BpePieces) -> Result<PromptTokenization> {
    let merge_count = select!(u32, tokenizer.clone().merge_bucket_count);
    let vocab_count = select!(u32, tokenizer.clone().vocab_bucket_count);
    let merges = select!(List<MergeBucket>, tokenizer.clone().merge_buckets);
    let vocab = select!(List<VocabBucket>, tokenizer.vocab_buckets);
    let pieces = select!(List<BpePiece>, merged_pieces.pieces);
    let best = call_seq!(find_best, pieces.clone(), merges, merge_count);
    call!(assert_merges_converged, best)?;
    let tokens = call_recur_seq!(
        sequence = resolve_piece,
        input = pieces,
        output = new!(PromptTokenization),
        args = (vocab, vocab_count)
    );
    Ok(tokens)
}
