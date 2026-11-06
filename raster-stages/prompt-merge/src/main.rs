use prompt_merge::input::*;
use prompt_merge::*;
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

#[sequence]
fn merge_once(pieces: List<BpePiece>, buckets: List<MergeBucket>, count: u32) -> MergeRound {
    let best = call_seq!(find_best, pieces.clone(), buckets, count);
    call_recur!(
        tile = apply_ranked_merge,
        input = pieces,
        chunk = 256,
        output,
        args = (best,)
    )
}

/// Eight merge rounds, each its own global best-pair scan and application.
#[sequence]
fn merge_eight(pieces: List<BpePiece>, buckets: List<MergeBucket>, count: u32) -> MergeRound {
    let round1 = call_seq!(merge_once, pieces, buckets.clone(), count.clone());
    let pieces1 = select!(List<BpePiece>, round1.pieces);
    let round2 = call_seq!(merge_once, pieces1, buckets.clone(), count.clone());
    let pieces2 = select!(List<BpePiece>, round2.pieces);
    let round3 = call_seq!(merge_once, pieces2, buckets.clone(), count.clone());
    let pieces3 = select!(List<BpePiece>, round3.pieces);
    let round4 = call_seq!(merge_once, pieces3, buckets.clone(), count.clone());
    let pieces4 = select!(List<BpePiece>, round4.pieces);
    let round5 = call_seq!(merge_once, pieces4, buckets.clone(), count.clone());
    let pieces5 = select!(List<BpePiece>, round5.pieces);
    let round6 = call_seq!(merge_once, pieces5, buckets.clone(), count.clone());
    let pieces6 = select!(List<BpePiece>, round6.pieces);
    let round7 = call_seq!(merge_once, pieces6, buckets.clone(), count.clone());
    let pieces7 = select!(List<BpePiece>, round7.pieces);
    call_seq!(merge_once, pieces7, buckets, count)
}

/// One tokenizer batch: 64 merge rounds.
///
/// A batch scans only the pieces it is given, so when run preparation cuts the
/// prompt into ranges each round costs the range's length, not the prompt's,
/// and a batch can afford many rounds. The count MUST match
/// `run_prep::MERGES_PER_STAGE` and the native kernel's `MERGES_PER_BATCH`.
#[sequence]
fn main(tokenizer: PromptTokenizer, initial_pieces: PromptState) -> PromptState {
    let count = select!(u32, tokenizer.clone().merge_bucket_count);
    let buckets = select!(List<MergeBucket>, tokenizer.merge_buckets);
    let context = select!(PromptContext, initial_pieces.clone().context);
    let pieces0 = select!(List<BpePiece>, initial_pieces.pieces);
    let batch1 = call_seq!(merge_eight, pieces0, buckets.clone(), count.clone());
    let pieces1 = select!(List<BpePiece>, batch1.pieces);
    let batch2 = call_seq!(merge_eight, pieces1, buckets.clone(), count.clone());
    let pieces2 = select!(List<BpePiece>, batch2.pieces);
    let batch3 = call_seq!(merge_eight, pieces2, buckets.clone(), count.clone());
    let pieces3 = select!(List<BpePiece>, batch3.pieces);
    let batch4 = call_seq!(merge_eight, pieces3, buckets.clone(), count.clone());
    let pieces4 = select!(List<BpePiece>, batch4.pieces);
    let batch5 = call_seq!(merge_eight, pieces4, buckets.clone(), count.clone());
    let pieces5 = select!(List<BpePiece>, batch5.pieces);
    let batch6 = call_seq!(merge_eight, pieces5, buckets.clone(), count.clone());
    let pieces6 = select!(List<BpePiece>, batch6.pieces);
    let batch7 = call_seq!(merge_eight, pieces6, buckets.clone(), count.clone());
    let pieces7 = select!(List<BpePiece>, batch7.pieces);
    let batch8 = call_seq!(merge_eight, pieces7, buckets, count);
    let merged = select!(List<BpePiece>, batch8.pieces);

    let opened = call!(open_pieces, context);
    call_recur!(
        tile = carry_pieces,
        input = merged,
        chunk = 256,
        output = opened,
        args = ()
    )
}
