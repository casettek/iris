//! `prompt-seed`: sequences, one stage per prompt range.
//!
//! ```text
//!   bound.pieces ──recur──▶ count, pieces either side of the cut ─▶ boundary lookup ─▶ context
//!   bound.pieces ──recur──▶ this range's pieces ──▶ 64 merge rounds ──▶ PromptState
//! ```

use prompt_seed::input::*;
use prompt_seed::*;
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

/// Stage entrypoint for one range: `bound` is every piece of the prompt,
/// `previous` the stage whose context this range continues, `cut` where this
/// range ends.
///
/// The merge count MUST match `prompt-merge`, `run_prep::MERGES_PER_STAGE` and
/// the native kernel's `MERGES_PER_BATCH`.
#[sequence]
fn main(
    tokenizer: PromptTokenizer,
    bound: PromptState,
    previous: PromptState,
    cut: RangeCut,
) -> PromptState {
    let context = select!(PromptContext, previous.context);
    let end = select!(u64, cut.end);
    let all = select!(List<BpePiece>, bound.pieces);

    let plan = call!(begin_slice, context.clone(), end);
    let observed = call_recur!(
        tile = fold_slice,
        input = all.clone(),
        chunk = 256,
        state = plan,
        args = ()
    );
    let start = select!(u64, observed.clone().start);
    let stop = select!(u64, observed.clone().end);
    let range = call_recur!(
        tile = copy_slice,
        input = all,
        chunk = 256,
        output,
        args = (start, stop)
    );

    let boundary_count = select!(u32, tokenizer.clone().boundary_bucket_count);
    let boundaries = select!(List<BoundaryBucket>, tokenizer.clone().boundary_buckets);
    let query = call!(boundary_query, observed.clone());
    let boundary_index = call!(boundary_bucket_index, query.clone(), boundary_count);
    let boundary = select!(BoundaryBucket, boundaries[boundary_index]);
    let pairs = select!(List<BoundaryPair>, boundary.pairs);
    let nothing = call!(empty_boundary_hit);
    let crossing = call_recur!(
        tile = scan_boundary_pairs,
        input = pairs,
        state = nothing,
        args = (query,)
    );
    let context = call!(finish_slice, context, observed, crossing);

    let count = select!(u32, tokenizer.clone().merge_bucket_count);
    let buckets = select!(List<MergeBucket>, tokenizer.merge_buckets);
    let pieces0 = select!(List<BpePiece>, range.pieces);
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
