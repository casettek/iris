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

#[sequence]
fn find_best(
    pieces: List<BpePiece>,
    merge_buckets: List<MergeBucket>,
    count: u32,
    header: SparseHeader,
    paths: List<SparsePath>,
) -> MergeCandidate {
    let empty = call!(empty_merge_scan);
    let scan = call_recur_seq!(
        sequence = scan_pair,
        input = pieces,
        state = empty,
        args = (merge_buckets, count, header, paths)
    );
    select!(MergeCandidate, scan.best)
}
#[sequence]
fn merge_once(
    pieces: List<BpePiece>,
    buckets: List<MergeBucket>,
    count: u32,
    header: SparseHeader,
    paths: List<SparsePath>,
    fault: u32,
) -> MergeRound {
    let best = call_seq!(find_best, pieces.clone(), buckets, count, header, paths);
    let opened = call!(identity_open_round, best.clone(), fault);
    call_recur!(
        tile = apply_ranked_merge,
        input = pieces,
        chunk = 256,
        output = opened,
        args = (best,)
    )
}
#[sequence]
fn merge_eight(
    pieces: List<BpePiece>,
    buckets: List<MergeBucket>,
    count: u32,
    header: SparseHeader,
    paths: List<SparsePath>,
    fault: u32,
) -> MergeRound {
    let round1 = call_seq!(
        merge_once,
        pieces,
        buckets.clone(),
        count.clone(),
        header.clone(),
        paths.clone(),
        fault
    );
    let fault1 = select!(u32, round1.clone().model_fault);
    let pieces1 = select!(List<BpePiece>, round1.pieces);
    let round2 = call_seq!(
        merge_once,
        pieces1,
        buckets.clone(),
        count.clone(),
        header.clone(),
        paths.clone(),
        fault1
    );
    let fault2 = select!(u32, round2.clone().model_fault);
    let pieces2 = select!(List<BpePiece>, round2.pieces);
    let round3 = call_seq!(
        merge_once,
        pieces2,
        buckets.clone(),
        count.clone(),
        header.clone(),
        paths.clone(),
        fault2
    );
    let fault3 = select!(u32, round3.clone().model_fault);
    let pieces3 = select!(List<BpePiece>, round3.pieces);
    let round4 = call_seq!(
        merge_once,
        pieces3,
        buckets.clone(),
        count.clone(),
        header.clone(),
        paths.clone(),
        fault3
    );
    let fault4 = select!(u32, round4.clone().model_fault);
    let pieces4 = select!(List<BpePiece>, round4.pieces);
    let round5 = call_seq!(
        merge_once,
        pieces4,
        buckets.clone(),
        count.clone(),
        header.clone(),
        paths.clone(),
        fault4
    );
    let fault5 = select!(u32, round5.clone().model_fault);
    let pieces5 = select!(List<BpePiece>, round5.pieces);
    let round6 = call_seq!(
        merge_once,
        pieces5,
        buckets.clone(),
        count.clone(),
        header.clone(),
        paths.clone(),
        fault5
    );
    let fault6 = select!(u32, round6.clone().model_fault);
    let pieces6 = select!(List<BpePiece>, round6.pieces);
    let round7 = call_seq!(
        merge_once,
        pieces6,
        buckets.clone(),
        count.clone(),
        header.clone(),
        paths.clone(),
        fault6
    );
    let fault7 = select!(u32, round7.clone().model_fault);
    let pieces7 = select!(List<BpePiece>, round7.pieces);
    call_seq!(merge_once, pieces7, buckets, count, header, paths, fault7)
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
    witness: TokenizerWitness,
    bound: PromptState,
    previous: PromptState,
    cut: RangeCut,
) -> PromptState {
    let identity = call_seq!(identity_root, tokenizer.clone(), witness.clone());
    let previous_context = select!(PromptContext, previous.context);
    let bound_context = select!(PromptContext, bound.clone().context);
    let context = call!(join_prompt_contexts, previous_context, bound_context);
    let context = call!(identity_prompt, context, identity);
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
    let empty_scan = call!(empty_merge_scan);
    let empty_best = select!(MergeCandidate, empty_scan.best);
    let zero_fault = call!(identity_zero_fault);
    let empty_range = call!(identity_open_round, empty_best, zero_fault);
    let range = call_recur!(
        tile = copy_slice,
        input = all,
        chunk = 256,
        output = empty_range,
        args = (start, stop)
    );

    let boundary_count = select!(u32, tokenizer.clone().boundary_bucket_count);
    let boundaries = select!(List<BoundaryBucket>, tokenizer.clone().boundary_buckets);
    let query = call!(boundary_query, observed.clone());
    let boundary_index = call!(boundary_bucket_index, query.clone(), boundary_count);
    let boundary_root = call_seq!(
        identity_boundarybucket_contents,
        boundaries.clone(),
        boundary_index.clone()
    );
    let boundary = select!(BoundaryBucket, boundaries[boundary_index]);
    let boundary_header = select!(SparseHeader, witness.clone().boundaries);
    let boundary_paths = select!(List<SparsePath>, witness.clone().boundary_paths);
    let boundary_proof = select!(SparsePath, boundary_paths[boundary_index]);
    let context = call!(
        identity_sparse_context,
        context,
        boundary_root,
        boundary_index,
        boundary_header,
        boundary_proof
    );
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
    let merge_header = select!(SparseHeader, witness.clone().merges);
    let merge_paths = select!(List<SparsePath>, witness.clone().merge_paths);
    let fault0 = call!(identity_zero_fault);
    let batch1 = call_seq!(
        merge_eight,
        pieces0,
        buckets.clone(),
        count.clone(),
        merge_header.clone(),
        merge_paths.clone(),
        fault0
    );
    let fault1 = select!(u32, batch1.clone().model_fault);
    let pieces1 = select!(List<BpePiece>, batch1.clone().pieces);
    let batch2 = call_seq!(
        merge_eight,
        pieces1,
        buckets.clone(),
        count.clone(),
        merge_header.clone(),
        merge_paths.clone(),
        fault1
    );
    let fault2 = select!(u32, batch2.clone().model_fault);
    let pieces2 = select!(List<BpePiece>, batch2.clone().pieces);
    let batch3 = call_seq!(
        merge_eight,
        pieces2,
        buckets.clone(),
        count.clone(),
        merge_header.clone(),
        merge_paths.clone(),
        fault2
    );
    let fault3 = select!(u32, batch3.clone().model_fault);
    let pieces3 = select!(List<BpePiece>, batch3.clone().pieces);
    let batch4 = call_seq!(
        merge_eight,
        pieces3,
        buckets.clone(),
        count.clone(),
        merge_header.clone(),
        merge_paths.clone(),
        fault3
    );
    let fault4 = select!(u32, batch4.clone().model_fault);
    let pieces4 = select!(List<BpePiece>, batch4.clone().pieces);
    let batch5 = call_seq!(
        merge_eight,
        pieces4,
        buckets.clone(),
        count.clone(),
        merge_header.clone(),
        merge_paths.clone(),
        fault4
    );
    let fault5 = select!(u32, batch5.clone().model_fault);
    let pieces5 = select!(List<BpePiece>, batch5.clone().pieces);
    let batch6 = call_seq!(
        merge_eight,
        pieces5,
        buckets.clone(),
        count.clone(),
        merge_header.clone(),
        merge_paths.clone(),
        fault5
    );
    let fault6 = select!(u32, batch6.clone().model_fault);
    let pieces6 = select!(List<BpePiece>, batch6.clone().pieces);
    let batch7 = call_seq!(
        merge_eight,
        pieces6,
        buckets.clone(),
        count.clone(),
        merge_header.clone(),
        merge_paths.clone(),
        fault6
    );
    let fault7 = select!(u32, batch7.clone().model_fault);
    let pieces7 = select!(List<BpePiece>, batch7.clone().pieces);
    let batch8 = call_seq!(
        merge_eight,
        pieces7,
        buckets,
        count,
        merge_header.clone(),
        merge_paths.clone(),
        fault7
    );
    let fault8 = select!(u32, batch8.clone().model_fault);
    let merged = select!(List<BpePiece>, batch8.clone().pieces);

    let context = call!(identity_round_fault, context, fault8);
    let opened = call!(open_pieces, context);
    call_recur!(
        tile = carry_pieces,
        input = merged,
        chunk = 256,
        output = opened,
        args = ()
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
