//! `prompt-check`: sequences, one stage per prompt range.
//!
//! ```text
//!   marks ──recur──▶ how many pieces the range covers
//!   previous.rest, bound.pieces ──recur──▶ how many pieces the range is given ──▶ its plan
//!   previous.pieces, previous.rest, bound.pieces ──recur──▶ the range's pieces, ending in an end piece
//!   pieces × marks ──recur seq──▶ every piece's token
//!   pass A, pass B: 64 × [ scan ──recur seq──▶ best pair per window
//!                          apply ──recur seq──▶ at most one merge per window ]
//!                   closing scan, and a fold over the pass's final pieces
//!   pass A's pieces ──▶ the range's tokens ──vocabulary──▶ token ids ──▶ PromptState
//!   previous.rest, bound.pieces ──recur──▶ the pieces past the range, for the next
//! ```

use prompt_check::check::*;
use prompt_check::input::*;
use prompt_check::*;
use raster::prelude::*;

/// Gives one piece its token: reads its mark and numbers it.
#[sequence(kind = recur)]
fn number_range_piece(
    input: RecurSequenceInput<RangePiece>,
    state: RecurSequenceState<NumberState>,
    output: RecurSequenceOutput<CheckList>,
    starts: List<u32>,
) -> (
    RecurSequenceState<NumberState>,
    RecurSequenceOutput<CheckList>,
) {
    let step = call!(begin_number, state, input);
    let index = select!(u64, step.clone().index);
    let mark = select!(u32, starts[index]);
    let numbered = call!(number, step, mark);
    let output = call!(push_check_piece, output, numbered.clone());
    let state = call!(numbered_state, numbered);
    (state, output)
}

/// One piece of a round's scan: looks up the pair it forms with the piece
/// before it, and appends the best pair of a window that has just ended.
#[sequence(kind = recur)]
fn scan_window_pair(
    input: RecurSequenceInput<CheckPiece>,
    state: RecurSequenceState<WindowScan>,
    output: RecurSequenceOutput<WindowBests>,
    merge_buckets: List<MergeBucket>,
    count: u32,
    header: SparseHeader,
    paths: List<SparsePath>,
    pass: u32,
) -> (
    RecurSequenceState<WindowScan>,
    RecurSequenceOutput<WindowBests>,
) {
    let step = call!(begin_window, state, input, pass);
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
    let output = call!(push_window_best, output, step.clone());
    let scan = call!(finish_window, step, hit);
    let state = call!(identity_window_scan, scan, root, index, header, proof);
    (state, output)
}

/// One piece of a round's apply: reads its window's best pair and is kept,
/// merged or dropped.
#[sequence(kind = recur)]
fn apply_window_piece(
    input: RecurSequenceInput<CheckPiece>,
    state: RecurSequenceState<ApplyState>,
    output: RecurSequenceOutput<CheckList>,
    bests: List<WindowBest>,
    pass: u32,
) -> (
    RecurSequenceState<ApplyState>,
    RecurSequenceOutput<CheckList>,
) {
    let step = call!(begin_apply, state, input, pass);
    let index = select!(u64, step.clone().index);
    let best = select!(WindowBest, bests[index]);
    let action = call!(apply, step, best);
    let output = call!(push_applied_piece, output, action.clone());
    let state = call!(applied_state, action);
    (state, output)
}

#[sequence]
fn scan_windows(
    pieces: List<CheckPiece>,
    buckets: List<MergeBucket>,
    count: u32,
    header: SparseHeader,
    paths: List<SparsePath>,
    pass: u32,
) -> WindowBests {
    let seed = call!(empty_window_scan);
    call_recur_seq!(
        sequence = scan_window_pair,
        input = pieces,
        state = seed,
        output,
        args = (buckets, count, header, paths, pass)
    )
}

/// One round: at most one merge in every window.
#[sequence]
fn check_round(
    pieces: List<CheckPiece>,
    buckets: List<MergeBucket>,
    count: u32,
    header: SparseHeader,
    paths: List<SparsePath>,
    pass: u32,
) -> CheckList {
    let bests = call_seq!(
        scan_windows,
        pieces.clone(),
        buckets,
        count,
        header,
        paths,
        pass.clone()
    );
    let entries = select!(List<WindowBest>, bests.entries);
    let seed = call!(empty_apply_state);
    call_recur_seq!(
        sequence = apply_window_piece,
        input = pieces,
        state = seed,
        output,
        args = (entries, pass)
    )
}

#[sequence]
fn check_eight(
    pieces: List<CheckPiece>,
    buckets: List<MergeBucket>,
    count: u32,
    header: SparseHeader,
    paths: List<SparsePath>,
    pass: u32,
) -> CheckList {
    let step1 = call_seq!(
        check_round,
        pieces,
        buckets.clone(),
        count.clone(),
        header.clone(),
        paths.clone(),
        pass.clone()
    );
    let pieces1 = select!(List<CheckPiece>, step1.pieces);
    let step2 = call_seq!(
        check_round,
        pieces1,
        buckets.clone(),
        count.clone(),
        header.clone(),
        paths.clone(),
        pass.clone()
    );
    let pieces2 = select!(List<CheckPiece>, step2.pieces);
    let step3 = call_seq!(
        check_round,
        pieces2,
        buckets.clone(),
        count.clone(),
        header.clone(),
        paths.clone(),
        pass.clone()
    );
    let pieces3 = select!(List<CheckPiece>, step3.pieces);
    let step4 = call_seq!(
        check_round,
        pieces3,
        buckets.clone(),
        count.clone(),
        header.clone(),
        paths.clone(),
        pass.clone()
    );
    let pieces4 = select!(List<CheckPiece>, step4.pieces);
    let step5 = call_seq!(
        check_round,
        pieces4,
        buckets.clone(),
        count.clone(),
        header.clone(),
        paths.clone(),
        pass.clone()
    );
    let pieces5 = select!(List<CheckPiece>, step5.pieces);
    let step6 = call_seq!(
        check_round,
        pieces5,
        buckets.clone(),
        count.clone(),
        header.clone(),
        paths.clone(),
        pass.clone()
    );
    let pieces6 = select!(List<CheckPiece>, step6.pieces);
    let step7 = call_seq!(
        check_round,
        pieces6,
        buckets.clone(),
        count.clone(),
        header.clone(),
        paths.clone(),
        pass.clone()
    );
    let pieces7 = select!(List<CheckPiece>, step7.pieces);
    call_seq!(check_round, pieces7, buckets, count, header, paths, pass)
}

/// One pass: 64 rounds. The count MUST match `check::ROUNDS_PER_PASS`.
#[sequence]
fn check_pass(
    pieces: List<CheckPiece>,
    buckets: List<MergeBucket>,
    count: u32,
    header: SparseHeader,
    paths: List<SparsePath>,
    pass: u32,
) -> CheckList {
    let step1 = call_seq!(
        check_eight,
        pieces,
        buckets.clone(),
        count.clone(),
        header.clone(),
        paths.clone(),
        pass.clone()
    );
    let pieces1 = select!(List<CheckPiece>, step1.pieces);
    let step2 = call_seq!(
        check_eight,
        pieces1,
        buckets.clone(),
        count.clone(),
        header.clone(),
        paths.clone(),
        pass.clone()
    );
    let pieces2 = select!(List<CheckPiece>, step2.pieces);
    let step3 = call_seq!(
        check_eight,
        pieces2,
        buckets.clone(),
        count.clone(),
        header.clone(),
        paths.clone(),
        pass.clone()
    );
    let pieces3 = select!(List<CheckPiece>, step3.pieces);
    let step4 = call_seq!(
        check_eight,
        pieces3,
        buckets.clone(),
        count.clone(),
        header.clone(),
        paths.clone(),
        pass.clone()
    );
    let pieces4 = select!(List<CheckPiece>, step4.pieces);
    let step5 = call_seq!(
        check_eight,
        pieces4,
        buckets.clone(),
        count.clone(),
        header.clone(),
        paths.clone(),
        pass.clone()
    );
    let pieces5 = select!(List<CheckPiece>, step5.pieces);
    let step6 = call_seq!(
        check_eight,
        pieces5,
        buckets.clone(),
        count.clone(),
        header.clone(),
        paths.clone(),
        pass.clone()
    );
    let pieces6 = select!(List<CheckPiece>, step6.pieces);
    let step7 = call_seq!(
        check_eight,
        pieces6,
        buckets.clone(),
        count.clone(),
        header.clone(),
        paths.clone(),
        pass.clone()
    );
    let pieces7 = select!(List<CheckPiece>, step7.pieces);
    call_seq!(check_eight, pieces7, buckets, count, header, paths, pass)
}

/// The scan after a pass's last round, which must find nothing to merge.
#[sequence]
fn closing_scan(
    pieces: List<CheckPiece>,
    buckets: List<MergeBucket>,
    count: u32,
    header: SparseHeader,
    paths: List<SparsePath>,
    pass: u32,
) -> ScanSummary {
    let bests = call_seq!(scan_windows, pieces, buckets, count, header, paths, pass);
    let entries = select!(List<WindowBest>, bests.entries);
    let none = call!(empty_scan_summary);
    call_recur!(
        tile = summarize_bests,
        input = entries,
        chunk = 256,
        state = none,
        args = ()
    )
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

#[sequence(kind = recur)]
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

/// Stage entrypoint for one range: `bound` is this range's bind, `previous`
/// the check of the range before or the empty state, and `marks` the claim
/// of where each of the range's tokens starts. The range's pieces are the
/// ones `previous` left, then `bound`'s.
#[sequence]
fn main(
    tokenizer: PromptTokenizer,
    witness: TokenizerWitness,
    bound: PromptState,
    previous: PromptState,
    marks: TokenMarks,
) -> PromptState {
    let identity = call_seq!(identity_root, tokenizer.clone(), witness.clone());
    let previous_context = select!(PromptContext, previous.clone().context);
    let bound_context = select!(PromptContext, bound.clone().context);
    let context = call!(join_prompt_contexts, previous_context, bound_context);
    let context = call!(identity_prompt, context, identity);

    // Which pieces the range takes.
    let starts = select!(List<u32>, marks.starts);
    let no_marks = call!(empty_mark_count);
    let marked = call_recur!(
        tile = count_mark_chunk,
        input = starts.clone(),
        chunk = 256,
        state = no_marks,
        args = ()
    );
    let end = call!(marked_end, context.clone(), marked);
    let rest = select!(List<BpePiece>, previous.clone().rest);
    let own = select!(List<BpePiece>, bound.clone().pieces);
    let slice = call!(begin_slice, context.clone(), end);
    let rest_base = select!(u64, slice.clone().position);
    let observed_rest = call_recur!(
        tile = fold_slice,
        input = rest.clone(),
        chunk = 256,
        state = slice,
        args = ()
    );
    let own_base = select!(u64, observed_rest.clone().position);
    let observed = call_recur!(
        tile = fold_slice,
        input = own.clone(),
        chunk = 256,
        state = observed_rest,
        args = ()
    );
    let plan = call!(plan_check_range, context.clone(), observed);

    // The range's pieces after the previous range's last token, numbered.
    let lead = select!(List<BpePiece>, previous.pieces);
    let with_lead = call_recur!(
        tile = copy_lead,
        input = lead,
        chunk = 256,
        output,
        args = (plan.clone(),)
    );
    let with_rest = call_recur!(
        tile = copy_range,
        input = rest.clone(),
        chunk = 256,
        output = with_lead,
        args = (plan.clone(), rest_base.clone())
    );
    let range = call_recur!(
        tile = copy_range,
        input = own.clone(),
        chunk = 256,
        output = with_rest,
        args = (plan.clone(), own_base.clone())
    );
    let range_pieces = select!(List<RangePiece>, range.pieces);
    let number_seed = call!(begin_numbering, plan.clone());
    let numbered = call_recur_seq!(
        sequence = number_range_piece,
        input = range_pieces,
        state = number_seed,
        output,
        args = (starts,)
    );
    let pieces = select!(List<CheckPiece>, numbered.pieces);

    let count = select!(u32, tokenizer.clone().merge_bucket_count);
    let buckets = select!(List<MergeBucket>, tokenizer.clone().merge_buckets);
    let header = select!(SparseHeader, witness.clone().merges);
    let paths = select!(List<SparsePath>, witness.clone().merge_paths);

    // Pass A pairs tokens (0,1), (2,3), …
    let first = call!(pass_a);
    let checked_a = call_seq!(
        check_pass,
        pieces.clone(),
        buckets.clone(),
        count.clone(),
        header.clone(),
        paths.clone(),
        first.clone()
    );
    let merged_a = select!(List<CheckPiece>, checked_a.pieces);
    let scan_a = call_seq!(
        closing_scan,
        merged_a.clone(),
        buckets.clone(),
        count.clone(),
        header.clone(),
        paths.clone(),
        first
    );
    let no_pass_a = call!(empty_pass_summary);
    let summary_a = call_recur!(
        tile = summarize_pass,
        input = merged_a.clone(),
        chunk = 256,
        state = no_pass_a,
        args = ()
    );

    // Pass B pairs tokens (1,2), (3,4), …
    let second = call!(pass_b);
    let checked_b = call_seq!(
        check_pass,
        pieces.clone(),
        buckets.clone(),
        count.clone(),
        header.clone(),
        paths.clone(),
        second.clone()
    );
    let merged_b = select!(List<CheckPiece>, checked_b.pieces);
    let scan_b = call_seq!(
        closing_scan,
        merged_b.clone(),
        buckets,
        count,
        header,
        paths,
        second
    );
    let no_pass_b = call!(empty_pass_summary);
    let summary_b = call_recur!(
        tile = summarize_pass,
        input = merged_b,
        chunk = 256,
        state = no_pass_b,
        args = ()
    );

    let bound_done = select!(bool, bound.bind.done);
    let context = call!(
        finish_check,
        context,
        plan.clone(),
        summary_a.clone(),
        scan_a,
        summary_b,
        scan_b,
        bound_done
    );

    // The range's tokens, as pass A left them, resolved to ids.
    let opened_tokens = call!(open_token_pieces);
    let tokens = call_recur!(
        tile = copy_token_pieces,
        input = merged_a,
        chunk = 256,
        output = opened_tokens,
        args = ()
    );
    let token_pieces = select!(List<BpePiece>, tokens.pieces);
    let vocab_count = select!(u32, tokenizer.clone().vocab_bucket_count);
    let vocab = select!(List<VocabBucket>, tokenizer.vocab_buckets);
    let vocab_header = select!(SparseHeader, witness.clone().vocab);
    let vocab_paths = select!(List<SparsePath>, witness.vocab_paths);
    let context = call_recur_seq!(
        sequence = verify_piece,
        input = token_pieces.clone(),
        state = context,
        args = (
            vocab.clone(),
            vocab_count.clone(),
            vocab_header,
            vocab_paths
        )
    );
    let opened = call!(open_tokens, context.clone());
    let resolved = call_recur_seq!(
        sequence = resolve_piece,
        input = token_pieces,
        output = opened,
        args = (vocab, vocab_count)
    );
    let ids = select!(List<u32>, resolved.token_ids);
    let context = call_recur!(
        tile = bind_prompt_token,
        input = ids.clone(),
        state = context,
        args = ()
    );
    let context = call!(finish_prompt_binding, context);
    let opened = call!(open_tokens, context);
    let with_ids = call_recur!(
        tile = copy_bound_token,
        input = ids,
        output = opened,
        args = ()
    );
    let last_token = select!(u32, summary_a.last_token);
    let with_last = call_recur!(
        tile = carry_last_token,
        input = pieces,
        chunk = 256,
        output = with_ids,
        args = (last_token,)
    );
    // What the range was given and did not take goes to the next range.
    let with_old_rest = call_recur!(
        tile = carry_rest,
        input = rest,
        chunk = 256,
        output = with_last,
        args = (plan.clone(), rest_base)
    );
    call_recur!(
        tile = carry_rest,
        input = own,
        chunk = 256,
        output = with_old_rest,
        args = (plan, own_base)
    )
}

#[sequence]
fn identity_root(value: PromptTokenizer, witness: TokenizerWitness) -> BytesPage {
    let vocab_bucket_count = select!(u32, value.clone().vocab_bucket_count);
    let vocab_bucket_count = call!(identity_hash_u32, vocab_bucket_count);
    let merge_bucket_count = select!(u32, value.clone().merge_bucket_count);
    let merge_bucket_count = call!(identity_hash_u32, merge_bucket_count);
    let profile = select!(TokenizerProfile, value.clone().profile);
    let profile = call!(identity_hash_tokenizerprofile, profile);
    let vocab_header = select!(SparseHeader, witness.clone().vocab);
    let vocab_buckets = call!(identity_sparse_root, vocab_header);
    let merge_header = select!(SparseHeader, witness.merges);
    let merge_buckets = call!(identity_sparse_root, merge_header);
    call!(
        identity_component,
        vocab_bucket_count,
        merge_bucket_count,
        profile,
        vocab_buckets,
        merge_buckets
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
