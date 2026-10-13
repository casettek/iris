//! Phase 1 — `prompt-prepare`: sequences.
//!
//! Orchestration only: these bodies name steps, select data, and bind results.
//! Both passes are recur sequences, because each element needs several tiles
//! *and* a scan of a second collection:
//!
//! ```text
//!   pieces ──recur seq──▶ [ pair · scan merge table · emit · advance ] ──▶ MergedPieces
//!   merged ──recur seq──▶ [ query · scan vocabulary · append        ] ──▶ PromptTokenization
//! ```
//!
//! The second collection (the merge table, the vocabulary) travels as a
//! recur-sequence arg: those are `AuthRef`s, never materialized, and each inner
//! `call_recur!` walks one of them a `Block` at a time with the current item as
//! a scalar arg.

use raster::prelude::*;

use prompt_prepare::input::*;
use prompt_prepare::*;

/// One prompt piece: pair it with the pending token, look the pair up in the
/// merge table, emit the finished token if the merge stopped, and carry the
/// result to the next piece.
#[sequence(kind = recur)]
fn merge_prompt_piece(
    input: RecurSequenceInput<String>,
    state: RecurSequenceState<MergeCursor>,
    output: RecurSequenceOutput<MergedPieces>,
    merge_buckets: List<MergeBucket>,
    merge_bucket_count: u32,
    terminator: String,
) -> (
    RecurSequenceState<MergeCursor>,
    RecurSequenceOutput<MergedPieces>,
) {
    let step = call!(begin_merge_step, state, input);

    // Hash the pair, then reach the one bucket it can be in. The index is an
    // authorized value, so this is a lookup the prover cannot steer; the scan
    // that follows walks a handful of rules instead of the whole table.
    let bucket_idx = call!(merge_bucket_index, step.clone(), merge_bucket_count);
    let bucket = select!(MergeBucket, merge_buckets[bucket_idx]);
    let rules = select!(List<BpeMerge>, bucket.rules);

    let hit_seed = call!(begin_merge_match);
    let hit = call_recur!(
        tile = scan_merge_rules,
        input = rules,
        state = hit_seed,
        args = (step.clone(),)
    );

    let output = call!(emit_merged_piece, output, step.clone(), hit.clone(), terminator);
    let state = call!(advance_merge_cursor, step, hit);

    (state, output)
}

/// One left-to-right merge pass over the pieces.
///
/// **BPE does not converge in a single pass.** A pass can merge `W`+`h` and
/// `a`+`t`, but the result `Wh`+`at` -> `What` is only available to the *next*
/// pass, because the pair it needs did not exist when the pass started. Running
/// this once is what made the chain tokenize "What" as two tokens while the
/// reference produced one.
///
/// Factored into its own sequence so `main` can apply it a fixed number of
/// times without repeating the body. Each application is a real recur over the
/// real pieces list — there is no fabricated round counter anywhere.
#[sequence]
fn merge_round(
    pieces: List<String>,
    merge_buckets: List<MergeBucket>,
    merge_bucket_count: u32,
    terminator: String,
) -> MergedPieces {
    let cursor = call!(begin_merge_cursor);
    call_recur_seq!(
        sequence = merge_prompt_piece,
        input = pieces,
        state = cursor,
        output,
        args = (merge_buckets, merge_bucket_count, terminator)
    )
}

/// Drops the terminator after the rounds have converged, so the vocabulary pass
/// never sees a sentinel that is in no vocabulary.
#[sequence(kind = recur)]
fn strip_terminator(
    input: RecurSequenceInput<String>,
    output: RecurSequenceOutput<MergedPieces>,
    terminator: String,
) -> RecurSequenceOutput<MergedPieces> {
    call!(append_unless_terminator, output, input, terminator)
}

/// Counts pairs that a further pass would still merge.
///
/// The fixed-point check: after the last round this must be zero, otherwise the
/// unroll below was too short for this prompt and the tokenization is truncated.
/// Reported as a committed error rather than silently accepted.
#[sequence(kind = recur)]
fn count_remaining_merges(
    input: RecurSequenceInput<String>,
    state: RecurSequenceState<MergeScan>,
    merge_buckets: List<MergeBucket>,
    merge_bucket_count: u32,
) -> RecurSequenceState<MergeScan> {
    let scan_step = call!(begin_scan_step, state, input);
    let step = select!(MergeStep, scan_step.clone().step);
    let bucket_idx = call!(scan_bucket_index, step.clone(), merge_bucket_count);
    let bucket = select!(MergeBucket, merge_buckets[bucket_idx]);
    let rules = select!(List<BpeMerge>, bucket.rules);
    let hit_seed = call!(begin_merge_match);
    let hit = call_recur!(
        tile = scan_merge_rules,
        input = rules,
        state = hit_seed,
        args = (step.clone(),)
    );
    call!(advance_scan, scan_step, hit)
}

/// One merged piece: resolve it against the vocabulary and append its id.
#[sequence(kind = recur)]
fn resolve_piece_token(
    input: RecurSequenceInput<String>,
    output: RecurSequenceOutput<PromptTokenization>,
    vocab_buckets: List<VocabBucket>,
    vocab_bucket_count: u32,
) -> RecurSequenceOutput<PromptTokenization> {
    let query = call!(begin_vocab_lookup, input);

    let bucket_idx = call!(vocab_bucket_index, query.clone(), vocab_bucket_count);
    let bucket = select!(VocabBucket, vocab_buckets[bucket_idx]);
    let entries = select!(List<TokenEntry>, bucket.entries);

    let hit_seed = call!(begin_vocab_match);
    let hit = call_recur!(
        tile = scan_vocab_chunk,
        input = entries,
        state = hit_seed,
        args = (query,)
    );

    call!(append_token_id, output, hit)
}

/// Phase 1 entrypoint.
///
/// `tokenizer` and `initial_pieces` are the committed entry arguments (bound by
/// name in `input.json` / `input_manifest.json`, and by the root `Raster.toml`
/// when the stage runs as part of the chain). The return value is the
/// authorized output the next chain stage consumes.
#[sequence]
fn main(tokenizer: PromptTokenizer, initial_pieces: BpePieces) -> Result<PromptTokenization> {
    let merge_bucket_count = select!(u32, tokenizer.clone().merge_bucket_count);
    let vocab_bucket_count = select!(u32, tokenizer.clone().vocab_bucket_count);
    let merge_buckets = select!(List<MergeBucket>, tokenizer.clone().merge_buckets);
    let vocab_buckets = select!(List<VocabBucket>, tokenizer.vocab_buckets);
    let pieces = select!(List<String>, initial_pieces.pieces);

    // BPE is a fixed point, not a single pass: a pass merges `W`+`h` and `a`+`t`,
    // and only the next one can merge the resulting `Wh`+`at` into `What`. The
    // CFS is linear, so the rounds are unrolled here rather than looped — the
    // count is a program constant, never a committed counter list.
    //
    // Each round is idempotent once nothing merges, so an over-long unroll is
    // harmless; a too-short one is caught by the convergence check below rather
    // than silently truncating the tokenization.
    let round1 = call_seq!(
        merge_round,
        pieces,
        merge_buckets.clone(),
        merge_bucket_count.clone(),
        "</w>".to_string()
    );
    let pieces1 = select!(List<String>, round1.pieces);
    let round2 = call_seq!(
        merge_round,
        pieces1,
        merge_buckets.clone(),
        merge_bucket_count.clone(),
        "</w>".to_string()
    );
    let pieces2 = select!(List<String>, round2.pieces);
    let round3 = call_seq!(
        merge_round,
        pieces2,
        merge_buckets.clone(),
        merge_bucket_count.clone(),
        "</w>".to_string()
    );
    let pieces3 = select!(List<String>, round3.pieces);
    let round4 = call_seq!(
        merge_round,
        pieces3,
        merge_buckets.clone(),
        merge_bucket_count.clone(),
        "</w>".to_string()
    );
    let pieces4 = select!(List<String>, round4.pieces);
    let round5 = call_seq!(
        merge_round,
        pieces4,
        merge_buckets.clone(),
        merge_bucket_count.clone(),
        "</w>".to_string()
    );
    let pieces5 = select!(List<String>, round5.pieces);
    let round6 = call_seq!(
        merge_round,
        pieces5,
        merge_buckets.clone(),
        merge_bucket_count.clone(),
        "</w>".to_string()
    );
    let pieces6 = select!(List<String>, round6.pieces);
    let round7 = call_seq!(
        merge_round,
        pieces6,
        merge_buckets.clone(),
        merge_bucket_count.clone(),
        "</w>".to_string()
    );
    let pieces7 = select!(List<String>, round7.pieces);
    let round8 = call_seq!(
        merge_round,
        pieces7,
        merge_buckets.clone(),
        merge_bucket_count.clone(),
        "</w>".to_string()
    );
    let pieces8 = select!(List<String>, round8.pieces);

    // Fixed-point check: nothing may still be mergeable.
    let scan_seed = call!(begin_merge_scan);
    let scan = call_recur_seq!(
        sequence = count_remaining_merges,
        input = pieces8.clone(),
        state = scan_seed,
        args = (merge_buckets, merge_bucket_count)
    );
    let remaining = select!(u32, scan.remaining);
    call!(assert_merges_converged, remaining)?;

    let stripped = call_recur_seq!(
        sequence = strip_terminator,
        input = pieces8,
        output,
        args = ("</w>".to_string(),)
    );
    let merged_pieces = select!(List<String>, stripped.pieces);

    let tokenization = call_recur_seq!(
        sequence = resolve_piece_token,
        input = merged_pieces,
        output,
        args = (vocab_buckets, vocab_bucket_count)
    );
    raster::println!("vocab pass → {:?}", tokenization);

    Ok(tokenization)
}
