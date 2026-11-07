#![no_std]
extern crate alloc;

use raster::prelude::*;
pub mod input;
use input::*;

#[tile(kind = iter)]
pub fn empty_merge_scan() -> MergeScan {
    MergeScan::default()
}

#[tile(kind = iter)]
pub fn empty_merge_match() -> MergeMatch {
    MergeMatch::default()
}

#[tile(kind = iter)]
pub fn begin_ranked_step(scan: MergeScan, piece: BpePiece) -> RankedStep {
    RankedStep {
        query: MergeStep {
            pending: scan.previous.text,
            has_pending: scan.has_previous && scan.previous.segment == piece.segment,
            piece: piece.text.clone(),
        },
        piece,
        position: scan.position,
        best: scan.best,
    }
}

#[tile(kind = iter)]
pub fn merge_bucket_index(step: MergeStep, bucket_count: u32) -> u32 {
    assert!(bucket_count > 0, "empty merge bucket index");
    merge_bucket_of(&step.pending, &step.piece, bucket_count)
}

#[tile(kind = recur)]
pub fn scan_merge_rules(
    input: RecurInput<BpeMerge>,
    state: RecurState<MergeMatch>,
    step: MergeStep,
) -> RecurState<MergeMatch> {
    let mut state = state;
    let rule = input.into_value();
    if step.has_pending
        && rule.left == step.pending
        && rule.right == step.piece
        && (!state.matched || rule.rank < state.rank)
    {
        state.matched = true;
        state.rank = rule.rank;
        state.merged = rule.merged;
    }
    state
}

#[tile(kind = iter)]
pub fn finish_ranked_step(step: RankedStep, hit: MergeMatch) -> MergeScan {
    let mut best = step.best;
    // The scan runs left to right; retaining equal ranks implements the tie break.
    if hit.matched && (!best.matched || hit.rank < best.rank) {
        best = MergeCandidate {
            model_fault: best.model_fault,
            matched: true,
            rank: hit.rank,
            left: step
                .position
                .checked_sub(1)
                .expect("merge without a left piece"),
            merged: hit.merged,
        };
    }
    MergeScan {
        previous: step.piece,
        has_previous: true,
        position: step
            .position
            .checked_add(1)
            .expect("piece position overflow"),
        best,
    }
}

/// The context once the range has been counted and checked for finished merges.
///
/// Pieces that could still merge are recorded rather than raised: the stage
/// must finish so it has an output to compare and replay, and such pieces are
/// all vocabulary tokens, so it can.
#[tile(kind = iter, description = "Record the range's tokens in the context")]
pub fn finish_tokens(context: PromptContext, best: MergeCandidate, tokens: u64) -> PromptContext {
    finish_token_context(&context, &best, tokens)
}

/// Opens the stage's output with the range's context.
#[tile(kind = iter, description = "Open the range's tokens with its context")]
pub fn open_tokens(context: PromptContext) -> Draft<PromptState> {
    let mut output = Draft::<PromptState>::new();
    output.context().set(context);
    output
}

#[tile(kind = iter)]
pub fn begin_vocab_lookup(piece: BpePiece) -> VocabQuery {
    VocabQuery { piece: piece.text }
}

#[tile(kind = iter)]
pub fn vocab_bucket_index(query: VocabQuery, count: u32) -> u32 {
    assert!(count > 0, "empty vocabulary bucket index");
    vocab_bucket_of(&query.piece, count)
}

#[tile(kind = iter)]
pub fn empty_vocab_match() -> VocabMatch {
    VocabMatch::default()
}

#[tile(kind = recur)]
pub fn scan_vocab_chunk(
    input: RecurInput<TokenEntry>,
    state: RecurState<VocabMatch>,
    query: VocabQuery,
) -> RecurState<VocabMatch> {
    let mut state = state;
    let entry = input.into_value();
    if entry.token == query.piece {
        state.found = true;
        state.token_id = entry.id;
    }
    state
}

#[tile(kind = iter)]
pub fn append_token_id(output: Draft<PromptState>, hit: VocabMatch) -> Draft<PromptState> {
    assert!(
        hit.found,
        "prepared or merged piece is absent from vocabulary"
    );
    let mut output = output;
    output.token_ids().push(hit.token_id);
    output
}

pub mod identity;
pub use identity::*;
