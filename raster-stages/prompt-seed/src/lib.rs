//! `prompt-seed`: tiles.
//!
//! Cuts one range out of the pieces `prompt-bind` derived, checks the cut, and
//! runs the range's first merge batch. Each range starts where the previous
//! prepared range's context says the last one ended, so ranges cannot be
//! omitted, repeated or reordered without showing in the context.
//!
//! A cut that is wrong is recorded in the context rather than raised: the stage
//! must finish so it has an output to compare and replay.
//!
//! `#![no_std]` so the same tiles compile into RISC0 replay guests.

#![no_std]

extern crate alloc;

use raster::prelude::*;

pub mod input;

use input::*;

/// The fold over the prompt's pieces, for a range ending at `end`.
#[tile(kind = iter, description = "Open the fold over the prompt's pieces")]
pub fn begin_slice(context: PromptContext, end: u64) -> SliceState {
    begin_slice_state(&context, end)
}

/// Counts a chunk of pieces and keeps the two either side of the cut.
#[tile(kind = recur, description = "Observe a chunk of the prompt's pieces")]
pub fn fold_slice(
    input: RecurInput<Block<BpePiece>>,
    state: RecurState<SliceState>,
) -> RecurState<SliceState> {
    let mut state = state;
    for piece in input.into_value().iter() {
        observe_piece(&mut state, piece);
    }
    state
}

/// Copies the pieces of a chunk that fall inside this range.
#[tile(kind = recur, description = "Copy a chunk's pieces in this range")]
pub fn copy_slice(
    input: RecurInput<Block<BpePiece>>,
    output: RecurOutput<MergeRound>,
    start: u64,
    end: u64,
) -> RecurOutput<MergeRound> {
    let mut output = output;
    let first = input
        .index()
        .checked_mul(256)
        .expect("piece position overflow");
    for (offset, piece) in input.into_value().iter().enumerate() {
        let index = first
            .checked_add(offset as u64)
            .expect("piece position overflow");
        if index >= start && index < end {
            output.pieces().push(piece.clone());
        }
    }
    output
}

/// The characters either side of the cut, if a merge could cross it at all.
#[tile(kind = iter, description = "The pair either side of the range's cut")]
pub fn boundary_query(state: SliceState) -> BoundaryQuery {
    boundary_query_of(&state)
}

#[tile(kind = iter, description = "Bucket of a boundary lookup")]
pub fn boundary_bucket_index(query: BoundaryQuery, count: u32) -> u32 {
    assert!(count > 0, "empty boundary bucket index");
    boundary_bucket_of(query.left, query.right, count)
}

#[tile(kind = iter, description = "Open a boundary lookup with nothing found")]
pub fn empty_boundary_hit() -> BoundaryHit {
    BoundaryHit::default()
}

#[tile(kind = recur, description = "Scan one boundary bucket")]
pub fn scan_boundary_pairs(
    input: RecurInput<BoundaryPair>,
    state: RecurState<BoundaryHit>,
    query: BoundaryQuery,
) -> RecurState<BoundaryHit> {
    let mut state = state;
    let pair = input.into_value();
    if query.active && pair.left == query.left && pair.right == query.right {
        state.found = true;
    }
    state
}

/// The context once the cut has been checked.
#[tile(kind = iter, description = "Record the range's cut in the context")]
pub fn finish_slice(
    context: PromptContext,
    state: SliceState,
    crossing: BoundaryHit,
) -> PromptContext {
    finish_slice_context(&context, &state, &crossing)
}

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

#[tile(kind = recur)]
pub fn apply_ranked_merge(
    input: RecurInput<Block<BpePiece>>,
    output: RecurOutput<MergeRound>,
    best: MergeCandidate,
) -> RecurOutput<MergeRound> {
    let mut output = output;
    let start = input
        .index()
        .checked_mul(256)
        .expect("piece position overflow");
    let chunk = input.into_value();
    for (offset, piece) in chunk.iter().enumerate() {
        let index = start
            .checked_add(offset as u64)
            .expect("piece position overflow");
        let mut piece = piece.clone();
        if best.matched && index == best.left {
            piece.text = best.merged.clone();
            output.pieces().push(piece);
        } else if !best.matched
            || index != best.left.checked_add(1).expect("merge position overflow")
        {
            output.pieces().push(piece);
        }
    }
    output
}

/// Opens the stage's output with the range's context.
#[tile(kind = iter, description = "Open the seeded range with its context")]
pub fn open_pieces(context: PromptContext) -> Draft<PromptState> {
    let mut output = Draft::<PromptState>::new();
    output.context().set(context);
    output
}

/// Copies the last merge round's pieces into the stage's output.
#[tile(kind = recur, description = "Carry merged pieces into the output")]
pub fn carry_pieces(
    input: RecurInput<Block<BpePiece>>,
    output: RecurOutput<PromptState>,
) -> RecurOutput<PromptState> {
    let mut output = output;
    for piece in input.into_value().iter() {
        output.pieces().push(piece.clone());
    }
    output
}

#[tile(kind = iter, description = "Join prompt model contexts without clearing faults")]
pub fn join_prompt_contexts(previous: PromptContext, bound: PromptContext) -> PromptContext {
    join_prompt_models(previous, &bound)
}

pub mod identity;
pub use identity::*;
