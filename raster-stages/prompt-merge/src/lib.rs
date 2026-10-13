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
    output: RecurOutput<BpePieces>,
    best: MergeCandidate,
) -> RecurOutput<BpePieces> {
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
