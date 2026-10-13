//! Phase 1 — `prompt-prepare`: tiles.
//!
//! The first stage of the `raster-chain-inference` chain: it turns a committed
//! prompt (pre-split into pieces) plus a committed tokenizer fixture into
//! prompt token ids.
//!
//! Every function here is one replay unit. Each takes a couple of small scalar
//! values — never a collection — and returns a small value or an appended
//! draft. The loop structure lives in `main.rs`, where the sequences make it
//! visible in the CFS.
//!
//! `#![no_std]` so the same tiles compile into RISC0 replay guests.

#![no_std]

extern crate alloc;

use alloc::string::String;
use raster::prelude::*;

pub mod input;

// The glob also brings the derive-generated `…DraftExt` traits into scope,
// which is what makes `output.pieces()` / `output.token_ids()` available.
use input::*;

// ---------------------------------------------------------------------------
// Merge pass — greedy left-to-right piece merging
// ---------------------------------------------------------------------------

/// The merge pass's opening cursor: nothing pending.
///
/// A stored seed, so the pass's state chain starts at a committed value rather
/// than a literal nothing pins.
#[tile(kind = iter, description = "Open a merge pass with nothing pending")]
pub fn begin_merge_cursor() -> MergeCursor {
    MergeCursor {
        pending: String::new(),
        has_pending: false,
    }
}

/// Pairs the loop-carried cursor with the incoming piece.
///
/// This is also what materializes the recur-sequence's item exactly once: the
/// resulting `MergeStep` is a small scalar value the three following steps can
/// each read, so the piece never has to be re-selected.
#[tile(kind = iter, description = "Pair the pending token with the next piece")]
pub fn begin_merge_step(cursor: MergeCursor, piece: String) -> MergeStep {
    MergeStep {
        pending: cursor.pending,
        has_pending: cursor.has_pending,
        piece,
    }
}

/// Bucket the current `(pending, piece)` pair hashes to.
///
/// Returned as a `u32` so the sequence can use it as a `select!` index: an
/// authorized value, hence a lookup a prover cannot steer.
#[tile(kind = iter, description = "Bucket index for the current merge pair")]
pub fn merge_bucket_index(step: MergeStep, bucket_count: u32) -> u32 {
    merge_bucket_of(&step.pending, &step.piece, bucket_count)
}

/// A merge-rule scan's opening state: no rule matched.
///
/// A stored seed, so the fold's state chain starts at a committed value
/// rather than a literal nothing pins.
#[tile(kind = iter, description = "Open a merge-rule scan with no match")]
pub fn begin_merge_match() -> MergeMatch {
    MergeMatch {
        matched: false,
        rank: 0,
        merged: String::new(),
    }
}

/// Scans one merge rule for `(pending, piece)`, keeping the lowest-ranked match.
///
/// The `input` is the *bucket's* rule list — the handful of rules whose pair
/// hashes to the same slot — so a replay unit touches one rule and one piece.
/// Rules that share a bucket but not the pair are rejected by the equality
/// test, exactly as they were when this scanned the whole table.
#[tile(kind = recur, description = "Scan a chunk of merge rules for the current pair")]
pub fn scan_merge_rules(
    input: RecurInput<BpeMerge>,
    state: RecurState<MergeMatch>,
    step: MergeStep,
) -> RecurState<MergeMatch> {
    let mut state = state;
    if !step.has_pending {
        // Nothing accumulated yet (first piece): no pair to merge.
        return state;
    }

    let rule = input.into_value();
    let applies = rule.left == step.pending && rule.right == step.piece;
    if applies && (!state.matched || rule.rank < state.rank) {
        state.matched = true;
        state.rank = rule.rank;
        state.merged = rule.merged;
    }

    state
}

/// Emits the completed token when the incoming piece does not extend it.
///
/// The output grows by one piece at most per iteration — the draft protocol
/// pays for the increment instead of re-committing the whole list.
#[tile(kind = iter, description = "Append the finished token when the merge stops")]
pub fn emit_merged_piece(
    output: Draft<MergedPieces>,
    step: MergeStep,
    hit: MergeMatch,
    terminator: String,
) -> Draft<MergedPieces> {
    let mut output = output;
    if step.has_pending && !hit.matched {
        output.pieces().push(step.pending);
    }
    // The pass flushes a completed token only when the *next* piece fails to
    // extend it, so whatever is pending after the last piece is never emitted.
    // That is exactly how the terminator gets consumed — and it is why the
    // terminator has to be re-emitted here: without it, round 2 has no
    // sentinel and silently drops its own last token instead.
    if step.piece == terminator {
        output.pieces().push(step.piece);
    }
    output
}

/// Drops the terminator once the rounds have converged.
#[tile(kind = iter, description = "Append a merged piece unless it is the terminator")]
pub fn append_unless_terminator(
    output: Draft<MergedPieces>,
    piece: String,
    terminator: String,
) -> Draft<MergedPieces> {
    let mut output = output;
    if piece != terminator {
        output.pieces().push(piece);
    }
    output
}

/// Advances the cursor: extend the pending token on a hit, otherwise start a
/// new pending token at the incoming piece.
#[tile(kind = iter, description = "Carry the merged or restarted token to the next piece")]
pub fn advance_merge_cursor(step: MergeStep, hit: MergeMatch) -> MergeCursor {
    if hit.matched {
        MergeCursor {
            pending: hit.merged,
            has_pending: true,
        }
    } else {
        MergeCursor {
            pending: step.piece,
            has_pending: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Vocabulary pass — merged piece -> token id
// ---------------------------------------------------------------------------

/// Materializes one merged piece into a lookup query.
#[tile(kind = iter, description = "Turn a merged piece into a vocabulary query")]
pub fn begin_vocab_lookup(piece: String) -> VocabQuery {
    VocabQuery { piece }
}

/// Bucket the queried piece hashes to.
#[tile(kind = iter, description = "Bucket index for a vocabulary lookup")]
pub fn vocab_bucket_index(query: VocabQuery, bucket_count: u32) -> u32 {
    vocab_bucket_of(&query.piece, bucket_count)
}

/// A vocabulary scan's opening state: not found, so the UNK id.
///
/// A stored seed, so the fold's state chain starts at a committed value
/// rather than a literal nothing pins.
#[tile(kind = iter, description = "Open a vocabulary scan at the UNK fallback")]
pub fn begin_vocab_match() -> VocabMatch {
    VocabMatch {
        found: false,
        token_id: UNK_TOKEN_ID,
    }
}

/// Scans one vocabulary entry for the queried piece.
///
/// Same shape as [`scan_merge_rules`]: the iterated collection is the bucket's
/// entry list, the query is a scalar arg, and the fold state is two small
/// fields. A bucket the piece does not occur in — including an empty one —
/// leaves the initial `found: false`, which is the UNK path.
#[tile(kind = recur, description = "Scan a chunk of the vocabulary for one piece")]
pub fn scan_vocab_chunk(
    input: RecurInput<TokenEntry>,
    state: RecurState<VocabMatch>,
    query: VocabQuery,
) -> RecurState<VocabMatch> {
    let mut state = state;
    let entry = input.into_value();
    if !state.found && entry.token == query.piece {
        state.found = true;
        state.token_id = entry.id;
    }
    state
}

/// Appends the resolved token id to the tokenization being built.
///
/// A piece the vocabulary does not contain keeps the initial state's
/// [`UNK_TOKEN_ID`], which is the standard tokenizer behaviour and keeps the
/// pass total — a fallible step here could not be propagated out of a recur
/// sequence.
#[tile(kind = iter, description = "Append one resolved token id")]
pub fn append_token_id(
    output: Draft<PromptTokenization>,
    hit: VocabMatch,
) -> Draft<PromptTokenization> {
    let mut output = output;
    output.token_ids().push(hit.token_id);
    output
}

// ---------------------------------------------------------------------------
// Fixed-point check — is another merge pass still needed?
// ---------------------------------------------------------------------------

/// The fixed-point check's opening state: no previous piece, nothing counted.
#[tile(kind = iter, description = "Open the fixed-point check")]
pub fn begin_merge_scan() -> MergeScan {
    MergeScan {
        previous: String::new(),
        has_previous: false,
        remaining: 0,
    }
}

/// Pairs the previous piece with the incoming one, carrying the running count.
///
/// The count has to ride along: this tile consumes the loop state to build the
/// pair, so nothing later in the iteration could recover it otherwise.
#[tile(kind = iter, description = "Pair the previous piece with the next, for the fixed-point check")]
pub fn begin_scan_step(scan: MergeScan, piece: String) -> ScanStep {
    ScanStep {
        step: MergeStep {
            pending: scan.previous,
            has_pending: scan.has_previous,
            piece,
        },
        remaining: scan.remaining,
    }
}

/// Bucket index for the pair under inspection.
#[tile(kind = iter, description = "Bucket index for the fixed-point check")]
pub fn scan_bucket_index(step: MergeStep, bucket_count: u32) -> u32 {
    merge_bucket_of(&step.pending, &step.piece, bucket_count)
}

/// Counts a still-mergeable pair and slides the window forward.
#[tile(kind = iter, description = "Count a pair that a further pass would merge")]
pub fn advance_scan(scan_step: ScanStep, hit: MergeMatch) -> MergeScan {
    MergeScan {
        previous: scan_step.step.piece,
        has_previous: true,
        // Saturating: a prompt that somehow overflowed would still report
        // "not converged" rather than wrapping to zero and reading as done.
        remaining: scan_step
            .remaining
            .saturating_add(if hit.matched { 1 } else { 0 }),
    }
}

/// Rejects a tokenization that has not reached its fixed point.
#[tile(kind = iter, description = "Reject a tokenization that is still mergeable")]
pub fn assert_merges_converged(remaining: u32) -> Result<u32> {
    if remaining == 0 {
        Ok(remaining)
    } else {
        Err(alloc::format!(
            "{remaining} adjacent pair(s) would still merge after the last round; \
             the merge unroll in `main` is too short for this prompt"
        ))
    }
}
