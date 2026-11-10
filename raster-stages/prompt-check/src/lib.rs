//! `prompt-check`: tiles.
//!
//! Checks one range of the prompt against the claim's token marks and
//! publishes the range's token ids. It replaces `prompt-seed`, the
//! `prompt-merge` batches and `prompt-prepare` for a range. The logic is in
//! [`check`]; these tiles are its steps.
//!
//! A claim that is wrong is recorded in the context rather than raised: the
//! stage must finish so it has an output to compare and replay.
//!
//! `#![no_std]` so the same tiles compile into RISC0 replay guests.

#![no_std]

extern crate alloc;

use raster::prelude::*;

pub mod check;
pub mod input;

use check::*;
use input::*;

// ---------------------------------------------------------------------------
// The range
// ---------------------------------------------------------------------------

#[tile(kind = iter, description = "Join prompt model contexts without clearing faults")]
pub fn join_prompt_contexts(previous: PromptContext, bound: PromptContext) -> PromptContext {
    join_prompt_models(previous, &bound)
}

/// Records the one merge step the stage schedule puts between a range's seed
/// and its prepare. This stage is all three.
#[tile(kind = iter, description = "Record the range's merge step in the stage schedule")]
pub fn continue_range(mut context: PromptContext) -> PromptContext {
    context.model.continue_prompt_range();
    context
}

#[tile(kind = iter)]
pub fn empty_mark_count() -> MarkCount {
    MarkCount::default()
}

#[tile(kind = recur, description = "Count a chunk of the claim's marks")]
pub fn count_mark_chunk(
    input: RecurInput<Block<u32>>,
    state: RecurState<MarkCount>,
) -> RecurState<MarkCount> {
    let mut state = state;
    count_marks(&mut state, input.into_value().iter().count());
    state
}

#[tile(kind = iter, description = "Where the marked range ends")]
pub fn marked_end(context: PromptContext, marks: MarkCount) -> u64 {
    range_end(&context, &marks)
}

/// The fold over the prompt's pieces, for a range ending at `end`.
#[tile(kind = iter, description = "Open the fold over the prompt's pieces")]
pub fn begin_slice(context: PromptContext, end: u64) -> SliceState {
    begin_slice_state(&context, end)
}

/// Counts a chunk of the prompt's pieces.
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

#[tile(kind = iter, description = "Which pieces the range takes")]
pub fn plan_check_range(context: PromptContext, observed: SliceState) -> RangePlan {
    plan_range(&context, observed.end, observed.position)
}

/// Copies the previous range's last token, the first token's neighbour.
#[tile(kind = recur, description = "Copy the previous range's last token")]
pub fn copy_lead(
    input: RecurInput<Block<BpePiece>>,
    output: RecurOutput<CheckRange>,
    plan: RangePlan,
) -> RecurOutput<CheckRange> {
    let mut output = output;
    if plan.lead {
        for piece in input.into_value().iter() {
            output.pieces().push(lead_piece(piece));
        }
    }
    output
}

/// Copies the pieces of a chunk that fall inside this range, and ends the
/// list after the range's last piece.
#[tile(kind = recur, description = "Copy a chunk's pieces in this range")]
pub fn copy_range(
    input: RecurInput<Block<BpePiece>>,
    output: RecurOutput<CheckRange>,
    plan: RangePlan,
) -> RecurOutput<CheckRange> {
    let mut output = output;
    let first = input
        .index()
        .checked_mul(256)
        .expect("piece position overflow");
    for (offset, piece) in input.into_value().iter().enumerate() {
        let index = first
            .checked_add(offset as u64)
            .expect("piece position overflow");
        if let Some((piece, last)) = range_piece(piece, index, &plan) {
            output.pieces().push(piece);
            if last {
                output.pieces().push(end_range_piece());
            }
        }
    }
    output
}

// ---------------------------------------------------------------------------
// Numbering
// ---------------------------------------------------------------------------

#[tile(kind = iter)]
pub fn begin_numbering(plan: RangePlan) -> NumberState {
    begin_number_state(&plan)
}

#[tile(kind = iter, description = "Which mark a piece reads")]
pub fn begin_number(state: NumberState, piece: RangePiece) -> NumberStep {
    begin_number_step(state, piece)
}

#[tile(kind = iter, description = "Give a piece its token")]
pub fn number(step: NumberStep, mark: u32) -> Numbered {
    number_piece(step, mark)
}

#[tile(kind = iter)]
pub fn push_check_piece(output: Draft<CheckList>, numbered: Numbered) -> Draft<CheckList> {
    let mut output = output;
    output.pieces().push(numbered.piece);
    output
}

#[tile(kind = iter)]
pub fn numbered_state(numbered: Numbered) -> NumberState {
    numbered.state
}

// ---------------------------------------------------------------------------
// A round
// ---------------------------------------------------------------------------

#[tile(kind = iter)]
pub fn pass_a() -> u32 {
    0
}

#[tile(kind = iter)]
pub fn pass_b() -> u32 {
    1
}

#[tile(kind = iter)]
pub fn empty_window_scan() -> WindowScan {
    WindowScan::default()
}

#[tile(kind = iter, description = "The pair a piece forms with the piece before it")]
pub fn begin_window(scan: WindowScan, piece: CheckPiece, pass: u32) -> WindowStep {
    begin_window_step(scan, &piece, pass)
}

#[tile(kind = iter)]
pub fn merge_bucket_index(step: MergeStep, bucket_count: u32) -> u32 {
    assert!(bucket_count > 0, "empty merge bucket index");
    merge_bucket_of(&step.pending, &step.piece, bucket_count)
}

#[tile(kind = iter)]
pub fn empty_merge_match() -> MergeMatch {
    MergeMatch::default()
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

#[tile(kind = iter, description = "Append the best pair of a window that has ended")]
pub fn push_window_best(output: Draft<WindowBests>, step: WindowStep) -> Draft<WindowBests> {
    let mut output = output;
    if step.push {
        output.entries().push(step.closed);
    }
    output
}

#[tile(kind = iter)]
pub fn finish_window(step: WindowStep, hit: MergeMatch) -> WindowScan {
    finish_window_step(step, hit)
}

#[tile(kind = iter)]
pub fn empty_apply_state() -> ApplyState {
    ApplyState::default()
}

#[tile(kind = iter, description = "Which window's best pair a piece reads")]
pub fn begin_apply(state: ApplyState, piece: CheckPiece, pass: u32) -> ApplyStep {
    begin_apply_step(state, piece, pass)
}

#[tile(kind = iter, description = "Keep, merge or drop a piece")]
pub fn apply(step: ApplyStep, best: WindowBest) -> ApplyAction {
    apply_piece(step, best)
}

#[tile(kind = iter)]
pub fn push_applied_piece(output: Draft<CheckList>, action: ApplyAction) -> Draft<CheckList> {
    let mut output = output;
    if action.push {
        output.pieces().push(action.piece);
    }
    output
}

#[tile(kind = iter)]
pub fn applied_state(action: ApplyAction) -> ApplyState {
    action.state
}

// ---------------------------------------------------------------------------
// After a pass
// ---------------------------------------------------------------------------

#[tile(kind = iter)]
pub fn empty_scan_summary() -> ScanSummary {
    ScanSummary::default()
}

#[tile(kind = recur, description = "Whether any window could still merge")]
pub fn summarize_bests(
    input: RecurInput<Block<WindowBest>>,
    state: RecurState<ScanSummary>,
) -> RecurState<ScanSummary> {
    let mut state = state;
    for best in input.into_value().iter() {
        observe_best(&mut state, best);
    }
    state
}

#[tile(kind = iter)]
pub fn empty_pass_summary() -> PassSummary {
    PassSummary::default()
}

#[tile(kind = recur, description = "Whether every token is one piece")]
pub fn summarize_pass(
    input: RecurInput<Block<CheckPiece>>,
    state: RecurState<PassSummary>,
) -> RecurState<PassSummary> {
    let mut state = state;
    for piece in input.into_value().iter() {
        observe_final_piece(&mut state, piece);
    }
    state
}

/// The context once the range is checked. Faults are recorded, not raised.
#[tile(kind = iter, description = "Record the range's check in the context")]
pub fn finish_check(
    context: PromptContext,
    plan: RangePlan,
    pass_a: PassSummary,
    scan_a: ScanSummary,
    pass_b: PassSummary,
    scan_b: ScanSummary,
) -> PromptContext {
    finish_check_context(&context, &plan, &pass_a, &scan_a, &pass_b, &scan_b)
}

// ---------------------------------------------------------------------------
// The range's token ids
// ---------------------------------------------------------------------------

#[tile(kind = iter)]
pub fn open_token_pieces() -> Draft<MergeRound> {
    let mut output = Draft::<MergeRound>::new();
    output.model_fault().set(0);
    output
}

/// Copies the range's own tokens out of a pass's final pieces.
#[tile(kind = recur, description = "Copy the range's tokens")]
pub fn copy_token_pieces(
    input: RecurInput<Block<CheckPiece>>,
    output: RecurOutput<MergeRound>,
) -> RecurOutput<MergeRound> {
    let mut output = output;
    for piece in input.into_value().iter() {
        if is_range_token(piece) {
            output.pieces().push(BpePiece {
                text: piece.text.clone(),
                segment: piece.segment,
            });
        }
    }
    output
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

/// Every piece here is one of `prompt-bind`'s, which are vocabulary entries,
/// or a merge's result, which the import checks is one. A claim cannot make
/// this fail.
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

#[tile(kind = recur)]
pub fn bind_prompt_token(
    input: RecurInput<u32>,
    state: RecurState<PromptContext>,
) -> RecurState<PromptContext> {
    let mut state = state;
    state.model.append_consumed_token(input.into_value());
    state
}

#[tile(kind = iter)]
pub fn finish_prompt_binding(mut context: PromptContext) -> PromptContext {
    context
        .model
        .finish_prompt_range(context.complete, context.prompt_tokens, context.fault);
    context
}

#[tile(kind = recur)]
pub fn copy_bound_token(
    input: RecurInput<u32>,
    output: RecurOutput<PromptState>,
) -> RecurOutput<PromptState> {
    let mut output = output;
    output.token_ids().push(input.into_value());
    output
}

/// Publishes the range's last token as the pieces it was claimed from, for
/// the next range to pair its first token with.
#[tile(kind = recur, description = "Carry the range's last token to the next range")]
pub fn carry_last_token(
    input: RecurInput<Block<CheckPiece>>,
    output: RecurOutput<PromptState>,
    last_token: u32,
) -> RecurOutput<PromptState> {
    let mut output = output;
    for piece in input.into_value().iter() {
        if is_last_token(piece, last_token) {
            output.pieces().push(BpePiece {
                text: piece.text.clone(),
                segment: piece.segment,
            });
        }
    }
    output
}

pub mod identity;
pub use identity::*;
