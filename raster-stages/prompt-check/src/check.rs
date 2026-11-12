//! The claimed-token check: its types and every step's logic.
//!
//! The claim marks where each token of a range starts. The stage confirms the
//! marks are the ranked merge's own result without replaying the merge over
//! the range: it merges each neighbouring pair of tokens on its own, and a
//! pair that does not come back as exactly those two tokens is a fault. See
//! `docs/proposals/claimed-prompt-tokens.md` for why that is enough.
//!
//! Everything here is a plain function, called from this stage's tiles (so it
//! is inside their image ids and replays in the guest) and from the native
//! twin. There is one copy of the logic, so the two cannot disagree.
//!
//! How a range is checked:
//!
//! ```text
//!   marks ──fold──▶ how many pieces the range covers
//!   pieces left by the check before, this range's bind ──fold──▶ how many there are
//!   previous range's last token, the range's pieces ──copy──▶ list ending in an end piece
//!   list × marks ──number──▶ every piece's token
//!   numbered list ──fold──▶ how many tokens the claim makes, and each one's text
//!   pass A, pass B:  64 × [ scan: one merge lookup per piece, best pair per window
//!                           apply: at most one merge per window; a window with
//!                                  nothing to merge is finished and leaves the list ]
//!                    then: nothing left to merge, every token one piece
//! ```
//!
//! A window's pieces change only when one of its own pairs merges, so a
//! window whose scan finds no pair will never merge again. The apply step
//! checks it there, that each of its tokens has become one piece, and drops
//! its pieces from the rounds that follow. Most windows finish within a few
//! rounds, so most rounds run over little more than the end piece. A window
//! that the last round still merged is checked after the pass, as before.
//!
//! The range's tokens are read from the numbered pieces and not from a
//! pass's result, which no longer holds the finished windows: a token's text
//! is its pieces joined. When every check passes that is exactly the one
//! piece the merge made of them. When one fails the claim is refused, and a
//! joined text that is no vocabulary entry is one more way it fails.
//!
//! Tokens are paired (0,1), (2,3), … in pass A and (1,2), (3,4), … in pass B,
//! so every neighbouring pair shares a window in exactly one pass. Token 0 is
//! the previous range's last token, kept only as a left neighbour. A piece may
//! merge with the piece before it only inside one window and one segment.
//!
//! A recur that builds a list gives its caller the list and not its final
//! state, so what a pass has learned travels on the list's last element, the
//! end piece, which every list here ends with.

use alloc::string::String;
use raster::List;
use serde::{Deserialize, Serialize};

use crate::input::{
    record_fault, BpePiece, MergeCandidate, MergeMatch, MergeStep, PromptContext,
    FAULT_CUT_BEYOND_TEXT, FAULT_EMPTY_RANGE, FAULT_MERGES_INCOMPLETE, FAULT_NONE,
    FAULT_NOT_THE_MERGE, FAULT_RANGE_TOO_LARGE, FAULT_TOKEN_SPLIT,
};

/// Merge rounds per pass. A window of two tokens needs two merges fewer than
/// it has pieces, so this covers tokens of up to 33 pieces each. MUST match
/// the rounds `main` unrolls.
pub const ROUNDS_PER_PASS: usize = 64;

/// The previous range's last token, present only as a left neighbour.
pub const LEAD_TOKEN: u32 = 0;
/// The first token of the range. Later tokens count up from it.
pub const FIRST_TOKEN: u32 = 1;
/// The end piece's token. It shares a window with nothing.
pub const END_TOKEN: u32 = u32::MAX;

pub const KIND_LEAD: u32 = 0;
pub const KIND_RANGE: u32 = 1;
pub const KIND_END: u32 = 2;

/// The claim for one range: for each piece, non-zero where a token starts.
/// The first piece of a range always starts one.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct TokenMarks {
    pub starts: List<u32>,
}

/// What a list's end piece carries: the first fault found so far, and
/// whether a tokenizer lookup failed its proof.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct Carried {
    pub fault: u32,
    pub fault_position: u64,
    pub model_fault: u32,
}

impl Carried {
    /// Keeps the first fault.
    pub fn record(&mut self, fault: u32, position: u64) {
        if self.fault == FAULT_NONE {
            self.fault = fault;
            self.fault_position = position;
        }
    }

    /// `self` happened before `later`.
    pub fn then(&self, later: &Carried) -> Carried {
        let mut joined = self.clone();
        joined.record(later.fault, later.fault_position);
        joined.model_fault |= later.model_fault;
        joined
    }
}

/// How many pieces the marks cover.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct MarkCount {
    pub pieces: u64,
}

pub fn count_marks(count: &mut MarkCount, marks: usize) {
    count.pieces = count.pieces.saturating_add(marks as u64);
}

/// Where the range the marks describe would end.
pub fn range_end(context: &PromptContext, marks: &MarkCount) -> u64 {
    context.next_piece.saturating_add(marks.pieces)
}

/// Which pieces the range takes, counted from the start of the prompt. The
/// check is given the pieces from `start` up to `total`: what the check
/// before it left, then its own range's bind.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct RangePlan {
    pub start: u64,
    /// Where the marks say the range ends.
    pub end: u64,
    /// Where it does end: `end`, or the last piece it was given if that is sooner.
    pub stop: u64,
    pub total: u64,
    /// The previous range's last token is kept as the first token's neighbour.
    pub lead: bool,
}

pub fn plan_range(context: &PromptContext, end: u64, total: u64) -> RangePlan {
    let start = context.next_piece;
    let stop = end.min(total);
    RangePlan {
        start,
        end,
        stop,
        total,
        lead: stop > start && start > 0,
    }
}

/// A piece on its way into the check: where it came from.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct RangePiece {
    pub text: String,
    pub segment: u32,
    pub kind: u32,
}

/// The pieces a range's check runs over, ending in an end piece.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct CheckRange {
    pub pieces: List<RangePiece>,
}

pub fn lead_piece(piece: &BpePiece) -> RangePiece {
    RangePiece {
        text: piece.text.clone(),
        segment: piece.segment,
        kind: KIND_LEAD,
    }
}

/// The range piece at `index` of the prompt, and whether it is the range's
/// last, after which the end piece goes.
pub fn range_piece(piece: &BpePiece, index: u64, plan: &RangePlan) -> Option<(RangePiece, bool)> {
    (index >= plan.start && index < plan.stop).then(|| {
        (
            RangePiece {
                text: piece.text.clone(),
                segment: piece.segment,
                kind: KIND_RANGE,
            },
            index + 1 == plan.stop,
        )
    })
}

/// Whether the piece at `index` is past the range, and so left for the
/// next range's check.
pub fn is_rest_piece(index: u64, plan: &RangePlan) -> bool {
    index >= plan.stop
}

pub fn end_range_piece() -> RangePiece {
    RangePiece {
        text: String::new(),
        segment: u32::MAX,
        kind: KIND_END,
    }
}

/// One piece with the claimed token it belongs to.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct CheckPiece {
    pub text: String,
    pub segment: u32,
    pub token: u32,
    /// Meaningful on the end piece only.
    pub carried: Carried,
}

/// A range's pieces during a pass.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct CheckList {
    pub pieces: List<CheckPiece>,
}

// ---------------------------------------------------------------------------
// Numbering: which token each piece belongs to
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct NumberState {
    /// The prompt index of the range's first piece.
    pub start: u64,
    /// Range pieces numbered so far.
    pub position: u64,
    pub token: u32,
    pub segment: u32,
    pub carried: Carried,
}

pub fn begin_number_state(plan: &RangePlan) -> NumberState {
    NumberState {
        start: plan.start,
        ..NumberState::default()
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct NumberStep {
    /// Which mark to read. A range piece reads its own. The others read the
    /// first, which exists whenever the list has a piece, and ignore it.
    pub index: u64,
    pub piece: RangePiece,
    pub state: NumberState,
}

pub fn begin_number_step(state: NumberState, piece: RangePiece) -> NumberStep {
    NumberStep {
        index: if piece.kind == KIND_RANGE {
            state.position
        } else {
            0
        },
        piece,
        state,
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct Numbered {
    pub piece: CheckPiece,
    pub state: NumberState,
}

pub fn number_piece(step: NumberStep, mark: u32) -> Numbered {
    let NumberStep {
        piece, mut state, ..
    } = step;
    let token = match piece.kind {
        KIND_LEAD => LEAD_TOKEN,
        KIND_RANGE => {
            let starts = state.position == 0 || mark != 0;
            if starts {
                state.token = if state.position == 0 {
                    FIRST_TOKEN
                } else {
                    state.token.saturating_add(1).min(END_TOKEN - 1)
                };
            } else if piece.segment != state.segment {
                state.carried.record(
                    FAULT_TOKEN_SPLIT,
                    state.start.saturating_add(state.position),
                );
            }
            state.segment = piece.segment;
            state.position += 1;
            state.token
        }
        _ => END_TOKEN,
    };
    Numbered {
        piece: CheckPiece {
            carried: if token == END_TOKEN {
                state.carried.clone()
            } else {
                Carried::default()
            },
            text: piece.text,
            segment: piece.segment,
            token,
        },
        state,
    }
}

// ---------------------------------------------------------------------------
// What the marks claim: the range's tokens
// ---------------------------------------------------------------------------

/// The most pieces of one claimed token whose text is joined. A merge's
/// result is at most 33 characters, so a token of more pieces is not one;
/// its text stops growing here and its lookup fails as it should.
pub const MAX_TOKEN_PIECES: u32 = 64;

/// The range's tokens as the marks number them.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct ClaimSummary {
    pub started: bool,
    pub token: u32,
    /// The range's own tokens.
    pub tokens: u64,
    /// The highest token number seen, for the next range's neighbour.
    pub last_token: u32,
}

pub fn observe_claimed_piece(summary: &mut ClaimSummary, piece: &CheckPiece) {
    if piece.token == END_TOKEN {
        return;
    }
    if piece.token != LEAD_TOKEN && (!summary.started || piece.token != summary.token) {
        summary.tokens = summary.tokens.saturating_add(1);
        summary.last_token = piece.token;
    }
    summary.started = true;
    summary.token = piece.token;
}

/// The claimed token being read off the numbered pieces: its text so far.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct TokenJoin {
    pub started: bool,
    pub token: u32,
    pub segment: u32,
    pub text: String,
    pub pieces: u32,
}

/// Adds a numbered piece to the token being read. Returns the token the
/// piece ends, when that is one of the range's own: every list ends with
/// the end piece, which ends the last.
pub fn join_claimed_piece(join: &mut TokenJoin, piece: &CheckPiece) -> Option<BpePiece> {
    let begins = !join.started || join.token != piece.token;
    let ended = (join.started && begins && join.token != LEAD_TOKEN).then(|| BpePiece {
        text: core::mem::take(&mut join.text),
        segment: join.segment,
    });
    if begins {
        join.started = true;
        join.token = piece.token;
        join.segment = piece.segment;
        join.text.clear();
        join.pieces = 0;
    }
    if piece.token != END_TOKEN && join.pieces < MAX_TOKEN_PIECES {
        join.text.push_str(&piece.text);
        join.pieces += 1;
    }
    ended
}

// ---------------------------------------------------------------------------
// A round's scan: the best pair of every window
// ---------------------------------------------------------------------------

/// The window a token is in during a pass: pass 0 pairs tokens (0,1), (2,3),
/// … and pass 1 pairs (1,2), (3,4), …. The end piece's window is its own.
pub fn window_of(token: u32, pass: u32) -> u32 {
    if token == END_TOKEN {
        u32::MAX
    } else {
        token.saturating_add(pass & 1) / 2
    }
}

/// A window's best pair: the lowest rank, and of equal ranks the leftmost.
/// `left` counts pieces from the window's first.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct WindowBest {
    pub matched: bool,
    pub left: u64,
    pub merged: String,
    /// Whether any lookup of the scan so far failed its proof.
    pub model_fault: u32,
}

/// One entry per window in order, after an unused first entry that makes the
/// list non-empty whenever a piece exists.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct WindowBests {
    pub entries: List<WindowBest>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct WindowScan {
    pub started: bool,
    pub previous: String,
    pub previous_segment: u32,
    pub window: u32,
    /// Pieces of the current window seen so far.
    pub local: u64,
    pub best: MergeCandidate,
    pub model_fault: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct WindowStep {
    pub query: MergeStep,
    /// Append `closed` to the list of bests.
    pub push: bool,
    pub closed: WindowBest,
    pub scan: WindowScan,
}

pub fn begin_window_step(scan: WindowScan, piece: &CheckPiece, pass: u32) -> WindowStep {
    let window = window_of(piece.token, pass);
    let same = scan.started && scan.window == window;
    // The first piece appends the unused first entry. A piece that starts a
    // later window appends the best of the window it closes.
    let closed = if scan.started && !same {
        WindowBest {
            matched: scan.best.matched,
            left: scan.best.left,
            merged: scan.best.merged.clone(),
            model_fault: scan.model_fault,
        }
    } else {
        WindowBest::default()
    };
    let query = MergeStep {
        pending: scan.previous.clone(),
        has_pending: same && scan.previous_segment == piece.segment,
        piece: piece.text.clone(),
    };
    let mut next = scan;
    let push = !same;
    if !same {
        next.window = window;
        next.local = 0;
        next.best = MergeCandidate::default();
    }
    next.started = true;
    next.previous = piece.text.clone();
    next.previous_segment = piece.segment;
    WindowStep {
        query,
        push,
        closed,
        scan: next,
    }
}

pub fn finish_window_step(step: WindowStep, hit: MergeMatch) -> WindowScan {
    let mut scan = step.scan;
    // The scan runs left to right; keeping equal ranks is the tie break.
    if hit.matched && (!scan.best.matched || hit.rank < scan.best.rank) {
        scan.best = MergeCandidate {
            model_fault: 0,
            matched: true,
            rank: hit.rank,
            left: scan.local.saturating_sub(1),
            merged: hit.merged,
        };
    }
    scan.local = scan.local.saturating_add(1);
    scan
}

// ---------------------------------------------------------------------------
// A round's apply: at most one merge per window
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct ApplyState {
    pub started: bool,
    pub window: u32,
    /// Windows begun so far, which is also the current window's entry.
    pub ordinal: u64,
    pub local: u64,
    /// The token of the window's piece before this one: the last kept, or
    /// in a finished window the last dropped.
    pub token: u32,
    pub carried: Carried,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct ApplyStep {
    /// Which entry of the bests to read: the piece's window's, or the unused
    /// first entry for the end piece.
    pub index: u64,
    pub piece: CheckPiece,
    pub state: ApplyState,
}

pub fn begin_apply_step(state: ApplyState, piece: CheckPiece, pass: u32) -> ApplyStep {
    let mut state = state;
    let window = window_of(piece.token, pass);
    if !state.started || state.window != window {
        state.window = window;
        state.local = 0;
        if piece.token != END_TOKEN {
            state.ordinal = state.ordinal.saturating_add(1);
        }
    }
    state.started = true;
    ApplyStep {
        index: if piece.token == END_TOKEN {
            0
        } else {
            state.ordinal
        },
        piece,
        state,
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct ApplyAction {
    pub push: bool,
    pub piece: CheckPiece,
    pub state: ApplyState,
}

pub fn apply_piece(step: ApplyStep, best: WindowBest) -> ApplyAction {
    let ApplyStep {
        mut piece,
        mut state,
        ..
    } = step;
    if piece.token == END_TOKEN {
        // What earlier rounds carried came first.
        piece.carried = piece.carried.then(&state.carried);
        return ApplyAction {
            push: true,
            piece,
            state,
        };
    }
    state.carried.model_fault |= best.model_fault;
    let local = state.local;
    state.local = local.saturating_add(1);
    if !best.matched {
        // Nothing in this window merges, now or in any later round: it is
        // finished. Each of its tokens must have become one piece, and its
        // pieces leave the list.
        if local > 0 && piece.token == state.token {
            state
                .carried
                .record(FAULT_NOT_THE_MERGE, u64::from(piece.token));
        }
        state.token = piece.token;
        return ApplyAction {
            push: false,
            piece,
            state,
        };
    }
    if best.matched && local == best.left.saturating_add(1) {
        // The right piece of the window's merge. Its left piece took the
        // merged text; joining two tokens is what the check refuses.
        if piece.token != state.token {
            state
                .carried
                .record(FAULT_NOT_THE_MERGE, u64::from(piece.token));
        }
        return ApplyAction {
            push: false,
            piece,
            state,
        };
    }
    if best.matched && local == best.left {
        piece.text = best.merged;
    }
    state.token = piece.token;
    ApplyAction {
        push: true,
        piece,
        state,
    }
}

// ---------------------------------------------------------------------------
// After a pass
// ---------------------------------------------------------------------------

/// The last scan of a pass, which must find nothing left to merge.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct ScanSummary {
    pub matched: bool,
    pub model_fault: u32,
}

pub fn observe_best(summary: &mut ScanSummary, best: &WindowBest) {
    summary.matched |= best.matched;
    summary.model_fault |= best.model_fault;
}

/// What is left of a pass: the end piece, with what the rounds carried, and
/// the pieces of any window the last round still merged. Each token among
/// those must be exactly one piece, as each token of a finished window was
/// when it left.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct PassSummary {
    pub started: bool,
    pub token: u32,
    pub split: Carried,
    pub carried: Carried,
}

pub fn observe_final_piece(summary: &mut PassSummary, piece: &CheckPiece) {
    if piece.token == END_TOKEN {
        summary.carried = piece.carried.clone();
        return;
    }
    if summary.started && piece.token == summary.token {
        summary
            .split
            .record(FAULT_NOT_THE_MERGE, u64::from(piece.token));
    }
    summary.started = true;
    summary.token = piece.token;
}

/// The context once the range is checked: where the range ends, how many
/// tokens it adds, and the first thing found wrong with it. `bound_done`
/// is whether the range's bind has seen the text end as claimed: the prompt
/// is complete once it has, and this check has left no piece.
pub fn finish_check_context(
    context: &PromptContext,
    plan: &RangePlan,
    claim: &ClaimSummary,
    pass_a: &PassSummary,
    scan_a: &ScanSummary,
    pass_b: &PassSummary,
    scan_b: &ScanSummary,
    bound_done: bool,
) -> PromptContext {
    let mut next = context.clone();
    let base = u64::from(next.prompt_tokens);
    if plan.end <= plan.start {
        record_fault(&mut next, FAULT_EMPTY_RANGE, plan.start);
    } else if plan.end > plan.total {
        record_fault(&mut next, FAULT_CUT_BEYOND_TEXT, plan.end);
    }
    for (pass, scan) in [(pass_a, scan_a), (pass_b, scan_b)] {
        for found in [&pass.carried, &pass.split] {
            if found.fault != FAULT_NONE {
                // A token is reported by its place in the prompt.
                let position = if found.fault == FAULT_NOT_THE_MERGE {
                    base.saturating_add(found.fault_position.saturating_sub(1))
                } else {
                    found.fault_position
                };
                record_fault(&mut next, found.fault, position);
            }
        }
        if scan.matched {
            record_fault(&mut next, FAULT_MERGES_INCOMPLETE, base);
        }
        next.model.fault |= pass.carried.model_fault | scan.model_fault;
    }
    if next.range_size > 0 && claim.tokens > u64::from(next.range_size) {
        record_fault(&mut next, FAULT_RANGE_TOO_LARGE, base);
    }
    if plan.stop > plan.start {
        next.next_piece = plan.stop;
    }
    next.complete = bound_done && next.next_piece >= plan.total;
    next.prompt_tokens = next
        .prompt_tokens
        .saturating_add(u32::try_from(claim.tokens).unwrap_or(u32::MAX));
    next
}

/// A claimed token whose text is no vocabulary entry is not a token the
/// merge produces. Every check of its pairs has already said so; this is
/// recorded as well so that the lookup's own result is never trusted.
pub fn require_vocabulary_token(context: &mut PromptContext, found: bool) {
    if !found {
        let position = u64::from(context.prompt_tokens);
        record_fault(context, FAULT_NOT_THE_MERGE, position);
    }
}

/// Whether a numbered piece belongs to the range's last token, whose pieces
/// the next range needs as its first token's neighbour.
pub fn is_last_token(piece: &CheckPiece, last_token: u32) -> bool {
    last_token != LEAD_TOKEN && piece.token == last_token
}
