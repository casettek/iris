//! The native twin of the `prompt-check` stage.
//!
//! It drives the same step functions the stage's tiles call
//! (`prompt_check::check`), in the order the stage's sequences run them, so
//! its output is the stage's output. What differs is only what cannot matter:
//! lookups read the tokenizer directly, and a pass stops once a scan finds
//! nothing to merge. The round after such a scan drops every window that is
//! left, and every later round sees only the end piece.

use anyhow::{bail, ensure, Result};
use prompt_check::check::{
    apply_piece, begin_apply_step, begin_number_state, begin_number_step, begin_window_step,
    count_marks, end_range_piece, finish_check_context, finish_window_step, is_last_token,
    is_rest_piece, join_claimed_piece, lead_piece, number_piece, observe_best,
    observe_claimed_piece, observe_final_piece, plan_range, range_end, range_piece,
    require_vocabulary_token, ApplyState, CheckPiece, ClaimSummary, MarkCount, PassSummary,
    ScanSummary, TokenJoin, WindowBest, WindowScan, ROUNDS_PER_PASS,
};
use prompt_check::input::{
    join_prompt_models, merge_bucket_of, vocab_bucket_of, BindCarry, BpePiece, MergeMatch,
    PromptState, PromptTokenizer,
};
use raster::{BytesPage, List};

/// What the stage's proofs of the tokenizer buckets it reads come to. The
/// staged pass checks each bucket against the witness; a caller with nothing
/// to check passes [`Trusted`].
pub trait BucketProofs {
    fn merge_bucket(&mut self, index: u32) -> bool;
    fn vocab_bucket(&mut self, index: u32) -> bool;
}

/// Accepts every bucket.
pub struct Trusted;

impl BucketProofs for Trusted {
    fn merge_bucket(&mut self, _index: u32) -> bool {
        true
    }
    fn vocab_bucket(&mut self, _index: u32) -> bool {
        true
    }
}

/// One scan over the pieces: the best pair of every window.
fn scan(
    pieces: &[CheckPiece],
    pass: u32,
    tokenizer: &PromptTokenizer,
    proofs: &mut dyn BucketProofs,
) -> Vec<WindowBest> {
    let mut state = WindowScan::default();
    let mut entries = Vec::new();
    for piece in pieces {
        let step = begin_window_step(state, piece, pass);
        let index = merge_bucket_of(
            &step.query.pending,
            &step.query.piece,
            tokenizer.merge_bucket_count,
        );
        let mut hit = MergeMatch::default();
        for rule in tokenizer.merge_buckets[index as usize].rules.iter() {
            if step.query.has_pending
                && rule.left == step.query.pending
                && rule.right == step.query.piece
                && (!hit.matched || rule.rank < hit.rank)
            {
                hit.matched = true;
                hit.rank = rule.rank;
                hit.merged = rule.merged.clone();
            }
        }
        if step.push {
            entries.push(step.closed.clone());
        }
        let mut next = finish_window_step(step, hit);
        next.model_fault |= u32::from(!proofs.merge_bucket(index));
        state = next;
    }
    entries
}

/// One pass: the closing scan, and the fold over what the pass has left.
fn pass(
    pieces: &[CheckPiece],
    pass: u32,
    tokenizer: &PromptTokenizer,
    proofs: &mut dyn BucketProofs,
) -> Result<(ScanSummary, PassSummary)> {
    let mut current = pieces.to_vec();
    let mut entries = scan(&current, pass, tokenizer, proofs);
    let mut settled = false;
    for _ in 0..ROUNDS_PER_PASS {
        if settled {
            // Only the end piece is left, and a round copies it as it is.
            break;
        }
        // After a scan that found nothing to merge, this round drops every
        // window that is left, and the scan that follows it is the last
        // that can differ.
        settled = !entries.iter().any(|best| best.matched);
        let mut state = ApplyState::default();
        let mut next = Vec::with_capacity(current.len());
        for piece in current {
            let step = begin_apply_step(state, piece, pass);
            let Some(best) = entries.get(step.index as usize).cloned() else {
                bail!("a piece's window has no entry in the round's best pairs");
            };
            let action = apply_piece(step, best);
            if action.push {
                next.push(action.piece.clone());
            }
            state = action.state;
        }
        current = next;
        entries = scan(&current, pass, tokenizer, proofs);
    }
    let mut closing = ScanSummary::default();
    for best in &entries {
        observe_best(&mut closing, best);
    }
    let mut summary = PassSummary::default();
    for piece in &current {
        observe_final_piece(&mut summary, piece);
    }
    Ok((closing, summary))
}

/// `prompt-check`: checks one range against the claim's marks and publishes
/// its token ids. `root` is the tokenizer's identity root, `bound` the
/// range's bind and `previous` the check of the range before, or the empty
/// state. The range's pieces are the ones `previous` left, then `bound`'s.
pub fn run_prompt_check(
    tokenizer: &PromptTokenizer,
    root: &BytesPage,
    bound: &PromptState,
    previous: &PromptState,
    marks: &[u32],
    proofs: &mut dyn BucketProofs,
) -> Result<PromptState> {
    ensure!(
        tokenizer.merge_bucket_count > 0
            && tokenizer.merge_bucket_count as usize == tokenizer.merge_buckets.len(),
        "invalid merge bucket count"
    );
    ensure!(
        tokenizer.vocab_bucket_count > 0
            && tokenizer.vocab_bucket_count as usize == tokenizer.vocab_buckets.len(),
        "invalid vocabulary bucket count"
    );
    let mut context = join_prompt_models(previous.context.clone(), &bound.context);
    context.model =
        model_identity::stream::verify(context.model, root, model_identity::Role::Tokenizer);
    context
        .model
        .require(context.model_id == context.model.model_id);

    // Which pieces the range takes.
    let mut marked = MarkCount::default();
    count_marks(&mut marked, marks.len());
    let end = range_end(&context, &marked);
    let given: Vec<&BpePiece> = previous.rest.iter().chain(bound.pieces.iter()).collect();
    let total = context.next_piece.saturating_add(given.len() as u64);
    let plan = plan_range(&context, end, total);

    // The range's pieces after the previous range's last token, numbered.
    let mut range = Vec::new();
    if plan.lead {
        range.extend(previous.pieces.iter().map(|piece| lead_piece(piece)));
    }
    for (offset, piece) in given.iter().enumerate() {
        if let Some((piece, last)) = range_piece(piece, plan.start + offset as u64, &plan) {
            range.push(piece);
            if last {
                range.push(end_range_piece());
            }
        }
    }
    let mut state = begin_number_state(&plan);
    let mut pieces = Vec::with_capacity(range.len());
    for piece in range {
        let step = begin_number_step(state, piece);
        let Some(mark) = marks.get(step.index as usize).copied() else {
            bail!("a range piece has no mark");
        };
        let numbered = number_piece(step, mark);
        pieces.push(numbered.piece);
        state = numbered.state;
    }

    // What the marks claim of the range: how many tokens, and the last.
    let mut claim = ClaimSummary::default();
    for piece in &pieces {
        observe_claimed_piece(&mut claim, piece);
    }

    let (scan_a, summary_a) = pass(&pieces, 0, tokenizer, proofs)?;
    let (scan_b, summary_b) = pass(&pieces, 1, tokenizer, proofs)?;
    let mut context = finish_check_context(
        &context,
        &plan,
        &claim,
        &summary_a,
        &scan_a,
        &summary_b,
        &scan_b,
        bound.bind.done,
    );

    // The range's tokens, each the text of its claimed pieces, resolved to
    // ids. A claimed token that is no vocabulary entry is a fault and is
    // given id 0.
    let mut join = TokenJoin::default();
    let mut tokens = Vec::new();
    for piece in &pieces {
        tokens.extend(join_claimed_piece(&mut join, piece));
    }
    let id_of = |token: &BpePiece| {
        let index = vocab_bucket_of(&token.text, tokenizer.vocab_bucket_count) as usize;
        tokenizer.vocab_buckets[index]
            .entries
            .iter()
            .find(|entry| entry.token == token.text)
            .map(|entry| entry.id)
    };
    for token in &tokens {
        let index = vocab_bucket_of(&token.text, tokenizer.vocab_bucket_count);
        context.model.require(proofs.vocab_bucket(index));
        require_vocabulary_token(&mut context, id_of(token).is_some());
    }
    let token_ids: Vec<u32> = tokens
        .iter()
        .map(|token| id_of(token).unwrap_or(0))
        .collect();
    for id in &token_ids {
        context.model.append_consumed_token(*id);
    }
    context
        .model
        .finish_prompt_range(context.complete, context.prompt_tokens, context.fault);

    // The range's last token, as the pieces it was claimed from.
    let carried = pieces
        .iter()
        .filter(|piece| is_last_token(piece, claim.last_token))
        .map(|piece| BpePiece {
            text: piece.text.clone(),
            segment: piece.segment,
        })
        .collect::<Vec<_>>();
    // What the range was given and did not take, for the next range.
    let rest = given
        .iter()
        .enumerate()
        .filter(|(offset, _)| is_rest_piece(plan.start + *offset as u64, &plan))
        .map(|(_, piece)| (*piece).clone())
        .collect::<Vec<_>>();
    Ok(PromptState {
        pieces: List::from(carried),
        rest: List::from(rest),
        token_ids: List::from(token_ids),
        context,
        bind: BindCarry::default(),
    })
}

/// The marks run preparation writes for a range: one per piece, 1 where a
/// token starts. `counts` is each token's number of pieces, in order.
pub fn marks_for(counts: impl IntoIterator<Item = usize>) -> Vec<u32> {
    counts
        .into_iter()
        .flat_map(|count| (0..count).map(|offset| u32::from(offset == 0)))
        .collect()
}
