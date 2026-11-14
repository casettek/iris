//! Native prompt preparation: the bind `prompt-bind` runs for a range, the ranked merge
//! every range's token check is held to, and the whole-prompt tokenization
//! direct inference uses.
use anyhow::{ensure, Result};
use prompt_check::input::{
    bind_range, merge_bucket_of, open_bind_range, record_fault, vocab_bucket_of, BpePiece,
    BpePieces, MergeCandidate, PromptContext, PromptState, PromptText, PromptTokenization,
    PromptTokenizer, RunConfig, RunSchedule, TextClaim, FAULT_MERGES_INCOMPLETE,
    FAULT_RANGE_TOO_LARGE,
};
use raster::{BytesPage, List};

use super::prompt_check_stage::BucketProofs;

pub struct PromptPrepareDirectInputs<'a> {
    pub tokenizer: &'a PromptTokenizer,
    pub initial_pieces: &'a BpePieces,
}

/// The lowest-ranked merge among neighbouring pieces of one segment, leftmost
/// on a tie.
pub fn best_merge(pieces: &[BpePiece], tokenizer: &PromptTokenizer) -> Result<MergeCandidate> {
    ensure!(
        tokenizer.merge_bucket_count as usize == tokenizer.merge_buckets.len()
            && tokenizer.merge_bucket_count > 0,
        "invalid merge bucket count"
    );
    let mut best = MergeCandidate::default();
    for (left, pair) in pieces.windows(2).enumerate() {
        if pair[0].segment != pair[1].segment {
            continue;
        }
        let index =
            merge_bucket_of(&pair[0].text, &pair[1].text, tokenizer.merge_bucket_count) as usize;
        for rule in tokenizer.merge_buckets[index].rules.iter() {
            if rule.left == pair[0].text
                && rule.right == pair[1].text
                && (!best.matched || rule.rank < best.rank)
            {
                best = MergeCandidate {
                    model_fault: 0,
                    matched: true,
                    rank: rule.rank,
                    left: left as u64,
                    merged: rule.merged.clone(),
                };
            }
        }
    }
    Ok(best)
}

fn apply(pieces: &mut Vec<BpePiece>, best: MergeCandidate) {
    let left = best.left as usize;
    pieces[left].text = best.merged;
    pieces.remove(left + 1);
}

/// What one range's `prompt-bind` is given.
pub struct PromptBindInputs<'a> {
    pub tokenizer: &'a PromptTokenizer,
    /// The tokenizer's identity root.
    pub root: &'a BytesPage,
    /// This range's slice of the original text.
    pub text: &'a PromptText,
    pub config: &'a RunConfig,
    pub descriptor: &'a [u8],
    pub schedule: &'a RunSchedule,
    pub claim: &'a TextClaim,
    /// The bind of the range before, or the empty state.
    pub previous: &'a PromptState,
}

/// `prompt-bind`: binds one range's text, continuing the range before.
pub fn run_prompt_bind(
    inputs: PromptBindInputs<'_>,
    proofs: &mut dyn BucketProofs,
) -> Result<PromptState> {
    let tokenizer = inputs.tokenizer;
    ensure!(
        tokenizer.vocab_bucket_count > 0
            && tokenizer.vocab_bucket_count as usize == tokenizer.vocab_buckets.len(),
        "invalid vocabulary bucket count"
    );
    let profile = &tokenizer.profile;
    let opened = open_bind_range(
        &inputs.previous.context,
        &inputs.previous.bind,
        inputs.config,
        inputs.descriptor,
        inputs.schedule,
        inputs.claim,
        inputs.root,
        profile,
    );
    let mut pieces = Vec::new();
    let closed = bind_range(
        opened,
        inputs.text.chunks.as_slice(),
        profile,
        |piece| {
            let index = vocab_bucket_of(piece, tokenizer.vocab_bucket_count);
            let found = tokenizer.vocab_buckets[index as usize]
                .entries
                .iter()
                .any(|entry| entry.token == piece);
            (found, proofs.vocab_bucket(index))
        },
        &mut |piece| pieces.push(piece),
    );
    Ok(PromptState {
        pieces: List::from(pieces),
        context: closed.context,
        bind: closed.carry,
        ..PromptState::default()
    })
}

/// The context after pieces have been merged and counted. Pieces that could
/// still merge are recorded rather than raised.
fn finish_token_context(
    context: &PromptContext,
    best: &MergeCandidate,
    tokens: u64,
) -> PromptContext {
    let mut next = context.clone();
    let start = next.prompt_tokens as u64;
    if best.matched {
        record_fault(&mut next, FAULT_MERGES_INCOMPLETE, start + best.left);
    }
    if next.range_size > 0 && tokens > next.range_size as u64 {
        record_fault(&mut next, FAULT_RANGE_TOO_LARGE, start);
    }
    next.prompt_tokens = next
        .prompt_tokens
        .saturating_add(u32::try_from(tokens).unwrap_or(u32::MAX));
    next
}

/// Resolves merged pieces to ids and records the outcome in the context.
pub fn finalize_prompt(inputs: PromptPrepareDirectInputs<'_>) -> Result<PromptTokenization> {
    let pieces = inputs
        .initial_pieces
        .pieces
        .iter()
        .cloned()
        .collect::<Vec<_>>();
    let best = best_merge(&pieces, inputs.tokenizer)?;
    let mut context =
        finish_token_context(&inputs.initial_pieces.context, &best, pieces.len() as u64);
    ensure!(
        inputs.tokenizer.vocab_bucket_count > 0
            && inputs.tokenizer.vocab_bucket_count as usize == inputs.tokenizer.vocab_buckets.len(),
        "invalid vocabulary bucket count"
    );
    let token_ids = pieces
        .iter()
        .map(|piece| {
            let index = vocab_bucket_of(&piece.text, inputs.tokenizer.vocab_bucket_count) as usize;
            inputs.tokenizer.vocab_buckets[index]
                .entries
                .iter()
                .find(|entry| entry.token == piece.text)
                .map(|entry| entry.id)
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "prepared or merged piece is absent from vocabulary: {:?}",
                        piece.text
                    )
                })
        })
        .collect::<Result<Vec<_>>>()?;
    for id in &token_ids {
        context.model.append_consumed_token(*id);
    }
    context
        .model
        .finish_prompt_range(context.complete, context.prompt_tokens, context.fault);
    Ok(PromptTokenization {
        token_ids: List::from(token_ids),
        context,
        ..PromptTokenization::default()
    })
}

/// A prompt whose ids direct inference was handed, with nothing a chain binds.
pub fn tokenization_of_ids(token_ids: Vec<u32>) -> PromptTokenization {
    PromptTokenization {
        token_ids: List::from(token_ids),
        ..PromptTokenization::default()
    }
}

/// Applies ranked merges until none is eligible.
pub fn merge_fully(
    mut pieces: Vec<BpePiece>,
    tokenizer: &PromptTokenizer,
) -> Result<Vec<BpePiece>> {
    loop {
        let best = best_merge(&pieces, tokenizer)?;
        if !best.matched {
            return Ok(pieces);
        }
        apply(&mut pieces, best);
    }
}

/// The whole prompt's tokens, as direct inference takes them: merged fully,
/// with no ranges and no checkpoints.
pub fn run_prompt_prepare_direct(
    inputs: PromptPrepareDirectInputs<'_>,
) -> Result<PromptTokenization> {
    let pieces = merge_fully(
        inputs.initial_pieces.pieces.iter().cloned().collect(),
        inputs.tokenizer,
    )?;
    finalize_prompt(PromptPrepareDirectInputs {
        tokenizer: inputs.tokenizer,
        initial_pieces: &BpePieces {
            pieces: pieces.into(),
            context: inputs.initial_pieces.context.clone(),
            ..BpePieces::default()
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use prompt_check::input::FAULT_NONE;

    #[test]
    fn unfinished_merges_and_an_oversized_range_are_recorded_first_fault_first() {
        let context = PromptContext {
            range_size: 2,
            ..PromptContext::default()
        };
        let best = MergeCandidate {
            matched: true,
            left: 1,
            ..MergeCandidate::default()
        };
        let next = finish_token_context(&context, &best, 3);
        assert_eq!(
            (next.fault, next.fault_position),
            (FAULT_MERGES_INCOMPLETE, 1)
        );
        assert_eq!(next.prompt_tokens, 3);

        let next = finish_token_context(&context, &MergeCandidate::default(), 3);
        assert_eq!(next.fault, FAULT_RANGE_TOO_LARGE);

        // With no range size, as in direct inference, no range limit applies.
        let next = finish_token_context(&PromptContext::default(), &MergeCandidate::default(), 3);
        assert_eq!(next.fault, FAULT_NONE);
    }
}
