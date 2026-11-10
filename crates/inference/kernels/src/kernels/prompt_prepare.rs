//! Native prompt preparation: the split `prompt-bind` runs, the ranked merge
//! every range's token check is held to, and the whole-prompt tokenization
//! direct inference uses.
use anyhow::{ensure, Result};
use prompt_check::input::{
    apply_atom, begin_scan_state, begin_text_fold_state, fold_text, merge_bucket_of,
    open_prompt_context, record_fault, scan_chunk_into, scan_plan_of, vocab_bucket_of, BpePiece,
    BpePieces, MergeCandidate, PromptContext, PromptState, PromptText, PromptTokenization,
    PromptTokenizer, RunConfig, SplitState, FAULT_MERGES_INCOMPLETE, FAULT_RANGE_TOO_LARGE,
};
use raster::List;

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

fn in_vocabulary(tokenizer: &PromptTokenizer, piece: &str) -> Result<bool> {
    ensure!(
        tokenizer.vocab_bucket_count > 0
            && tokenizer.vocab_bucket_count as usize == tokenizer.vocab_buckets.len(),
        "invalid vocabulary bucket count"
    );
    let index = vocab_bucket_of(piece, tokenizer.vocab_bucket_count) as usize;
    Ok(tokenizer.vocab_buckets[index]
        .entries
        .iter()
        .any(|entry| entry.token == piece))
}

/// `prompt-bind`: hashes the original text, renders it, and splits it.
pub fn run_prompt_bind(
    tokenizer: &PromptTokenizer,
    text: &PromptText,
    config: &RunConfig,
) -> Result<PromptState> {
    let profile = &tokenizer.profile;
    let fold = text
        .chunks
        .iter()
        .fold(begin_text_fold_state(), |fold, chunk| {
            fold_text(fold, chunk)
        });
    let plan = scan_plan_of(&fold, config, profile);
    let mut scan = begin_scan_state(&plan);
    let mut atoms = Vec::new();
    for chunk in text.chunks.iter() {
        scan = scan_chunk_into(scan, chunk, &plan, profile, &mut |atom| atoms.push(atom));
    }
    let mut pieces = Vec::new();
    let mut split = SplitState::default();
    for atom in &atoms {
        let known = in_vocabulary(tokenizer, &atom.text)?;
        split = apply_atom(&split, atom, known, profile, &mut |piece| {
            pieces.push(piece)
        });
    }
    Ok(PromptState {
        pieces: List::from(pieces),
        token_ids: List::new(),
        context: open_prompt_context(&fold, config),
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
        pieces: List::new(),
        token_ids: List::from(token_ids),
        context,
    })
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
            token_ids: List::new(),
            context: inputs.initial_pieces.context.clone(),
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
