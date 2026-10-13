use anyhow::{bail, ensure, Result};
use prompt_prepare::input::{
    merge_bucket_of, vocab_bucket_of, BpePiece, BpePieces, MergeCandidate, PromptTokenization,
    PromptTokenizer,
};
use raster::List;

pub const MERGES_PER_BATCH: usize = 8;

pub struct PromptPrepareDirectInputs<'a> {
    pub tokenizer: &'a PromptTokenizer,
    pub initial_pieces: &'a BpePieces,
}

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

pub fn run_merge_batch(inputs: PromptPrepareDirectInputs<'_>) -> Result<BpePieces> {
    let mut pieces = inputs
        .initial_pieces
        .pieces
        .iter()
        .cloned()
        .collect::<Vec<_>>();
    for _ in 0..MERGES_PER_BATCH {
        let best = best_merge(&pieces, inputs.tokenizer)?;
        if !best.matched {
            break;
        }
        apply(&mut pieces, best);
    }
    Ok(BpePieces {
        pieces: List::from(pieces),
    })
}

pub fn finalize_prompt(inputs: PromptPrepareDirectInputs<'_>) -> Result<PromptTokenization> {
    let pieces = inputs
        .initial_pieces
        .pieces
        .iter()
        .cloned()
        .collect::<Vec<_>>();
    let best = best_merge(&pieces, inputs.tokenizer)?;
    if best.matched {
        bail!(
            "tokenization budget exhausted: eligible merge at position {} (rank {})",
            best.left,
            best.rank
        );
    }
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
    Ok(PromptTokenization {
        token_ids: List::from(token_ids),
    })
}

/// Direct inference has no checkpoint budget, but obeys the same ranked ordering.
pub fn run_prompt_prepare_direct(
    inputs: PromptPrepareDirectInputs<'_>,
) -> Result<PromptTokenization> {
    let mut pieces = inputs
        .initial_pieces
        .pieces
        .iter()
        .cloned()
        .collect::<Vec<_>>();
    loop {
        let best = best_merge(&pieces, inputs.tokenizer)?;
        if !best.matched {
            break;
        }
        apply(&mut pieces, best);
    }
    finalize_prompt(PromptPrepareDirectInputs {
        tokenizer: inputs.tokenizer,
        initial_pieces: &BpePieces {
            pieces: pieces.into(),
        },
    })
}
