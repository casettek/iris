use anyhow::{ensure, Result};
use prompt_prepare::input::{
    apply_atom, begin_scan_state, begin_slice_state, begin_text_fold_state, boundary_bucket_of,
    boundary_query_of, finish_slice_context, finish_token_context, fold_text, merge_bucket_of,
    observe_piece, open_prompt_context, scan_chunk_into, scan_plan_of, vocab_bucket_of,
    BoundaryHit, BpePiece, BpePieces, MergeCandidate, PromptContext, PromptState, PromptText,
    PromptTokenization, PromptTokenizer, RangeCut, RunConfig, SplitState,
};
use raster::List;

/// Merge rounds per `prompt-merge` stage. MUST match its `main`.
pub const MERGES_PER_BATCH: usize = 64;

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

fn merge_batch(pieces: &mut Vec<BpePiece>, tokenizer: &PromptTokenizer) -> Result<()> {
    for _ in 0..MERGES_PER_BATCH {
        let best = best_merge(pieces, tokenizer)?;
        if !best.matched {
            break;
        }
        apply(pieces, best);
    }
    Ok(())
}

pub fn run_merge_batch(inputs: PromptPrepareDirectInputs<'_>) -> Result<BpePieces> {
    let mut pieces = inputs
        .initial_pieces
        .pieces
        .iter()
        .cloned()
        .collect::<Vec<_>>();
    merge_batch(&mut pieces, inputs.tokenizer)?;
    Ok(BpePieces {
        pieces: List::from(pieces),
        token_ids: List::new(),
        context: inputs.initial_pieces.context.clone(),
    })
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

/// `prompt-seed`: cuts one range out of the prompt's pieces, checks the cut,
/// and runs the range's first merge batch.
pub fn run_prompt_seed(
    tokenizer: &PromptTokenizer,
    bound: &PromptState,
    previous: &PromptContext,
    cut: &RangeCut,
) -> Result<PromptState> {
    let joined = ::prompt_prepare::input::join_prompt_models(previous.clone(), &bound.context);
    let previous = &joined;
    let mut slice = begin_slice_state(previous, cut.end);
    for piece in bound.pieces.iter() {
        observe_piece(&mut slice, piece);
    }
    let query = boundary_query_of(&slice);
    ensure!(
        tokenizer.boundary_bucket_count > 0
            && tokenizer.boundary_bucket_count as usize == tokenizer.boundary_buckets.len(),
        "invalid boundary bucket count"
    );
    let index =
        boundary_bucket_of(query.left, query.right, tokenizer.boundary_bucket_count) as usize;
    let crossing = BoundaryHit {
        found: query.active
            && tokenizer.boundary_buckets[index]
                .pairs
                .iter()
                .any(|pair| pair.left == query.left && pair.right == query.right),
    };
    let context = finish_slice_context(previous, &slice, &crossing);

    let mut pieces = bound
        .pieces
        .iter()
        .enumerate()
        .filter(|(index, _)| (*index as u64) >= slice.start && (*index as u64) < slice.end)
        .map(|(_, piece)| piece.clone())
        .collect::<Vec<_>>();
    merge_batch(&mut pieces, tokenizer)?;
    Ok(PromptState {
        pieces: List::from(pieces),
        token_ids: List::new(),
        context,
    })
}

/// `prompt-prepare`: resolves the merged pieces to ids and records the range's
/// outcome in the context. Pieces that could still merge are recorded rather
/// than raised, so the stage finishes with an output to compare and replay.
pub fn finalize_prompt(inputs: PromptPrepareDirectInputs<'_>) -> Result<PromptTokenization> {
    let pieces = inputs
        .initial_pieces
        .pieces
        .iter()
        .cloned()
        .collect::<Vec<_>>();
    let best = best_merge(&pieces, inputs.tokenizer)?;
    let context = finish_token_context(&inputs.initial_pieces.context, &best, pieces.len() as u64);
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

/// Direct inference has no checkpoint budget, but obeys the same ranked ordering.
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
