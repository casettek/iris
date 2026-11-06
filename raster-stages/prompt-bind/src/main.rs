//! `prompt-bind`: sequences, one stage per run.
//!
//! ```text
//!   text.chunks ──recur──▶ sha256, trim bounds ─┐
//!   text.chunks ──recur──▶ atoms ◀── template ──┤
//!   atoms ──recur seq──▶ [ vocabulary lookup · pieces · next position ] ──▶ PromptState
//! ```

use prompt_bind::input::*;
use prompt_bind::*;
use raster::prelude::*;

/// One atom: look it up, append the pieces it becomes, move the split on.
#[sequence(kind = recur)]
fn resolve_atom(
    input: RecurSequenceInput<PromptAtom>,
    state: RecurSequenceState<SplitState>,
    output: RecurSequenceOutput<PromptState>,
    vocab_buckets: List<VocabBucket>,
    count: u32,
    profile: TokenizerProfile,
) -> (
    RecurSequenceState<SplitState>,
    RecurSequenceOutput<PromptState>,
) {
    let step = call!(begin_atom_step, state, input);
    let query = select!(VocabQuery, step.clone().query);
    let index = call!(vocab_bucket_index, query.clone(), count);
    let bucket = select!(VocabBucket, vocab_buckets[index]);
    let entries = select!(List<TokenEntry>, bucket.entries);
    let empty = call!(empty_vocab_match);
    let hit = call_recur!(
        tile = scan_vocab_chunk,
        input = entries,
        state = empty,
        args = (query,)
    );
    let output = call!(
        append_atom_pieces,
        output,
        step.clone(),
        hit.clone(),
        profile.clone()
    );
    let state = call!(advance_split, step, hit, profile);
    (state, output)
}

/// Stage entrypoint: `text` is the original prompt, `config` the run's
/// configuration, `tokenizer` the committed tokenizer.
#[sequence]
fn main(tokenizer: PromptTokenizer, text: PromptText, config: RunConfig) -> PromptState {
    let profile = select!(TokenizerProfile, tokenizer.clone().profile);
    let count = select!(u32, tokenizer.clone().vocab_bucket_count);
    let vocab = select!(List<VocabBucket>, tokenizer.vocab_buckets);
    let chunks = select!(List<String>, text.chunks);

    let fold_seed = call!(begin_text_fold);
    let fold = call_recur!(
        tile = fold_text_chunk,
        input = chunks.clone(),
        state = fold_seed,
        args = ()
    );

    let plan = call!(plan_scan, fold.clone(), config.clone(), profile.clone());
    let scan_seed = call!(begin_scan, plan.clone());
    let scanned = call_recur!(
        tile = scan_prompt_chunk,
        input = chunks,
        state = scan_seed,
        output,
        args = (plan, profile.clone())
    );
    let atoms = select!(List<PromptAtom>, scanned.atoms);

    let opened = call!(open_bound_prompt, fold, config);
    let split_seed = call!(begin_split);
    let bound = call_recur_seq!(
        sequence = resolve_atom,
        input = atoms,
        state = split_seed,
        output = opened,
        args = (vocab, count, profile)
    );
    raster::println!("prompt bind → {:?}", bound);
    bound
}
