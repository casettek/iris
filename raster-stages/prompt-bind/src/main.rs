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
/// configuration, `schedule` its stage schedule, `tokenizer` the committed
/// tokenizer.
#[sequence]
fn main(
    tokenizer: PromptTokenizer,
    witness: TokenizerWitness,
    text: PromptText,
    config: RunConfig,
    descriptor: BytesPage,
    schedule: RunSchedule,
) -> PromptState {
    let profile = select!(TokenizerProfile, tokenizer.clone().profile);
    let count = select!(u32, tokenizer.clone().vocab_bucket_count);
    let vocab = select!(List<VocabBucket>, tokenizer.clone().vocab_buckets);
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

    let root = call_seq!(identity_root, tokenizer, witness.clone());
    let context = call!(identity_bound_context, fold, config, descriptor, root);
    let context = call!(bind_schedule, context, schedule);
    let vocab_header = select!(SparseHeader, witness.clone().vocab);
    let vocab_paths = select!(List<SparsePath>, witness.vocab_paths);
    let context = call_recur_seq!(
        sequence = verify_atom,
        input = atoms.clone(),
        state = context,
        args = (vocab.clone(), count.clone(), vocab_header, vocab_paths)
    );
    let opened = call!(identity_open_bound, context);
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

#[sequence]
fn identity_root(value: PromptTokenizer, witness: TokenizerWitness) -> BytesPage {
    let vocab_bucket_count = select!(u32, value.clone().vocab_bucket_count);
    let vocab_bucket_count = call!(identity_hash_u32, vocab_bucket_count);
    let merge_bucket_count = select!(u32, value.clone().merge_bucket_count);
    let merge_bucket_count = call!(identity_hash_u32, merge_bucket_count);
    let boundary_bucket_count = select!(u32, value.clone().boundary_bucket_count);
    let boundary_bucket_count = call!(identity_hash_u32, boundary_bucket_count);
    let profile = select!(TokenizerProfile, value.clone().profile);
    let profile = call!(identity_hash_tokenizerprofile, profile);
    let vocab_header = select!(SparseHeader, witness.clone().vocab);
    let vocab_buckets = call!(identity_sparse_root, vocab_header);
    let merge_header = select!(SparseHeader, witness.clone().merges);
    let merge_buckets = call!(identity_sparse_root, merge_header);
    let boundary_header = select!(SparseHeader, witness.boundaries);
    let boundary_buckets = call!(identity_sparse_root, boundary_header);
    call!(
        identity_component,
        vocab_bucket_count,
        merge_bucket_count,
        boundary_bucket_count,
        profile,
        vocab_buckets,
        merge_buckets,
        boundary_buckets
    )
}

#[sequence]
fn identity_vocabbucket_contents(buckets: List<VocabBucket>, index: u32) -> BytesPage {
    let input = select!(VocabBucket, buckets[index]);
    let entries = select!(List<TokenEntry>, input.entries);
    let seed = call!(identity_empty);
    let folded = call_recur!(
        tile = identity_vocabbucket_entry,
        input = entries,
        state = seed,
        args = ()
    );
    let root = call!(identity_finish, folded);
    let root = call!(identity_vocabbucket_root, root);
    root
}

#[sequence]
fn identity_mergebucket_contents(buckets: List<MergeBucket>, index: u32) -> BytesPage {
    let input = select!(MergeBucket, buckets[index]);
    let entries = select!(List<BpeMerge>, input.rules);
    let seed = call!(identity_empty);
    let folded = call_recur!(
        tile = identity_mergebucket_entry,
        input = entries,
        state = seed,
        args = ()
    );
    let root = call!(identity_finish, folded);
    let root = call!(identity_mergebucket_root, root);
    root
}

#[sequence]
fn identity_boundarybucket_contents(buckets: List<BoundaryBucket>, index: u32) -> BytesPage {
    let input = select!(BoundaryBucket, buckets[index]);
    let entries = select!(List<BoundaryPair>, input.pairs);
    let seed = call!(identity_empty);
    let folded = call_recur!(
        tile = identity_boundarybucket_entry,
        input = entries,
        state = seed,
        args = ()
    );
    let root = call!(identity_finish, folded);
    let root = call!(identity_boundarybucket_root, root);
    root
}

#[sequence(kind=recur)]
fn verify_atom(
    input: RecurSequenceInput<PromptAtom>,
    state: RecurSequenceState<PromptContext>,
    buckets: List<VocabBucket>,
    count: u32,
    header: SparseHeader,
    paths: List<SparsePath>,
) -> RecurSequenceState<PromptContext> {
    let query = call!(identity_atom_query, input);
    let index = call!(vocab_bucket_index, query, count);
    let root = call_seq!(identity_vocabbucket_contents, buckets, index.clone());
    let proof = select!(SparsePath, paths[index]);
    call!(identity_sparse_context, state, root, index, header, proof)
}
