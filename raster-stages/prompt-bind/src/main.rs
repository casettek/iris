//! `prompt-bind`: sequences, one stage per prompt range.
//!
//! ```text
//!   previous range's bind ─▶ context, fold, scan and split where it stopped
//!   text.chunks ──recur──▶ sha256, trim bounds, scan position
//!   text.chunks ──recur──▶ atoms
//!   atoms ──recur seq──▶ [ bucket · one walk: hash and compare ] ──▶ known atoms
//!   known atoms ──recur──▶ where the split ends, every lookup proven
//!   known atoms ──recur──▶ pieces ──▶ PromptState
//! ```

use prompt_bind::input::*;
use prompt_bind::*;
use raster::prelude::*;

/// One atom: walk its bucket once, hashing every entry and comparing each
/// with the atom, and record what was found.
#[sequence(kind = recur)]
fn look_up_atom(
    input: RecurSequenceInput<PromptAtom>,
    output: RecurSequenceOutput<KnownAtoms>,
    vocab_buckets: List<VocabBucket>,
    count: u32,
    header: SparseHeader,
    paths: List<SparsePath>,
) -> RecurSequenceOutput<KnownAtoms> {
    let lookup = call!(begin_atom_lookup, input, count);
    let index = select!(u32, lookup.clone().index);
    let bucket = select!(VocabBucket, vocab_buckets[index]);
    let entries = select!(List<TokenEntry>, bucket.entries);
    let piece = select!(String, lookup.clone().atom.text);
    let seed = call!(begin_walk);
    let walk = call_recur!(
        tile = walk_vocab_bucket,
        input = entries,
        state = seed,
        args = (piece,)
    );
    let path = select!(SparsePath, paths[index]);
    call!(append_known_atom, output, lookup, walk, header, path)
}

/// Stage entrypoint for one range: `text` is the range's slice of the original
/// prompt and `previous` the bind of the range before, or the empty state.
/// `config`, `descriptor`, `schedule` and `claim` are the run's, read by the
/// first range only. `tokenizer` is the committed tokenizer.
#[sequence]
fn main(
    tokenizer: PromptTokenizer,
    witness: TokenizerWitness,
    text: PromptText,
    config: RunConfig,
    descriptor: BytesPage,
    schedule: RunSchedule,
    claim: TextClaim,
    previous: PromptState,
) -> PromptState {
    let profile = select!(TokenizerProfile, tokenizer.clone().profile);
    let count = select!(u32, tokenizer.clone().vocab_bucket_count);
    let vocab = select!(List<VocabBucket>, tokenizer.clone().vocab_buckets);
    let chunks = select!(List<String>, text.chunks);

    let root = call_seq!(identity_root, tokenizer, witness.clone());
    let previous_context = select!(PromptContext, previous.clone().context);
    let previous_carry = select!(BindCarry, previous.bind);
    let opened = call!(
        open_bind,
        previous_context,
        previous_carry,
        config,
        descriptor,
        schedule,
        claim,
        root,
        profile.clone()
    );
    let context = select!(PromptContext, opened.clone().context);
    let before = select!(BindCarry, opened.carry);

    let text_claim = select!(TextClaim, before.clone().claim);
    let raw_prompt = select!(bool, context.clone().raw_prompt);
    let plan = call!(plan_scan, text_claim, raw_prompt, profile.clone());
    let pass_seed = select!(TextPass, before.clone().text);
    let pass = call_recur!(
        tile = fold_text_chunk,
        input = chunks.clone(),
        state = pass_seed,
        args = (plan.clone(), profile.clone())
    );
    let scan_seed = select!(ScanState, before.clone().text.scan);
    let scanned = call_recur!(
        tile = scan_prompt_chunk,
        input = chunks,
        state = scan_seed,
        output,
        args = (plan, profile.clone())
    );
    let atoms = select!(List<PromptAtom>, scanned.atoms);

    let vocab_header = select!(SparseHeader, witness.clone().vocab);
    let vocab_paths = select!(List<SparsePath>, witness.vocab_paths);
    let known = call_recur_seq!(
        sequence = look_up_atom,
        input = atoms,
        output,
        args = (vocab, count, vocab_header, vocab_paths)
    );
    let known_atoms = select!(List<KnownAtom>, known.atoms);

    let split_seed = select!(SplitState, before.clone().split);
    let tail_seed = call!(begin_tail, split_seed.clone());
    let tail = call_recur!(
        tile = advance_tail,
        input = known_atoms.clone(),
        chunk = 256,
        state = tail_seed,
        args = (profile.clone(),)
    );
    let closed = call!(close_bind, context, before, pass, tail, profile.clone());
    let bound = call_recur!(
        tile = append_pieces,
        input = known_atoms,
        chunk = 256,
        state = split_seed,
        output = closed,
        args = (profile,)
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
    let profile = select!(TokenizerProfile, value.clone().profile);
    let profile = call!(identity_hash_tokenizerprofile, profile);
    let vocab_header = select!(SparseHeader, witness.clone().vocab);
    let vocab_buckets = call!(identity_sparse_root, vocab_header);
    let merge_header = select!(SparseHeader, witness.merges);
    let merge_buckets = call!(identity_sparse_root, merge_header);
    call!(
        identity_component,
        vocab_bucket_count,
        merge_bucket_count,
        profile,
        vocab_buckets,
        merge_buckets
    )
}
