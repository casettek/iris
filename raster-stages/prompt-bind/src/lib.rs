//! `prompt-bind`: tiles.
//!
//! Binds one prompt range's text: hashes it, renders it through the chat
//! template, and splits it into pieces, continuing from where the range before
//! stopped. Everything the prompt stages consume after this comes from text a
//! bind stage hashed, so pieces cannot be substituted for it.
//!
//! The logic is in `input`, shared with the native kernel and with run
//! preparation. A tile calls it, so it is inside the tile's image id.
//!
//! `#![no_std]` so the same tiles compile into RISC0 replay guests.

#![no_std]

extern crate alloc;

use alloc::string::String;
use raster::prelude::*;

pub mod input;

use input::*;

/// Opens the range: the first binds the model, the statement and the schedule;
/// a later one continues the bind before it.
#[allow(clippy::too_many_arguments)]
#[tile(kind = iter, description = "Open one range's bind from the range before")]
pub fn open_bind(
    previous: PromptContext,
    carry: BindCarry,
    config: RunConfig,
    descriptor: BytesPage,
    schedule: RunSchedule,
    claim: TextClaim,
    root: BytesPage,
    profile: TokenizerProfile,
) -> BindOpen {
    open_bind_range(
        &previous,
        &carry,
        &config,
        descriptor.as_slice(),
        &schedule,
        &claim,
        &root,
        &profile,
    )
}

/// How the text is rendered: what the claim says the trim keeps, and the
/// template.
#[tile(kind = iter, description = "The scan the text's claim calls for")]
pub fn plan_scan(claim: TextClaim, raw_prompt: bool, profile: TokenizerProfile) -> ScanPlan {
    claimed_scan_plan(&claim, raw_prompt, &profile)
}

// The three loops below each start from part of what the range before left.
// Each part is returned by a tile so the loop's seed is a whole stored value:
// Raster ties a loop's first step to a seed only when it is one, and takes the
// first step's word for a seed selected out of a larger value.

/// The text fold and the scan's position the range before left.
#[tile(kind = iter, description = "Where the range's text fold starts")]
pub fn text_pass_seed(before: BindCarry) -> TextPass {
    before.text
}

/// The scan's position the range before left.
#[tile(kind = iter, description = "Where the range's scan starts")]
pub fn scan_state_seed(before: BindCarry) -> ScanState {
    before.text.scan
}

/// The split the range before left.
#[tile(kind = iter, description = "Where the range's split starts")]
pub fn split_state_seed(before: BindCarry) -> SplitState {
    before.split
}

/// Hashes and measures one chunk of the range's text, and moves the scan's
/// position past it.
#[tile(kind = recur, description = "Hash and measure one chunk of the prompt")]
pub fn fold_text_chunk(
    input: RecurInput<String>,
    state: RecurState<TextPass>,
    plan: ScanPlan,
    profile: TokenizerProfile,
) -> RecurState<TextPass> {
    let mut state = state;
    let chunk = input.into_value();
    *state = advance_text_pass((*state).clone(), &chunk, &plan, &profile);
    state
}

/// Turns one chunk of text into atoms: special tokens and single characters.
#[tile(kind = recur, description = "Scan one chunk of the prompt into atoms")]
pub fn scan_prompt_chunk(
    input: RecurInput<String>,
    state: RecurState<ScanState>,
    output: RecurOutput<PromptAtoms>,
    plan: ScanPlan,
    profile: TokenizerProfile,
) -> (RecurState<ScanState>, RecurOutput<PromptAtoms>) {
    let mut state = state;
    let mut output = output;
    let chunk = input.into_value();
    *state = scan_chunk_into((*state).clone(), &chunk, &plan, &profile, &mut |atom| {
        output.atoms().push(atom)
    });
    (state, output)
}

/// One atom and the vocabulary bucket it would be in.
#[derive(
    Debug, Clone, Default, serde::Serialize, serde::Deserialize, PartialEq, Eq, raster::Selectable,
)]
pub struct AtomLookup {
    pub atom: PromptAtom,
    pub index: u32,
}

#[tile(kind = iter, description = "Bucket of an atom's vocabulary lookup")]
pub fn begin_atom_lookup(atom: PromptAtom, count: u32) -> AtomLookup {
    assert!(count > 0, "empty vocabulary bucket index");
    AtomLookup {
        index: vocab_bucket_of(&atom.text, count),
        atom,
    }
}

#[tile(kind = iter, description = "Open the walk of a vocabulary bucket")]
pub fn begin_walk() -> BucketWalk {
    begin_bucket_walk()
}

/// Hashes one entry of the bucket and compares it with the atom.
#[tile(kind = recur, description = "Hash and compare one vocabulary entry")]
pub fn walk_vocab_bucket(
    input: RecurInput<TokenEntry>,
    state: RecurState<BucketWalk>,
    piece: String,
) -> RecurState<BucketWalk> {
    let mut state = state;
    let entry = input.into_value();
    *state = walk_bucket_entry((*state).clone(), &entry, &piece);
    state
}

/// Records what the bucket said of the atom, and whether the bucket is the
/// committed one.
#[tile(kind = iter, description = "Append one looked-up atom")]
pub fn append_known_atom(
    output: Draft<KnownAtoms>,
    lookup: AtomLookup,
    walk: BucketWalk,
    header: SparseHeader,
    path: SparsePath,
) -> Draft<KnownAtoms> {
    let mut output = output;
    output.atoms().push(KnownAtom {
        proven: bucket_walk_proven(&walk, lookup.index, &header, &path),
        found: walk.found,
        text: lookup.atom.text,
        special: lookup.atom.special,
    });
    output
}

#[tile(kind = iter, description = "Open the split's tail at the range's start")]
pub fn begin_tail(split: SplitState) -> BindTail {
    BindTail {
        split,
        unproven: false,
    }
}

/// Moves the split past a chunk of atoms without emitting their pieces, to
/// learn where it ends and whether every lookup was proven.
#[tile(kind = recur, description = "Advance the split past a chunk of atoms")]
pub fn advance_tail(
    input: RecurInput<Block<KnownAtom>>,
    state: RecurState<BindTail>,
    profile: TokenizerProfile,
) -> RecurState<BindTail> {
    let mut state = state;
    for atom in input.into_value().iter() {
        *state = advance_bind_tail((*state).clone(), atom, &profile);
    }
    state
}

/// Opens the stage's output with the range's context and what the next
/// range's bind continues from.
///
/// Set here so a range with no pieces still has a complete output.
#[tile(kind = iter, description = "Close the range's bind and open its output")]
pub fn close_bind(
    context: PromptContext,
    before: BindCarry,
    pass: TextPass,
    tail: BindTail,
    profile: TokenizerProfile,
) -> Draft<PromptState> {
    let closed = close_bind_range(context, &before, pass, tail, &profile);
    let mut output = Draft::<PromptState>::new();
    output.context().set(closed.context);
    output.bind().set(closed.carry);
    output
}

/// Appends the pieces a chunk of atoms becomes.
#[tile(kind = recur, description = "Append a chunk of atoms' pieces")]
pub fn append_pieces(
    input: RecurInput<Block<KnownAtom>>,
    state: RecurState<SplitState>,
    output: RecurOutput<PromptState>,
    profile: TokenizerProfile,
) -> (RecurState<SplitState>, RecurOutput<PromptState>) {
    let mut state = state;
    let mut output = output;
    for atom in input.into_value().iter() {
        *state = split_known_atom(&state, atom, &profile, &mut |piece| {
            output.pieces().push(piece)
        });
    }
    (state, output)
}

pub mod identity;
pub use identity::*;
