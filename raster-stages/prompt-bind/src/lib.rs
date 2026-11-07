//! `prompt-bind`: tiles.
//!
//! Hashes the original prompt, renders it through the chat template, and splits
//! it into pieces. Everything the prompt stages consume after this comes from
//! the text this stage hashed, so pieces cannot be substituted for it.
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

/// The hash and measurements before any text.
#[tile(kind = iter, description = "Open the fold over the prompt text")]
pub fn begin_text_fold() -> TextFold {
    begin_text_fold_state()
}

/// Hashes and measures one chunk of the original text.
#[tile(kind = recur, description = "Hash and measure one chunk of the prompt")]
pub fn fold_text_chunk(
    input: RecurInput<String>,
    state: RecurState<TextFold>,
) -> RecurState<TextFold> {
    let mut state = state;
    let chunk = input.into_value();
    *state = fold_text((*state).clone(), &chunk);
    state
}

/// Decides how the text is rendered: what the trim keeps, and the template.
#[tile(kind = iter, description = "Plan the prompt's rendering")]
pub fn plan_scan(fold: TextFold, config: RunConfig, profile: TokenizerProfile) -> ScanPlan {
    scan_plan_of(&fold, &config, &profile)
}

/// The scan before any text: the template's prefix is already pending.
#[tile(kind = iter, description = "Open the scan over the rendered prompt")]
pub fn begin_scan(plan: ScanPlan) -> ScanState {
    begin_scan_state(&plan)
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

/// Opens the stage's output with the context a run starts from.
///
/// Set here so a prompt with no pieces still has a complete output.
#[tile(kind = iter, description = "Open the bound prompt with its context")]
pub fn open_bound_prompt(
    fold: TextFold,
    config: RunConfig,
    descriptor: BytesPage,
    root: BytesPage,
) -> Draft<PromptState> {
    let mut output = Draft::<PromptState>::new();
    let mut context = open_prompt_context(&fold, &config);
    context.model = model_identity::stream::verify(
        model_identity::ModelContext::bind(descriptor.as_slice(), &config.model_id),
        &root,
        model_identity::Role::Tokenizer,
    );
    context.model.bind_statement(
        &context.input_text_sha256,
        context.raw_prompt,
        context.max_new_tokens,
        context.range_size,
    );
    output.context().set(context);
    output
}

/// The split before any atom.
#[tile(kind = iter, description = "Open the split at the first segment")]
pub fn begin_split() -> SplitState {
    SplitState::default()
}

/// Pairs one atom with the split's position and its vocabulary lookup.
#[tile(kind = iter, description = "Begin splitting one atom")]
pub fn begin_atom_step(state: SplitState, atom: PromptAtom) -> AtomStep {
    AtomStep {
        query: VocabQuery {
            piece: atom.text.clone(),
        },
        atom,
        state,
    }
}

#[tile(kind = iter, description = "Bucket of a vocabulary lookup")]
pub fn vocab_bucket_index(query: VocabQuery, count: u32) -> u32 {
    assert!(count > 0, "empty vocabulary bucket index");
    vocab_bucket_of(&query.piece, count)
}

#[tile(kind = iter, description = "Open a vocabulary lookup with nothing found")]
pub fn empty_vocab_match() -> VocabMatch {
    VocabMatch::default()
}

#[tile(kind = recur, description = "Scan one vocabulary bucket")]
pub fn scan_vocab_chunk(
    input: RecurInput<TokenEntry>,
    state: RecurState<VocabMatch>,
    query: VocabQuery,
) -> RecurState<VocabMatch> {
    let mut state = state;
    let entry = input.into_value();
    if entry.token == query.piece {
        state.found = true;
        state.token_id = entry.id;
    }
    state
}

/// Appends the pieces one atom becomes.
#[tile(kind = iter, description = "Append one atom's pieces")]
pub fn append_atom_pieces(
    output: Draft<PromptState>,
    step: AtomStep,
    hit: VocabMatch,
    profile: TokenizerProfile,
) -> Draft<PromptState> {
    let mut output = output;
    apply_atom(&step.state, &step.atom, hit.found, &profile, &mut |piece| {
        output.pieces().push(piece)
    });
    output
}

/// Moves the split past one atom.
#[tile(kind = iter, description = "Advance the split past one atom")]
pub fn advance_split(step: AtomStep, hit: VocabMatch, profile: TokenizerProfile) -> SplitState {
    apply_atom(&step.state, &step.atom, hit.found, &profile, &mut |_| {})
}

pub mod identity;
pub use identity::*;
