//! Prompt types and pure preparation logic.
//!
//! Everything a prompt stage computes lives here as plain functions, called
//! from that stage's tiles (so it is inside their image ids and replays in the
//! guest), from the native kernels, and from run preparation. There is one
//! copy of the logic, so the three cannot disagree.
//!
//! The chain establishes this relationship over the prompt:
//!
//! ```text
//!   original text ─ sha256 ─▶ input_text_sha256
//!        │
//!        └ template, split ─▶ pieces ─ cut ─▶ ranges ─ merge ─▶ token ids
//! ```
//!
//! A preparation that is wrong does not abort: an aborting stage has no output
//! to compare or replay. It runs to completion and records why in
//! [`PromptContext::fault`], and the context travels down the prompt stages.
//!
//! Raster reads a stage's interface types from the structs in that stage's own
//! `src/`, so a stage cannot import them from another crate. Each prompt stage
//! compiles this one file as its own module through a symlink in its `src/`.
//! Their types are distinct Rust types with identical structure, which is what
//! the chain links stages by.

use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use raster::{BytesPage, List};
use serde::{Deserialize, Serialize};
use sha2::digest::generic_array::GenericArray;

/// The word scanner, for a tokenizer whose profile asks for it.
#[path = "tekken.rs"]
pub mod tekken;

/// Token id reserved for pieces that are not in the vocabulary.
pub const UNK_TOKEN_ID: u32 = 0;

/// `PromptContext::fault` values. Zero is a clean preparation.
pub const FAULT_NONE: u32 = 0;
/// The committed text has no chunks at all, not even an empty one.
pub const FAULT_NO_TEXT: u32 = 1;
/// A range's cut does not move past the previous range's.
pub const FAULT_EMPTY_RANGE: u32 = 2;
/// A range's cut is past the end of the prompt's pieces.
pub const FAULT_CUT_BEYOND_TEXT: u32 = 3;
// 4 was a cut at a boundary a merge could cross. No stage records it now that
// a range's tokens are checked against the merge itself; the value stays unused.
/// A range has more tokens than the run's range size.
pub const FAULT_RANGE_TOO_LARGE: u32 = 5;
/// A range's pieces could still merge.
pub const FAULT_MERGES_INCOMPLETE: u32 = 6;
/// A prompt range continues another statement or loses an inherited fault.
pub const FAULT_CONTEXT_MISMATCH: u32 = 7;
/// A claimed token's pieces are in two segments.
pub const FAULT_TOKEN_SPLIT: u32 = 8;
/// The claimed tokens are not what the ranked merge produces.
pub const FAULT_NOT_THE_MERGE: u32 = 9;
/// The text is not what the run claimed of it: its hash, its length or what
/// the template's trim keeps.
pub const FAULT_TEXT_CLAIM: u32 = 10;
/// A range was given text after the text had ended.
pub const FAULT_TEXT_AFTER_END: u32 = 11;
/// The text has a run the word scanner could not hold until it knew which
/// word the run is in, so its words may not be the tokenizer's.
pub const FAULT_WORD_SPLIT: u32 = 12;

/// What a `PromptContext::fault` value means.
pub fn fault_description(fault: u32) -> &'static str {
    match fault {
        FAULT_NONE => "no fault",
        FAULT_CONTEXT_MISMATCH => "prompt predecessors describe different statements",
        FAULT_NO_TEXT => "the committed text has no chunks",
        FAULT_EMPTY_RANGE => "a range's cut does not move past the previous range's",
        FAULT_CUT_BEYOND_TEXT => "a range's cut is past the end of the prompt's pieces",
        FAULT_RANGE_TOO_LARGE => "a range has more tokens than the run's range size",
        FAULT_MERGES_INCOMPLETE => "a range's pieces could still merge",
        FAULT_TOKEN_SPLIT => "a claimed token's pieces are in two segments",
        FAULT_NOT_THE_MERGE => "the claimed tokens are not what the merge produces",
        FAULT_TEXT_CLAIM => "the text is not what the run claimed of it",
        FAULT_TEXT_AFTER_END => "a range was given text after the text had ended",
        FAULT_WORD_SPLIT => "the text has a run too long for the word scanner to place",
        _ => "an unknown fault",
    }
}

// ---------------------------------------------------------------------------
// Tokenizer tables
// ---------------------------------------------------------------------------

/// One vocabulary entry: a token string and the id it maps to.
///
/// All-scalar, so it is `Materializable`: a tile may take one whole, and a
/// `chunk = N` recur step may take a `Block` of them.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct TokenEntry {
    pub token: String,
    pub id: u32,
}

/// One BPE merge rule: `left + right -> merged`, ranked by `rank`
/// (lower rank wins across all eligible adjacent pairs).
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct BpeMerge {
    pub rank: u32,
    pub left: String,
    pub right: String,
    pub merged: String,
}

/// One vocabulary bucket: the entries whose token hashes to this slot.
///
/// `entries` may be empty. Both bucket scans are state-only recurs, and an
/// empty recur source skips the step and yields the initial state — which is
/// exactly "not found", so an empty bucket needs no special case.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct VocabBucket {
    pub entries: List<TokenEntry>,
}

/// One merge bucket: the rules whose `(left, right)` pair hashes to this slot.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct MergeBucket {
    pub rules: List<BpeMerge>,
}

/// The tokenizer's text-handling behaviour, as committed data.
///
/// All-scalar and small, so a tile may take it whole.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct TokenizerProfile {
    /// What the chat template puts before the trimmed prompt. A prompt that is
    /// not raw is rendered through the template when this or the suffix is not
    /// empty.
    pub template_prefix: String,
    /// What the chat template puts after the trimmed prompt.
    pub template_suffix: String,
    /// Spaces normalize to `▁`.
    pub normalize_spaces: bool,
    /// A literal space ends its segment.
    pub split_spaces: bool,
    pub byte_fallback: bool,
    pub fuse_unk: bool,
    pub has_unk: bool,
    pub unk_token: String,
    /// Special tokens longest first, separated by NUL.
    pub specials: String,
    /// 64 hex digits: bit `b` is set when `<0xBB>` is in the vocabulary.
    pub fallback_bytes: String,
    /// How text becomes atoms: [`SCANNER_CHARACTERS`] or [`SCANNER_WORDS`].
    pub scanner: u32,
    /// The longest token in bytes, for the word scanner: a longer word is no
    /// token, and is put out in parts.
    pub longest_token: u32,
}

/// One atom a character, with the profile's space and fallback rules.
pub const SCANNER_CHARACTERS: u32 = 0;
/// One atom a word, each written as its bytes. See [`tekken`].
pub const SCANNER_WORDS: u32 = 1;

impl TokenizerProfile {
    /// Whether the model has a chat template at all.
    pub fn templated(&self) -> bool {
        !self.template_prefix.is_empty() || !self.template_suffix.is_empty()
    }
}

/// Committed tokenizer fixture: the vocabulary and the merge table, each
/// bucketed by a hash of its key, plus the profile.
///
/// The bucketing is what makes a lookup affordable. A flat `List` can only be
/// searched by scanning it, and a scan is one replay unit per entry — 262,144
/// of them per piece for a real vocabulary. Hashing to a bucket turns that into
/// one tile (the hash), one dynamic-index `select!` (one authenticated node),
/// and a recur over the handful of entries that share the slot.
///
/// `*_bucket_count` is the modulus the hash is reduced by, and it MUST equal
/// the corresponding list's length: it is what guarantees the computed index is
/// in range, and an out-of-range dynamic index aborts the run. They travel in
/// one committed value so the commitment binds them together.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct PromptTokenizer {
    pub vocab_bucket_count: u32,
    pub merge_bucket_count: u32,
    pub profile: TokenizerProfile,
    pub vocab_buckets: List<VocabBucket>,
    pub merge_buckets: List<MergeBucket>,
}

/// FNV-1a offset basis and prime.
///
/// FNV is the right hash here for one reason: it is pure integer arithmetic
/// over bytes, so it is bit-identical on the host that writes the fixture and
/// in the RISC0 guest that replays the tile. A `DefaultHasher` (or anything
/// seeded, floating-point, or endianness-dependent) would put host and guest in
/// different buckets and silently tokenize to UNK.
pub const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
pub const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// Folds `bytes` into a running FNV-1a hash.
pub fn fnv1a64_continue(mut hash: u64, bytes: &[u8]) -> u64 {
    for byte in bytes {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// FNV-1a over a byte string.
pub fn fnv1a64(bytes: &[u8]) -> u64 {
    fnv1a64_continue(FNV_OFFSET_BASIS, bytes)
}

/// Bucket a vocabulary token lands in.
///
/// Shared verbatim by the tile and by the fixture generator — the one place
/// host and guest must agree byte-for-byte.
pub fn vocab_bucket_of(token: &str, bucket_count: u32) -> u32 {
    if bucket_count == 0 {
        return 0;
    }
    (fnv1a64(token.as_bytes()) % bucket_count as u64) as u32
}

/// Bucket a merge rule's `(left, right)` pair lands in.
///
/// The `0xff` separator is not a legal UTF-8 byte, so no pair of tokens can
/// forge another pair's hash by straddling the boundary: `("ab", "c")` and
/// `("a", "bc")` are distinct inputs here.
pub fn merge_bucket_of(left: &str, right: &str, bucket_count: u32) -> u32 {
    if bucket_count == 0 {
        return 0;
    }
    let mut hash = fnv1a64_continue(FNV_OFFSET_BASIS, left.as_bytes());
    hash = fnv1a64_continue(hash, &[0xff]);
    hash = fnv1a64_continue(hash, right.as_bytes());
    (hash % bucket_count as u64) as u32
}

// ---------------------------------------------------------------------------
// The run's committed inputs
// ---------------------------------------------------------------------------

/// The original UTF-8 prompt, in chunks cut at character boundaries.
///
/// The chunks concatenate to the text the commitment identifies. Where they
/// are cut changes nothing the chain computes.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct PromptText {
    pub chunks: List<String>,
}

/// What the run claims about the whole text, as one committed value.
///
/// The first range's bind needs each of these, and only the range that reaches
/// the end of the text can know them. So the run supplies them, every range
/// uses them, and the range the text ends in checks them against its fold.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct TextClaim {
    /// SHA-256 of the original UTF-8 prompt bytes, in lowercase hex.
    pub input_text_sha256: String,
    /// The text's length in bytes.
    pub total: u64,
    /// The bytes the template's trim keeps: from `keep_start` up to `keep_end`.
    pub keep_start: u64,
    pub keep_end: u64,
}

/// The run's configuration, as one committed value.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct RunConfig {
    /// The model's identifier: a hash of its imported commitments by role.
    pub model_id: String,
    pub raw_prompt: bool,
    pub max_new_tokens: u32,
    pub range_size: u32,
}

/// The run's stage schedule, as one committed value: which stages run.
///
/// Run preparation chooses it. The first range's `prompt-bind` checks it and
/// stores it in the context, and it travels from there with the statement.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct RunSchedule {
    pub prompt_ranges: u32,
    pub claimed_ranges: u32,
}

/// The context after the run's schedule has been checked and stored.
pub fn bind_run_schedule(mut context: PromptContext, schedule: &RunSchedule) -> PromptContext {
    context
        .model
        .bind_schedule(model_identity::pipeline_core::StageSchedule {
            prompt_ranges: schedule.prompt_ranges,
            claimed_ranges: schedule.claimed_ranges,
        });
    context
}

// ---------------------------------------------------------------------------
// The prompt stages' shared output
// ---------------------------------------------------------------------------

/// Explicit segmentation: adjacent pieces may merge only within one segment.
/// Special tokens occupy their own segment. There is no sentinel token.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct BpePiece {
    pub text: String,
    pub segment: u32,
}

/// What the prompt stages have established so far.
///
/// All-scalar and small, so it travels through every prompt stage.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct PromptContext {
    pub model: model_identity::ModelContext,
    /// SHA-256 of the original UTF-8 prompt bytes, in lowercase hex.
    pub input_text_sha256: String,
    pub model_id: String,
    pub raw_prompt: bool,
    pub max_new_tokens: u32,
    /// Zero outside the ranged chain, where no range limit applies.
    pub range_size: u32,
    /// The piece the next range starts at.
    pub next_piece: u64,
    /// Tokens produced by the ranges prepared so far.
    pub prompt_tokens: u32,
    /// The ranges so far cover every piece of the prompt.
    pub complete: bool,
    /// The first thing that went wrong, or [`FAULT_NONE`].
    pub fault: u32,
    /// Where it went wrong: a piece index, or a token position for faults found
    /// after merging.
    pub fault_position: u64,
}

/// The one output type of every prompt stage, so each may follow any other.
///
/// `prompt-bind` publishes the pieces of one range's text, and in `bind` what
/// the next range's bind continues from. `prompt-check` publishes one range's
/// token ids, with the pieces of the range's last token for the next range to
/// check against, and in `rest` the pieces it was given and did not take.
/// Only the last range's context covers the prompt.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct PromptState {
    pub pieces: List<BpePiece>,
    /// Pieces a check was given past the end of its range, in order. The next
    /// range's check takes them before its own bind's.
    pub rest: List<BpePiece>,
    pub token_ids: List<u32>,
    pub context: PromptContext,
    pub bind: BindCarry,
}

/// What one range's bind hands the next: the claim, and the three loops over
/// the text exactly where this range left them.
///
/// Each loop's result depends only on its state and the chunks it is fed, in
/// order. So binding the text in ranges, each from the state before, gives the
/// pieces binding it whole gives, wherever the ranges are cut.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct BindCarry {
    pub claim: TextClaim,
    /// The text fold and the scan's position.
    pub text: TextPass,
    pub split: SplitState,
    /// The text has ended, in this range or an earlier one, and is what the
    /// claim said.
    pub done: bool,
}

/// The stage output as a list of pieces.
pub type BpePieces = PromptState;

/// The stage output as a range's token ids.
pub type PromptTokenization = PromptState;

/// One merge round's pieces, inside a stage.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct MergeRound {
    pub model_fault: u32,
    pub pieces: List<BpePiece>,
}

impl PromptState {
    /// A prompt state holding only pieces, as an unranged run prepares them.
    pub fn of_pieces(pieces: Vec<BpePiece>) -> Self {
        Self {
            pieces: List::from(pieces),
            ..Self::default()
        }
    }

    /// A prompt state holding only token ids.
    pub fn of_token_ids(token_ids: Vec<u32>) -> Self {
        Self {
            token_ids: List::from(token_ids),
            ..Self::default()
        }
    }
}

// ---------------------------------------------------------------------------
// Ranked-merge working values
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct MergeCandidate {
    pub model_fault: u32,
    pub matched: bool,
    pub rank: u32,
    pub left: u64,
    pub merged: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct MergeScan {
    pub previous: BpePiece,
    pub has_previous: bool,
    pub position: u64,
    pub best: MergeCandidate,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct MergeStep {
    pub pending: String,
    pub has_pending: bool,
    pub piece: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct MergeMatch {
    pub matched: bool,
    pub rank: u32,
    pub merged: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct VocabQuery {
    pub piece: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct VocabMatch {
    pub found: bool,
    pub token_id: u32,
}

// ---------------------------------------------------------------------------
// Hashing and measuring the original text
// ---------------------------------------------------------------------------

/// A fold over the text's chunks: its running SHA-256, and where the text
/// starts and ends once the template's trim removes surrounding whitespace.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct TextFold {
    /// The eight SHA-256 chaining words, big-endian.
    pub sha_words: BytesPage,
    /// Bytes waiting for a whole 64-byte block.
    pub sha_tail: BytesPage,
    pub total: u64,
    pub chunks: u32,
    pub has_text: bool,
    /// Byte offset of the first non-whitespace character.
    pub first_text: u64,
    /// Byte offset just past the last non-whitespace character.
    pub text_end: u64,
}

const SHA256_INIT: [u32; 8] = [
    0x6a09_e667,
    0xbb67_ae85,
    0x3c6e_f372,
    0xa54f_f53a,
    0x510e_527f,
    0x9b05_688c,
    0x1f83_d9ab,
    0x5be0_cd19,
];

fn page(bytes: Vec<u8>) -> BytesPage {
    BytesPage::__from_parts(0, 0, bytes)
}

fn words_page(words: &[u32; 8]) -> BytesPage {
    page(words.iter().flat_map(|word| word.to_be_bytes()).collect())
}

fn page_words(page: &BytesPage) -> [u32; 8] {
    let mut words = [0_u32; 8];
    for (word, bytes) in words.iter_mut().zip(page.as_slice().chunks_exact(4)) {
        *word = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    }
    words
}

fn compress_blocks(words: &mut [u32; 8], bytes: &[u8]) -> usize {
    let mut used = 0;
    for block in bytes.chunks_exact(64) {
        let block = GenericArray::clone_from_slice(block);
        sha2::compress256(words, core::slice::from_ref(&block));
        used += 64;
    }
    used
}

impl Default for TextFold {
    fn default() -> Self {
        begin_text_fold_state()
    }
}

/// The fold before any chunk.
pub fn begin_text_fold_state() -> TextFold {
    TextFold {
        sha_words: words_page(&SHA256_INIT),
        sha_tail: page(Vec::new()),
        total: 0,
        chunks: 0,
        has_text: false,
        first_text: 0,
        text_end: 0,
    }
}

/// Adds one chunk of the original text to the fold.
pub fn fold_text(mut fold: TextFold, chunk: &str) -> TextFold {
    let mut words = page_words(&fold.sha_words);
    let mut waiting = fold.sha_tail.as_slice().to_vec();
    waiting.extend_from_slice(chunk.as_bytes());
    let used = compress_blocks(&mut words, &waiting);
    fold.sha_words = words_page(&words);
    fold.sha_tail = page(waiting[used..].to_vec());

    for (offset, ch) in chunk.char_indices() {
        if ch.is_whitespace() {
            continue;
        }
        let at = fold.total + offset as u64;
        if !fold.has_text {
            fold.has_text = true;
            fold.first_text = at;
        }
        fold.text_end = at + ch.len_utf8() as u64;
    }
    fold.total += chunk.len() as u64;
    fold.chunks += 1;
    fold
}

/// The SHA-256 of every byte the fold has seen, in lowercase hex.
pub fn text_sha256_hex(fold: &TextFold) -> String {
    let mut words = page_words(&fold.sha_words);
    let mut waiting = fold.sha_tail.as_slice().to_vec();
    waiting.push(0x80);
    while waiting.len() % 64 != 56 {
        waiting.push(0);
    }
    waiting.extend_from_slice(&(fold.total * 8).to_be_bytes());
    compress_blocks(&mut words, &waiting);
    let mut hex = String::with_capacity(64);
    for word in words {
        hex.push_str(&format!("{word:08x}"));
    }
    hex
}

/// The prompt context a run starts from. The text's hash is the claim's,
/// until the range the text ends in has checked it.
pub fn open_prompt_context(claim: &TextClaim, config: &RunConfig) -> PromptContext {
    PromptContext {
        input_text_sha256: claim.input_text_sha256.clone(),
        model_id: config.model_id.clone(),
        raw_prompt: config.raw_prompt,
        max_new_tokens: config.max_new_tokens,
        range_size: config.range_size,
        ..PromptContext::default()
    }
}

/// Keeps the first fault: the later ones are often its consequences.
pub fn record_fault(context: &mut PromptContext, fault: u32, position: u64) {
    if context.fault == FAULT_NONE {
        context.fault = fault;
        context.fault_position = position;
    }
}

// ---------------------------------------------------------------------------
// Rendering and scanning the text into atoms
// ---------------------------------------------------------------------------

/// How the scan renders the text: the part the trim keeps, and the template.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct ScanPlan {
    pub total: u64,
    pub keep_start: u64,
    pub keep_end: u64,
    pub templated: bool,
}

/// One special token, one normalized character, or one word.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct PromptAtom {
    pub text: String,
    pub special: bool,
    /// [`ATOM_WHOLE`], or which part of a word too long for one atom.
    pub part: u32,
}

/// The atom is a whole unit: its lookup says whether it is one piece.
pub const ATOM_WHOLE: u32 = 0;
/// The atom starts a word too long to be a token. It is its bytes.
pub const ATOM_FIRST_PART: u32 = 1;
/// The atom continues that word, in the same segment.
pub const ATOM_LATER_PART: u32 = 2;

/// The scan's output.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct PromptAtoms {
    pub atoms: List<PromptAtom>,
}

/// The scan's position in the text, and rendered text it cannot decide yet.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct ScanState {
    pub pending: String,
    pub offset: u64,
    pub finished: bool,
    /// The rest is the word scanner's, and stays zero for any other.
    ///
    /// Where it is in the pattern: one of [`tekken`]'s places.
    pub mode: u32,
    /// Bytes at the start of `pending` that are in the word so far.
    pub word: u32,
    /// Bytes after those that it cannot place in a word yet.
    pub held: u32,
    /// The word so far has already put out a part.
    pub long: bool,
    /// It has had to place held text without knowing where the text belongs.
    pub overflowed: bool,
}

/// The scan the claim calls for. Every range scans with it.
pub fn claimed_scan_plan(
    claim: &TextClaim,
    raw_prompt: bool,
    profile: &TokenizerProfile,
) -> ScanPlan {
    ScanPlan {
        total: claim.total,
        keep_start: claim.keep_start,
        keep_end: claim.keep_end,
        templated: !raw_prompt && profile.templated(),
    }
}

/// What the run may claim of a text it has folded to its end.
pub fn text_claim_of(fold: &TextFold, raw_prompt: bool, profile: &TokenizerProfile) -> TextClaim {
    let plan = scan_plan_of(fold, raw_prompt, profile);
    TextClaim {
        input_text_sha256: text_sha256_hex(fold),
        total: plan.total,
        keep_start: plan.keep_start,
        keep_end: plan.keep_end,
    }
}

/// The scan a whole text's fold and the run's configuration call for.
pub fn scan_plan_of(fold: &TextFold, raw_prompt: bool, profile: &TokenizerProfile) -> ScanPlan {
    let templated = !raw_prompt && profile.templated();
    let (keep_start, keep_end) = match (templated, fold.has_text) {
        (false, _) => (0, fold.total),
        (true, true) => (fold.first_text, fold.text_end),
        (true, false) => (0, 0),
    };
    ScanPlan {
        total: fold.total,
        keep_start,
        keep_end,
        templated,
    }
}

/// The scan before any chunk: the template's prefix is already pending.
pub fn begin_scan_state(plan: &ScanPlan, profile: &TokenizerProfile) -> ScanState {
    ScanState {
        pending: if plan.templated {
            profile.template_prefix.clone()
        } else {
            String::new()
        },
        ..ScanState::default()
    }
}

fn specials_of(profile: &TokenizerProfile) -> impl Iterator<Item = &str> {
    profile.specials.split('\0').filter(|text| !text.is_empty())
}

/// Turns pending text into atoms. Until the text ends, it holds back anything
/// shorter than the longest special token, which could still become one.
fn drain_atoms(
    pending: &mut String,
    last: bool,
    profile: &TokenizerProfile,
    emit: &mut impl FnMut(PromptAtom),
) {
    let lookahead = specials_of(profile).map(str::len).max().unwrap_or(0);
    let mut start = 0;
    while start < pending.len() {
        let rest = &pending[start..];
        if !last && rest.len() < lookahead {
            break;
        }
        if let Some(special) = specials_of(profile).find(|text| rest.starts_with(text)) {
            emit(PromptAtom {
                text: special.to_string(),
                special: true,
                part: ATOM_WHOLE,
            });
            start += special.len();
            continue;
        }
        let ch = rest.chars().next().expect("nonempty pending text");
        start += ch.len_utf8();
        let text = if profile.normalize_spaces && ch == ' ' {
            String::from("▁")
        } else {
            ch.to_string()
        };
        emit(PromptAtom {
            text,
            special: false,
            part: ATOM_WHOLE,
        });
    }
    pending.drain(..start);
}

/// Scans one chunk of the original text. The chunk that reaches the end of the
/// text also adds the template's suffix and drains everything.
pub fn scan_chunk_into(
    mut state: ScanState,
    chunk: &str,
    plan: &ScanPlan,
    profile: &TokenizerProfile,
    emit: &mut impl FnMut(PromptAtom),
) -> ScanState {
    for (offset, ch) in chunk.char_indices() {
        let at = state.offset + offset as u64;
        if at >= plan.keep_start && at < plan.keep_end {
            state.pending.push(ch);
        }
    }
    state.offset += chunk.len() as u64;
    let last = !state.finished && state.offset == plan.total;
    if last {
        if plan.templated {
            state.pending.push_str(&profile.template_suffix);
        }
        state.finished = true;
    }
    if profile.scanner == SCANNER_WORDS {
        tekken::drain_words(&mut state, last, profile, emit);
    } else {
        drain_atoms(&mut state.pending, last, profile, emit);
    }
    state
}

// ---------------------------------------------------------------------------
// Splitting atoms into pieces
// ---------------------------------------------------------------------------

/// The split's position: its segment and whether the last atom was unknown.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct SplitState {
    pub segment: u32,
    pub previous_unknown: bool,
}

/// One atom and the vocabulary lookup it needs.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct AtomStep {
    pub query: VocabQuery,
    pub atom: PromptAtom,
    pub state: SplitState,
}

pub fn fallback_has(profile: &TokenizerProfile, byte: u8) -> bool {
    let digit = profile
        .fallback_bytes
        .as_bytes()
        .get(byte as usize / 4)
        .copied()
        .unwrap_or(b'0');
    let nibble = match digit {
        b'0'..=b'9' => digit - b'0',
        b'a'..=b'f' => digit - b'a' + 10,
        _ => 0,
    };
    nibble & (1 << (byte % 4)) != 0
}

/// `fallback_bytes` for the bytes the predicate accepts.
pub fn fallback_mask(has: impl Fn(u8) -> bool) -> String {
    (0..64_u32)
        .map(|digit| {
            let nibble = (0..4_u32)
                .filter(|bit| has((digit * 4 + bit) as u8))
                .fold(0_u32, |nibble, bit| nibble | (1 << bit));
            char::from_digit(nibble, 16).expect("a nibble is one hex digit")
        })
        .collect()
}

/// The vocabulary token of one fallback byte.
pub fn byte_token(byte: u8) -> String {
    format!("<0x{byte:02X}>")
}

/// Applies one atom: the pieces it becomes, and the split's next position.
///
/// Supported tokenizer profile: Gemma's literal space normalization/splitting,
/// atomic added tokens, character pieces, byte fallback and fused unknowns.
pub fn apply_atom(
    state: &SplitState,
    atom: &PromptAtom,
    in_vocabulary: bool,
    profile: &TokenizerProfile,
    emit: &mut impl FnMut(BpePiece),
) -> SplitState {
    let mut next = state.clone();
    if atom.special {
        next.segment = next.segment.saturating_add(1);
        emit(BpePiece {
            text: atom.text.clone(),
            segment: next.segment,
        });
        next.segment = next.segment.saturating_add(1);
        next.previous_unknown = false;
        return next;
    }
    if profile.scanner == SCANNER_WORDS {
        return tekken::apply_word(state, atom, in_vocabulary, emit);
    }
    if in_vocabulary {
        emit(BpePiece {
            text: atom.text.clone(),
            segment: next.segment,
        });
        next.previous_unknown = false;
    } else if profile.byte_fallback && atom.text.bytes().all(|byte| fallback_has(profile, byte)) {
        for byte in atom.text.bytes() {
            emit(BpePiece {
                text: byte_token(byte),
                segment: next.segment,
            });
        }
        next.previous_unknown = false;
    } else if profile.has_unk {
        if !profile.fuse_unk || !next.previous_unknown {
            emit(BpePiece {
                text: profile.unk_token.clone(),
                segment: next.segment,
            });
        }
        next.previous_unknown = true;
    } else {
        // BPE without an unknown token drops characters outside its vocabulary
        // when byte fallback cannot represent them.
        next.previous_unknown = false;
    }
    // Split(MergedWithPrevious) runs after normalization. In production spaces
    // have become ▁, so it introduces no new boundaries.
    if profile.split_spaces && atom.text == " " {
        next.segment = next.segment.saturating_add(1);
        next.previous_unknown = false;
    }
    next
}

// ---------------------------------------------------------------------------
// Binding the text one range at a time
// ---------------------------------------------------------------------------

/// What the first range's bind starts from: the claim, and each loop before
/// any text.
pub fn begin_bind_carry(
    claim: &TextClaim,
    raw_prompt: bool,
    profile: &TokenizerProfile,
) -> BindCarry {
    BindCarry {
        claim: claim.clone(),
        text: TextPass {
            fold: begin_text_fold_state(),
            scan: begin_scan_state(&claimed_scan_plan(claim, raw_prompt, profile), profile),
        },
        split: SplitState::default(),
        done: false,
    }
}

/// Closes one range's bind, once its text has been folded and scanned.
///
/// `before` is what the range started from. The range the text ends in holds
/// the text to its claim; a range after that one may be given no text.
pub fn finish_bind_range(
    context: &mut PromptContext,
    before: &BindCarry,
    carry: &mut BindCarry,
    profile: &TokenizerProfile,
) {
    if carry.text.fold.chunks == 0 {
        record_fault(context, FAULT_NO_TEXT, 0);
    }
    if carry.text.scan.overflowed {
        record_fault(context, FAULT_WORD_SPLIT, carry.text.fold.total);
    }
    if before.done {
        if carry.text.fold.chunks != before.text.fold.chunks {
            record_fault(context, FAULT_TEXT_AFTER_END, before.text.fold.total);
        }
        return;
    }
    if !carry.text.scan.finished {
        return;
    }
    let raw = context.raw_prompt;
    let as_claimed = scan_plan_of(&carry.text.fold, raw, profile)
        == claimed_scan_plan(&carry.claim, raw, profile)
        && text_sha256_hex(&carry.text.fold) == carry.claim.input_text_sha256;
    if as_claimed {
        carry.done = true;
    } else {
        record_fault(context, FAULT_TEXT_CLAIM, carry.text.fold.total);
    }
}

/// What one range's bind starts from: its context and the state it continues.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct BindOpen {
    pub context: PromptContext,
    pub carry: BindCarry,
}

/// Opens one range's bind.
///
/// The first range has nothing before it (`previous` is the empty state): it
/// binds the model, the statement and the schedule, and starts the three loops.
/// Every later range continues the bind before it, and reads none of the run's
/// inputs again. Each checks the tokenizer it was given against the model.
#[allow(clippy::too_many_arguments)]
pub fn open_bind_range(
    previous: &PromptContext,
    previous_carry: &BindCarry,
    config: &RunConfig,
    descriptor: &[u8],
    schedule: &RunSchedule,
    claim: &TextClaim,
    tokenizer_root: &BytesPage,
    profile: &TokenizerProfile,
) -> BindOpen {
    use model_identity::{stream, ModelContext, Role};
    if *previous == PromptContext::default() {
        let mut context = open_prompt_context(claim, config);
        context.model = ModelContext::bind(descriptor, &config.model_id);
        context.model = stream::verify(context.model, tokenizer_root, Role::Tokenizer);
        context.model.bind_statement(
            &context.input_text_sha256,
            context.raw_prompt,
            context.max_new_tokens,
            context.range_size,
        );
        let context = bind_run_schedule(context, schedule);
        let carry = begin_bind_carry(claim, context.raw_prompt, profile);
        BindOpen { context, carry }
    } else {
        let mut context = previous.clone();
        context.model = stream::verify(context.model, tokenizer_root, Role::Tokenizer);
        context
            .model
            .require(context.model_id == context.model.model_id);
        context.model.continue_bind();
        BindOpen {
            context,
            carry: previous_carry.clone(),
        }
    }
}

/// The two loops over a range's text that end in a state: the fold, and the
/// scan's position. The scan's atoms are built by a loop of their own.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct TextPass {
    pub fold: TextFold,
    pub scan: ScanState,
}

/// Adds one chunk to the fold and moves the scan past it.
pub fn advance_text_pass(
    pass: TextPass,
    chunk: &str,
    plan: &ScanPlan,
    profile: &TokenizerProfile,
) -> TextPass {
    TextPass {
        fold: fold_text(pass.fold, chunk),
        scan: scan_chunk_into(pass.scan, chunk, plan, profile, &mut |_| {}),
    }
}

/// One walk of a vocabulary bucket: its entries hashed in order, and whether
/// one of them is the piece looked for.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct BucketWalk {
    pub entries: BytesPage,
    pub found: bool,
    pub token_id: u32,
}

/// The walk before any entry.
pub fn begin_bucket_walk() -> BucketWalk {
    BucketWalk {
        entries: model_identity::stream::empty(),
        found: false,
        token_id: 0,
    }
}

/// Adds one entry to the walk.
pub fn walk_bucket_entry(mut walk: BucketWalk, entry: &TokenEntry, piece: &str) -> BucketWalk {
    use model_identity::stream;
    walk.entries = stream::append(&walk.entries, &stream::digest(entry));
    if entry.token == piece {
        walk.found = true;
        walk.token_id = entry.id;
    }
    walk
}

/// Whether the bucket a walk went through is bucket `index` of the committed
/// vocabulary.
pub fn bucket_walk_proven(
    walk: &BucketWalk,
    index: u32,
    header: &SparseHeader,
    path: &SparsePath,
) -> bool {
    use model_identity::{sparse, stream};
    let root = stream::structure(&[("entries", stream::finish(&walk.entries))]);
    sparse::verify(&header.root, header.count, u64::from(index), path, &root)
}

/// An atom with what its bucket said of it.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct KnownAtom {
    pub text: String,
    pub special: bool,
    pub part: u32,
    /// The atom is a vocabulary entry.
    pub found: bool,
    /// The bucket that answered is the committed one.
    pub proven: bool,
}

/// A range's atoms, looked up.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct KnownAtoms {
    pub atoms: List<KnownAtom>,
}

/// Where the split ends, and whether every lookup on the way was proven.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct BindTail {
    pub split: SplitState,
    pub unproven: bool,
}

/// The tail before any of this range's atoms.
pub fn begin_bind_tail(before: &BindCarry) -> BindTail {
    BindTail {
        split: before.split.clone(),
        unproven: false,
    }
}

/// Moves the split past one atom, emitting its pieces.
pub fn split_known_atom(
    split: &SplitState,
    atom: &KnownAtom,
    profile: &TokenizerProfile,
    emit: &mut impl FnMut(BpePiece),
) -> SplitState {
    let plain = PromptAtom {
        text: atom.text.clone(),
        special: atom.special,
        part: atom.part,
    };
    apply_atom(split, &plain, atom.found, profile, emit)
}

/// Moves the tail past one atom.
pub fn advance_bind_tail(tail: BindTail, atom: &KnownAtom, profile: &TokenizerProfile) -> BindTail {
    BindTail {
        split: split_known_atom(&tail.split, atom, profile, &mut |_| {}),
        unproven: tail.unproven || !atom.proven,
    }
}

/// Closes one range's bind: what the next range continues from, and the
/// context with anything this range found wrong.
pub fn close_bind_range(
    mut context: PromptContext,
    before: &BindCarry,
    pass: TextPass,
    tail: BindTail,
    profile: &TokenizerProfile,
) -> BindOpen {
    context.model.require(!tail.unproven);
    let mut carry = BindCarry {
        claim: before.claim.clone(),
        text: pass,
        split: tail.split,
        done: before.done,
    };
    finish_bind_range(&mut context, before, &mut carry, profile);
    BindOpen { context, carry }
}

/// Binds one range's text from what `open_bind_range` gave, as a `prompt-bind`
/// stage does, step for step. `lookup` answers for one atom: whether it is a
/// vocabulary entry, and whether that answer is proven.
///
/// For the native kernel and run preparation. A stage runs the same functions
/// from its tiles, with the lookup an authenticated walk of the atom's bucket.
pub fn bind_range<S: AsRef<str>>(
    opened: BindOpen,
    chunks: &[S],
    profile: &TokenizerProfile,
    mut lookup: impl FnMut(&str) -> (bool, bool),
    emit: &mut impl FnMut(BpePiece),
) -> BindOpen {
    let BindOpen {
        context,
        carry: before,
    } = opened;
    let plan = claimed_scan_plan(&before.claim, context.raw_prompt, profile);
    let mut pass = before.text.clone();
    for chunk in chunks {
        pass = advance_text_pass(pass, chunk.as_ref(), &plan, profile);
    }
    let mut atoms = Vec::new();
    let mut scan = before.text.scan.clone();
    for chunk in chunks {
        scan = scan_chunk_into(scan, chunk.as_ref(), &plan, profile, &mut |atom| {
            let (found, proven) = lookup(&atom.text);
            atoms.push(KnownAtom {
                text: atom.text,
                special: atom.special,
                part: atom.part,
                found,
                proven,
            })
        });
    }
    let mut tail = begin_bind_tail(&before);
    let mut split = before.split.clone();
    for atom in &atoms {
        tail = advance_bind_tail(tail, atom, profile);
        split = split_known_atom(&split, atom, profile, emit);
    }
    close_bind_range(context, &before, pass, tail, profile)
}

/// Splits a rendered string into pieces, given a vocabulary membership test.
pub fn split_rendered(
    rendered: &str,
    profile: &TokenizerProfile,
    in_vocabulary: impl Fn(&str) -> bool,
) -> Vec<BpePiece> {
    let plan = ScanPlan {
        total: rendered.len() as u64,
        keep_start: 0,
        keep_end: rendered.len() as u64,
        templated: false,
    };
    let mut atoms = Vec::new();
    scan_chunk_into(
        begin_scan_state(&plan, profile),
        rendered,
        &plan,
        profile,
        &mut |atom| atoms.push(atom),
    );
    let mut pieces = Vec::new();
    let mut state = SplitState::default();
    for atom in &atoms {
        state = apply_atom(
            &state,
            atom,
            in_vocabulary(&atom.text),
            profile,
            &mut |piece| pieces.push(piece),
        );
    }
    pieces
}

// ---------------------------------------------------------------------------
// Cutting one range out of the prompt's pieces
// ---------------------------------------------------------------------------

/// A fold over the pieces a range's check is given: the ones the check
/// before it left, then its own bind's. Positions count from the start of the
/// prompt, so `position` ends at the number of pieces bound so far.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct SliceState {
    pub start: u64,
    pub end: u64,
    pub position: u64,
    pub has_left: bool,
    pub left: BpePiece,
    pub has_right: bool,
    pub right: BpePiece,
}

/// The fold before any piece, for a range ending at `end`. The first piece
/// the check is given is the one the range starts at.
pub fn begin_slice_state(context: &PromptContext, end: u64) -> SliceState {
    SliceState {
        start: context.next_piece,
        end,
        position: context.next_piece,
        ..SliceState::default()
    }
}

/// Adds one piece to the fold.
pub fn observe_piece(state: &mut SliceState, piece: &BpePiece) {
    if state.end > 0 && state.position == state.end - 1 {
        state.has_left = true;
        state.left = piece.clone();
    }
    if state.position == state.end {
        state.has_right = true;
        state.right = piece.clone();
    }
    state.position += 1;
}

/// The context a range's token check starts from.
///
/// A check continues the check before it, which holds the tokens so far. The
/// first range has none: it continues from its own bind, and `previous` is the
/// empty state. Either way the range is held to its bind and to what the
/// schedule puts before it.
pub fn join_prompt_models(previous: PromptContext, bound: &PromptContext) -> PromptContext {
    let first = previous == PromptContext::default();
    let mut joined = if first {
        bound.clone()
    } else {
        previous.clone()
    };
    if !first {
        if previous.input_text_sha256 != bound.input_text_sha256
            || previous.model_id != bound.model_id
            || previous.raw_prompt != bound.raw_prompt
            || previous.max_new_tokens != bound.max_new_tokens
            || previous.range_size != bound.range_size
        {
            let position = previous.next_piece;
            record_fault(&mut joined, FAULT_CONTEXT_MISMATCH, position);
        }
        if bound.fault != FAULT_NONE {
            record_fault(&mut joined, bound.fault, bound.fault_position);
        }
        joined.model.same_model(&bound.model);
    }
    let valid = joined.model.valid()
        && joined.model.producer == model_identity::context::PROMPT
        && bound.model.producer == model_identity::context::PROMPT
        && joined.model.model_id == joined.model_id
        && bound.model.model_id == bound.model_id;
    joined.model.require(valid);
    joined.model.require_execution(
        bound
            .model
            .execution_context()
            .is_some_and(|e| e.phase == model_identity::pipeline_core::BOUND),
    );
    joined.model.begin_prompt_range();
    joined.model.require_bound(&bound.model);
    joined.model.require_previous_check(&previous.model);
    joined
}

pub use model_identity::sparse::{SparseHeader, SparsePath, SparseWitness};
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct TokenizerWitness {
    pub vocab: SparseHeader,
    pub vocab_paths: List<SparsePath>,
    pub merges: SparseHeader,
    pub merge_paths: List<SparsePath>,
}
pub fn tokenizer_witness(
    tokenizer: &PromptTokenizer,
) -> core::result::Result<TokenizerWitness, model_identity::ModelFault> {
    let vocab = model_identity::sparse::witness(tokenizer.vocab_buckets.as_slice())?;
    let merges = model_identity::sparse::witness(tokenizer.merge_buckets.as_slice())?;
    Ok(TokenizerWitness {
        vocab: vocab.header,
        vocab_paths: vocab.paths,
        merges: merges.header,
        merge_paths: merges.paths,
    })
}
