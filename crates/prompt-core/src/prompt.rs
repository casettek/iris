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

/// Token id reserved for pieces that are not in the vocabulary.
pub const UNK_TOKEN_ID: u32 = 0;

/// What the chat template adds before and after the trimmed prompt.
pub const PROMPT_PREFIX: &str = "<bos><|turn>user\n";
pub const PROMPT_SUFFIX: &str = "<turn|>\n<|turn>model\n";

/// `PromptContext::fault` values. Zero is a clean preparation.
pub const FAULT_NONE: u32 = 0;
/// The committed text has no chunks at all, not even an empty one.
pub const FAULT_NO_TEXT: u32 = 1;
/// A range's cut does not move past the previous range's.
pub const FAULT_EMPTY_RANGE: u32 = 2;
/// A range's cut is past the end of the prompt's pieces.
pub const FAULT_CUT_BEYOND_TEXT: u32 = 3;
/// A range's cut is at a boundary a merge could cross.
pub const FAULT_UNSAFE_CUT: u32 = 4;
/// A range has more tokens than the run's range size.
pub const FAULT_RANGE_TOO_LARGE: u32 = 5;
/// A range's pieces could still merge: too few tokenizer batches ran.
pub const FAULT_MERGES_INCOMPLETE: u32 = 6;

/// What a `PromptContext::fault` value means.
pub fn fault_description(fault: u32) -> &'static str {
    match fault {
        FAULT_NONE => "no fault",
        FAULT_NO_TEXT => "the committed text has no chunks",
        FAULT_EMPTY_RANGE => "a range's cut does not move past the previous range's",
        FAULT_CUT_BEYOND_TEXT => "a range's cut is past the end of the prompt's pieces",
        FAULT_UNSAFE_CUT => "a range's cut is at a boundary a merge could cross",
        FAULT_RANGE_TOO_LARGE => "a range has more tokens than the run's range size",
        FAULT_MERGES_INCOMPLETE => "a range's pieces could still merge",
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

/// Two characters that sit next to each other inside some vocabulary token.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct BoundaryPair {
    pub left: u32,
    pub right: u32,
}

/// One boundary bucket: the pairs that hash to this slot.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct BoundaryBucket {
    pub pairs: List<BoundaryPair>,
}

/// The tokenizer's text-handling behaviour, as committed data.
///
/// All-scalar and small, so a tile may take it whole.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct TokenizerProfile {
    /// The vocabulary has the chat template's tokens, so a prompt that is not
    /// raw is rendered through the template.
    pub templated: bool,
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
}

/// Committed tokenizer fixture: the vocabulary, the merge table and the
/// boundary table, each bucketed by a hash of its key, plus the profile.
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
    pub boundary_bucket_count: u32,
    pub profile: TokenizerProfile,
    pub vocab_buckets: List<VocabBucket>,
    pub merge_buckets: List<MergeBucket>,
    pub boundary_buckets: List<BoundaryBucket>,
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

/// Bucket a boundary pair lands in.
pub fn boundary_bucket_of(left: u32, right: u32, bucket_count: u32) -> u32 {
    if bucket_count == 0 {
        return 0;
    }
    let hash = fnv1a64_continue(FNV_OFFSET_BASIS, &left.to_le_bytes());
    let hash = fnv1a64_continue(hash, &right.to_le_bytes());
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

/// The run's configuration, as one committed value.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct RunConfig {
    /// The model's identifier: a hash of its imported commitments by role.
    pub model_id: String,
    pub raw_prompt: bool,
    pub max_new_tokens: u32,
    pub range_size: u32,
}

/// Where one prompt range ends, as an index into the prompt's pieces.
///
/// Run preparation chooses it. The chain checks it rather than trusting it.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct RangeCut {
    pub end: u64,
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
/// `prompt-bind` publishes every piece of the prompt. `prompt-seed` and
/// `prompt-merge` publish one range's pieces. `prompt-prepare` publishes that
/// range's token ids. Only the last prompt stage's context covers the prompt.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct PromptState {
    pub pieces: List<BpePiece>,
    pub token_ids: List<u32>,
    pub context: PromptContext,
}

/// The stage output as a list of pieces.
pub type BpePieces = PromptState;

/// The stage output as a range's token ids.
pub type PromptTokenization = PromptState;

/// One merge round's pieces, inside a stage.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct MergeRound {
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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct RankedStep {
    pub query: MergeStep,
    pub piece: BpePiece,
    pub position: u64,
    pub best: MergeCandidate,
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

/// The prompt context a run starts from.
pub fn open_prompt_context(fold: &TextFold, config: &RunConfig) -> PromptContext {
    let mut context = PromptContext {
        input_text_sha256: text_sha256_hex(fold),
        model_id: config.model_id.clone(),
        raw_prompt: config.raw_prompt,
        max_new_tokens: config.max_new_tokens,
        range_size: config.range_size,
        ..PromptContext::default()
    };
    if fold.chunks == 0 {
        record_fault(&mut context, FAULT_NO_TEXT, 0);
    }
    context
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

/// One special token, or one normalized character.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct PromptAtom {
    pub text: String,
    pub special: bool,
}

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
}

/// The scan a fold's text and the run's configuration call for.
pub fn scan_plan_of(fold: &TextFold, config: &RunConfig, profile: &TokenizerProfile) -> ScanPlan {
    let templated = !config.raw_prompt && profile.templated;
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

/// The scan before any chunk.
pub fn begin_scan_state(plan: &ScanPlan) -> ScanState {
    ScanState {
        pending: if plan.templated {
            String::from(PROMPT_PREFIX)
        } else {
            String::new()
        },
        offset: 0,
        finished: false,
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
            state.pending.push_str(PROMPT_SUFFIX);
        }
        state.finished = true;
    }
    drain_atoms(&mut state.pending, last, profile, emit);
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
    scan_chunk_into(begin_scan_state(&plan), rendered, &plan, profile, &mut |atom| {
        atoms.push(atom)
    });
    let mut pieces = Vec::new();
    let mut state = SplitState::default();
    for atom in &atoms {
        state = apply_atom(&state, atom, in_vocabulary(&atom.text), profile, &mut |piece| {
            pieces.push(piece)
        });
    }
    pieces
}

// ---------------------------------------------------------------------------
// Cutting one range out of the prompt's pieces
// ---------------------------------------------------------------------------

/// A fold over every piece of the prompt: how many there are, and the two
/// pieces either side of this range's cut.
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

/// The two characters either side of a cut, if a merge could cross it at all.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct BoundaryQuery {
    pub active: bool,
    pub left: u32,
    pub right: u32,
}

/// Whether the boundary table holds the query's pair.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct BoundaryHit {
    pub found: bool,
}

/// The fold before any piece, for a range ending at `end`.
pub fn begin_slice_state(context: &PromptContext, end: u64) -> SliceState {
    SliceState {
        start: context.next_piece,
        end,
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

/// The pair a safe cut must not be able to merge across.
pub fn boundary_query_of(state: &SliceState) -> BoundaryQuery {
    let edges = state
        .left
        .text
        .chars()
        .last()
        .zip(state.right.text.chars().next());
    match edges {
        Some((left, right))
            if state.has_left && state.has_right && state.left.segment == state.right.segment =>
        {
            BoundaryQuery {
                active: true,
                left: left as u32,
                right: right as u32,
            }
        }
        _ => BoundaryQuery::default(),
    }
}

/// The context after this range's cut has been checked.
pub fn finish_slice_context(
    context: &PromptContext,
    state: &SliceState,
    crossing: &BoundaryHit,
) -> PromptContext {
    let mut next = context.clone();
    let total = state.position;
    if state.end <= state.start {
        record_fault(&mut next, FAULT_EMPTY_RANGE, state.start);
    } else if state.end > total {
        record_fault(&mut next, FAULT_CUT_BEYOND_TEXT, state.end);
    } else if crossing.found {
        record_fault(&mut next, FAULT_UNSAFE_CUT, state.end);
    }
    if state.end > state.start {
        next.next_piece = state.end.min(total);
    }
    next.complete = next.next_piece >= total;
    next
}

/// The context after a range has been merged and counted.
pub fn finish_token_context(
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
