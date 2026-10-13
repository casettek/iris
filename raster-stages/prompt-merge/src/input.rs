//! Authenticated tokenizer tables and bounded ranked-BPE working values.

extern crate alloc;

use alloc::string::String;
use raster::List;
use serde::{Deserialize, Serialize};

/// Token id reserved for pieces that are not in the vocabulary.
pub const UNK_TOKEN_ID: u32 = 0;

/// One vocabulary entry: a token string and the id it maps to.
///
/// All-scalar, so it is `Materializable`: a tile may take one whole, and a
/// `chunk = N` recur step may take a `Block` of them.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct TokenEntry {
    pub token: String,
    pub id: u32,
}

/// One BPE merge rule: `left + right -> merged`, ranked by `rank`
/// (lower rank wins across all eligible adjacent pairs).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
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
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct VocabBucket {
    pub entries: List<TokenEntry>,
}

/// One merge bucket: the rules whose `(left, right)` pair hashes to this slot.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct MergeBucket {
    pub rules: List<BpeMerge>,
}

/// Committed tokenizer fixture: the vocabulary and the merge table, each
/// bucketed by a hash of its key.
///
/// The bucketing is what makes a lookup affordable. A flat `List` can only be
/// searched by scanning it, and a scan is one replay unit per entry — 262,144
/// of them per piece for a real vocabulary. Hashing to a bucket turns that into
/// one tile (the hash), one dynamic-index `select!` (one authenticated node),
/// and a recur over the handful of entries that share the slot.
///
/// `*_bucket_count` is the modulus the hash is reduced by, and it MUST equal
/// the corresponding list's length: it is what guarantees the computed index is
/// in range, and an out-of-range dynamic index aborts the run. The two travel
/// in one committed value so the commitment binds them together.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct PromptTokenizer {
    pub vocab_bucket_count: u32,
    pub merge_bucket_count: u32,
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
/// host and guest must agree byte-for-byte, which is why it lives in the
/// `no_std` library rather than being written twice.
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

/// Explicit segmentation: adjacent pieces may merge only within one segment.
/// Special tokens occupy their own segment. There is no sentinel token.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct BpePiece {
    pub text: String,
    pub segment: u32,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct BpePieces {
    pub pieces: List<BpePiece>,
}

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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct VocabQuery {
    pub piece: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct VocabMatch {
    pub found: bool,
    pub token_id: u32,
}

/// Structural interface shared with input-embedding.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct PromptTokenization {
    pub token_ids: List<u32>,
}
