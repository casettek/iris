//! Source and result types and pure logic.
//!
//! A run with a source builds its prompt from a document a signer vouches for:
//!
//! ```text
//!   each attestation ─ signature ─▶ the signer says this document came from this server
//!   each document ─ sha256 ─▶ it is the attested document
//!        │
//!        └ one scan per field of it ─▶ values
//!   recipe: base, then lead and value per field, then tail ─ sha256 ─▶ text_sha256
//! ```
//!
//! Everything here is plain functions over small states, one function per step
//! a stage would take, so direct, the native stage and the tiles run one copy.
//! Nothing aborts: a step that cannot go on records the first fault in its
//! state and the rest run to the end.
//!
//! See `docs/proposals/authenticated-prompt.md`.

use alloc::string::String;
use alloc::vec::Vec;
use k256::ecdsa::signature::Verifier;
use k256::ecdsa::{Signature, VerifyingKey};
use raster::{BytesPage, List};
use serde::{Deserialize, Serialize};
use sha2::digest::generic_array::GenericArray;
use sha2::{Digest, Sha256};

/// `SourceRecord::fault` values. Zero is a clean source.
pub const FAULT_NONE: u32 = 0;
/// The scheme, cipher, key, hash or signature is not one this code reads.
pub const FAULT_ATTESTATION: u32 = 1;
/// The signature is not the signer's over this attestation.
pub const FAULT_SIGNATURE: u32 = 2;
/// A document's hash or length is not the attested one.
pub const FAULT_DOCUMENT: u32 = 3;
/// The documents given are not the recipe's: their servers and targets
/// differ, or there are more or fewer of them.
pub const FAULT_SERVER: u32 = 4;
/// A part of the recipe is over its size limit, a path has an empty step, or
/// a field is of a document the recipe does not name.
pub const FAULT_RECIPE: u32 = 5;
/// The document has no member at a field's path.
pub const FAULT_FIELD_MISSING: u32 = 6;
/// A field's value is an object, an array, or no JSON scalar.
pub const FAULT_FIELD_NOT_SCALAR: u32 = 7;
/// A field's value is over [`MAX_VALUE`] bytes.
pub const FAULT_FIELD_TOO_LONG: u32 = 8;
/// A field's value is not UTF-8, or a string with a control byte in it.
pub const FAULT_FIELD_NOT_TEXT: u32 = 9;

pub fn fault_description(fault: u32) -> &'static str {
    match fault {
        FAULT_NONE => "none",
        FAULT_ATTESTATION => "the attestation cannot be read",
        FAULT_SIGNATURE => "the signature is not the signer's",
        FAULT_DOCUMENT => "a document is not the attested one",
        FAULT_SERVER => "the documents given are not the ones the recipe names",
        FAULT_RECIPE => "a part of the recipe is over its limit or malformed",
        FAULT_FIELD_MISSING => "the document has no member at a field's path",
        FAULT_FIELD_NOT_SCALAR => "a field's value is not a JSON scalar",
        FAULT_FIELD_TOO_LONG => "a field's value is too long",
        FAULT_FIELD_NOT_TEXT => "a field's value is not plain text",
        _ => "unknown",
    }
}

/// ECDSA over secp256k1, the message hashed with SHA-256.
pub const SCHEME_SECP256K1: u32 = 1;
/// The records are the response body as received.
pub const CIPHER_NONE: u32 = 0;

/// The longest lead, tail, base chunk and form text, in bytes. A lead often
/// holds a URL, so this is well over a line.
pub const MAX_PART: usize = 1024;
/// The size [`text_chunks`] cuts a base prompt to, in bytes.
pub const BASE_CHUNK: usize = 256;
/// The longest field value, in bytes.
pub const MAX_VALUE: usize = 1024;
/// The longest field path, in bytes.
pub const MAX_PATH: usize = 256;
/// What separates a path's steps. A JSON key cannot hold a raw newline, so no
/// key is cut by it.
pub const PATH_SEPARATOR: char = '\n';
/// The page a document is scanned in. The scan's result does not depend on
/// where pages are cut.
pub const DOCUMENT_PAGE: usize = 4096;

// ---------------------------------------------------------------------------
// Inputs
// ---------------------------------------------------------------------------

/// The key attestations are checked against. Pinned, not the run's to choose.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct SignerKey {
    pub scheme: u32,
    /// The compressed public key, in lowercase hex.
    pub key: String,
}

/// A signer's statement that a server answered a request with these records.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct Attestation {
    pub scheme: u32,
    pub cipher: u32,
    /// The signer's clock, in Unix seconds.
    pub time: u64,
    /// The host name.
    pub server: String,
    /// The request path and query.
    pub target: String,
    /// SHA-256 of the records, in lowercase hex.
    pub records_sha256: String,
    pub records_length: u64,
    /// The signature over [`attestation_message`], in lowercase hex.
    pub signature: String,
}

/// A document the recipe takes values from, as its attestation must name it.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct DocumentName {
    pub server: String,
    pub target: String,
}

/// One value taken from a document: `lead` is written before it.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct Field {
    pub lead: String,
    /// Which of the recipe's documents the value is in.
    pub document: u32,
    /// The steps from the document's root to the value, separated by
    /// [`PATH_SEPARATOR`]. A step into an object is a key. A step into an
    /// array is an element's index in decimal.
    pub path: String,
}

/// How the prompt is made from the documents. The text is `base`, then each
/// field's lead and value in order, then `tail`.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct Recipe {
    /// The base prompt, in chunks of at most [`MAX_PART`] bytes. Where it is
    /// cut changes nothing.
    pub base: List<String>,
    /// The documents a run must give, in order.
    pub documents: List<DocumentName>,
    pub fields: List<Field>,
    pub tail: String,
}

/// A piece of one document's bytes. A run's documents are given as one list
/// of these, each document's chunks together and in order, so a step that
/// reads documents loops over a single list.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct DocumentChunk {
    /// Which document, counting from zero.
    pub document: u32,
    /// The document ends with this chunk. Every document has one such chunk,
    /// which may be empty.
    pub last: bool,
    /// At most [`DOCUMENT_PAGE`] bytes.
    pub bytes: BytesPage,
}

/// The documents a run gives, with the attestation over each.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct SourceDocuments {
    /// One per document, in order.
    pub attestations: List<Attestation>,
    pub chunks: List<DocumentChunk>,
}

// ---------------------------------------------------------------------------
// Outputs
// ---------------------------------------------------------------------------

/// What the source step establishes. Small, so a tile may take it whole.
///
/// The first bind takes this record and the statement names its digest. Both
/// are in the shared prompt source (`SourceRecord` and `source_id` there),
/// which declares the same fields in the same order.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct SourceRecord {
    /// False in the constant a run without a source uses.
    pub present: bool,
    /// SHA-256 and length of the prompt text the recipe and documents give.
    pub text_sha256: String,
    pub text_total: u64,
    /// A hash over every part of the recipe.
    pub recipe_sha256: String,
    /// How many documents, and a hash over each one's attested hash, length,
    /// server, target and time, in order.
    pub documents: u32,
    pub documents_sha256: String,
    /// The signer's key, as in [`SignerKey::key`].
    pub signer: String,
    /// The first thing that went wrong, or [`FAULT_NONE`].
    pub fault: u32,
    /// Where: a document's index for a fault in a document, a field's index
    /// for a fault in a field, zero otherwise.
    pub fault_position: u64,
}

// ---------------------------------------------------------------------------
// Hashing
// ---------------------------------------------------------------------------

pub fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(DIGITS[(byte >> 4) as usize] as char);
        out.push(DIGITS[(byte & 15) as usize] as char);
    }
    out
}

/// Lowercase hex only, so one value has one spelling.
pub fn unhex(text: &str) -> Option<Vec<u8>> {
    fn nibble(byte: u8) -> Option<u8> {
        match byte {
            b'0'..=b'9' => Some(byte - b'0'),
            b'a'..=b'f' => Some(byte - b'a' + 10),
            _ => None,
        }
    }
    let bytes = text.as_bytes();
    if bytes.len() % 2 != 0 {
        return None;
    }
    let mut out = Vec::with_capacity(bytes.len() / 2);
    for pair in bytes.chunks_exact(2) {
        out.push(nibble(pair[0])? << 4 | nibble(pair[1])?);
    }
    Some(out)
}

pub fn sha256(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

/// SHA-256 over bytes given a piece at a time, as a value a loop can carry.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct ByteFold {
    /// The eight state words, big-endian.
    pub sha_words: BytesPage,
    /// Bytes waiting for a whole 64-byte block.
    pub sha_tail: BytesPage,
    pub total: u64,
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

impl Default for ByteFold {
    fn default() -> Self {
        begin_byte_fold()
    }
}

/// The fold before any byte.
pub fn begin_byte_fold() -> ByteFold {
    ByteFold {
        sha_words: words_page(&SHA256_INIT),
        sha_tail: page(Vec::new()),
        total: 0,
    }
}

/// Adds bytes to the fold.
pub fn fold_bytes(mut fold: ByteFold, bytes: &[u8]) -> ByteFold {
    let mut words = page_words(&fold.sha_words);
    let mut waiting = fold.sha_tail.as_slice().to_vec();
    waiting.extend_from_slice(bytes);
    let used = compress_blocks(&mut words, &waiting);
    fold.sha_words = words_page(&words);
    fold.sha_tail = page(waiting[used..].to_vec());
    fold.total += bytes.len() as u64;
    fold
}

/// The SHA-256 of every byte the fold has seen. The fold can go on after.
pub fn fold_sha256(fold: &ByteFold) -> [u8; 32] {
    let mut words = page_words(&fold.sha_words);
    let mut waiting = fold.sha_tail.as_slice().to_vec();
    waiting.push(0x80);
    while waiting.len() % 64 != 56 {
        waiting.push(0);
    }
    waiting.extend_from_slice(&(fold.total * 8).to_be_bytes());
    compress_blocks(&mut words, &waiting);
    let mut out = [0_u8; 32];
    for (bytes, word) in out.chunks_exact_mut(4).zip(words) {
        bytes.copy_from_slice(&word.to_be_bytes());
    }
    out
}

// ---------------------------------------------------------------------------
// The attestation
// ---------------------------------------------------------------------------

const ATTESTATION_TAG: &[u8] = b"iris/attestation/v1\0";

/// The bytes a signer signs, or `None` when a field does not fit its place.
pub fn attestation_message(attestation: &Attestation) -> Option<Vec<u8>> {
    let scheme = u8::try_from(attestation.scheme).ok()?;
    let cipher = u8::try_from(attestation.cipher).ok()?;
    let server = u16::try_from(attestation.server.len()).ok()?;
    let target = u16::try_from(attestation.target.len()).ok()?;
    let records = unhex(&attestation.records_sha256)?;
    if records.len() != 32 {
        return None;
    }
    let mut message =
        Vec::with_capacity(ATTESTATION_TAG.len() + 86 + server as usize + target as usize);
    message.extend_from_slice(ATTESTATION_TAG);
    message.push(scheme);
    message.push(cipher);
    message.extend_from_slice(&attestation.time.to_le_bytes());
    message.extend_from_slice(&server.to_le_bytes());
    message.extend_from_slice(attestation.server.as_bytes());
    message.extend_from_slice(&target.to_le_bytes());
    message.extend_from_slice(attestation.target.as_bytes());
    message.extend_from_slice(&records);
    message.extend_from_slice(&attestation.records_length.to_le_bytes());
    Some(message)
}

/// Whether the attestation is the signer's: a fault, or [`FAULT_NONE`].
pub fn verify_attestation(signer: &SignerKey, attestation: &Attestation) -> u32 {
    if signer.scheme != SCHEME_SECP256K1
        || attestation.scheme != signer.scheme
        || attestation.cipher != CIPHER_NONE
    {
        return FAULT_ATTESTATION;
    }
    let Some(message) = attestation_message(attestation) else {
        return FAULT_ATTESTATION;
    };
    let Some(Ok(key)) = unhex(&signer.key).map(|key| VerifyingKey::from_sec1_bytes(&key)) else {
        return FAULT_ATTESTATION;
    };
    let Some(Ok(signature)) =
        unhex(&attestation.signature).map(|signature| Signature::from_slice(&signature))
    else {
        return FAULT_ATTESTATION;
    };
    match key.verify(&message, &signature) {
        Ok(()) => FAULT_NONE,
        Err(_) => FAULT_SIGNATURE,
    }
}

/// The public key of a secret key, or `None` when the bytes are no key.
#[cfg(any(test, feature = "sign"))]
pub fn signer_key(secret: &[u8]) -> Option<SignerKey> {
    let key = k256::ecdsa::SigningKey::from_slice(secret).ok()?;
    Some(SignerKey {
        scheme: SCHEME_SECP256K1,
        key: hex(key.verifying_key().to_encoded_point(true).as_bytes()),
    })
}

/// Signs the attestation in place. This is the host signer's whole job: it
/// trusts whoever gives it the document's hash.
#[cfg(any(test, feature = "sign"))]
pub fn sign_attestation(secret: &[u8], attestation: &mut Attestation) -> Option<()> {
    use k256::ecdsa::signature::Signer;
    let key = k256::ecdsa::SigningKey::from_slice(secret).ok()?;
    let message = attestation_message(attestation)?;
    let signature: Signature = key.sign(&message);
    attestation.signature = hex(&signature.to_bytes());
    Some(())
}

// ---------------------------------------------------------------------------
// Finding one field's value in the document
// ---------------------------------------------------------------------------

const PHASE_START: u32 = 0;
/// In the object the path runs through: a key, a comma or its end comes next.
const PHASE_MEMBER: u32 = 1;
const PHASE_KEY: u32 = 2;
const PHASE_COLON: u32 = 3;
const PHASE_VALUE: u32 = 4;
const PHASE_SKIP_STRING: u32 = 5;
const PHASE_SKIP_SCALAR: u32 = 6;
const PHASE_SKIP_NESTED: u32 = 7;
const PHASE_TAKE_STRING: u32 = 8;
const PHASE_TAKE_SCALAR: u32 = 9;
/// In the array the path runs through: an element, a comma or its end comes
/// next.
const PHASE_ELEMENT: u32 = 10;

const KEY_DIFFERS: u32 = u32::MAX;

/// One field's scan of the document, as a value a loop over pages can carry.
///
/// The scan walks the path's objects and arrays only. Any other value is
/// skipped by counting its brackets, with strings and escapes respected. A key
/// is a path's key when its bytes between the quotes are the key's bytes:
/// escapes are not decoded. The first member with a matching key is the one
/// taken.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct FieldScan {
    pub phase: u32,
    /// Objects and arrays open at this byte.
    pub depth: u32,
    /// Steps of the path whose objects and arrays the scan is inside.
    pub matched: u32,
    /// The innermost of them is an array.
    pub in_array: bool,
    /// Elements of that array already passed.
    pub index_at: u32,
    pub in_string: bool,
    pub escape: bool,
    /// Bytes of the current key that agree with the path's key, or
    /// `u32::MAX` once one has differed.
    pub key_at: u32,
    /// The value so far, exactly as the document has it.
    pub value: String,
    /// The bytes of a character a page boundary has cut, low byte first.
    pub pending: u32,
    pub pending_len: u32,
    pub fault: u32,
    /// Nothing later in the document can change the outcome.
    pub done: bool,
}

/// One field's outcome.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct FieldValue {
    /// Empty when there is a fault.
    pub value: String,
    pub fault: u32,
}

fn path_keys(path: &str) -> core::str::Split<'_, char> {
    path.split(PATH_SEPARATOR)
}

/// A step as an array index: decimal digits with no leading zero.
fn step_index(step: &[u8]) -> Option<u32> {
    if step.is_empty() || step.len() > 9 || (step.len() > 1 && step[0] == b'0') {
        return None;
    }
    let mut index = 0_u32;
    for byte in step {
        if !byte.is_ascii_digit() {
            return None;
        }
        index = index * 10 + (byte - b'0') as u32;
    }
    Some(index)
}

/// What comes next in the innermost object or array of the path.
fn next_in_container(scan: &FieldScan) -> u32 {
    if scan.in_array {
        PHASE_ELEMENT
    } else {
        PHASE_MEMBER
    }
}

/// Starts skipping a value the path does not go through.
fn skip_value(scan: &mut FieldScan, byte: u8) {
    if byte == b'{' || byte == b'[' {
        scan.depth += 1;
        scan.phase = PHASE_SKIP_NESTED;
    } else if byte == b'"' {
        scan.phase = PHASE_SKIP_STRING;
    } else {
        scan.phase = PHASE_SKIP_SCALAR;
    }
}

/// Starts on the value the path's current step names: the field's value when
/// the step is the last, and otherwise the object or array the next step is
/// looked for in.
fn enter_value(scan: &mut FieldScan, byte: u8, steps: u32) {
    if scan.matched + 1 == steps {
        if byte == b'{' || byte == b'[' {
            stop(scan, FAULT_FIELD_NOT_SCALAR);
        } else {
            scan.phase = if byte == b'"' {
                PHASE_TAKE_STRING
            } else {
                PHASE_TAKE_SCALAR
            };
            take(scan, byte);
        }
    } else if byte == b'{' || byte == b'[' {
        scan.matched += 1;
        scan.depth += 1;
        scan.in_array = byte == b'[';
        scan.index_at = 0;
        scan.phase = next_in_container(scan);
    } else {
        stop(scan, FAULT_FIELD_MISSING);
    }
}

/// The scan before any page.
pub fn begin_field_scan(path: &str) -> FieldScan {
    let mut scan = FieldScan::default();
    if path.is_empty() || path.len() > MAX_PATH || path_keys(path).any(str::is_empty) {
        scan.fault = FAULT_RECIPE;
        scan.done = true;
    }
    scan
}

fn is_space(byte: u8) -> bool {
    matches!(byte, b' ' | b'\t' | b'\n' | b'\r')
}

fn stop(scan: &mut FieldScan, fault: u32) {
    scan.fault = fault;
    scan.value = String::new();
    scan.pending = 0;
    scan.pending_len = 0;
    scan.done = true;
}

/// Adds one byte of the value. Whole characters go to `value`; the bytes of
/// one a page has cut wait in `pending`.
fn take(scan: &mut FieldScan, byte: u8) {
    if scan.value.len() + scan.pending_len as usize >= MAX_VALUE {
        return stop(scan, FAULT_FIELD_TOO_LONG);
    }
    scan.pending |= (byte as u32) << (8 * scan.pending_len);
    scan.pending_len += 1;
    let bytes = scan.pending.to_le_bytes();
    let wanted = match bytes[0] {
        0x00..=0x7f => 1,
        0xc0..=0xdf => 2,
        0xe0..=0xef => 3,
        0xf0..=0xf7 => 4,
        _ => return stop(scan, FAULT_FIELD_NOT_TEXT),
    };
    if scan.pending_len < wanted {
        return;
    }
    match core::str::from_utf8(&bytes[..wanted as usize]) {
        Ok(character) => {
            scan.value.push_str(character);
            scan.pending = 0;
            scan.pending_len = 0;
        }
        Err(_) => stop(scan, FAULT_FIELD_NOT_TEXT),
    }
}

fn is_scalar(text: &str) -> bool {
    if matches!(text, "true" | "false" | "null") {
        return true;
    }
    let bytes = text.as_bytes();
    matches!(bytes.first(), Some(b'-' | b'0'..=b'9'))
        && bytes
            .iter()
            .all(|byte| matches!(byte, b'0'..=b'9' | b'-' | b'+' | b'.' | b'e' | b'E'))
}

fn finish_value(scan: &mut FieldScan, scalar: bool) {
    if scan.pending_len != 0 {
        return stop(scan, FAULT_FIELD_NOT_TEXT);
    }
    if scalar && !is_scalar(&scan.value) {
        return stop(scan, FAULT_FIELD_NOT_SCALAR);
    }
    scan.done = true;
}

/// Moves the scan over one page of the document.
pub fn scan_field_page(mut scan: FieldScan, page: &[u8], path: &str) -> FieldScan {
    if scan.done {
        return scan;
    }
    let steps = path_keys(path).count() as u32;
    let step_of = |matched: u32| {
        path_keys(path)
            .nth(matched as usize)
            .unwrap_or("")
            .as_bytes()
    };
    let mut key = step_of(scan.matched);
    for &byte in page {
        match scan.phase {
            PHASE_START => {
                if byte == b'{' {
                    scan.depth = 1;
                    scan.phase = PHASE_MEMBER;
                } else if !is_space(byte) {
                    stop(&mut scan, FAULT_FIELD_MISSING);
                }
            }
            PHASE_MEMBER => {
                if byte == b'"' {
                    scan.key_at = 0;
                    scan.phase = PHASE_KEY;
                } else if !is_space(byte) && byte != b',' {
                    // The object has ended without the key, or is not JSON.
                    stop(&mut scan, FAULT_FIELD_MISSING);
                }
            }
            PHASE_KEY => {
                if !scan.escape && byte == b'"' {
                    scan.phase = PHASE_COLON;
                } else {
                    scan.escape = !scan.escape && byte == b'\\';
                    if scan.key_at != KEY_DIFFERS && key.get(scan.key_at as usize) == Some(&byte) {
                        scan.key_at += 1;
                    } else {
                        scan.key_at = KEY_DIFFERS;
                    }
                }
            }
            PHASE_COLON => {
                if byte == b':' {
                    scan.phase = PHASE_VALUE;
                } else if !is_space(byte) {
                    stop(&mut scan, FAULT_FIELD_MISSING);
                }
            }
            PHASE_VALUE => {
                if is_space(byte) {
                    continue;
                }
                if scan.key_at as usize == key.len() {
                    enter_value(&mut scan, byte, steps);
                    key = step_of(scan.matched);
                } else {
                    skip_value(&mut scan, byte);
                }
            }
            PHASE_ELEMENT => {
                if is_space(byte) || byte == b',' {
                    continue;
                }
                if byte == b']' {
                    // The array has ended before the element.
                    stop(&mut scan, FAULT_FIELD_MISSING);
                } else if step_index(key) == Some(scan.index_at) {
                    enter_value(&mut scan, byte, steps);
                    key = step_of(scan.matched);
                } else if step_index(key).is_none() {
                    // The step is a key, and this is an array.
                    stop(&mut scan, FAULT_FIELD_MISSING);
                } else {
                    scan.index_at += 1;
                    skip_value(&mut scan, byte);
                }
            }
            PHASE_SKIP_STRING => {
                if !scan.escape && byte == b'"' {
                    scan.phase = next_in_container(&scan);
                } else {
                    scan.escape = !scan.escape && byte == b'\\';
                }
            }
            PHASE_SKIP_SCALAR => {
                if byte == b',' {
                    scan.phase = next_in_container(&scan);
                } else if byte == b'}' || byte == b']' {
                    // The object or array has ended without the step.
                    stop(&mut scan, FAULT_FIELD_MISSING);
                }
            }
            PHASE_SKIP_NESTED => {
                if scan.in_string {
                    if !scan.escape && byte == b'"' {
                        scan.in_string = false;
                    } else {
                        scan.escape = !scan.escape && byte == b'\\';
                    }
                } else if byte == b'"' {
                    scan.in_string = true;
                } else if byte == b'{' || byte == b'[' {
                    scan.depth += 1;
                } else if byte == b'}' || byte == b']' {
                    scan.depth -= 1;
                    if scan.depth == scan.matched + 1 {
                        scan.phase = next_in_container(&scan);
                    }
                }
            }
            PHASE_TAKE_STRING => {
                if byte < 0x20 {
                    stop(&mut scan, FAULT_FIELD_NOT_TEXT);
                } else {
                    let closes = !scan.escape && byte == b'"';
                    scan.escape = !scan.escape && byte == b'\\';
                    take(&mut scan, byte);
                    if closes && !scan.done {
                        finish_value(&mut scan, false);
                    }
                }
            }
            PHASE_TAKE_SCALAR => {
                if is_space(byte) || matches!(byte, b',' | b'}' | b']') {
                    finish_value(&mut scan, true);
                } else {
                    take(&mut scan, byte);
                }
            }
            _ => stop(&mut scan, FAULT_FIELD_MISSING),
        }
        if scan.done {
            break;
        }
    }
    scan
}

/// Moves a field's scan over one chunk of the documents. Chunks of other
/// documents are passed over, and once the field's own document is behind
/// nothing can be found.
pub fn scan_field_chunk(
    mut scan: FieldScan,
    chunk: &DocumentChunk,
    document: u32,
    path: &str,
) -> FieldScan {
    if scan.done || chunk.document < document {
        return scan;
    }
    if chunk.document > document {
        stop(&mut scan, FAULT_FIELD_MISSING);
        return scan;
    }
    scan_field_page(scan, chunk.bytes.as_slice(), path)
}

/// The field's outcome once the pages are over: a document that ended before
/// the scan was done has no such field.
pub fn finish_field_scan(scan: FieldScan) -> FieldValue {
    if !scan.done {
        return FieldValue {
            value: String::new(),
            fault: FAULT_FIELD_MISSING,
        };
    }
    FieldValue {
        value: scan.value,
        fault: scan.fault,
    }
}

// ---------------------------------------------------------------------------
// The documents
// ---------------------------------------------------------------------------

fn chain(previous: &[u8], parts: &[&[u8]]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(previous);
    for part in parts {
        hasher.update((part.len() as u32).to_le_bytes());
        hasher.update(part);
    }
    hasher.finalize().into()
}

fn chain_start() -> BytesPage {
    page([0_u8; 32].to_vec())
}

/// What the source step carries from chunk to chunk of the documents.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct BodiesFold {
    /// Documents that have ended.
    pub count: u32,
    /// The bytes of the document the next chunk continues.
    pub body: ByteFold,
    /// A hash chained over the hash and length of every document that has
    /// ended: 32 bytes.
    pub chain: BytesPage,
    /// A chunk was out of order or over its size.
    pub fault: bool,
}

pub fn begin_bodies_fold() -> BodiesFold {
    BodiesFold {
        count: 0,
        body: begin_byte_fold(),
        chain: chain_start(),
        fault: false,
    }
}

/// Adds one chunk. A document's hash joins the chain when its last chunk has.
pub fn fold_body_chunk(mut fold: BodiesFold, chunk: &DocumentChunk) -> BodiesFold {
    if chunk.document != fold.count || chunk.bytes.len() > DOCUMENT_PAGE {
        fold.fault = true;
    }
    fold.body = fold_bytes(fold.body, chunk.bytes.as_slice());
    if chunk.last {
        let length = fold.body.total.to_le_bytes();
        fold.chain = page(
            chain(
                fold.chain.as_slice(),
                &[hex(&fold_sha256(&fold.body)).as_bytes(), &length[..]],
            )
            .to_vec(),
        );
        fold.count += 1;
        fold.body = begin_byte_fold();
    }
    fold
}

/// What the source step carries from attestation to attestation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct DocumentsFold {
    pub count: u32,
    /// A hash chained over every document's attested facts so far: its hash,
    /// length, server, target and time. 32 bytes.
    pub facts: BytesPage,
    /// A hash chained over every document's server and target so far, to hold
    /// against the recipe's. 32 bytes.
    pub names: BytesPage,
    /// A hash chained over every document's attested hash and length so far,
    /// to hold against the documents given. 32 bytes.
    pub bodies: BytesPage,
    pub fault: u32,
    /// The index of the attestation a signature fault is in, zero otherwise.
    pub fault_position: u64,
}

pub fn begin_documents_fold() -> DocumentsFold {
    DocumentsFold {
        count: 0,
        facts: chain_start(),
        names: chain_start(),
        bodies: chain_start(),
        fault: FAULT_NONE,
        fault_position: 0,
    }
}

/// Adds one attestation, which must be the signer's.
pub fn fold_attestation(
    mut fold: DocumentsFold,
    signer: &SignerKey,
    attestation: &Attestation,
) -> DocumentsFold {
    let fault = verify_attestation(signer, attestation);
    if fold.fault == FAULT_NONE && fault != FAULT_NONE {
        fold.fault = fault;
        fold.fault_position = fold.count as u64;
    }
    let length = attestation.records_length.to_le_bytes();
    let time = attestation.time.to_le_bytes();
    fold.facts = page(
        chain(
            fold.facts.as_slice(),
            &[
                attestation.records_sha256.as_bytes(),
                &length[..],
                attestation.server.as_bytes(),
                attestation.target.as_bytes(),
                &time[..],
            ],
        )
        .to_vec(),
    );
    fold.names = page(
        chain(
            fold.names.as_slice(),
            &[attestation.server.as_bytes(), attestation.target.as_bytes()],
        )
        .to_vec(),
    );
    fold.bodies = page(
        chain(
            fold.bodies.as_slice(),
            &[attestation.records_sha256.as_bytes(), &length[..]],
        )
        .to_vec(),
    );
    fold.count += 1;
    fold
}

/// Holds the documents given to the attestations: as many, each with the
/// attested hash and length, in the same order. Equal chains say all of that.
pub fn check_bodies(mut documents: DocumentsFold, bodies: &BodiesFold) -> DocumentsFold {
    let unfinished = bodies.body.total != 0;
    if documents.fault == FAULT_NONE
        && (bodies.fault
            || unfinished
            || bodies.count != documents.count
            || bodies.chain != documents.bodies)
    {
        documents.fault = FAULT_DOCUMENT;
        documents.fault_position = 0;
    }
    documents
}

// ---------------------------------------------------------------------------
// The prompt text and the record
// ---------------------------------------------------------------------------

/// What the source step carries from part to part of the recipe.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct SourceFold {
    /// The prompt text so far.
    pub text: ByteFold,
    /// A hash chained over every part of the recipe so far: 32 bytes.
    pub recipe: BytesPage,
    /// A hash chained over the server and target of every document the recipe
    /// names so far: 32 bytes.
    pub names: BytesPage,
    pub document_count: u32,
    pub field_count: u32,
    pub fault: u32,
    pub fault_position: u64,
}

fn record_fault(fold: &mut SourceFold, fault: u32, position: u64) {
    if fold.fault == FAULT_NONE && fault != FAULT_NONE {
        fold.fault = fault;
        fold.fault_position = position;
    }
}

/// The fold before the base prompt.
pub fn begin_source_fold() -> SourceFold {
    SourceFold {
        text: begin_byte_fold(),
        recipe: chain_start(),
        names: chain_start(),
        document_count: 0,
        field_count: 0,
        fault: FAULT_NONE,
        fault_position: 0,
    }
}

/// Adds one chunk of the base prompt.
pub fn fold_base_chunk(mut fold: SourceFold, chunk: &str) -> SourceFold {
    if chunk.len() > MAX_PART {
        record_fault(&mut fold, FAULT_RECIPE, 0);
    }
    fold.text = fold_bytes(fold.text, chunk.as_bytes());
    fold
}

/// Closes the base prompt: the text so far is exactly it.
pub fn fold_base_end(mut fold: SourceFold) -> SourceFold {
    let base = fold_sha256(&fold.text);
    fold.recipe = page(chain(fold.recipe.as_slice(), &[&b"base"[..], &base[..]]).to_vec());
    fold
}

/// Adds one document the recipe names.
pub fn fold_document_name(mut fold: SourceFold, name: &DocumentName) -> SourceFold {
    let parts: [&[u8]; 2] = [name.server.as_bytes(), name.target.as_bytes()];
    fold.recipe = page(
        chain(
            fold.recipe.as_slice(),
            &[&b"document"[..], parts[0], parts[1]],
        )
        .to_vec(),
    );
    fold.names = page(chain(fold.names.as_slice(), &parts).to_vec());
    fold.document_count += 1;
    fold
}

/// Holds the documents given to the ones the recipe names: a fault among the
/// documents comes first, and then they must be the recipe's servers and
/// targets, as many and in the same order.
pub fn check_documents(mut fold: SourceFold, documents: &DocumentsFold) -> SourceFold {
    record_fault(&mut fold, documents.fault, documents.fault_position);
    if documents.count != fold.document_count || documents.names != fold.names {
        record_fault(&mut fold, FAULT_SERVER, 0);
    }
    fold
}

/// Adds one field: its lead and its value to the text, and what the recipe
/// says of it to the recipe's hash. A field of a document the recipe does not
/// name is a fault, and its value should be empty.
pub fn fold_field(mut fold: SourceFold, field: &Field, value: &FieldValue) -> SourceFold {
    let index = fold.field_count as u64;
    if field.lead.len() > MAX_PART || field.document >= fold.document_count {
        record_fault(&mut fold, FAULT_RECIPE, index);
    }
    record_fault(&mut fold, value.fault, index);
    fold.text = fold_bytes(fold.text, field.lead.as_bytes());
    fold.text = fold_bytes(fold.text, value.value.as_bytes());
    fold.recipe = page(
        chain(
            fold.recipe.as_slice(),
            &[
                &b"field"[..],
                field.lead.as_bytes(),
                &field.document.to_le_bytes()[..],
                field.path.as_bytes(),
            ],
        )
        .to_vec(),
    );
    fold.field_count += 1;
    fold
}

/// Adds the tail and states the record.
pub fn close_source(
    mut fold: SourceFold,
    documents: &DocumentsFold,
    signer: &SignerKey,
    tail: &str,
) -> SourceRecord {
    if tail.len() > MAX_PART {
        let index = fold.field_count as u64;
        record_fault(&mut fold, FAULT_RECIPE, index);
    }
    fold.text = fold_bytes(fold.text, tail.as_bytes());
    let recipe = chain(fold.recipe.as_slice(), &[&b"tail"[..], tail.as_bytes()]);

    SourceRecord {
        present: true,
        text_sha256: hex(&fold_sha256(&fold.text)),
        text_total: fold.text.total,
        recipe_sha256: hex(&recipe),
        documents: documents.count,
        documents_sha256: hex(documents.facts.as_slice()),
        signer: signer.key.clone(),
        fault: fold.fault,
        fault_position: fold.fault_position,
    }
}

/// One document as a run gives it.
#[derive(Debug, Clone, Copy)]
pub struct SourceDocument<'a> {
    pub attestation: &'a Attestation,
    pub body: &'a [u8],
}

/// The source step from end to end, and the prompt text it stands for.
pub struct SourceRun {
    pub record: SourceRecord,
    /// The text whose hash the record states. Run preparation gives it to the
    /// prompt stages; the source step itself keeps only the hash.
    pub text: String,
}

/// A run's documents as the source step takes them: every document cut into
/// chunks of [`DOCUMENT_PAGE`] bytes, the last of each marked.
pub fn source_documents(sources: &[SourceDocument]) -> SourceDocuments {
    let mut chunks = Vec::new();
    for (index, source) in sources.iter().enumerate() {
        let mut pieces: Vec<&[u8]> = source.body.chunks(DOCUMENT_PAGE).collect();
        if pieces.is_empty() {
            pieces.push(&[]);
        }
        let last = pieces.len() - 1;
        for (at, piece) in pieces.into_iter().enumerate() {
            chunks.push(DocumentChunk {
                document: index as u32,
                last: at == last,
                bytes: page(piece.to_vec()),
            });
        }
    }
    SourceDocuments {
        attestations: sources
            .iter()
            .map(|source| source.attestation.clone())
            .collect::<Vec<_>>()
            .into(),
        chunks: chunks.into(),
    }
}

/// Runs every step in the order the stage takes them, over the inputs the
/// stage takes.
pub fn run_source_steps(
    signer: &SignerKey,
    recipe: &Recipe,
    documents: &SourceDocuments,
) -> SourceRun {
    let mut bodies = begin_bodies_fold();
    for chunk in documents.chunks.iter() {
        bodies = fold_body_chunk(bodies, chunk);
    }
    let mut attested = begin_documents_fold();
    for attestation in documents.attestations.iter() {
        attested = fold_attestation(attested, signer, attestation);
    }
    let attested = check_bodies(attested, &bodies);

    let mut text = String::new();
    let mut fold = begin_source_fold();
    for chunk in recipe.base.iter() {
        fold = fold_base_chunk(fold, chunk);
        text.push_str(chunk);
    }
    fold = fold_base_end(fold);
    for name in recipe.documents.iter() {
        fold = fold_document_name(fold, name);
    }
    fold = check_documents(fold, &attested);
    for field in recipe.fields.iter() {
        let mut scan = begin_field_scan(&field.path);
        for chunk in documents.chunks.iter() {
            if scan.done {
                break;
            }
            scan = scan_field_chunk(scan, chunk, field.document, &field.path);
        }
        let value = finish_field_scan(scan);
        fold = fold_field(fold, field, &value);
        text.push_str(&field.lead);
        text.push_str(&value.value);
    }
    text.push_str(&recipe.tail);
    let record = close_source(fold, &attested, signer, &recipe.tail);
    SourceRun { record, text }
}

/// The source step over documents as a run holds them.
pub fn run_source(signer: &SignerKey, recipe: &Recipe, sources: &[SourceDocument]) -> SourceRun {
    run_source_steps(signer, recipe, &source_documents(sources))
}

/// Cuts text into chunks of at most [`BASE_CHUNK`] bytes at character
/// boundaries, for [`Recipe::base`].
pub fn text_chunks(text: &str) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut rest = text;
    while !rest.is_empty() {
        let mut end = rest.len().min(BASE_CHUNK);
        while !rest.is_char_boundary(end) {
            end -= 1;
        }
        chunks.push(String::from(&rest[..end]));
        rest = &rest[end..];
    }
    chunks
}
