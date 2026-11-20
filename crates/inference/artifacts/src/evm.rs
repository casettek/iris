//! What iris sends to its contracts: a configuration, a claim and a
//! challenge in the ABI encoding `contracts/src/IrisTypes.sol` declares, and
//! the calldata of the calls that carry them.
//!
//! The contracts take each as one `bytes` argument and decode it themselves,
//! so a call is a selector and one byte string. The structs here are written
//! field for field as the Solidity ones; a field out of order is a claim the
//! contract reads wrongly, and the end-to-end run in `devstack/` is what
//! holds the two together.

use anyhow::{bail, ensure, Context, Result};
use sha2::{Digest, Sha256};

use crate::availability::{
    AvailabilityProof, BlobContents, PostedBlob, StatedRecords, NAMESPACE_BYTES,
};
use crate::checkpoint::Checkpoint;
use crate::contract::{
    ChallengeSubmission, ClaimSubmission, SubmittedConsensus, SubmittedRecord, SubmittedResult,
};
use crate::encoding::{parse_hash, Encoder};
use crate::registry::Configuration;

/// One ABI value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Token {
    /// A static value in one 32-byte word.
    Word([u8; 32]),
    /// `bytes` or `string`.
    Bytes(Vec<u8>),
    /// `T[]`.
    Array(Vec<Token>),
    /// A struct.
    Tuple(Vec<Token>),
}

impl Token {
    pub fn number(value: u64) -> Self {
        let mut word = [0; 32];
        word[24..].copy_from_slice(&value.to_be_bytes());
        Self::Word(word)
    }

    pub fn flag(value: bool) -> Self {
        Self::number(u64::from(value))
    }

    pub fn text(value: &str) -> Self {
        Self::Bytes(value.as_bytes().to_vec())
    }

    /// `bytesN`: left-aligned.
    pub fn fixed(bytes: &[u8]) -> Result<Self> {
        ensure!(bytes.len() <= 32, "a fixed byte string is at most 32 bytes");
        let mut word = [0; 32];
        word[..bytes.len()].copy_from_slice(bytes);
        Ok(Self::Word(word))
    }

    /// `bytes32`, from 64 hex digits.
    pub fn hash(hex_digits: &str) -> Result<Self> {
        Ok(Self::Word(parse_hash(hex_digits).with_context(|| {
            format!("`{hex_digits}` is not 32 bytes of lowercase hex")
        })?))
    }

    /// `address`, from `0x` and 40 hex digits.
    pub fn address(text: &str) -> Result<Self> {
        let mut word = [0; 32];
        word[12..].copy_from_slice(&address_bytes(text)?);
        Ok(Self::Word(word))
    }

    fn is_dynamic(&self) -> bool {
        match self {
            Self::Word(_) => false,
            Self::Bytes(_) | Self::Array(_) => true,
            Self::Tuple(members) => members.iter().any(Self::is_dynamic),
        }
    }

    /// The bytes this value takes in the head of the sequence it is in.
    fn head_size(&self) -> usize {
        match self {
            Self::Tuple(members) if !self.is_dynamic() => members.iter().map(Self::head_size).sum(),
            _ => 32,
        }
    }

    fn encode_into(&self, out: &mut Vec<u8>) {
        match self {
            Self::Word(word) => out.extend(word),
            Self::Bytes(bytes) => {
                out.extend(Self::number(bytes.len() as u64).word());
                out.extend(bytes);
                out.resize(out.len() + (32 - bytes.len() % 32) % 32, 0);
            }
            Self::Array(elements) => {
                out.extend(Self::number(elements.len() as u64).word());
                out.extend(encode_sequence(elements));
            }
            Self::Tuple(members) => out.extend(encode_sequence(members)),
        }
    }

    fn word(&self) -> [u8; 32] {
        match self {
            Self::Word(word) => *word,
            _ => unreachable!("only a word is taken as one"),
        }
    }
}

/// A sequence of values as a tuple encodes them: the static ones in place,
/// and each dynamic one as an offset in the head and its bytes after it.
fn encode_sequence(tokens: &[Token]) -> Vec<u8> {
    let head_size: usize = tokens.iter().map(Token::head_size).sum();
    let mut head = Vec::with_capacity(head_size);
    let mut tail = Vec::new();
    for token in tokens {
        if token.is_dynamic() {
            head.extend(Token::number((head_size + tail.len()) as u64).word());
            token.encode_into(&mut tail);
        } else {
            token.encode_into(&mut head);
        }
    }
    head.extend(tail);
    head
}

/// `abi.encode(value)`: what `abi.decode(bytes, (T))` reads.
pub fn abi_encode(value: &Token) -> Vec<u8> {
    encode_sequence(std::slice::from_ref(value))
}

fn address_bytes(text: &str) -> Result<[u8; 20]> {
    let mut bytes = [0; 20];
    let digits = text
        .strip_prefix("0x")
        .filter(|digits| digits.len() == 40)
        .with_context(|| format!("`{text}` is not an address"))?;
    hex::decode_to_slice(digits, &mut bytes)
        .with_context(|| format!("`{text}` is not an address"))?;
    Ok(bytes)
}

// ---------------------------------------------------------------------------
// Keccak-256, for a call's selector
// ---------------------------------------------------------------------------

const ROUND_CONSTANTS: [u64; 24] = [
    0x0000000000000001,
    0x0000000000008082,
    0x800000000000808a,
    0x8000000080008000,
    0x000000000000808b,
    0x0000000080000001,
    0x8000000080008081,
    0x8000000000008009,
    0x000000000000008a,
    0x0000000000000088,
    0x0000000080008009,
    0x000000008000000a,
    0x000000008000808b,
    0x800000000000008b,
    0x8000000000008089,
    0x8000000000008003,
    0x8000000000008002,
    0x8000000000000080,
    0x000000000000800a,
    0x800000008000000a,
    0x8000000080008081,
    0x8000000000008080,
    0x0000000080000001,
    0x8000000080008008,
];
const ROTATIONS: [u32; 24] = [
    1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14, 27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44,
];
const PERMUTATION: [usize; 24] = [
    10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4, 15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1,
];

fn keccak_f(state: &mut [u64; 25]) {
    for constant in ROUND_CONSTANTS {
        let mut column = [0u64; 5];
        for x in 0..5 {
            column[x] = state[x] ^ state[x + 5] ^ state[x + 10] ^ state[x + 15] ^ state[x + 20];
        }
        for x in 0..5 {
            let mixed = column[(x + 4) % 5] ^ column[(x + 1) % 5].rotate_left(1);
            for y in 0..5 {
                state[x + 5 * y] ^= mixed;
            }
        }
        let mut carried = state[1];
        for (rotation, position) in ROTATIONS.iter().zip(PERMUTATION) {
            let next = state[position];
            state[position] = carried.rotate_left(*rotation);
            carried = next;
        }
        for y in 0..5 {
            let row: [u64; 5] = std::array::from_fn(|x| state[x + 5 * y]);
            for x in 0..5 {
                state[x + 5 * y] = row[x] ^ (!row[(x + 1) % 5] & row[(x + 2) % 5]);
            }
        }
        state[0] ^= constant;
    }
}

/// Keccak-256 as the EVM computes it (the padding before SHA-3 changed it).
pub fn keccak256(bytes: &[u8]) -> [u8; 32] {
    const RATE: usize = 136;
    let mut state = [0u64; 25];
    let mut padded = bytes.to_vec();
    padded.push(0x01);
    padded.resize(padded.len().div_ceil(RATE) * RATE, 0);
    *padded.last_mut().expect("the padding is never empty") |= 0x80;
    for block in padded.chunks(RATE) {
        for (lane, word) in block.chunks(8).enumerate() {
            state[lane] ^= u64::from_le_bytes(word.try_into().expect("a lane is 8 bytes"));
        }
        keccak_f(&mut state);
    }
    let mut digest = [0; 32];
    for (lane, word) in digest.chunks_mut(8).enumerate() {
        word.copy_from_slice(&state[lane].to_le_bytes());
    }
    digest
}

/// The calldata of a call that takes one `bytes` argument: the selector of
/// `name(bytes)` and the argument.
pub fn call_with_bytes(name: &str, argument: &[u8]) -> Vec<u8> {
    let mut calldata = keccak256(format!("{name}(bytes)").as_bytes())[..4].to_vec();
    calldata.extend(abi_encode(&Token::Bytes(argument.to_vec())));
    calldata
}

// ---------------------------------------------------------------------------
// A configuration
// ---------------------------------------------------------------------------

/// The programs of a configuration as its id encodes them: what the
/// Solidity `Configuration.programs` holds and the registry walks.
pub fn programs_encoding(configuration: &Configuration) -> Vec<u8> {
    let mut out = Vec::new();
    let count = |out: &mut Vec<u8>, items: usize| out.extend((items as u32).to_le_bytes());
    let text = |out: &mut Vec<u8>, text: &str| {
        out.extend((text.len() as u32).to_le_bytes());
        out.extend(text.as_bytes());
    };
    let programs = &configuration.programs.stage_programs;
    count(&mut out, programs.len());
    for program in programs {
        text(&mut out, &program.stage);
        out.extend(hash_field(&program.program_commitment));
        count(&mut out, program.chained_inputs.len());
        for input in &program.chained_inputs {
            text(&mut out, &input.name);
            count(&mut out, input.producers.len());
            for producer in &input.producers {
                text(&mut out, producer);
            }
            count(&mut out, input.empty_values.len());
            for value in &input.empty_values {
                out.extend(hash_field(value));
            }
        }
        count(&mut out, program.run_inputs.len());
        for input in &program.run_inputs {
            text(&mut out, input);
        }
        count(&mut out, program.pinned_inputs.len());
        for input in &program.pinned_inputs {
            text(&mut out, &input.name);
            count(&mut out, input.values.len());
            for value in &input.values {
                out.extend(hash_field(value));
            }
        }
    }
    out
}

/// A hash as the encoding writes it when it is one: a byte 1 and 32 bytes.
/// Anything else is written as the encoder writes it, which no contract
/// reads as a hash.
fn hash_field(hex_digits: &str) -> Vec<u8> {
    match parse_hash(hex_digits) {
        Some(bytes) => {
            let mut out = vec![1];
            out.extend(bytes);
            out
        }
        None if hex_digits.is_empty() => vec![0],
        None => {
            let mut out = vec![2];
            out.extend((hex_digits.len() as u32).to_le_bytes());
            out.extend(hex_digits.as_bytes());
            out
        }
    }
}

/// A configuration as `IrisRegistry.register` takes it.
pub fn configuration_token(configuration: &Configuration) -> Result<Token> {
    let models = configuration
        .models
        .iter()
        .map(|model| {
            let mut commitments = (model.model_commitments.len() as u32)
                .to_le_bytes()
                .to_vec();
            for commitment in &model.model_commitments {
                commitments.extend((commitment.role.len() as u32).to_le_bytes());
                commitments.extend(commitment.role.as_bytes());
                commitments.extend(hash_field(&commitment.commitment));
            }
            Ok(Token::Tuple(vec![
                Token::hash(&model.model_id)?,
                Token::Bytes(commitments),
                Token::number(u64::from(model.range_size)),
            ]))
        })
        .collect::<Result<Vec<_>>>()?;
    let recipe = configuration
        .source
        .as_ref()
        .and_then(|source| source.recipe_sha256.as_deref());
    Ok(Token::Tuple(vec![
        Token::number(u64::from(configuration.version)),
        Token::text(configuration.pipeline.name()),
        Token::text(&configuration.programs.family),
        Token::Bytes(programs_encoding(configuration)),
        Token::Array(models),
        Token::flag(configuration.raw_prompt),
        Token::number(u64::from(configuration.max_new_tokens)),
        Token::flag(configuration.source.is_some()),
        Token::text(
            configuration
                .source
                .as_ref()
                .map_or("", |source| source.signer.as_str()),
        ),
        Token::flag(recipe.is_some()),
        match recipe {
            Some(recipe) => Token::hash(recipe)?,
            None => Token::Word([0; 32]),
        },
        Token::Array(
            configuration
                .forms
                .iter()
                .map(|form| {
                    Token::Tuple(vec![
                        Token::text(&form.text),
                        Token::number(u64::from(form.code)),
                    ])
                })
                .collect(),
        ),
    ]))
}

// ---------------------------------------------------------------------------
// What a blob's attester signs
// ---------------------------------------------------------------------------

fn encode_record(encoder: &mut Encoder, record: &SubmittedRecord) {
    encoder
        .hash(&record.statement_sha256)
        .hash(&record.output_text_sha256)
        .byte(u8::from(record.valid));
}

/// The digest a blob's contents list a stated-records entry by.
pub fn stated_records_digest(stated: &StatedRecords) -> [u8; 32] {
    let mut encoder = Encoder::new(b"iris/stated-records/v1\0");
    encoder
        .hash(&stated.checkpoints_root)
        .hash(&stated.result_program)
        .count(stated.result_inputs.len());
    for (name, position) in &stated.result_inputs {
        encoder.text(name).number(*position);
    }
    encoder.count(stated.results.inferences.len());
    for inference in &stated.results.inferences {
        encode_record(&mut encoder, &inference.record);
        encoder
            .number(inference.result.code)
            .hash(&inference.result.forms_sha256);
    }
    match &stated.results.consensus {
        None => {
            encoder.byte(0);
        }
        Some(consensus) => {
            encoder
                .byte(1)
                .byte(u8::from(consensus.same_input))
                .byte(u8::from(consensus.agree))
                .number(consensus.code);
        }
    }
    encoder.count(stated.checks.len());
    for check in &stated.checks {
        encoder.number(check.checkpoint).hash(&check.program);
        encode_record(&mut encoder, &check.record);
    }
    encoder.finish()
}

/// What the attester signs for a blob: where it is, the root of its block,
/// and what it holds. `IrisEncoding.attestationDigest` computes the same.
pub fn attestation_digest(blob: &PostedBlob, data_root: &str) -> Result<[u8; 32]> {
    let namespace = hex::decode(&blob.pointer.namespace)
        .ok()
        .filter(|bytes| bytes.len() == NAMESPACE_BYTES)
        .with_context(|| format!("`{}` is not a namespace", blob.pointer.namespace))?;
    let mut encoder = Encoder::new(b"iris/blob-attestation/v2\0");
    encoder
        .long(blob.pointer.height)
        .raw(&namespace)
        .hash(&blob.pointer.commitment)
        .long(blob.length)
        .hash(data_root)
        .raw(&address_bytes(&blob.contents.submitter)?);
    for list in [
        &blob.contents.run_inputs,
        &blob.contents.checkpoint_lists,
        &blob.contents.trace_bodies,
    ] {
        encoder.count(list.len());
        for hash in list {
            encoder.hash(hash);
        }
    }
    encoder.count(blob.contents.records.len());
    for stated in &blob.contents.records {
        encoder.hash(&hex::encode(stated_records_digest(stated)));
    }
    Ok(encoder.finish())
}

// ---------------------------------------------------------------------------
// A claim and a challenge
// ---------------------------------------------------------------------------

fn record_token(record: &SubmittedRecord) -> Result<Token> {
    Ok(Token::Tuple(vec![
        Token::hash(&record.statement_sha256)?,
        Token::hash(&record.output_text_sha256)?,
        Token::flag(record.valid),
    ]))
}

fn result_token(result: &SubmittedResult) -> Result<Token> {
    Ok(Token::Tuple(vec![
        Token::number(u64::from(result.code)),
        Token::hash(&result.forms_sha256)?,
    ]))
}

fn consensus_token(consensus: Option<&SubmittedConsensus>) -> Token {
    let consensus = consensus.copied().unwrap_or(SubmittedConsensus {
        same_input: false,
        agree: false,
        code: 0,
    });
    Token::Tuple(vec![
        Token::flag(consensus.same_input),
        Token::flag(consensus.agree),
        Token::number(u64::from(consensus.code)),
    ])
}

fn hashes_token(hashes: &[String]) -> Result<Token> {
    Ok(Token::Array(
        hashes
            .iter()
            .map(|hash| Token::hash(hash))
            .collect::<Result<_>>()?,
    ))
}

fn named_positions(positions: &std::collections::BTreeMap<String, u32>) -> Token {
    Token::Array(
        positions
            .iter()
            .map(|(name, position)| {
                Token::Tuple(vec![Token::text(name), Token::number(u64::from(*position))])
            })
            .collect(),
    )
}

/// A stated-records entry, or the empty one for a claim whose proof states
/// none: its digest is in no blob, which is how a contract finds that out.
fn stated_token(stated: Option<&StatedRecords>) -> Result<Token> {
    let Some(stated) = stated else {
        return Ok(Token::Tuple(vec![
            Token::Word([0; 32]),
            Token::Word([0; 32]),
            Token::Array(Vec::new()),
            Token::Array(Vec::new()),
            Token::flag(false),
            consensus_token(None),
            Token::Array(Vec::new()),
        ]));
    };
    Ok(Token::Tuple(vec![
        Token::hash(&stated.checkpoints_root)?,
        Token::hash(&stated.result_program)?,
        named_positions(&stated.result_inputs),
        Token::Array(
            stated
                .results
                .inferences
                .iter()
                .map(|inference| {
                    Ok(Token::Tuple(vec![
                        record_token(&inference.record)?,
                        result_token(&inference.result)?,
                    ]))
                })
                .collect::<Result<_>>()?,
        ),
        Token::flag(stated.results.consensus.is_some()),
        consensus_token(stated.results.consensus.as_ref()),
        Token::Array(
            stated
                .checks
                .iter()
                .map(|check| {
                    Ok(Token::Tuple(vec![
                        Token::number(u64::from(check.checkpoint)),
                        Token::hash(&check.program)?,
                        record_token(&check.record)?,
                    ]))
                })
                .collect::<Result<_>>()?,
        ),
    ]))
}

fn contents_token(contents: &BlobContents) -> Result<Token> {
    Ok(Token::Tuple(vec![
        Token::address(&contents.submitter)?,
        hashes_token(&contents.run_inputs)?,
        hashes_token(&contents.checkpoint_lists)?,
        hashes_token(&contents.trace_bodies)?,
        Token::Array(
            contents
                .records
                .iter()
                .map(|stated| Token::Word(stated_records_digest(stated)))
                .collect(),
        ),
    ]))
}

/// A posted blob with its attestation. A blob nobody attested is sent with
/// no root and no signature, which no contract accepts.
fn blob_token(blob: &PostedBlob) -> Result<Token> {
    let AvailabilityProof::HostChecked { attestation, .. } = &blob.proof;
    let namespace = hex::decode(&blob.pointer.namespace)
        .with_context(|| format!("`{}` is not a namespace", blob.pointer.namespace))?;
    Ok(Token::Tuple(vec![
        Token::number(blob.pointer.height),
        Token::fixed(&namespace)?,
        Token::hash(&blob.pointer.commitment)?,
        Token::number(blob.length),
        match attestation {
            Some(attestation) => Token::hash(&attestation.data_root)?,
            None => Token::Word([0; 32]),
        },
        contents_token(&blob.contents)?,
        Token::Bytes(match attestation {
            Some(attestation) => hex::decode(&attestation.signature)
                .context("an attestation's signature is not hex")?,
            None => Vec::new(),
        }),
    ]))
}

fn blobs_token(blobs: &[PostedBlob]) -> Result<Token> {
    Ok(Token::Array(
        blobs.iter().map(blob_token).collect::<Result<_>>()?,
    ))
}

/// A claim as `IrisClaims.submitClaim` takes it.
pub fn claim_token(submission: &ClaimSubmission) -> Result<Token> {
    let source = submission.source.clone().unwrap_or_default();
    let stated = submission
        .availability
        .iter()
        .flat_map(|blob| &blob.contents.records)
        .find(|stated| stated.checkpoints_root == submission.checkpoints_root);
    Ok(Token::Tuple(vec![
        Token::hash(&submission.configuration)?,
        Token::address(&submission.claimer)?,
        Token::hash(&submission.input_text_sha256)?,
        Token::hash(&submission.source_sha256)?,
        Token::flag(submission.source.is_some()),
        Token::Tuple(vec![
            Token::flag(source.present),
            Token::text(&source.text_sha256),
            Token::number(source.text_total),
            Token::text(&source.recipe_sha256),
            Token::number(u64::from(source.documents)),
            Token::text(&source.documents_sha256),
            Token::text(&source.signer),
            Token::number(u64::from(source.fault)),
            Token::number(source.fault_position),
        ]),
        Token::Array(
            submission
                .inferences
                .iter()
                .map(|inference| {
                    Ok(Token::Tuple(vec![
                        Token::text(&inference.output_text),
                        record_token(&inference.record)?,
                        result_token(&inference.result)?,
                    ]))
                })
                .collect::<Result<_>>()?,
        ),
        Token::flag(submission.consensus.is_some()),
        consensus_token(submission.consensus.as_ref()),
        Token::number(u64::from(submission.checkpoint_count)),
        Token::hash(&submission.checkpoints_root)?,
        stated_token(stated)?,
        blobs_token(&submission.availability)?,
    ]))
}

/// The id a contract gives a claim: the SHA-256 of its ABI encoding, which
/// the contract makes again from the claim it decoded.
pub fn claim_id(submission: &ClaimSubmission) -> Result<[u8; 32]> {
    Ok(Sha256::digest(abi_encode(&claim_token(submission)?)).into())
}

fn checkpoint_token(checkpoint: &Checkpoint) -> Result<Token> {
    Ok(Token::Tuple(vec![
        Token::text(&checkpoint.stage),
        Token::hash(&checkpoint.program_commitment)?,
        Token::hash(&checkpoint.input_commitment)?,
        named_positions(&checkpoint.input_sources),
        Token::hash(&checkpoint.output_commitment)?,
        Token::hash(&checkpoint.output_sha256)?,
    ]))
}

/// A challenge as `IrisClaims.submitChallenge` takes it: of the claim a
/// contract knows by `claim_id`, with the claim's checkpoint at the disputed
/// stage and the path that opens it.
pub fn challenge_token(
    claim_id: &[u8; 32],
    submission: &ChallengeSubmission,
    checkpoint: &Checkpoint,
    path: &[String],
) -> Result<Token> {
    let claimed_output: [u8; 32] = submission
        .challenge
        .claimed_output
        .as_slice()
        .try_into()
        .ok()
        .context("the challenge's claimed output is not 32 bytes")?;
    if path.is_empty() {
        bail!("a challenge opens its checkpoint with a path");
    }
    Ok(Token::Tuple(vec![
        Token::Word(*claim_id),
        Token::address(&submission.challenger)?,
        Token::number(u64::from(submission.stage)),
        checkpoint_token(checkpoint)?,
        hashes_token(path)?,
        trace_header_token(&submission.challenge.trace_commitment_header)?,
        Token::Word(claimed_output),
        blobs_token(&submission.availability)?,
    ]))
}

fn word(bytes: &[u8], what: &str) -> Result<Token> {
    let word: [u8; 32] = bytes
        .try_into()
        .ok()
        .with_context(|| format!("the trace header's {what} is not 32 bytes"))?;
    Ok(Token::Word(word))
}

/// The header of a challenger's trace commitment, as a challenge gives it:
/// whole, so that a contract can hold its shape to the protocol's and make
/// the digest the posted trace body is found by.
pub fn trace_header_token(
    header: &raster_core::transition::TraceCommitmentHeader,
) -> Result<Token> {
    Ok(Token::Tuple(vec![
        Token::number(header.bits_packer.0 as u64),
        Token::number(header.fingerprint_len),
        word(&header.fingerprint_root, "fingerprint root")?,
        word(
            &header.revealed_items_commitment,
            "revealed items commitment",
        )?,
        Token::number(header.window_size),
        word(
            &header.revealed_tail_roots_commitment,
            "revealed tail roots commitment",
        )?,
    ]))
}

// ---------------------------------------------------------------------------
// A fault pointed at, with no replay
// ---------------------------------------------------------------------------

/// A stage that names a program outside the approved set, as
/// `IrisClaims.voidForProgram` takes it: the claim's own checkpoint at that
/// stage, with the path that opens it.
pub fn program_fault_token(
    claim_id: &[u8; 32],
    stage: u32,
    checkpoint: &Checkpoint,
    path: &[String],
) -> Result<Token> {
    Ok(Token::Tuple(vec![
        Token::Word(*claim_id),
        Token::number(u64::from(stage)),
        checkpoint_token(checkpoint)?,
        hashes_token(path)?,
    ]))
}

/// A stage whose input is not what the claim says an earlier stage produced,
/// as `IrisClaims.voidForInput` takes it: the stage's checkpoint, the bytes
/// of the input list it commits to, the input, and the checkpoint the stage's
/// own checkpoint names as the producer, when it names one.
pub fn input_fault_token(
    claim_id: &[u8; 32],
    stage: u32,
    checkpoint: &Checkpoint,
    path: &[String],
    input_manifest: &[u8],
    input: &str,
    producer: Option<(&Checkpoint, &[String])>,
    posted: &[String],
) -> Result<Token> {
    let (has_producer, producer, producer_path) = match producer {
        Some((producer, path)) => (true, checkpoint_token(producer)?, hashes_token(path)?),
        None => (
            false,
            Token::Tuple(vec![
                Token::text(""),
                Token::Word([0; 32]),
                Token::Word([0; 32]),
                Token::Array(Vec::new()),
                Token::Word([0; 32]),
                Token::Word([0; 32]),
            ]),
            Token::Array(Vec::new()),
        ),
    };
    Ok(Token::Tuple(vec![
        Token::Word(*claim_id),
        Token::number(u64::from(stage)),
        checkpoint_token(checkpoint)?,
        hashes_token(path)?,
        Token::Bytes(input_manifest.to_vec()),
        Token::text(input),
        Token::flag(has_producer),
        producer,
        producer_path,
        hashes_token(posted)?,
    ]))
}

/// The id a contract gives a challenge: the SHA-256 of its ABI encoding,
/// as a claim's is.
pub fn challenge_id(
    claim_id: &[u8; 32],
    submission: &ChallengeSubmission,
    checkpoint: &Checkpoint,
    path: &[String],
) -> Result<[u8; 32]> {
    Ok(Sha256::digest(abi_encode(&challenge_token(
        claim_id, submission, checkpoint, path,
    )?))
    .into())
}

// ---------------------------------------------------------------------------
// A fault proof
// ---------------------------------------------------------------------------

/// A fault proof as `IrisClaims.submitFaultProof` takes it: the challenge it
/// answers, the journal of Raster's transition program as RISC Zero wrote it,
/// and the Groth16 seal as RISC Zero's verifier takes it.
pub fn fault_proof_token(challenge_id: &[u8; 32], journal: &[u8], seal: &[u8]) -> Token {
    Token::Tuple(vec![
        Token::Word(*challenge_id),
        Token::Bytes(journal.to_vec()),
        Token::Bytes(seal.to_vec()),
    ])
}

/// What a contract reads from a fault proof's journal: `RasterJournal.read`,
/// byte for byte.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FaultVerdict {
    pub transition_image_id: [u8; 32],
    pub authorization_image_id: [u8; 32],
    pub input_manifest: [u8; 32],
    pub program: [u8; 32],
    pub refuted_trace: [u8; 32],
}

/// Read a transition journal's bytes as `RasterJournal.read` does: `None`
/// for bytes that are not one (the contract's `MalformedJournal`),
/// `Some(None)` for a journal that has not reached the fault verdict, and
/// the fields otherwise.
///
/// The journal is a run of little-endian 32-bit words. It begins with the
/// window's opening state, whose size depends on where the window opens, so
/// the fields are found by walking it. The layout is the transition
/// program's: `iris evm fault-proof` checks this reading against the decoded
/// journal every time it writes a call.
pub fn read_fault_journal(journal: &[u8]) -> Option<Option<FaultVerdict>> {
    struct Cursor<'a>(&'a [u8], usize);
    impl Cursor<'_> {
        fn skip(&mut self, bytes: usize) -> Option<()> {
            self.1 = self.1.checked_add(bytes)?;
            (self.1 <= self.0.len()).then_some(())
        }
        fn word(&mut self) -> Option<usize> {
            let at = self.1;
            self.skip(4)?;
            Some(u32::from_le_bytes(self.0[at..at + 4].try_into().ok()?) as usize)
        }
        fn bytes(&mut self) -> Option<&[u8]> {
            let length = self.word()?;
            let at = self.1;
            self.skip(length.checked_add(3)? / 4 * 4)?;
            Some(&self.0[at..at + length])
        }
        fn hash(&mut self) -> Option<[u8; 32]> {
            self.bytes()?.try_into().ok()
        }
        fn frontier(&mut self) -> Option<()> {
            self.skip(8)?;
            self.bytes()?;
            for _ in 0..self.word()? {
                self.bytes()?;
            }
            Some(())
        }
    }
    const FINISHED: usize = 2;
    let mut at = Cursor(journal, 0);
    at.frontier()?;
    at.frontier()?;
    at.bytes()?;
    at.bytes()?;
    at.skip(8)?;
    let count = at.word()?;
    at.skip(count.checked_mul(8)?.checked_add(8)?)?;
    if at.word()? != FINISHED {
        return Some(None);
    }
    Some(Some(FaultVerdict {
        transition_image_id: at.hash()?,
        authorization_image_id: at.hash()?,
        input_manifest: at.hash()?,
        program: at.hash()?,
        refuted_trace: at.hash()?,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keccak_is_the_evms() {
        assert_eq!(
            hex::encode(keccak256(b"")),
            "c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470"
        );
        assert_eq!(
            hex::encode(keccak256(b"abc")),
            "4e03657aea45a94fc7d47ba826c8d667c0d1e6e33a64a036ec44f58fa12d6c45"
        );
        // A well-known selector, and input longer than one block.
        assert_eq!(
            hex::encode(&keccak256(b"transfer(address,uint256)")[..4]),
            "a9059cbb"
        );
        // The selectors of the contracts' own calls, as `cast sig` gives them.
        for (name, selector) in [
            ("register", "82fbdc9c"),
            ("submitClaim", "ddfdfbb0"),
            ("submitChallenge", "5e9c0523"),
            ("submitFaultProof", "f8389708"),
            ("voidForProgram", "592cf381"),
            ("voidForInput", "8cd1e035"),
        ] {
            assert_eq!(hex::encode(&call_with_bytes(name, b"")[..4]), selector);
        }
    }

    #[test]
    fn values_encode_as_the_abi_lays_them_out() {
        // (uint256 7, string "hi"): the number in place, the string after.
        let encoded = abi_encode(&Token::Tuple(vec![Token::number(7), Token::text("hi")]));
        let words: Vec<String> = encoded.chunks(32).map(hex::encode).collect();
        let word = |value: u64| hex::encode(Token::number(value).word());
        assert_eq!(
            words,
            vec![
                word(0x20), // where the tuple starts
                word(7),
                word(0x40), // where the string starts, within the tuple
                word(2),
                format!("6869{}", "00".repeat(30)),
            ]
        );
        // A static struct is laid out in place, and an array of them packed.
        let pair = |a, b| Token::Tuple(vec![Token::number(a), Token::flag(b)]);
        let encoded = encode_sequence(&[Token::Array(vec![pair(1, true), pair(2, false)])]);
        let words: Vec<String> = encoded.chunks(32).map(hex::encode).collect();
        assert_eq!(
            words,
            vec![word(0x20), word(2), word(1), word(1), word(2), word(0)]
        );
        // Fixed bytes sit at the left of their word, an address at the right.
        assert_eq!(
            hex::encode(Token::fixed(&[0xab; 29]).unwrap().word()),
            format!("{}000000", "ab".repeat(29))
        );
        assert_eq!(
            hex::encode(
                Token::address("0x00000000000000000000000000000000000a11ce")
                    .unwrap()
                    .word()
            ),
            format!("{}0a11ce", "00".repeat(29))
        );
        assert!(Token::address("a11ce").is_err());
    }
}
