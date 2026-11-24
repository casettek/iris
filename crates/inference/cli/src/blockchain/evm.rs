//! What iris hands to whoever sends transactions: the calldata of the calls
//! its contracts take, and the attester's signature over each posted blob.
//!
//! iris holds no wallet. Every command here reads files and prints bytes;
//! iris-backend sends them, or `iris test devstack submit` on the local stack,
//! through an account the development node keeps unlocked.
//! The one key iris uses is the attester's, read from `IRIS_ATTESTER_KEY`: the
//! trusted host's, standing in for an availability proof.

use std::path::Path;

use anyhow::{ensure, Context, Result};
use inference_artifacts::availability::{AvailabilityProof, HostAttestation, PostedBlob};
use inference_artifacts::contract::{ChallengeSubmission, ClaimSubmission};
use inference_artifacts::{
    checkpoint_opening, read_checkpoint_hashes, read_checkpoint_trace, read_json, CHECKPOINTS_TXT,
    CHECKPOINT_TRACE_JSON,
};
use iris_evm::{
    abi_encode, attestation_digest, call_with_bytes, challenge_id, challenge_token, claim_id,
    claim_token, configuration_token, fault_proof_token, read_fault_journal, FaultVerdict,
};
use serde_json::{json, Value};

use crate::posting::PostingLayer;

/// The attester's secret key: 32 bytes of hex, with or without `0x`.
pub const ATTESTER_KEY_ENV: &str = "IRIS_ATTESTER_KEY";

pub(crate) fn hex0x(bytes: &[u8]) -> String {
    format!("0x{}", hex::encode(bytes))
}

/// A call as a sender needs it: what the contract decodes, and the whole
/// calldata of `name(bytes)` over it.
pub(crate) fn call(name: &str, encoded: &[u8]) -> Value {
    json!({
        "function": format!("{name}(bytes)"),
        "argument": hex0x(encoded),
        "calldata": hex0x(&call_with_bytes(name, encoded)),
    })
}

/// The call that registers a configuration, with the id it will have.
pub fn configuration_call(
    configuration: &inference_artifacts::registry::Configuration,
) -> Result<Value> {
    let mut call = call(
        "register",
        &abi_encode(&configuration_token(configuration)?),
    );
    call["configuration"] = json!(format!("0x{}", configuration.id()));
    Ok(call)
}

/// The call that submits a claim, with the id the contract will give it.
pub fn claim_call(submission_path: &Path) -> Result<Value> {
    let submission: ClaimSubmission = read_json(submission_path)?;
    let mut call = call("submitClaim", &abi_encode(&claim_token(&submission)?));
    call["claim"] = json!(hex0x(&claim_id(&submission)?));
    call["heights"] = json!(heights(&submission.availability));
    Ok(call)
}

/// The call that submits a challenge of a claim. The claim's checkpoint at
/// the disputed stage and the path that opens it are read from the claim's
/// directory, where its submission is.
pub fn challenge_call(challenge_path: &Path, claim_submission_path: &Path) -> Result<Value> {
    let opened = OpenedChallenge::read(challenge_path, claim_submission_path)?;
    let mut call = call("submitChallenge", &abi_encode(&opened.token()?));
    call["claim"] = json!(hex0x(&opened.claim_id));
    call["challenge"] = json!(hex0x(&opened.id()?));
    call["heights"] = json!(heights(&opened.challenge.availability));
    Ok(call)
}

/// A challenge with the checkpoint it disputes and the path that opens it,
/// read from the claim's directory.
struct OpenedChallenge {
    claim_id: [u8; 32],
    challenge: ChallengeSubmission,
    checkpoint: inference_artifacts::Checkpoint,
    path: Vec<String>,
}

impl OpenedChallenge {
    fn token(&self) -> Result<iris_evm::Token> {
        challenge_token(
            &self.claim_id,
            &self.challenge,
            &self.checkpoint,
            &self.path,
        )
    }

    /// The id the contract gives the challenge.
    fn id(&self) -> Result<[u8; 32]> {
        challenge_id(
            &self.claim_id,
            &self.challenge,
            &self.checkpoint,
            &self.path,
        )
    }

    fn read(challenge_path: &Path, claim_submission_path: &Path) -> Result<Self> {
        let challenge: ChallengeSubmission = read_json(challenge_path)?;
        let claim: ClaimSubmission = read_json(claim_submission_path)?;
        ensure!(
            challenge.checkpoints_root == claim.checkpoints_root,
            "the challenge is of checkpoint list {}, and the claim's is {}",
            challenge.checkpoints_root,
            claim.checkpoints_root
        );
        let dir = claim_submission_path
            .parent()
            .context("the claim submission has no directory")?;
        let trace = read_checkpoint_trace(&dir.join(CHECKPOINT_TRACE_JSON))?;
        let hashes = read_checkpoint_hashes(&dir.join(CHECKPOINTS_TXT))?;
        let stage = challenge.stage as usize;
        let checkpoint = trace
            .checkpoints
            .get(stage)
            .with_context(|| format!("the claim has no checkpoint {stage}"))?;
        // On chain a challenge is of the checkpoint it opens, and carries no
        // commitment of its own to say otherwise. One that was built about
        // another checkpoint is not the challenge these bytes would state.
        ensure!(
            inference_artifacts::contract::challenge_is_about(checkpoint, &challenge.challenge),
            "the challenge is not about the claim's checkpoint {stage}"
        );
        Ok(Self {
            claim_id: claim_id(&claim)?,
            checkpoint: checkpoint.clone(),
            path: checkpoint_opening(&hashes, stage)?,
            challenge,
        })
    }
}

/// The ids of Raster's two proof programs, which a contract pins: the
/// transition program a fault proof is a proof of, and the authorization
/// program a fault proof's inputs were checked by.
pub fn proof_programs() -> Value {
    json!({
        "transition_image_id": hex0x(&raster_prover::chain_fraud::transition_guest_image_id()),
        "authorization_image_id":
            hex0x(&raster_prover::authorization::authorization_guest_image_id()),
    })
}

/// A fault proof as a contract is given it.
struct SealedFaultProof {
    /// The journal's bytes as RISC Zero wrote them.
    journal: Vec<u8>,
    /// The verifier's selector, then the Groth16 proof.
    seal: Vec<u8>,
    decoded: raster_core::transition::TransitionJournal,
    /// The journal as the contract reads it, or `None` short of a verdict.
    verdict: Option<FaultVerdict>,
}

/// Read a fault proof (`fault.receipt`) and lay it out for a contract. Only a
/// Groth16 proof is verified on chain, and only one this build of Raster
/// verifies is worth sending.
fn sealed_fault_proof(receipt_path: &Path) -> Result<SealedFaultProof> {
    let bytes = std::fs::read(receipt_path)
        .with_context(|| format!("reading the fault proof {}", receipt_path.display()))?;
    let receipt: risc0_zkvm::Receipt =
        postcard::from_bytes(&bytes).context("the fault proof is not a RISC Zero receipt")?;
    let risc0_zkvm::InnerReceipt::Groth16(inner) = &receipt.inner else {
        anyhow::bail!("the fault proof is not a Groth16 proof, the one kind a contract verifies");
    };
    receipt
        .verify(raster_prover::TRANSITION_GUEST_ID)
        .context("the fault proof does not verify as one of Raster's transition program")?;
    let decoded: raster_core::transition::TransitionJournal = receipt
        .journal
        .decode()
        .context("the fault proof's journal is not a transition journal")?;
    let journal = receipt.journal.bytes.clone();
    // The contract finds its fields by walking the journal's bytes. That
    // reading is held to the decoded journal here, so a journal laid out
    // another way is caught before anything is sent.
    let verdict = read_fault_journal(&journal)
        .context("the journal's bytes are not laid out as the contract reads them")?;
    let finished = matches!(
        decoded.current_state,
        raster_core::transition::TransitionState::Finished
    );
    ensure!(
        match &verdict {
            None => !finished,
            Some(read) =>
                finished
                    && read.transition_image_id.as_slice() == decoded.transition_image_id
                    && read.authorization_image_id.as_slice() == decoded.authorization_image_id
                    && read.input_manifest.as_slice() == decoded.input_manifest_commitment
                    && read.program.as_slice() == decoded.program_commitment
                    && read.refuted_trace.as_slice() == decoded.refuted_trace_commitment,
        },
        "the contract would read other fields from the journal than it holds"
    );
    let mut seal = inner.verifier_parameters.as_bytes()[..4].to_vec();
    seal.extend(&inner.seal);
    Ok(SealedFaultProof {
        journal,
        seal,
        decoded,
        verdict,
    })
}

fn fault_proof_fields(proof: &SealedFaultProof) -> Value {
    json!({
        "journal": hex0x(&proof.journal),
        "seal": hex0x(&proof.seal),
        "verdict": proof.verdict.as_ref().map(|verdict| json!({
            "transition_image_id": hex0x(&verdict.transition_image_id),
            "authorization_image_id": hex0x(&verdict.authorization_image_id),
            "input_manifest": hex0x(&verdict.input_manifest),
            "program": hex0x(&verdict.program),
            "refuted_trace": hex0x(&verdict.refuted_trace),
        })),
    })
}

/// A fault proof's journal and seal as a contract takes them, and what the
/// contract reads from the journal. Of no challenge: this is the proof alone.
pub fn fault_proof_parts(receipt_path: &Path) -> Result<Value> {
    Ok(fault_proof_fields(&sealed_fault_proof(receipt_path)?))
}

/// The call that answers a challenge with a fault proof of its trace.
///
/// `rejection` is the contract error the proof would be refused with, by the
/// checks `IrisClaims.submitFaultProof` makes of the journal, or null. The
/// call is written either way: whether a contract agrees is the contract's to
/// say.
pub fn fault_proof_call(
    challenge_path: &Path,
    claim_submission_path: &Path,
    receipt_path: &Path,
) -> Result<Value> {
    let opened = OpenedChallenge::read(challenge_path, claim_submission_path)?;
    let proof = sealed_fault_proof(receipt_path)?;
    let rejection = inference_artifacts::contract::check_fault_proof(
        &opened.challenge.challenge,
        &proof.decoded,
        &raster_prover::chain_fraud::transition_guest_image_id(),
        &raster_prover::authorization::authorization_guest_image_id(),
    )
    .err();
    let id = opened.id()?;
    let mut call = call(
        "submitFaultProof",
        &abi_encode(&fault_proof_token(&id, &proof.journal, &proof.seal)),
    );
    call["challenge"] = json!(hex0x(&id));
    call["claim"] = json!(hex0x(&opened.claim_id));
    call["rejection"] = json!(rejection.as_ref().map(|rejection| rejection.name()));
    call["proof"] = fault_proof_fields(&proof);
    Ok(call)
}

/// The heights of the blocks a submission's blobs are in: the ones a
/// contract's oracle has to know before the submission arrives.
fn heights(blobs: &[PostedBlob]) -> Vec<u64> {
    let mut heights: Vec<u64> = blobs.iter().map(|blob| blob.pointer.height).collect();
    heights.sort_unstable();
    heights.dedup();
    heights
}

/// Sign a 32-byte digest with a secp256k1 key, as the EVM recovers it: `r`,
/// `s` and a recovery byte of 27 or 28, in hex.
pub fn sign_digest(key_hex: &str, digest: &[u8; 32]) -> Result<String> {
    let key = hex::decode(key_hex.trim().trim_start_matches("0x"))
        .context("the attester's key is not hex")?;
    let key = k256::ecdsa::SigningKey::from_slice(&key)
        .map_err(|_| anyhow::anyhow!("the attester's key is not a secp256k1 key"))?;
    let (signature, recovery) = key
        .sign_prehash_recoverable(digest)
        .map_err(|error| anyhow::anyhow!("failed to sign: {error}"))?;
    let mut bytes = signature.to_bytes().to_vec();
    bytes.push(27 + recovery.to_byte());
    Ok(hex::encode(bytes))
}

/// Attest each blob as the trusted host: check that it is in its block and
/// holds what it states, then sign where it is, its block's root and what it
/// holds. A blob that does not check is not signed, and nothing is.
pub fn attest_blobs(
    layer: &dyn PostingLayer,
    blobs: &mut [PostedBlob],
    key_hex: &str,
) -> Result<()> {
    for (index, blob) in blobs.iter().enumerate() {
        ensure!(
            layer.included_at(&blob.pointer, blob.length).is_some(),
            "blob {index} is not in the block it names"
        );
        ensure!(
            layer.proves(blob),
            "blob {index} does not hold what it states"
        );
    }
    for blob in blobs.iter_mut() {
        let root = layer.block_root(blob.pointer.height)?;
        let signature = sign_digest(key_hex, &attestation_digest(blob, &root.data_root)?)?;
        let AvailabilityProof::HostChecked { attestation, .. } = &mut blob.proof;
        *attestation = Some(HostAttestation {
            data_root: root.data_root,
            signature,
        });
    }
    Ok(())
}

/// Attest the blobs of a claim's or a challenge's submission, in place.
pub fn attest_submission(path: &Path, layer: &dyn PostingLayer) -> Result<usize> {
    let key = std::env::var(ATTESTER_KEY_ENV)
        .with_context(|| format!("{ATTESTER_KEY_ENV} is not set: the attester's key, in hex"))?;
    // A claim's submission, or else a challenge's.
    if let Ok(mut claim) = read_json::<ClaimSubmission>(path) {
        attest_blobs(layer, &mut claim.availability, &key)?;
        inference_artifacts::write_json(path, &claim)?;
        return Ok(claim.availability.len());
    }
    let mut challenge: ChallengeSubmission = read_json(path)
        .with_context(|| format!("{} is no claim or challenge submission", path.display()))?;
    attest_blobs(layer, &mut challenge.availability, &key)?;
    inference_artifacts::write_json(path, &challenge)?;
    Ok(challenge.availability.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    use raster_core::fingerprint::{BitPacker, Fingerprint};
    use raster_core::transition::{
        EntrypointAuthorization, InitTransition, OutputAuthorization, SerializableFrontier,
        TransitionJournal, TransitionState,
    };

    /// A journal whose opening state has the given sizes: what moves the
    /// fields a contract reads.
    fn journal(
        ommers: usize,
        leaf: usize,
        bits: usize,
        state: TransitionState,
    ) -> TransitionJournal {
        let frontier = |position: u64| SerializableFrontier {
            position,
            leaf: vec![0x11; leaf],
            ommers: (0..ommers)
                .map(|index| raster_core::merkle::to_node_hash(&[index as u8; 32]))
                .collect(),
        };
        TransitionJournal {
            init_state: InitTransition {
                init_frontier: frontier(u64::MAX - 3),
                init_storage_frontier: frontier(7),
                init_storage_root: vec![0x22; 32],
                init_storage_index_root: vec![0x33; 32],
                fingerprint: Fingerprint {
                    bits_packer: BitPacker::new(1),
                    bits: vec![u64::MAX; bits],
                    len: bits * 64,
                },
            },
            current_state: state,
            transition_image_id: vec![0x44; 32],
            authorization_image_id: vec![0x55; 32],
            input_manifest_commitment: vec![0x66; 32],
            program_commitment: vec![0x77; 32],
            refuted_trace_commitment: vec![0x88; 32],
            entrypoint_authorization: EntrypointAuthorization::Established,
            output_authorization: OutputAuthorization::Established {
                output_commitment: vec![0x99; 32],
            },
            window_is_terminal: true,
            final_committed_root: Some(vec![0xaa; 32]),
        }
    }

    /// The journal's bytes as RISC Zero's guest commits them.
    fn committed(journal: &TransitionJournal) -> Vec<u8> {
        risc0_zkvm::serde::to_vec(journal)
            .unwrap()
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect()
    }

    #[test]
    fn a_journal_is_read_as_risc_zero_wrote_it() {
        // Wherever the window opens, the fields are the journal's own. If
        // Raster's journal changes shape this fails, and the contract's
        // `RasterJournal.read` has to change with it.
        for (ommers, leaf, bits) in [(0, 32, 0), (1, 32, 1), (5, 32, 2), (31, 30, 3), (64, 0, 9)] {
            let bytes = committed(&journal(ommers, leaf, bits, TransitionState::Finished));
            assert_eq!(
                read_fault_journal(&bytes),
                Some(Some(FaultVerdict {
                    transition_image_id: [0x44; 32],
                    authorization_image_id: [0x55; 32],
                    input_manifest: [0x66; 32],
                    program: [0x77; 32],
                    refuted_trace: [0x88; 32],
                })),
                "{ommers} ommers, a leaf of {leaf}, {bits} fingerprint words"
            );
            // Cut short anywhere before the last field read, it is no journal.
            let read_to = bytes
                .windows(32)
                .position(|window| window == [0x88; 32])
                .unwrap()
                + 32;
            for length in (0..read_to).step_by(4) {
                assert_eq!(
                    read_fault_journal(&bytes[..length]),
                    None,
                    "cut at {length}"
                );
            }
            assert!(read_fault_journal(&bytes[..read_to]).is_some());
        }
        // A window that matched its trace to the end is read as no verdict.
        let matched = committed(&journal(3, 32, 2, TransitionState::TerminalMatch));
        assert_eq!(read_fault_journal(&matched), Some(None));
        // A field that should be a hash and is not.
        let mut short = journal(3, 32, 2, TransitionState::Finished);
        short.program_commitment = vec![0x77; 31];
        assert_eq!(read_fault_journal(&committed(&short)), None);
    }

    #[test]
    fn a_signature_is_65_bytes_and_the_same_each_time() {
        // A test key of the local EVM node, published with it.
        let key = "0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d";
        let signature = sign_digest(key, &[7; 32]).unwrap();
        assert_eq!(signature.len(), 130);
        assert!(signature.ends_with("1b") || signature.ends_with("1c"));
        assert_eq!(sign_digest(key, &[7; 32]).unwrap(), signature);
        assert_ne!(sign_digest(key, &[8; 32]).unwrap(), signature);
        assert!(sign_digest("zz", &[7; 32]).is_err());
    }
}
