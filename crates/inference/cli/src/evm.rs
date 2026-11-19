//! What iris hands to whoever sends transactions: the calldata of the calls
//! its contracts take, and the attester's signature over each posted blob.
//!
//! iris sends nothing itself and holds no wallet. Every command here reads
//! files and prints bytes; iris-backend (or a script with `cast`) sends them.
//! The one key iris uses is the attester's, read from `IRIS_ATTESTER_KEY`: the
//! trusted host's, standing in for an availability proof.

use std::path::Path;

use anyhow::{ensure, Context, Result};
use inference_artifacts::availability::{AvailabilityProof, HostAttestation, PostedBlob};
use inference_artifacts::contract::{ChallengeSubmission, ClaimSubmission};
use inference_artifacts::evm::{
    abi_encode, attestation_digest, call_with_bytes, challenge_token, claim_id, claim_token,
    configuration_token,
};
use inference_artifacts::{
    checkpoint_opening, read_checkpoint_hashes, read_checkpoint_trace, read_json, CHECKPOINTS_TXT,
    CHECKPOINT_TRACE_JSON,
};
use serde_json::{json, Value};

use crate::availability::PostingLayer;

/// The attester's secret key: 32 bytes of hex, with or without `0x`.
pub const ATTESTER_KEY_ENV: &str = "IRIS_ATTESTER_KEY";

fn hex0x(bytes: &[u8]) -> String {
    format!("0x{}", hex::encode(bytes))
}

/// A call as a sender needs it: what the contract decodes, and the whole
/// calldata of `name(bytes)` over it.
fn call(name: &str, encoded: &[u8]) -> Value {
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
    let token = challenge_token(
        &claim_id(&claim)?,
        &challenge,
        checkpoint,
        &checkpoint_opening(&hashes, stage)?,
    )?;
    let mut call = call("submitChallenge", &abi_encode(&token));
    call["claim"] = json!(hex0x(&claim_id(&claim)?));
    call["heights"] = json!(heights(&challenge.availability));
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
