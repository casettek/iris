//! Send a claim, a challenge, a fault proof or a fault to the local
//! contracts, with what has to be there first: the blobs attested, their
//! blocks' roots relayed, and the claim's configuration registered.

use std::path::Path;

use anyhow::{bail, Context, Result};
use serde_json::{json, Value};

use super::{bytes_of, call_with_words, returned_u64, word_of_hex, word_of_u64, Node};
use crate::posting::PostingLayer;

/// What to do before a submission is sent. All of it by default.
#[derive(Debug, Clone, Copy)]
pub struct Before {
    /// Sign each blob as the trusted host.
    pub attest: bool,
    /// Post each blob's block root to the oracle, as the trusted relayer.
    pub relay: bool,
    /// Register the claim's configuration when it is not registered.
    pub register: bool,
}

impl Default for Before {
    fn default() -> Self {
        Self {
            attest: true,
            relay: true,
            register: true,
        }
    }
}

/// Post each block's root to the oracle.
pub fn relay(node: &Node, layer: &dyn PostingLayer, heights: &[u64]) -> Result<()> {
    for height in heights {
        let root = layer.block_root(*height)?;
        let data = call_with_words(
            "post(uint64,bytes32,uint64)",
            &[
                word_of_u64(*height),
                word_of_hex(&root.data_root)?,
                word_of_u64(root.time),
            ],
        );
        if let Err(revert) = node.send(&node.deployment.oracle, &data)? {
            bail!("the oracle refused block {height}: {revert}");
        }
    }
    Ok(())
}

/// The heights a call's blobs are at.
pub(crate) fn heights_of(call: &Value) -> Vec<u64> {
    call["heights"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(Value::as_u64)
        .collect()
}

/// Register the run's configuration unless it is registered. Its id.
pub fn register(node: &Node, run_spec: &Path) -> Result<String> {
    let configuration = crate::checks::spec_configuration(run_spec, None)?;
    let call = crate::evm::configuration_call(&configuration)?;
    let id = call["configuration"]
        .as_str()
        .context("the configuration call has no id")?
        .to_string();
    let registered = node
        .call(
            &node.deployment.registry,
            &call_with_words("isRegistered(bytes32)", &[word_of_hex(&id)?]),
            None,
        )?
        .map_err(|revert| anyhow::anyhow!("the registry does not answer: {revert}"))?;
    if returned_u64(&registered, 0)? == 0 {
        if let Err(revert) = node.send(&node.deployment.registry, &bytes_of(&call, "calldata")?)? {
            bail!("the registry refused the configuration: {revert}");
        }
    }
    Ok(id)
}

/// Send a built call to the claims contract. What happened, as the outcome
/// every submission prints: `accepted`, and the gas used or the error.
fn send(node: &Node, call: &Value, mut outcome: Value) -> Result<Value> {
    match node.send(&node.deployment.claims, &bytes_of(call, "calldata")?)? {
        Ok(sent) => {
            outcome["accepted"] = json!(true);
            outcome["gas_used"] = json!(sent.gas_used);
            outcome["calldata_bytes"] = json!(bytes_of(call, "calldata")?.len());
        }
        Err(revert) => {
            outcome["accepted"] = json!(false);
            outcome["error"] = json!(revert.error);
        }
    }
    Ok(outcome)
}

/// Submit a posted claim.
pub fn claim(
    node: &Node,
    layer: &dyn PostingLayer,
    run_spec: Option<&Path>,
    submission: &Path,
    before: Before,
) -> Result<Value> {
    if before.attest {
        crate::evm::attest_submission(submission, layer)?;
    }
    let call = crate::evm::claim_call(submission)?;
    let mut outcome = json!({"claim": call["claim"]});
    if before.register {
        let run_spec = run_spec.context("registering a claim's configuration needs --run")?;
        outcome["configuration"] = json!(register(node, run_spec)?);
    }
    if before.relay {
        relay(node, layer, &heights_of(&call))?;
    }
    send(node, &call, outcome)
}

/// Submit a posted challenge of a claim the contract holds.
pub fn challenge(
    node: &Node,
    layer: &dyn PostingLayer,
    claim_submission: &Path,
    challenge: &Path,
    before: Before,
) -> Result<Value> {
    if before.attest {
        crate::evm::attest_submission(challenge, layer)?;
    }
    let call = crate::evm::challenge_call(challenge, claim_submission)?;
    let outcome = json!({"claim": call["claim"], "challenge": call["challenge"]});
    if before.relay {
        relay(node, layer, &heights_of(&call))?;
    }
    send(node, &call, outcome)
}

/// Answer a challenge with a fault proof. `expected_rejection` is what iris
/// expects the contract to say of the proof's journal.
pub fn fault_proof(
    node: &Node,
    claim_submission: &Path,
    challenge: &Path,
    receipt: &Path,
) -> Result<Value> {
    let call = crate::evm::fault_proof_call(challenge, claim_submission, receipt)?;
    let outcome = json!({
        "challenge": call["challenge"],
        "expected_rejection": call["rejection"],
    });
    send(node, &call, outcome)
}

/// Send one of the calls `iris claim audit --calls` wrote: point the
/// contract at a fault.
pub fn fault(node: &Node, calls: &Path, index: usize) -> Result<Value> {
    let calls: Vec<Value> = inference_artifacts::read_json(calls)?;
    let call = calls
        .get(index)
        .with_context(|| format!("there is no call {index}: the audit wrote {}", calls.len()))?;
    let outcome = json!({
        "checkpoint": call["checkpoint"],
        "stage": call["stage"],
        "input": call["input"],
        "expected": call["expected"],
    });
    send(node, call, outcome)
}
