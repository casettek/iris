//! The local chains, as a test drives them: iris sending its own calls to
//! the contracts `devstack/deploy.sh` deployed, and holding the contracts'
//! answers to its own.
//!
//! Everywhere else iris sends nothing and holds no wallet: it writes the
//! calldata of a call and whoever runs it sends that. Here it sends, through
//! an account the local EVM node keeps unlocked, so it still holds no key.
//! That works against a development node and nothing else, which is the
//! point: these commands are for the local stack and its tests.

pub mod replay;
pub mod submit;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{bail, ensure, Context, Result};
use inference_artifacts::evm::keccak256;
use serde::Deserialize;
use serde_json::{json, Value};

/// Names the file `devstack/deploy.sh` wrote: where the contracts are.
pub const DEPLOYMENT_ENV: &str = "IRIS_CHAIN_DEPLOYMENT";
/// That file, from the repository's root.
pub const DEFAULT_DEPLOYMENT: &str = "devstack/deployment.json";

/// Where the contracts are and what they were deployed with.
#[derive(Debug, Clone, Deserialize)]
pub struct Deployment {
    /// The EVM node, as `http://host:port`.
    pub rpc: String,
    pub oracle: String,
    pub registry: String,
    pub claims: String,
    /// The account that sends: it relays block roots and owns the registry.
    pub relayer: String,
    pub retention_secs: u64,
    pub challenge_period_secs: u64,
    /// The contracts' errors by selector, eight hex digits each.
    #[serde(default)]
    pub errors: BTreeMap<String, String>,
}

impl Deployment {
    pub fn path() -> PathBuf {
        std::env::var_os(DEPLOYMENT_ENV)
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(DEFAULT_DEPLOYMENT))
    }

    pub fn read() -> Result<Self> {
        let path = Self::path();
        inference_artifacts::read_json(&path).with_context(|| {
            format!(
                "no deployment at {}: deploy the contracts (`just devstack-deploy`), and name \
                 the file with {DEPLOYMENT_ENV} when this is not run from the repository's root",
                path.display()
            )
        })
    }
}

/// Why the contract refused a call: the error it reverted with, by name when
/// it is one of the contracts' own.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Revert {
    pub error: String,
}

impl Revert {
    /// The error's name, without its arguments.
    pub fn name(&self) -> &str {
        self.error.split('(').next().unwrap_or(&self.error)
    }
}

impl std::fmt::Display for Revert {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.error)
    }
}

/// A transaction the node mined.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Sent {
    pub gas_used: u64,
}

/// The local EVM node with the contracts on it.
pub struct Node {
    /// `host:port`.
    address: String,
    pub deployment: Deployment,
}

impl Node {
    pub fn open() -> Result<Self> {
        let deployment = Deployment::read()?;
        let address = deployment
            .rpc
            .strip_prefix("http://")
            .with_context(|| format!("{} is not an http:// address", deployment.rpc))?
            .trim_end_matches('/')
            .to_string();
        Ok(Self {
            address,
            deployment,
        })
    }

    /// One JSON-RPC request: its result, or the error the node answered.
    fn rpc(&self, method: &str, params: Value) -> Result<std::result::Result<Value, Value>> {
        let body = json!({"jsonrpc": "2.0", "id": 1, "method": method, "params": params});
        let response: Value = serde_json::from_slice(
            &crate::celestia::http_post(&self.address, &body.to_string())
                .with_context(|| format!("{method}: is the EVM node up (`just devstack-up`)?"))?,
        )
        .with_context(|| format!("{method}: the node's answer is not JSON"))?;
        if let Some(error) = response.get("error") {
            return Ok(Err(error.clone()));
        }
        response
            .get("result")
            .cloned()
            .map(Ok)
            .with_context(|| format!("{method}: the node's answer has no result"))
    }

    fn transaction(&self, to: &str, data: &[u8]) -> Value {
        json!({
            "from": self.deployment.relayer,
            "to": to,
            "data": format!("0x{}", hex::encode(data)),
        })
    }

    /// A call that changes nothing: what it returns, or the error it
    /// reverts with. With `at`, as if the block's time were that, in Unix
    /// seconds.
    pub fn call(
        &self,
        to: &str,
        data: &[u8],
        at: Option<u64>,
    ) -> Result<std::result::Result<Vec<u8>, Revert>> {
        let mut params = vec![self.transaction(to, data), json!("latest")];
        if let Some(at) = at {
            // No state is overridden; the block's time is.
            params.push(json!({}));
            params.push(json!({"time": format!("{at:#x}")}));
        }
        match self.rpc("eth_call", Value::Array(params))? {
            Ok(result) => {
                let text = result
                    .as_str()
                    .context("eth_call: the result is no string")?;
                Ok(Ok(hex::decode(text.trim_start_matches("0x"))
                    .context("eth_call: the result is not hex")?))
            }
            Err(error) => Ok(Err(self.revert(&error))),
        }
    }

    /// Send a transaction from the node's unlocked account and wait for it.
    /// A call the contract would refuse is not sent.
    pub fn send(&self, to: &str, data: &[u8]) -> Result<std::result::Result<Sent, Revert>> {
        if let Err(revert) = self.call(to, data, None)? {
            return Ok(Err(revert));
        }
        let hash = match self.rpc("eth_sendTransaction", json!([self.transaction(to, data)]))? {
            Ok(hash) => hash,
            Err(error) => return Ok(Err(self.revert(&error))),
        };
        for _ in 0..600 {
            let receipt = self
                .rpc("eth_getTransactionReceipt", json!([hash]))?
                .map_err(|error| anyhow::anyhow!("eth_getTransactionReceipt: {error}"))?;
            if receipt.is_null() {
                std::thread::sleep(std::time::Duration::from_millis(100));
                continue;
            }
            ensure!(
                receipt["status"] == "0x1",
                "transaction {hash} was mined and failed, after a call of it passed"
            );
            let gas_used = receipt["gasUsed"]
                .as_str()
                .and_then(|gas| u64::from_str_radix(gas.trim_start_matches("0x"), 16).ok())
                .context("the receipt states no gas")?;
            return Ok(Ok(Sent { gas_used }));
        }
        bail!("transaction {hash} was not mined within a minute")
    }

    /// The contract error in a node's error answer: the first selector in it
    /// that is one of the contracts' own, else the node's message.
    fn revert(&self, error: &Value) -> Revert {
        let text = error.to_string();
        let named = text
            .split(|ch: char| !ch.is_ascii_alphanumeric())
            .filter_map(|word| word.strip_prefix("0x"))
            .filter(|digits| digits.len() >= 8)
            .find_map(|digits| {
                self.deployment
                    .errors
                    .get(&digits[..8].to_ascii_lowercase())
            });
        Revert {
            error: match named {
                Some(name) => name.clone(),
                None => error["message"].as_str().map(String::from).unwrap_or(text),
            },
        }
    }
}

/// The calldata of a call whose arguments are all one word each.
pub fn call_with_words(signature: &str, words: &[[u8; 32]]) -> Vec<u8> {
    let mut data = keccak256(signature.as_bytes())[..4].to_vec();
    for word in words {
        data.extend_from_slice(word);
    }
    data
}

pub fn word_of_u64(value: u64) -> [u8; 32] {
    let mut word = [0; 32];
    word[24..].copy_from_slice(&value.to_be_bytes());
    word
}

/// A 32-byte value in hex, with or without `0x`.
pub fn word_of_hex(hex_text: &str) -> Result<[u8; 32]> {
    hex::decode(hex_text.trim_start_matches("0x"))
        .ok()
        .and_then(|bytes| bytes.try_into().ok())
        .with_context(|| format!("{hex_text} is not 32 bytes of hex"))
}

/// The `index`th word of what a call returned, as a number.
pub fn returned_u64(returned: &[u8], index: usize) -> Result<u64> {
    let word = returned
        .get(index * 32..index * 32 + 32)
        .with_context(|| format!("the call returned no word {index}"))?;
    ensure!(
        word[..24].iter().all(|byte| *byte == 0),
        "word {index} of what the call returned is no 64-bit number"
    );
    Ok(u64::from_be_bytes(
        word[24..].try_into().expect("eight bytes"),
    ))
}

/// `0x` and the hex of a call's `calldata` or `argument` field.
pub(crate) fn bytes_of(call: &Value, field: &str) -> Result<Vec<u8>> {
    let text = call[field]
        .as_str()
        .with_context(|| format!("the call has no {field}"))?;
    hex::decode(text.trim_start_matches("0x")).with_context(|| format!("{field} is not hex"))
}

/// Write the deployment file: where the contracts are, what they were
/// deployed with, and their errors by selector, read from what `forge
/// inspect <contract> errors --json` printed for each contract, one JSON
/// object after another in `errors_text`.
pub fn write_deployment(path: &Path, mut fields: Value, errors_text: &str) -> Result<usize> {
    let mut errors = BTreeMap::new();
    for listed in serde_json::Deserializer::from_str(errors_text).into_iter::<Value>() {
        let listed = listed.context("the errors are not JSON objects")?;
        for (signature, selector) in listed
            .as_object()
            .context("a contract's errors are not an object")?
        {
            let selector = selector
                .as_str()
                .context("an error's selector is no string")?
                .trim_start_matches("0x")
                .to_ascii_lowercase();
            // The selector is the signature's: nothing is taken on trust.
            ensure!(
                hex::encode(&keccak256(signature.as_bytes())[..4]) == selector,
                "`{signature}` does not have the selector {selector}"
            );
            errors.insert(selector, signature.clone());
        }
    }
    let count = errors.len();
    fields["errors"] = json!(errors);
    inference_artifacts::write_json(path, &fields)?;
    Ok(count)
}
