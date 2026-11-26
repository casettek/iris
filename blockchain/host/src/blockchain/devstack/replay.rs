//! Hold the local contracts to iris's own verdict, case by case.
//!
//! [`claim`] takes a claim that `iris claim post` has posted. It asks iris's
//! own checks and the contract about the claim as it is, then about one
//! changed copy after another, and compares the two answers: both take it, or
//! both refuse it at the same check. The claim itself is then submitted.
//!
//! [`challenge`] does the same for a challenge that `iris challenge post` has
//! posted. The claim it disputes is submitted first if the contract does not
//! hold it.
//!
//! [`faults`] takes a claim in the directory `iris claim fetch` wrote, and
//! asks the contract everything `iris claim audit` asks of it with no replay.
//! The contract must answer each as iris does: refuse to void an honest
//! stage, with the same error, and take a fault iris found. A fault found is
//! then sent, and the claim must read as void of that fault.
//!
//! A changed copy is only ever called, never sent, so nothing but the claim
//! and the challenge themselves reaches the contract's state. A few cases are
//! the contract's alone, where it checks something the Rust rules have no
//! part for yet: the host's signature, and what the contract already holds.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{bail, ensure, Context, Result};
use blockchain_rules::checks::CLAIM_FAULTS;
use serde_json::{json, Value};

use super::submit::{heights_of, register, relay};
use super::{bytes_of, call_with_words, returned_u64, word_of_hex, Node};
use crate::claim_check::{
    check_challenge_submission, check_claim, ChallengeCheckOptions, ClaimCheckOptions,
    ClaimCheckReport,
};
use crate::posting::PostingLayer;

/// The check of iris's that each contract error belongs to: the start of the
/// check's name in a [`ClaimCheckReport`].
const CHECK_OF: &[(&str, &str)] = &[
    ("UnregisteredProcedure", "submission: procedure"),
    ("AnotherShape", "submission: procedure kind"),
    ("StatementMismatch", "submission: statement"),
    ("OutputMismatch", "submission: statement"),
    ("NotValid", "submission: statement"),
    ("AnotherSourceRecord", "submission: source record"),
    ("UnexpectedSource", "submission: source record"),
    ("MissingSource", "submission: source record"),
    ("SourceFault", "submission: source record"),
    ("AnotherSigner", "submission: source record"),
    ("AnotherRecipe", "submission: source record"),
    ("AnotherRelayed", "submission: source record"),
    ("AnotherForms", "submission: source record"),
    ("AnotherResult", "submission: source record"),
    ("AnotherConsensus", "submission: source record"),
    ("NothingPosted", "posted"),
    ("NotIncluded", "posted"),
    ("ExpiresTooSoon", "posted"),
    ("NotProven", "posted"),
    ("AnotherSubmitter", "posted"),
    ("CheckpointsNotPosted", "posted"),
    ("TraceBodyNotPosted", "posted"),
    ("RecordsNotStated", "submission: records and results"),
    ("NotTheResultStage", "submission: records and results"),
    ("AnotherStatedResult", "submission: records and results"),
    ("NoCheck", "submission: records and results"),
    ("NotACheck", "submission: records and results"),
    ("AnotherStatedRecord", "submission: records and results"),
    ("NotTheCheckpoint", "challenge: of the claim's checkpoint"),
    (
        "UnapprovedProgram",
        "challenge admission: the stage names an approved program",
    ),
    ("OutputNotDisputed", "challenge: Raster admits it"),
    ("TraceTooShort", "challenge: Raster admits it"),
    ("WindowSizeNotProtocol", "challenge: Raster admits it"),
    ("NonCanonicalBitsPerItem", "challenge: Raster admits it"),
];

/// An address no claim or challenge here is made for.
const SOMEONE_ELSE: &str = "0x00000000000000000000000000000000000c0b1e";

fn check_of(contract_error: &str) -> Option<&'static str> {
    CHECK_OF
        .iter()
        .find(|(error, _)| *error == contract_error)
        .map(|(_, check)| *check)
}

/// The name a check goes by in [`CHECK_OF`].
fn check_name(check: &str) -> String {
    let check = check
        .replace("the first inference's ", "")
        .replace("the second inference's ", "");
    if check.contains("posted, proven and retrievable") {
        return String::from("posted");
    }
    if check.starts_with("challenge: about the claim's checkpoint") {
        return String::from("challenge: of the claim's checkpoint");
    }
    // The longest name the check starts with: one name is the start of
    // another (`submission: procedure` and `submission: procedure kind`).
    CHECK_OF
        .iter()
        .map(|(_, name)| *name)
        .filter(|name| check.starts_with(name))
        .max_by_key(|name| name.len())
        .map(String::from)
        .unwrap_or(check)
}

/// What iris decides: `None` when it takes the submission, otherwise the
/// first check it refuses it at.
fn refused_at(report: &ClaimCheckReport) -> Option<String> {
    report
        .lines
        .iter()
        // The checks of a submission, and not the comparisons a later
        // challenge of the claim would meet.
        .filter(|line| !line.check.starts_with("challenge admission: every "))
        .find(|line| line.rejection.is_some())
        .map(|line| check_name(line.check))
}

/// The cases of one replay, printed as they are made.
#[derive(Default)]
struct Replay {
    rows: Vec<(String, bool)>,
}

impl Replay {
    fn row(&mut self, name: &str, iris: &str, contract: &str, agree: bool) {
        println!(
            "{}  {name:24} iris: {iris:58} contract: {contract}",
            if agree { "ok  " } else { "DIFF" }
        );
        self.rows.push((name.to_string(), agree));
    }

    /// One case: the two verdicts, and whether they are the same. A case
    /// that is the contract's alone has no verdict of iris's.
    fn record(
        &mut self,
        name: &str,
        iris: Option<&Option<String>>,
        contract: &Option<String>,
        contract_only: Option<&str>,
    ) {
        let (iris_text, agree) = match (contract_only, iris) {
            (Some(expected), _) => (
                String::from("(no part)"),
                contract.as_deref() == Some(expected),
            ),
            (None, Some(None)) => (String::from("accepted"), contract.is_none()),
            (None, Some(Some(check))) => (
                format!("refused at {check}"),
                contract
                    .as_deref()
                    .is_some_and(|error| check_of(error) == Some(check.as_str())),
            ),
            (None, None) => (String::from("(not asked)"), false),
        };
        self.row(
            name,
            &iris_text,
            contract.as_deref().unwrap_or("accepted"),
            agree,
        );
    }

    fn finish(self) -> Result<()> {
        let differing: Vec<&str> = self
            .rows
            .iter()
            .filter(|(_, agree)| !agree)
            .map(|(name, _)| name.as_str())
            .collect();
        println!(
            "{} of {} cases agree",
            self.rows.len() - differing.len(),
            self.rows.len()
        );
        ensure!(
            differing.is_empty(),
            "the contract and iris differ on: {}",
            differing.join(", ")
        );
        Ok(())
    }
}

/// What the contract decides of a call, by a call that changes nothing:
/// `None` when it takes it, otherwise the error it reverts with.
fn contract_verdict(node: &Node, call: &Value, at: Option<u64>) -> Result<Option<String>> {
    Ok(node
        .call(&node.deployment.claims, &bytes_of(call, "calldata")?, at)?
        .err()
        .map(|revert| revert.name().to_string()))
}

/// Submit a claim's call unless the contract holds the claim.
fn hold_claim(node: &Node, call: &Value) -> Result<()> {
    match contract_verdict(node, call, None)?.as_deref() {
        None => {
            if let Err(revert) = node.send(&node.deployment.claims, &bytes_of(call, "calldata")?)? {
                bail!("the contract refused the claim it had taken in a call: {revert}");
            }
            Ok(())
        }
        Some("AlreadySubmitted") => Ok(()),
        Some(error) => bail!("the contract does not take the claim: {error}"),
    }
}

/// Write a changed copy of a submission beside it. Its path.
fn changed(
    path: &Path,
    name: &str,
    original: &Value,
    change: impl FnOnce(&mut Value),
) -> Result<PathBuf> {
    let mut copied = original.clone();
    change(&mut copied);
    let file = path
        .file_name()
        .and_then(|file| file.to_str())
        .context("the submission has no file name")?;
    let out = path.with_file_name(format!("replay-{name}.{file}"));
    procedure_definition::write_json(&out, &copied)?;
    Ok(out)
}

/// Take the host's signature off every blob of a submission.
fn unsign(submission: &mut Value) {
    for blob in submission["availability"]
        .as_array_mut()
        .into_iter()
        .flatten()
    {
        if let Some(proof) = blob["proof"].as_object_mut() {
            proof.remove("attestation");
        }
    }
}

fn append(text: &mut Value, more: &str) {
    *text = json!(format!("{}{more}", text.as_str().unwrap_or_default()));
}

fn add(number: &mut Value, more: u64) {
    *number = json!(number.as_u64().unwrap_or_default() + more);
}

fn hex_of(byte: &str) -> Value {
    json!(byte.repeat(32))
}

/// When the contract accepted a claim, in Unix seconds.
fn accepted_at(node: &Node, claim_id: &str) -> Result<u64> {
    let held = node
        .call(
            &node.deployment.claims,
            &call_with_words("claims(bytes32)", &[word_of_hex(claim_id)?]),
            None,
        )?
        .map_err(|revert| {
            anyhow::anyhow!("the contract does not answer for claim {claim_id}: {revert}")
        })?;
    returned_u64(&held, 4)
}

/// The fault the contract holds a claim void of, or `None`.
fn voided(node: &Node, claim_id: &str) -> Result<Option<&'static str>> {
    let held = node
        .call(
            &node.deployment.claims,
            &call_with_words("voided(bytes32)", &[word_of_hex(claim_id)?]),
            None,
        )?
        .map_err(|revert| {
            anyhow::anyhow!("the contract does not answer for claim {claim_id}: {revert}")
        })?;
    let fault = returned_u64(&held, 0)? as usize;
    ensure!(
        fault < CLAIM_FAULTS.len(),
        "the contract holds claim {claim_id} void of fault {fault}, which iris has no name for"
    );
    Ok((fault != 0).then(|| CLAIM_FAULTS[fault]))
}

fn id_of(call: &Value, field: &str) -> Result<String> {
    call[field]
        .as_str()
        .map(String::from)
        .with_context(|| format!("the call has no {field}"))
}

/// Hold the contract to iris on a posted claim and changed copies of it,
/// then submit the claim.
pub fn claim(
    node: &Node,
    layer: &dyn PostingLayer,
    specification: &Path,
    path: &Path,
) -> Result<()> {
    let base_dir = std::env::current_dir().context("failed to resolve current directory")?;
    let mut replay = Replay::default();

    crate::blockchain::evm::attest_submission(path, layer)?;
    let honest: Value = procedure_definition::read_json(path)?;
    let built = crate::blockchain::evm::claim_call(path)?;
    register(node, specification)?;
    relay(node, layer, &heights_of(&built))?;

    let verdicts = |submission: &Path, at: Option<u64>, ask_iris: bool| -> Result<_> {
        let iris = if ask_iris {
            Some(refused_at(&check_claim(ClaimCheckOptions {
                base_dir: base_dir.clone(),
                specification_path: specification.to_path_buf(),
                submission_path: submission.to_path_buf(),
                raster_run: None,
                pins_path: None,
                registry_path: None,
                da_dir: None,
                submitted_at: at,
            })?))
        } else {
            None
        };
        let contract =
            contract_verdict(node, &crate::blockchain::evm::claim_call(submission)?, at)?;
        Ok((iris, contract))
    };

    let (iris, contract) = verdicts(path, None, true)?;
    let already = contract.as_deref() == Some("AlreadySubmitted");
    replay.record(
        "as built",
        iris.as_ref(),
        if already { &None } else { &contract },
        None,
    );
    if contract.is_none() {
        // The id iris works out for a claim is the one the contract gives it.
        let given = node
            .call(
                &node.deployment.claims,
                &bytes_of(&built, "calldata")?,
                None,
            )?
            .ok()
            .map(|id| format!("0x{}", hex::encode(id)));
        let ours = id_of(&built, "claim")?;
        let agree = given
            .as_deref()
            .is_some_and(|given| given.eq_ignore_ascii_case(&ours));
        replay.row("id", &ours, given.as_deref().unwrap_or("none"), agree);
    }

    let mut case = |name: &str,
                    change: &dyn Fn(&mut Value),
                    at: Option<u64>,
                    attest: bool,
                    contract_only: Option<&str>|
     -> Result<()> {
        let copy = changed(path, name, &honest, change)?;
        if attest {
            crate::blockchain::evm::attest_submission(&copy, layer)?;
        }
        let (iris, contract) = verdicts(&copy, at, contract_only.is_none())?;
        replay.record(name, iris.as_ref(), &contract, contract_only);
        Ok(())
    };
    case(
        "other-procedure",
        &|s| s["procedure"] = hex_of("00"),
        None,
        false,
        None,
    )?;
    case(
        "other-output",
        &|s| append(&mut s["inferences"][0]["output_text"], "!"),
        None,
        false,
        None,
    )?;
    case(
        "not-valid",
        &|s| s["inferences"][0]["record"]["valid"] = json!(false),
        None,
        false,
        None,
    )?;
    case(
        "other-statement",
        &|s| s["input_text_sha256"] = hex_of("11"),
        None,
        false,
        None,
    )?;
    case(
        "other-result",
        &|s| add(&mut s["inferences"][0]["result"]["code"], 7),
        None,
        false,
        None,
    )?;
    if honest["inferences"].as_array().map_or(0, Vec::len) > 1 {
        case(
            "other-second-output",
            &|s| append(&mut s["inferences"][1]["output_text"], "!"),
            None,
            false,
            None,
        )?;
    }
    if honest["relays"]
        .as_array()
        .is_some_and(|relays| !relays.is_empty())
    {
        // The first output swapped for another, with a record that commits
        // to it: the relay's record names the output it was built on, and
        // this is not it.
        case(
            "other-relayed-output",
            &|s| {
                use sha2::{Digest, Sha256};
                let output = "another answer";
                s["inferences"][0]["output_text"] = json!(output);
                s["inferences"][0]["record"]["output_text_sha256"] =
                    json!(hex::encode(Sha256::digest(output.as_bytes())));
            },
            None,
            false,
            None,
        )?;
        // Another relay record than the one the second statement names.
        case(
            "other-relay-record",
            &|s| add(&mut s["relays"][0]["documents"], 1),
            None,
            false,
            None,
        )?;
        case(
            "no-relay-record",
            &|s| s["relays"] = json!([]),
            None,
            false,
            None,
        )?;
    }
    if !honest["source"].is_null() {
        case(
            "other-source-record",
            &|s| add(&mut s["source"]["documents"], 1),
            None,
            false,
            None,
        )?;
        case(
            "missing-source",
            &|s| {
                s.as_object_mut().map(|s| s.remove("source"));
            },
            None,
            false,
            None,
        )?;
    }
    if !honest["consensus"].is_null() {
        case(
            "other-agreement",
            &|s| {
                let agree = s["consensus"]["agree"].as_bool().unwrap_or_default();
                s["consensus"]["agree"] = json!(!agree);
            },
            None,
            false,
            None,
        )?;
        case(
            "no-agreement",
            &|s| {
                s.as_object_mut().map(|s| s.remove("consensus"));
            },
            None,
            false,
            None,
        )?;
    }
    case(
        "copied",
        &|s| s["claimer"] = json!(SOMEONE_ELSE),
        None,
        false,
        None,
    )?;
    case(
        "nothing-posted",
        &|s| s["availability"] = json!([]),
        None,
        false,
        None,
    )?;
    case(
        "not-included",
        &|s| add(&mut s["availability"][0]["pointer"]["height"], 1),
        None,
        false,
        None,
    )?;
    let height = honest["availability"][0]["pointer"]["height"]
        .as_u64()
        .context("the claim posted no blob")?;
    case(
        "expires",
        &|_| {},
        Some(
            layer.block_root(height)?.time + node.deployment.retention_secs
                - node.deployment.challenge_period_secs
                + 1,
        ),
        false,
        None,
    )?;
    // The contract's alone: iris reads the claim's directory, and stops
    // before its checks when the list there is not the one the submission
    // names.
    case(
        "other-root",
        &|s| s["checkpoints_root"] = hex_of("22"),
        None,
        false,
        Some("CheckpointsNotPosted"),
    )?;
    case(
        "overstated-contents",
        &|s| {
            let contents = &mut s["availability"][0]["contents"];
            if !contents["run_inputs"].is_array() {
                contents["run_inputs"] = json!([]);
            }
            if let Some(run_inputs) = contents["run_inputs"].as_array_mut() {
                run_inputs.push(hex_of("33"));
            }
        },
        None,
        false,
        None,
    )?;
    // The host signs what the blobs hold when no output is read: no records.
    case(
        "records-not-stated",
        &|s| {
            for posted in s["availability"].as_array_mut().into_iter().flatten() {
                posted["contents"]["records"] = json!([]);
                posted["proof"]["outputs"] = json!([]);
            }
            unsign(s);
        },
        None,
        true,
        None,
    )?;
    // The contract's alone: the host's signature is how it learns what the
    // Rust rules read from the layer itself.
    case("unsigned", &unsign, None, false, Some("NotIncluded"))?;

    if !already {
        if let Err(revert) = node.send(&node.deployment.claims, &bytes_of(&built, "calldata")?)? {
            bail!("the contract refused the claim it had taken in a call: {revert}");
        }
    }
    // The contract's alone: the Rust rules hold no earlier claims.
    replay.record(
        "again",
        None,
        &contract_verdict(node, &built, None)?,
        Some("AlreadySubmitted"),
    );
    println!("{{\"claim\": \"{}\"}}", id_of(&built, "claim")?);
    replay.finish()
}

/// Hold the contract to iris on a posted challenge and changed copies of
/// it, then submit the challenge.
pub fn challenge(
    node: &Node,
    layer: &dyn PostingLayer,
    specification: &Path,
    claim_path: &Path,
    path: &Path,
) -> Result<()> {
    let mut replay = Replay::default();

    // The disputed claim, as the contract holds it. A contract takes a claim
    // with a false checkpoint: that is what a challenge is for.
    crate::blockchain::evm::attest_submission(claim_path, layer)?;
    let claim = crate::blockchain::evm::claim_call(claim_path)?;
    register(node, specification)?;
    relay(node, layer, &heights_of(&claim))?;
    hold_claim(node, &claim)?;

    crate::blockchain::evm::attest_submission(path, layer)?;
    let honest: Value = procedure_definition::read_json(path)?;
    let built = crate::blockchain::evm::challenge_call(path, claim_path)?;
    relay(node, layer, &heights_of(&built))?;

    let verdicts = |submission: &Path, at: Option<u64>, ask_iris: bool| -> Result<_> {
        let iris = if ask_iris {
            Some(refused_at(&check_challenge_submission(
                ChallengeCheckOptions {
                    specification_path: specification.to_path_buf(),
                    pins_path: None,
                    challenge_path: submission.to_path_buf(),
                    submission_path: claim_path.to_path_buf(),
                    da_dir: None,
                    submitted_at: at,
                },
            )?))
        } else {
            None
        };
        let contract = contract_verdict(
            node,
            &crate::blockchain::evm::challenge_call(submission, claim_path)?,
            at,
        )?;
        Ok((iris, contract))
    };

    let (iris, contract) = verdicts(path, None, true)?;
    let already = contract.as_deref() == Some("AlreadySubmitted");
    replay.record(
        "as built",
        iris.as_ref(),
        if already { &None } else { &contract },
        None,
    );

    let mut case = |name: &str,
                    change: &dyn Fn(&mut Value),
                    at: Option<u64>,
                    contract_only: Option<&str>|
     -> Result<()> {
        let copy = changed(path, name, &honest, change)?;
        let (iris, contract) = verdicts(&copy, at, contract_only.is_none())?;
        replay.record(name, iris.as_ref(), &contract, contract_only);
        Ok(())
    };

    case(
        "copied",
        &|s| s["challenger"] = json!(SOMEONE_ELSE),
        None,
        None,
    )?;
    case(
        "nothing-posted",
        &|s| s["availability"] = json!([]),
        None,
        None,
    )?;
    // The output the claim's own checkpoint states: nothing is disputed.
    let stage = honest["stage"]
        .as_u64()
        .context("the challenge names no stage")? as usize;
    let trace = procedure_definition::read_checkpoint_list(
        &claim_path.with_file_name(procedure_definition::CHECKPOINT_LIST_JSON),
    )?;
    let stated = hex::decode(
        &trace
            .checkpoints
            .get(stage)
            .with_context(|| format!("the claim has no checkpoint {stage}"))?
            .output_sha256,
    )
    .context("the checkpoint's output hash is not hex")?;
    case(
        "undisputed",
        &|s| s["challenge"]["claimed_output"] = json!(stated),
        None,
        None,
    )?;
    // A trace in another shape than the protocol fixes for its length: a
    // window the challenger chose, another packing, or too short to have
    // one.
    case(
        "another-window",
        &|s| {
            let header = &mut s["challenge"]["trace_commitment_header"];
            header["window_size"] = json!(header["window_size"].as_u64().unwrap_or_default() / 2);
        },
        None,
        None,
    )?;
    case(
        "another-packing",
        &|s| {
            add(
                &mut s["challenge"]["trace_commitment_header"]["bits_packer"],
                1,
            )
        },
        None,
        None,
    )?;
    case(
        "short-trace",
        &|s| s["challenge"]["trace_commitment_header"]["fingerprint_len"] = json!(2),
        None,
        None,
    )?;
    case(
        "overstated-contents",
        &|s| {
            if let Some(bodies) = s["availability"][0]["contents"]["trace_bodies"].as_array_mut() {
                bodies.push(hex_of("33"));
            }
        },
        None,
        None,
    )?;
    case("unsigned", &unsign, None, Some("NotIncluded"))?;
    // The contract's alone: a challenge names its claim by id on chain, and
    // the Rust rules hold no claims to look one up in.
    let unknown = changed(
        claim_path,
        "unknown",
        &procedure_definition::read_json::<Value>(claim_path)?,
        |s| s["claimer"] = json!(SOMEONE_ELSE),
    )?;
    replay.record(
        "unknown-claim",
        None,
        &contract_verdict(
            node,
            &crate::blockchain::evm::challenge_call(path, &unknown)?,
            None,
        )?,
        Some("UnknownClaim"),
    );
    // The contract's alone: the Rust rules keep no clock for a claim yet.
    // The period runs from when the contract accepted the claim, which can
    // be long after the challenge's trace was posted.
    let mut case = |name: &str,
                    change: &dyn Fn(&mut Value),
                    at: Option<u64>,
                    contract_only: Option<&str>|
     -> Result<()> {
        let copy = changed(path, name, &honest, change)?;
        let (iris, contract) = verdicts(&copy, at, contract_only.is_none())?;
        replay.record(name, iris.as_ref(), &contract, contract_only);
        Ok(())
    };
    case(
        "period-over",
        &|_| {},
        Some(
            accepted_at(node, &id_of(&built, "claim")?)?
                + node.deployment.challenge_period_secs
                + 3600,
        ),
        Some("ChallengePeriodOver"),
    )?;

    if !already {
        if let Err(revert) = node.send(&node.deployment.claims, &bytes_of(&built, "calldata")?)? {
            bail!("the contract refused the challenge it had taken in a call: {revert}");
        }
    }
    // The contract's alone: the Rust rules hold no earlier challenges.
    replay.record(
        "again",
        None,
        &contract_verdict(node, &built, None)?,
        Some("AlreadySubmitted"),
    );
    println!(
        "{{\"claim\": \"{}\", \"stage\": {stage}}}",
        id_of(&built, "claim")?
    );
    replay.finish()
}

/// Ask the contract every on-the-spot question `iris claim audit` asks of a
/// fetched claim, and send the first fault found.
pub fn faults(
    node: &Node,
    layer: &dyn PostingLayer,
    specification: &Path,
    path: &Path,
) -> Result<()> {
    let mut replay = Replay::default();

    let claim = crate::blockchain::evm::claim_call(path)?;
    let claim_id = id_of(&claim, "claim")?;
    register(node, specification)?;
    relay(node, layer, &heights_of(&claim))?;
    hold_claim(node, &claim)?;

    let audit = crate::audit::audit_claim(crate::audit::AuditClaimOptions {
        submission_path: path.to_path_buf(),
        specification_path: specification.to_path_buf(),
        pins_path: None,
    })?;
    procedure_definition::write_json(
        &path.with_file_name("replay-audit-calls.json"),
        &audit.calls,
    )?;
    println!("checkpoints: {}", audit.checkpoints);
    println!("programs approved: {}", audit.programs_approved);
    println!("inputs held to their origin: {}", audit.inputs_linked);
    println!("faults: {}", audit.faults.len());
    let calls = &audit.calls;
    let void_of = |call: &Value| call["expected"]["void"].as_str().map(String::from);
    let found: Vec<&Value> = calls
        .iter()
        .filter(|call| void_of(call).is_some())
        .collect();
    let asked = |call: &Value| contract_verdict(node, call, None);

    if let Some(before) = voided(node, &claim_id)? {
        // Sent on an earlier replay. A void claim answers nothing else.
        let expected = found.first().and_then(|fault| void_of(fault));
        replay.row(
            "already void",
            expected.as_deref().unwrap_or("no fault"),
            before,
            expected.as_deref() == Some(before),
        );
        return replay.finish();
    }

    // Every question, by a call that changes nothing.
    let mut answers: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    let mut differing = Vec::new();
    for call in calls {
        let expected = call["expected"]["revert"].as_str().map(String::from);
        let got = asked(call)?;
        let counted = answers
            .entry(expected.clone().unwrap_or_else(|| String::from("void")))
            .or_default();
        counted.0 += 1;
        if got == expected {
            counted.1 += 1;
        } else {
            let mut place = format!("checkpoint {} ({})", call["checkpoint"], call["stage"]);
            if let Some(input) = call["input"].as_str() {
                place.push_str(&format!(" input {input}"));
            }
            differing.push((
                place,
                call["expected"].to_string(),
                got.unwrap_or_else(|| String::from("would void")),
            ));
        }
    }
    for (expected, (count, same)) in &answers {
        let name: String = format!("{count} x {expected}").chars().take(24).collect();
        replay.row(
            &name,
            if expected == "void" {
                "a fault"
            } else {
                "refuses to void"
            },
            &format!("{same} of {count} the same"),
            same == count,
        );
    }
    for (place, expected, got) in differing.iter().take(20) {
        replay.row("differs", &format!("{place}: {expected}"), got, false);
    }

    // A fault iris found is one the contract takes, and records as iris
    // named it.
    match found.first() {
        Some(fault) => {
            let expected = void_of(fault).expect("filtered above");
            let function = fault["function"]
                .as_str()
                .and_then(|function| function.split('(').next())
                .unwrap_or("void");
            let calldata = bytes_of(fault, "calldata")?;
            let sent = node.send(&node.deployment.claims, &calldata)?;
            let name: String = format!("send {function}").chars().take(24).collect();
            match &sent {
                Ok(sent) => replay.row(
                    &name,
                    &expected,
                    &format!(
                        "sent, {} gas, {} bytes of call data",
                        sent.gas_used,
                        calldata.len()
                    ),
                    true,
                ),
                Err(revert) => replay.row(&name, &expected, &revert.error, false),
            }
            let now = voided(node, &claim_id)?;
            replay.row(
                "void of",
                &expected,
                now.unwrap_or("None"),
                now == Some(expected.as_str()),
            );
            let again = asked(fault)?;
            replay.row(
                "again",
                "(no part)",
                again.as_deref().unwrap_or("would void"),
                again.as_deref() == Some("AlreadyVoided"),
            );
        }
        None => {
            let now = voided(node, &claim_id)?;
            replay.row("not void", "no fault", now.unwrap_or("None"), now.is_none());
        }
    }
    println!(
        "{{\"claim\": \"{claim_id}\", \"questions\": {}, \"faults\": {}}}",
        calls.len(),
        found.len()
    );
    replay.finish()
}
