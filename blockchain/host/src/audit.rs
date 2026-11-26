//! What is wrong with a fetched claim that no replay shows: the questions
//! a contract can be asked of each checkpoint on the spot, each with what
//! the contract is expected to answer.

use std::fs;
use std::path::PathBuf;

use anyhow::{ensure, Context, Result};
use blockchain_rules::checks::ClaimSubmission;
use procedure_definition::{
    read_checkpoint_hashes, read_checkpoint_list, read_json, CHECKPOINTS_TXT, CHECKPOINT_LIST_JSON,
};

use crate::posted::{parent, stage_dir_name, INPUT_MANIFEST_JSON};

#[derive(Debug, Clone)]
pub struct AuditClaimOptions {
    /// The claim's `claim_submission.json`, in a directory that holds its
    /// checkpoint list, checkpoints and input lists: the one `fetch_claim`
    /// wrote.
    pub submission_path: PathBuf,
    /// A specification of the procedure the claim names.
    pub specification_path: PathBuf,
    pub pins_path: Option<PathBuf>,
}

/// One thing wrong with a claim that no replay shows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuditFault {
    /// The position in [`ClaimAudit::calls`] of the call that points at it.
    pub call: usize,
    pub checkpoint: u32,
    pub stage: String,
    /// The input, for a fault of an input.
    pub input: Option<String>,
    /// The fault a contract records: `IrisClaims.ClaimFault`.
    pub fault: String,
}

#[derive(Debug, Clone)]
pub struct ClaimAudit {
    pub checkpoints: usize,
    pub programs_approved: usize,
    /// Inputs that are what the procedure says they must be: an earlier
    /// stage's output, a pinned commitment, or a value the claim posted.
    pub inputs_linked: usize,
    pub faults: Vec<AuditFault>,
    /// Every question a contract can be asked of the claim with no replay:
    /// for each checkpoint whether its program is approved, and for each
    /// input its program takes from an earlier stage whether it is that
    /// stage's output. Each is the contract call with `expected`: the fault
    /// it records (`void`) or the error it reverts with (`revert`).
    pub calls: Vec<serde_json::Value>,
}

/// What is wrong with a claim that no replay shows: a checkpoint that names
/// a program outside the approved set, or an input that is not what an
/// earlier checkpoint produced, not a pinned commitment, or not a value the
/// claim posted.
///
/// A replay runs the program a checkpoint names on the inputs it names, so a
/// challenger asks these two things first, of every checkpoint, from the
/// claim's own files and the procedure alone. Nothing is run. A fault
/// found here is pointed at with `voidClaim`, and the
/// claim is void: there is no challenge to post and nothing to answer.
pub fn audit_claim(options: AuditClaimOptions) -> Result<ClaimAudit> {
    use blockchain_evm::{abi_encode, checkpoint_fault_token, claim_id, InputQuestion};
    use blockchain_rules::checks::{CheckpointHashes, LinkCheck, StageAdmission};

    let submission: ClaimSubmission = read_json(&options.submission_path)?;
    let procedure = crate::claim_check::spec_procedure(
        &options.specification_path,
        options.pins_path.as_deref(),
    )?;
    ensure!(
        procedure.id() == submission.procedure,
        "the claim is of procedure {}, and {} describes {}: give the specification of the \
         procedure the claim names",
        submission.procedure,
        options.specification_path.display(),
        procedure.id()
    );
    let pins = &procedure.programs;
    let dir = parent(&options.submission_path)?;
    let hashes_path = dir.join(CHECKPOINTS_TXT);
    let opened = CheckpointHashes::open(
        &submission,
        &fs::read(&hashes_path)
            .with_context(|| format!("failed to read {}", hashes_path.display()))?,
    )
    .map_err(|refusal| anyhow::anyhow!("{}: {refusal}", hashes_path.display()))?;
    let hashes = read_checkpoint_hashes(&hashes_path)?;
    let list = read_checkpoint_list(&dir.join(CHECKPOINT_LIST_JSON))?;
    ensure!(
        list.checkpoints.len() == hashes.len(),
        "{} lists {} checkpoints, and {CHECKPOINT_LIST_JSON} holds {}",
        hashes_path.display(),
        hashes.len(),
        list.checkpoints.len()
    );
    let claim = claim_id(&submission)?;
    let posted = submission.posted_run_inputs();

    let mut audit = ClaimAudit {
        checkpoints: hashes.len(),
        programs_approved: 0,
        inputs_linked: 0,
        faults: Vec::new(),
        calls: Vec::new(),
    };
    for (index, checkpoint) in list.checkpoints.iter().enumerate() {
        let position = index as u32;
        let path = procedure_definition::checkpoint_opening(&hashes, index)?;
        let refused =
            |refusal| anyhow::anyhow!("checkpoint {index} (`{}`): {refusal}", checkpoint.stage);
        let approved = matches!(
            opened
                .admit_stage_challenge(pins, position, checkpoint)
                .map_err(refused)?,
            StageAdmission::Disputed { .. }
        );
        let mut asked = crate::blockchain::evm::call(
            "voidClaim",
            &abi_encode(&checkpoint_fault_token(
                &claim, position, checkpoint, &path, None,
            )?),
        );
        asked["checkpoint"] = serde_json::json!(position);
        asked["stage"] = serde_json::json!(checkpoint.stage);
        asked["input"] = serde_json::Value::Null;
        asked["expected"] = if approved {
            serde_json::json!({ "revert": "NoFault" })
        } else {
            serde_json::json!({ "void": "UnapprovedProgram" })
        };
        audit.calls.push(asked);
        let Some(program) = pins.program(&checkpoint.program_commitment) else {
            audit.faults.push(AuditFault {
                call: audit.calls.len() - 1,
                checkpoint: position,
                stage: checkpoint.stage.clone(),
                input: None,
                fault: String::from("UnapprovedProgram"),
            });
            // What an unapproved program takes from earlier stages is not
            // defined, so its inputs are not asked about.
            continue;
        };
        audit.programs_approved += 1;
        // Every input of the program: the ones taken from an earlier stage,
        // the ones bound to fixed data, and the ones the run gives.
        let held: std::collections::BTreeSet<&str> = program
            .piped_inputs
            .iter()
            .map(|input| input.name.as_str())
            .chain(
                program
                    .pinned_inputs
                    .iter()
                    .map(|input| input.name.as_str()),
            )
            .chain(program.run_inputs.iter().map(String::as_str))
            .collect();
        if held.is_empty() {
            continue;
        }
        let manifest_path = dir
            .join(stage_dir_name(&checkpoint.stage)?)
            .join(INPUT_MANIFEST_JSON);
        let manifest = fs::read(&manifest_path).with_context(|| {
            format!(
                "failed to read {}: audit the directory `iris claim fetch` wrote",
                manifest_path.display()
            )
        })?;
        for input in held {
            let producer = checkpoint.input_sources.get(input).and_then(|producer| {
                let listed = list.checkpoints.get(*producer as usize)?;
                Some((*producer as usize, listed))
            });
            let verdict = opened.check_link(
                pins,
                position,
                checkpoint,
                &manifest,
                input,
                producer.map(|(_, listed)| listed),
            );
            let producer_opening = producer
                .map(|(producer, listed)| {
                    procedure_definition::checkpoint_opening(&hashes, producer)
                        .map(|path| (listed, path))
                })
                .transpose()?;
            let mut asked = crate::blockchain::evm::call(
                "voidClaim",
                &abi_encode(&checkpoint_fault_token(
                    &claim,
                    position,
                    checkpoint,
                    &path,
                    Some(&InputQuestion {
                        input,
                        input_manifest: &manifest,
                        producer: producer_opening
                            .as_ref()
                            .map(|(listed, path)| (*listed, path.as_slice())),
                        posted: &posted,
                    }),
                )?),
            );
            asked["checkpoint"] = serde_json::json!(position);
            asked["stage"] = serde_json::json!(checkpoint.stage);
            asked["input"] = serde_json::json!(input);
            asked["expected"] = match &verdict {
                Ok(LinkCheck::Linked) => {
                    audit.inputs_linked += 1;
                    serde_json::json!({ "revert": "NoFault" })
                }
                Ok(LinkCheck::Void(fault)) => {
                    audit.faults.push(AuditFault {
                        call: audit.calls.len(),
                        checkpoint: position,
                        stage: checkpoint.stage.clone(),
                        input: Some(input.to_string()),
                        fault: fault.name().to_string(),
                    });
                    serde_json::json!({ "void": fault.name() })
                }
                Err(refusal) => serde_json::json!({ "revert": refusal.name() }),
            };
            audit.calls.push(asked);
        }
    }
    Ok(audit)
}
