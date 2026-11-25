//! The claims a lying claimer would submit, for tests of the paths that
//! catch them. Nothing here is anything an honest party runs.
//!
//! [`corrupt_claim`] takes an honest claim and makes one checkpoint of it
//! false, in one of the ways a claim can be: another output, an unapproved
//! program, an input from nowhere, a held value in the wrong place, or a
//! verdict the check did not reach. The claim it writes is ready to post.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{ensure, Context, Result};
use blockchain_rules::availability::{check_holds, RequiredPosting};
use blockchain_rules::checks::{is_address, ClaimSubmission, CLAIM_SUBMISSION_JSON};
use procedure_definition::{
    checkpoint_hashes, checkpoints_root, read_checkpoint_hashes, read_checkpoint_list, read_json,
    write_checkpoint_hashes, write_json, CHECKPOINTS_TXT, CHECKPOINT_LIST_JSON,
};
use sha2::{Digest, Sha256};

use blockchain_host::claim::{corrupt_hash, select_checkpoint_index, CorruptCheckpointSelection};
use blockchain_host::posted::{fresh_dir, parent, stage_dir_name, INPUT_MANIFEST_JSON};
use blockchain_host::posting::PreparedPosting;

/// The file `corrupt_claim` writes beside the lying claim: which checkpoint
/// it changed, in the form `test corrupt-checkpoints --checkpoints` writes.
pub const CORRUPTION_JSON: &str = "checkpoints.corruption.json";

/// What a lying claim says falsely of the checkpoint it lies about.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum CorruptFault {
    /// Another output than the stage's: a lie about the stage's execution,
    /// which a replay of the stage shows.
    #[default]
    Output,
    /// A program outside the approved set.
    Program,
    /// An input that is not from where the procedure says it comes: not
    /// what its producer made, not a pinned commitment, or not a value the
    /// claim posted, by the kind of input. The named input, or the first the
    /// checkpoint names a producer for.
    Input(Option<String>),
    /// An input the claim is allowed to name, in a place the chain does not
    /// give it: the value an earlier stage of the same program is given for
    /// that input, such as another layer's weights, another range's marks or
    /// another stage's output. Every such value is one a challenger holds,
    /// so no comparison voids the claim. The stage's own checks say what it
    /// was given, and the checkpoint states the output of a run that was
    /// given the right value: a replay of the stage on the inputs the
    /// checkpoint names shows it. The named input, or the first that an
    /// earlier stage of the program is given another value for.
    Arrangement(Option<String>),
    /// The verdict: the check's record states that every claimed token is
    /// the greedy selection and that the record is valid, where the check
    /// run on the claim's own inputs does not, and the last stage is run on
    /// that record. It is the lie of a claimer whose claimed output is not
    /// the model's: every other stage is run as it is, so the claim arrives
    /// with a valid record, and a replay of the check shows it.
    Verdict,
}

#[derive(Debug, Clone)]
pub struct CorruptClaimOptions {
    /// The honest claim's `claim_submission.json`, in its directory.
    pub submission_path: PathBuf,
    pub selection: CorruptCheckpointSelection,
    pub fault: CorruptFault,
    /// The address the lying claim is attributed to. With none, the honest
    /// claim's.
    pub claimer: Option<String>,
    pub out_dir: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CorruptedClaim {
    pub submission_path: PathBuf,
    pub hashes_path: PathBuf,
    pub manifest_path: PathBuf,
    pub checkpoint_index: usize,
    pub stage: String,
    pub original_hash: String,
    pub corrupted_hash: String,
}

/// The claim a lying claimer submits in place of an honest one: the same run
/// and the same results, with one checkpoint false. By default it states
/// another output. It can instead name a program outside the approved set,
/// or commit to an input list in which one input is not its producer's
/// output; neither is a lie a replay shows, and a contract takes the claim
/// all the same unless the checkpoint is one its records are read from.
///
/// Checkpoint hashes are posted with the checkpoints they name, so a changed
/// hash alone is no claim: the lie is a checkpoint that states something
/// false, and the hashes and their root follow from it. The directory
/// written is ready for `claim post`: the submission, the hashes, the
/// checkpoints, and the honest claim's blobs with the lying hashes and
/// checkpoints in place of the honest ones.
pub fn corrupt_claim(options: CorruptClaimOptions) -> Result<CorruptedClaim> {
    let honest: ClaimSubmission = read_json(&options.submission_path)?;
    let dir = parent(&options.submission_path)?;
    let honest_hashes_path = dir.join(CHECKPOINTS_TXT);
    let honest_hashes = read_checkpoint_hashes(&honest_hashes_path)?;
    let mut list = read_checkpoint_list(&dir.join(CHECKPOINT_LIST_JSON))?;
    ensure!(
        honest_hashes.len() == list.checkpoints.len(),
        "{} lists {} checkpoints, and {CHECKPOINT_LIST_JSON} holds {}",
        honest_hashes_path.display(),
        honest_hashes.len(),
        list.checkpoints.len()
    );
    let index = select_checkpoint_index(&options.selection, &honest_hashes, Some(&list))?;
    let stage = list.checkpoints[index].stage.clone();
    // Input lists the lying claim posts beside the honest ones, each with
    // its stage, and the input the lie is about.
    let mut lying_manifests: Vec<(String, Vec<u8>)> = Vec::new();
    let mut lied_input = None;
    // Outputs the lying claim's records are read from, in place of the
    // honest ones at those checkpoints.
    let mut forged_outputs: Vec<(usize, staged_infer::artifact_io::EncodedArtifact)> = Vec::new();
    let read_list = |checkpoint: &procedure_definition::Checkpoint,
                     position: usize|
     -> Result<BTreeMap<String, BTreeMap<String, String>>> {
        let path = dir
            .join(stage_dir_name(&checkpoint.stage)?)
            .join(INPUT_MANIFEST_JSON);
        let list = fs::read(&path).with_context(|| format!("failed to read {}", path.display()))?;
        ensure!(
            hex::encode(Sha256::digest(&list)) == checkpoint.input_commitment,
            "{} is not the input list checkpoint {position} commits to",
            path.display()
        );
        serde_json::from_slice(&list).with_context(|| format!("failed to parse {}", path.display()))
    };
    match &options.fault {
        CorruptFault::Output => {
            list.checkpoints[index].output_sha256 =
                corrupt_hash(&list.checkpoints[index].output_sha256);
        }
        CorruptFault::Program => {
            list.checkpoints[index].program_commitment =
                corrupt_hash(&list.checkpoints[index].program_commitment);
        }
        CorruptFault::Input(named) => {
            let checkpoint = &mut list.checkpoints[index];
            let input = match named {
                Some(input) => input.clone(),
                None => checkpoint
                    .input_sources
                    .keys()
                    .next()
                    .cloned()
                    .with_context(|| {
                        format!(
                        "checkpoint {index} (`{stage}`) names no producer for any input: pick a \
                         later stage, or name the input"
                    )
                    })?,
            };
            let manifest_path = dir.join(stage_dir_name(&stage)?).join(INPUT_MANIFEST_JSON);
            let honest_manifest = fs::read(&manifest_path)
                .with_context(|| format!("failed to read {}", manifest_path.display()))?;
            ensure!(
                hex::encode(Sha256::digest(&honest_manifest)) == checkpoint.input_commitment,
                "{} is not the input list checkpoint {index} commits to",
                manifest_path.display()
            );
            let mut entries: BTreeMap<String, BTreeMap<String, String>> =
                serde_json::from_slice(&honest_manifest)
                    .with_context(|| format!("failed to parse {}", manifest_path.display()))?;
            let loaded = entries
                .get_mut(&input)
                .and_then(|entry| entry.get_mut("commitment"))
                .with_context(|| format!("stage `{stage}` has no input `{input}`"))?;
            *loaded = corrupt_hash(loaded);
            let lying = serde_json::to_vec_pretty(&entries)?;
            checkpoint.input_commitment = hex::encode(Sha256::digest(&lying));
            lying_manifests.push((stage.clone(), lying));
            lied_input = Some(input);
        }
        CorruptFault::Arrangement(named) => {
            let mut entries = read_list(&list.checkpoints[index], index)?;
            let program = &list.checkpoints[index].program_commitment;
            // The nearest earlier stage of the same program that is given
            // another value for an input.
            let mut found = None;
            'donors: for donor in (0..index).rev() {
                if &list.checkpoints[donor].program_commitment != program {
                    continue;
                }
                let given = read_list(&list.checkpoints[donor], donor)?;
                for (input, entry) in &entries {
                    if named.as_ref().is_some_and(|named| named != input) {
                        continue;
                    }
                    let other = given.get(input).and_then(|entry| entry.get("commitment"));
                    if other.is_some() && other != entry.get("commitment") {
                        found = Some((donor, input.clone(), other.cloned().expect("checked")));
                        break 'donors;
                    }
                }
            }
            let (donor, input, value) = found.with_context(|| {
                format!(
                    "no earlier stage of the program of checkpoint {index} (`{stage}`) is given \
                     another value for {}: pick a later stage, or another input",
                    named
                        .as_ref()
                        .map_or(String::from("any input"), |input| format!("`{input}`"))
                )
            })?;
            *entries
                .get_mut(&input)
                .and_then(|entry| entry.get_mut("commitment"))
                .expect("found above") = value;
            // The pointer goes with the value: the stage the donor took it
            // from, or none for a value no stage made.
            let pointer = list.checkpoints[donor].input_sources.get(&input).copied();
            let checkpoint = &mut list.checkpoints[index];
            match pointer {
                Some(producer) => checkpoint.input_sources.insert(input.clone(), producer),
                None => checkpoint.input_sources.remove(&input),
            };
            let lying = serde_json::to_vec_pretty(&entries)?;
            checkpoint.input_commitment = hex::encode(Sha256::digest(&lying));
            lying_manifests.push((stage.clone(), lying));
            lied_input = Some(input);
        }
        CorruptFault::Verdict => {
            let last = list.checkpoints.len() - 1;
            let reads: Vec<String> = list.checkpoints[last]
                .input_sources
                .iter()
                .filter(|(_, producer)| **producer as usize == index)
                .map(|(input, _)| input.clone())
                .collect();
            ensure!(
                index != last && !reads.is_empty(),
                "checkpoint {index} (`{stage}`) is not a check the claim's last stage reads: \
                 pick the last check"
            );
            ensure!(
                list.checkpoints[..last].iter().all(|checkpoint| checkpoint
                    .input_sources
                    .values()
                    .all(|producer| *producer as usize != index)),
                "a stage before the last reads the output of checkpoint {index} (`{stage}`), \
                 and only the last stage is run again here"
            );
            let result_stage = list.checkpoints[last].stage.clone();
            let forged = forge_valid_verdict(
                &dir.join(stage_dir_name(&stage)?),
                &dir.join(stage_dir_name(&result_stage)?),
            )?;
            let mut entries = read_list(&list.checkpoints[last], last)?;
            for input in &reads {
                *entries
                    .get_mut(input)
                    .and_then(|entry| entry.get_mut("commitment"))
                    .with_context(|| format!("stage `{result_stage}` has no input `{input}`"))? =
                    forged.check.structural_commitment.clone();
            }
            let lying = serde_json::to_vec_pretty(&entries)?;
            let stated =
                |checkpoint: &mut procedure_definition::Checkpoint,
                 output: &staged_infer::artifact_io::EncodedArtifact| {
                    checkpoint.output_commitment = output.structural_commitment.clone();
                    checkpoint.output_sha256 = hex::encode(Sha256::digest(&output.data));
                };
            stated(&mut list.checkpoints[index], &forged.check);
            stated(&mut list.checkpoints[last], &forged.result);
            list.checkpoints[last].input_commitment = hex::encode(Sha256::digest(&lying));
            lying_manifests.push((result_stage, lying));
            forged_outputs.push((index, forged.check));
            forged_outputs.push((last, forged.result));
        }
    }
    let hashes = checkpoint_hashes(&list)?;

    let claimer = options.claimer.unwrap_or_else(|| honest.claimer.clone());
    ensure!(
        is_address(&claimer),
        "`{claimer}` is not an address: give `0x` and 40 lowercase hex digits"
    );
    let mut claim = honest;
    claim.claimer = claimer.clone();
    claim.checkpoint_count = hashes.len() as u32;
    claim.checkpoints_root = checkpoints_root(&hashes)?;
    claim.availability.clear();

    fresh_dir(&options.out_dir)?;
    let hashes_path = options.out_dir.join(CHECKPOINTS_TXT);
    write_checkpoint_hashes(&hashes_path, &hashes)?;
    let list_path = options.out_dir.join(CHECKPOINT_LIST_JSON);
    write_json(&list_path, &list)?;
    for (stage, manifest) in &lying_manifests {
        let stage_dir = options.out_dir.join(stage_dir_name(stage)?);
        fs::create_dir_all(&stage_dir)
            .with_context(|| format!("failed to create {}", stage_dir.display()))?;
        fs::write(stage_dir.join(INPUT_MANIFEST_JSON), manifest)?;
    }
    let mut posting = PreparedPosting::read(dir)?
        .with_checkpoints(&claimer, &fs::read(&hashes_path)?, &fs::read(&list_path)?)?
        .with_input_manifests(
            lying_manifests
                .into_iter()
                .map(|(_, manifest)| manifest)
                .collect(),
        )?;
    // A forged output is given to the proof in place of the honest one, and
    // the submission states what it holds.
    for (position, output) in &forged_outputs {
        let given = posting
            .outputs
            .iter_mut()
            .find(|given| given.checkpoint as usize == *position)
            .with_context(|| {
                format!("the claim's records are not read from checkpoint {position}")
            })?;
        given.payload = hex::encode(&output.data);
        given.index = hex::encode(&output.index);
    }
    if let Some((_, result)) = forged_outputs.last() {
        let stated = blockchain_host::posting::stated_results(&result.data, &result.index)
            .context("the forged result is not a result record")?;
        ensure!(
            stated.inferences.len() == claim.inferences.len(),
            "the forged result states {} inferences, and the claim has {}",
            stated.inferences.len(),
            claim.inferences.len()
        );
        for (inference, stated) in claim.inferences.iter_mut().zip(stated.inferences) {
            inference.record = stated.record;
            inference.result = stated.result;
        }
        claim.consensus = stated.consensus;
    }
    // The proof of a claim's checkpoints reads its records out of the last
    // stage's output and the checks', each held to its checkpoint's hash. A
    // lie about one of those has no output behind it, so no proof states it
    // and no contract takes the claim.
    let contents = posting.contents().with_context(|| {
        format!(
            "checkpoint {index} (`{stage}`) is one the claim's records are read from: a claim \
             that states another output for it is refused when it arrives, so there is nothing \
             to challenge"
        )
    })?;
    check_holds(&contents, &RequiredPosting::of(&claim)).map_err(|rejection| {
        anyhow::anyhow!("the lying claim's data is not its own: {rejection}")
    })?;
    posting.write(&options.out_dir)?;
    let submission_path = options.out_dir.join(CLAIM_SUBMISSION_JSON);
    write_json(&submission_path, &claim)?;

    let manifest_path = options.out_dir.join(CORRUPTION_JSON);
    write_json(
        &manifest_path,
        &serde_json::json!({
            "version": 1,
            "source_hashes_path": honest_hashes_path,
            "corrupted_hashes_path": hashes_path,
            "checkpoint_index": index,
            "checkpoint_number": index + 1,
            "stage": stage,
            "fault": match &options.fault {
                CorruptFault::Output => "output",
                CorruptFault::Program => "program",
                CorruptFault::Input(_) => "input",
                CorruptFault::Arrangement(_) => "arrangement",
                CorruptFault::Verdict => "verdict",
            },
            "input": lied_input,
            "original_hash": honest_hashes[index],
            "corrupted_hash": hashes[index],
        }),
    )?;
    Ok(CorruptedClaim {
        submission_path,
        hashes_path,
        manifest_path,
        checkpoint_index: index,
        stage,
        original_hash: honest_hashes[index].clone(),
        corrupted_hash: hashes[index].clone(),
    })
}

/// The outputs of a claimer who states a valid verdict its check did not
/// reach: the check's own output with nothing claimed differing and the
/// record valid, and the last stage's output on that.
struct ForgedVerdict {
    check: staged_infer::artifact_io::EncodedArtifact,
    result: staged_infer::artifact_io::EncodedArtifact,
}

/// Run the check in `check_dir` on its own inputs and state its verdict as
/// valid, then run the result stage in `result_dir` on that record. Both
/// directories are the claimer's own, with the files each stage was given.
fn forge_valid_verdict(check_dir: &Path, result_dir: &Path) -> Result<ForgedVerdict> {
    use staged_infer::artifact_io::encode_output;
    use staged_infer::routines::{claim_result, sequence_check, StageKind};

    ensure!(
        StageKind::from_stage_dir(result_dir)? == StageKind::ClaimResult,
        "{} is not the result stage of a procedure kind of one inference: a forged verdict is built \
         for that procedure kind only",
        result_dir.display()
    );
    let files = |dir: &Path| (dir.join("input.json"), dir.join(INPUT_MANIFEST_JSON));
    let cached = staged_infer::cache::CachedInputs::new();
    let (input, list) = files(check_dir);
    let mut edge = sequence_check::run_direct(&sequence_check::load_inputs_from_paths(
        &input, &list, &cached,
    )?)?;
    ensure!(
        !edge.claim.valid,
        "the check in {} states a valid record as it is, so there is no verdict to forge: \
         corrupt a claim whose claimed output is not the model's",
        check_dir.display()
    );
    edge.claim_mismatches = 0;
    edge.first_claim_mismatch_position = 0;
    edge.claim.valid = true;
    let check = encode_output(&edge)?;

    let (input, list) = files(result_dir);
    let mut inputs = claim_result::load_inputs_from_paths_with_cache(&input, &list, &cached, None)?;
    // The two stages declare the edge field for field: one encoding.
    inputs.check = postcard::from_bytes(&postcard::to_allocvec(&edge)?)
        .context("the check's output is not the result stage's input")?;
    let result = encode_output(&claim_result::run_direct(&inputs)?)?;
    Ok(ForgedVerdict { check, result })
}
