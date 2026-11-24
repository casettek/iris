//! The contract rules' tests, by subject, over one set of fixtures: a
//! contract's pins and a small claim with its checkpoints.

use super::*;

use std::path::{Path, PathBuf};

use raster_core::chain::{
    ChainCommitment, ChainShape, DefenceRefusal, InputBindingSource, StageCheckpoint,
};
use raster_core::transition::{TraceCommitmentHeader, TransitionJournal};

use sha2::{Digest, Sha256};

use crate::availability::{BlobContents, PostedBlob};
use model_manifest::direct::{model_id_of, ModelCommitment, ModelManifest};
use procedure_definition::checkpoint::{checkpoints_root, Checkpoint, CheckpointList};

use procedure_definition::procedure_kind::ProcedureKind;
use raster_core::chain::DefenceReason;
use raster_core::fingerprint::{BitPacker, Fingerprint};
use raster_core::transition::{
    EntrypointAuthorization, InitTransition, OutputAuthorization, SerializableFrontier,
    TransitionState,
};

mod checkpoints;
mod dispute;
mod pins;
mod submission;

const REQUIRED: RequiredParameters = RequiredParameters {
    raw_prompt: true,
    max_new_tokens: 8,
};

const BIND: [u8; 32] = [1; 32];

const CHECK: [u8; 32] = [2; 32];

const RESULT: [u8; 32] = [0x1e; 32];

const LAYER: [u8; 32] = [4; 32];

fn empty_edge() -> String {
    hex::encode([0xee; 32])
}

/// Four programs: the first takes nothing from the chain, a layer takes
/// `activations` from a bind or another layer, the check takes
/// `activations` from a layer and `prior` from a check or the empty edge,
/// and the result takes `check` from a check.
fn pins() -> ContractPins {
    let model_commitments = vec![ModelCommitment {
        role: String::from("tokenizer"),
        commitment: hex::encode([7; 32]),
    }];
    let program = |stage: &str, id: [u8; 32], chained_inputs| StageProgram {
        stage: stage.to_string(),
        program_commitment: hex::encode(id),
        chained_inputs,
        run_inputs: vec![String::from("text")],
        pinned_inputs: vec![PinnedInput {
            name: String::from("layer"),
            values: vec![hex::encode([8; 32]), hex::encode([0x1b; 32])],
        }],
    };
    let input = |name: &str, producers: &[&str], empty_values: Vec<String>| ChainedInput {
        name: name.to_string(),
        producers: producers.iter().map(|p| p.to_string()).collect(),
        empty_values,
    };
    ContractPins {
        version: CONTRACT_PINS_VERSION,
        procedure_kind: ProcedureKind::Single,
        programs: ProgramPins {
            family: String::from("gemma-dense"),
            stage_programs: vec![
                program("prompt-bind", BIND, vec![]),
                program(
                    "prefill-range",
                    LAYER,
                    vec![input(
                        "activations",
                        &["prompt-bind", "prefill-range"],
                        vec![],
                    )],
                ),
                program(
                    "sequence-check",
                    CHECK,
                    vec![
                        input("activations", &["prefill-range"], vec![]),
                        input("prior", &["sequence-check"], vec![empty_edge()]),
                    ],
                ),
                program(
                    "claim-result",
                    RESULT,
                    vec![input("check", &["sequence-check"], vec![])],
                ),
            ],
        },
        models: vec![ModelPins {
            model_id: model_id_of(&model_commitments),
            model_commitments,
            range_size: 512,
        }],
    }
}

/// What an inference on this model submits for this output of the
/// fixture's input.
fn inference(model: &ModelPins, input: &str, output: &str) -> SubmittedInference {
    SubmittedInference {
        output_text: output.to_string(),
        record: SubmittedRecord {
            statement_sha256: hex::encode(
                expected_statement(model, REQUIRED, input, &"00".repeat(32)).unwrap(),
            ),
            output_text_sha256: hex::encode(Sha256::digest(output.as_bytes())),
            valid: true,
        },
        result: no_source(forms()).result_of(output),
    }
}

/// A stage's input list as the chain writes it.
fn manifest(inputs: &[(&str, &str)]) -> Vec<u8> {
    let entries = inputs
            .iter()
            .map(|(name, commitment)| {
                (
                    name.to_string(),
                    serde_json::json!({"type": "sha256", "encoding": "raster", "commitment": commitment}),
                )
            })
            .collect::<serde_json::Map<_, _>>();
    serde_json::to_vec_pretty(&serde_json::Value::Object(entries)).unwrap()
}

fn output_commitment(output: &[u8]) -> String {
    hex::encode(Sha256::digest([b"structural:".as_slice(), output].concat()))
}

fn checkpoint(
    stage: &str,
    program: [u8; 32],
    inputs: &[u8],
    sources: &[(&str, u32)],
    output: &[u8],
) -> Checkpoint {
    Checkpoint {
        stage: stage.to_string(),
        program_commitment: hex::encode(program),
        input_commitment: hex::encode(Sha256::digest(inputs)),
        input_sources: sources
            .iter()
            .map(|(input, index)| (input.to_string(), *index))
            .collect(),
        output_commitment: output_commitment(output),
        output_sha256: hex::encode(Sha256::digest(output)),
    }
}

fn checkpoints_txt(list: &CheckpointList) -> Vec<u8> {
    let mut text = procedure_definition::checkpoint::checkpoint_hashes(list)
        .unwrap()
        .join("\n");
    text.push('\n');
    text.into_bytes()
}

/// An honest claim of four stages, bind then layer then check then
/// result, with each stage's input list.
struct Claim {
    submission: ClaimSubmission,
    list: CheckpointList,
    hashes: Vec<u8>,
    manifests: Vec<Vec<u8>>,
}

impl Claim {
    fn of(checkpoints: Vec<Checkpoint>, manifests: Vec<Vec<u8>>) -> Self {
        let pins = pins();
        let input = hex::encode([5; 32]);
        let list = CheckpointList { checkpoints };
        let hashes = checkpoints_txt(&list);
        let submission = ClaimSubmission {
            version: CLAIM_SUBMISSION_VERSION,
            procedure_kind: ProcedureKind::Single,
            procedure: String::new(),
            claimer: NO_ADDRESS.to_string(),
            inferences: vec![inference(&pins.models[0], &input, "ZK")],
            consensus: None,
            input_text_sha256: input,
            source_sha256: "00".repeat(32),
            source: None,
            checkpoint_count: list.checkpoints.len() as u32,
            checkpoints_root: checkpoints_root(
                &procedure_definition::checkpoint::checkpoint_hashes(&list).unwrap(),
            )
            .unwrap(),
            // The run inputs the claim posted: its text and its forms.
            availability: vec![PostedBlob {
                pointer: crate::availability::BlobPointer {
                    height: 1,
                    namespace: "00".repeat(29),
                    commitment: "00".repeat(32),
                },
                length: 1,
                contents: crate::availability::BlobContents {
                    submitter: NO_ADDRESS.to_string(),
                    run_inputs: vec![hex::encode([9; 32]), hex::encode([6; 32])],
                    ..Default::default()
                },
                proof: crate::availability::AvailabilityProof::host_checked(),
            }],
        };
        Self {
            submission,
            list,
            hashes,
            manifests,
        }
    }

    fn honest() -> Self {
        let manifests = vec![
            manifest(&[("text", &hex::encode([9; 32]))]),
            manifest(&[
                ("activations", &output_commitment(b"bound")),
                ("layer", &hex::encode([8; 32])),
            ]),
            manifest(&[
                ("activations", &output_commitment(b"layered")),
                ("prior", &empty_edge()),
            ]),
            manifest(&[
                ("check", &output_commitment(b"last")),
                ("forms", &hex::encode([6; 32])),
            ]),
        ];
        Self::of(
            vec![
                checkpoint("prompt_bind_p0", BIND, &manifests[0], &[], b"bound"),
                checkpoint(
                    "range_p0_l0",
                    LAYER,
                    &manifests[1],
                    &[("activations", 0)],
                    b"layered",
                ),
                checkpoint(
                    "check_c0",
                    CHECK,
                    &manifests[2],
                    &[("activations", 1)],
                    b"last",
                ),
                checkpoint("result", RESULT, &manifests[3], &[("check", 2)], b"result"),
            ],
            manifests,
        )
    }

    fn opened(&self) -> CheckpointHashes {
        CheckpointHashes::open(&self.submission, &self.hashes).unwrap()
    }

    /// The link check as a challenger pointing at `stage` and `input`
    /// would have a contract make it.
    fn link(&self, stage: u32, input: &str) -> std::result::Result<LinkCheck, ChallengeRefusal> {
        let checkpoint = &self.list.checkpoints[stage as usize];
        let producer = checkpoint
            .input_sources
            .get(input)
            .and_then(|index| self.list.checkpoints.get(*index as usize));
        self.opened().check_link(
            &pins().programs,
            stage,
            checkpoint,
            &self.manifests[stage as usize],
            input,
            producer,
        )
    }
}

/// What a proof states of a claim: what its last stage states, taken to
/// be the submission's, and each record in the checks that stage names,
/// taken in the order of the procedure kind's inferences.
fn stated_records(claim: &Claim) -> crate::availability::StatedRecords {
    let last = claim.list.checkpoints.last().unwrap();
    let inputs = claim.submission.procedure_kind.result_inputs();
    crate::availability::StatedRecords {
        checkpoints_root: claim.submission.checkpoints_root.clone(),
        result_program: last.program_commitment.clone(),
        result_inputs: last.input_sources.clone(),
        results: claim.submission.stated(),
        checks: inputs
            .iter()
            .enumerate()
            .filter_map(|(inference, input)| {
                let position = *last.input_sources.get(*input)?;
                Some(crate::availability::StatedCheck {
                    checkpoint: position,
                    program: claim.list.checkpoints[position as usize]
                        .program_commitment
                        .clone(),
                    record: claim.submission.inferences.get(inference)?.record.clone(),
                })
            })
            .collect(),
    }
}

fn posted(records: crate::availability::StatedRecords) -> BlobContents {
    BlobContents {
        records: vec![records],
        ..BlobContents::default()
    }
}

const OTHER_CHECK_OUTPUT: &[u8] = b"second last";

const CONSENSUS: [u8; 32] = [0x2e; 32];

/// The second model of a consensus contract, and the programs of the
/// consensus procedure kind: the single one's without its result program, and
/// the consensus result, which takes `first` and `second` from a check.
fn consensus_pins() -> (ProgramPins, [ModelPins; 2]) {
    let single = pins();
    let model_commitments = vec![ModelCommitment {
        role: String::from("tokenizer"),
        commitment: hex::encode([0x17; 32]),
    }];
    let second = ModelPins {
        model_id: model_id_of(&model_commitments),
        model_commitments,
        range_size: 512,
    };
    let mut programs = single.programs;
    programs
        .stage_programs
        .retain(|program| program.stage != "claim-result");
    let check = |name: &str| ChainedInput {
        name: name.to_string(),
        producers: vec![String::from("sequence-check")],
        empty_values: Vec::new(),
    };
    programs.stage_programs.push(StageProgram {
        stage: String::from("consensus-result"),
        program_commitment: hex::encode(CONSENSUS),
        chained_inputs: vec![check("first"), check("second")],
        run_inputs: vec![String::from("forms")],
        pinned_inputs: Vec::new(),
    });
    (programs, [single.models[0].clone(), second])
}

/// An honest consensus claim: the single fixture's three stages for each
/// of two models, then the consensus result reading both checks. The
/// first inference answers `ZK` and the second ` ZK`, the same form.
fn consensus_claim(second_output: &str) -> Claim {
    let (_, models) = consensus_pins();
    let single = Claim::honest();
    let mut manifests = single.manifests[..3].to_vec();
    manifests.extend(single.manifests[..3].iter().cloned());
    manifests.push(manifest(&[
        ("first", &output_commitment(b"last")),
        ("second", &output_commitment(OTHER_CHECK_OUTPUT)),
        ("forms", &hex::encode([6; 32])),
    ]));
    let mut checkpoints = single.list.checkpoints[..3].to_vec();
    for (stage, name) in ["prompt_bind_p0", "range_p0_l0", "check_c0"]
        .iter()
        .enumerate()
    {
        checkpoints[stage].stage = format!("m0_{name}");
    }
    checkpoints.extend([
        checkpoint("m1_prompt_bind_p0", BIND, &manifests[3], &[], b"bound"),
        checkpoint(
            "m1_range_p0_l0",
            LAYER,
            &manifests[4],
            &[("activations", 3)],
            b"layered",
        ),
        checkpoint(
            "m1_check_c0",
            CHECK,
            &manifests[5],
            &[("activations", 4)],
            OTHER_CHECK_OUTPUT,
        ),
        checkpoint(
            "result",
            CONSENSUS,
            &manifests[6],
            &[("first", 2), ("second", 5)],
            b"agreed",
        ),
    ]);
    let mut claim = Claim::of(checkpoints, manifests);
    let input = claim.submission.input_text_sha256.clone();
    claim.submission.procedure_kind = ProcedureKind::Consensus;
    claim
        .submission
        .inferences
        .push(inference(&models[1], &input, second_output));
    claim.submission.consensus = Some(no_source(forms()).consensus_of("ZK", second_output));
    claim
}

/// The fixture's forms: its output, `ZK`, is the second.
fn forms() -> Vec<result_core::Form> {
    [("no", 2), ("ZK", 1)]
        .into_iter()
        .map(|(text, code)| result_core::Form {
            text: text.to_string(),
            code,
        })
        .collect()
}

fn no_source(forms: Vec<result_core::Form>) -> RequiredOrigin {
    RequiredOrigin {
        source: None,
        forms,
    }
}
