//! A dispute about one stage, and the fault proof that answers it.

use super::*;

#[test]
fn a_checkpoint_is_the_boundary_a_raster_run_commits() {
    let trace = Claim::honest().trace;
    let stage = |checkpoint: &Checkpoint| StageCheckpoint {
        name: checkpoint.stage.clone(),
        program_commitment: hex::decode(&checkpoint.program_commitment).unwrap(),
        input_manifest_commitment: hex::decode(&checkpoint.input_commitment).unwrap(),
        input_bindings: checkpoint
            .input_sources
            .iter()
            .map(|(input, index)| {
                (
                    input.clone(),
                    InputBindingSource::Chained {
                        stage: *index as usize,
                    },
                )
            })
            .chain([(String::from("layer"), InputBindingSource::External)])
            .collect(),
        output_payload_commitment: hex::decode(&checkpoint.output_sha256).unwrap(),
        output_structural_commitment: hex::decode(&checkpoint.output_commitment).unwrap(),
    };
    let chain = |stages: Vec<StageCheckpoint>| ChainCommitment {
        stages,
        shape: ChainShape {
            spec_digest: vec![9; 32],
            repeats: Vec::new(),
        },
    };
    let raster = chain(trace.checkpoints.iter().map(stage).collect());
    assert_eq!(raster_boundary_mismatch(&trace, &raster), None);

    let mut other_program = raster.clone();
    other_program.stages[2].program_commitment = vec![3; 32];
    assert!(raster_boundary_mismatch(&trace, &other_program)
        .unwrap()
        .contains("stage 2 (check_c0): the checkpoint's program"));

    let mut other_input = raster.clone();
    other_input.stages[0].input_manifest_commitment = vec![3; 32];
    assert!(raster_boundary_mismatch(&trace, &other_input)
        .unwrap()
        .contains("input manifest commitment"));

    let mut other_producer = raster.clone();
    other_producer.stages[2].input_bindings.insert(
        String::from("activations"),
        InputBindingSource::Chained { stage: 0 },
    );
    assert!(raster_boundary_mismatch(&trace, &other_producer)
        .unwrap()
        .contains("producers"));

    let shorter = chain(vec![stage(&trace.checkpoints[0])]);
    assert!(raster_boundary_mismatch(&trace, &shorter)
        .unwrap()
        .contains("4 checkpoints"));
}

const TRANSITION_ID: [u8; 32] = [0x7a; 32];

fn trace_header() -> TraceCommitmentHeader {
    TraceCommitmentHeader {
        bits_packer: BitPacker::new(1),
        fingerprint_len: 200,
        fingerprint_root: vec![1; 32],
        revealed_items_commitment: vec![2; 32],
        window_size: 128,
        revealed_tail_roots_commitment: vec![3; 32],
    }
}

/// The journal of a window proven against `trace_header()` over this
/// checkpoint's program and inputs, ending in `state` with `output`.
fn journal(checkpoint: &Checkpoint, state: TransitionState, output: &[u8]) -> TransitionJournal {
    let frontier = || SerializableFrontier {
        position: 0,
        leaf: vec![0; 32],
        ommers: Vec::new(),
    };
    TransitionJournal {
        init_state: InitTransition {
            init_frontier: frontier(),
            init_storage_frontier: frontier(),
            init_storage_root: vec![0; 32],
            init_storage_index_root: vec![0; 32],
            fingerprint: Fingerprint {
                bits_packer: BitPacker::new(1),
                bits: Vec::new(),
                len: 0,
            },
        },
        current_state: state,
        transition_image_id: TRANSITION_ID.to_vec(),
        authorization_image_id: vec![0xa0; 32],
        input_manifest_commitment: hex::decode(&checkpoint.input_commitment).unwrap(),
        program_commitment: hex::decode(&checkpoint.program_commitment).unwrap(),
        refuted_trace_commitment: trace_header().digest(),
        entrypoint_authorization: EntrypointAuthorization::Established,
        output_authorization: OutputAuthorization::Established {
            output_commitment: output.to_vec(),
        },
        window_is_terminal: true,
        final_committed_root: None,
    }
}

#[test]
fn a_dispute_commitment_is_the_checkpoint_and_nothing_else() {
    let checkpoint = Claim::honest().trace.checkpoints[2].clone();
    let bytes = dispute_commitment(&checkpoint).unwrap();

    // The layout a contract writes out by hand.
    let mut expected = vec![1, checkpoint.stage.len() as u8];
    expected.extend_from_slice(checkpoint.stage.as_bytes());
    for (field, last) in [
        (&checkpoint.program_commitment, false),
        (&checkpoint.input_commitment, true),
    ] {
        expected.push(32);
        expected.extend_from_slice(&hex::decode(field).unwrap());
        if last {
            expected.push(0);
        }
    }
    for field in [&checkpoint.output_sha256, &checkpoint.output_commitment] {
        expected.push(32);
        expected.extend_from_slice(&hex::decode(field).unwrap());
    }
    expected.extend_from_slice(&[0, 0]);
    assert_eq!(bytes, expected);
    assert_eq!(
        dispute_digest(&checkpoint).unwrap().as_slice(),
        Sha256::digest(&expected).as_slice()
    );

    // It is a Raster chain commitment of one stage, so Raster's own
    // dispute code reads it.
    let chain: ChainCommitment = postcard::from_bytes(&bytes).unwrap();
    assert_eq!(chain.stages.len(), 1);
    assert_eq!(
        hex::encode(&chain.stages[0].program_commitment),
        checkpoint.program_commitment
    );

    // Each of the three things a dispute reads moves the digest; where the
    // inputs came from does not, since the input manifest hash covers them.
    let digest = |change: fn(&mut Checkpoint)| {
        let mut changed = Claim::honest().trace.checkpoints[2].clone();
        change(&mut changed);
        dispute_digest(&changed)
    };
    let base = digest(|_| {});
    assert_ne!(
        digest(|c| c.program_commitment = hex::encode([3; 32])),
        base
    );
    assert_ne!(digest(|c| c.input_commitment = hex::encode([3; 32])), base);
    assert_ne!(digest(|c| c.output_sha256 = hex::encode([3; 32])), base);
    assert_eq!(digest(|c| c.input_sources.clear()), base);
    // A checkpoint that names no program cannot be disputed by a replay:
    // the program check voids it instead.
    assert_eq!(digest(|c| c.program_commitment.clear()), None);
}

#[test]
fn a_proof_dismisses_a_challenge_only_when_it_is_about_the_checkpoint() {
    let checkpoint = Claim::honest().trace.checkpoints[1].clone();
    let stated = hex::encode([0x13; 32]);
    let challenge = stage_challenge(&checkpoint, trace_header(), &stated).unwrap();
    let defence = |journal: &TransitionJournal| {
        check_defence(&checkpoint, &challenge, journal, &TRANSITION_ID)
    };

    // A window that diverges from the challenger's trace, proven over the
    // checkpoint's program and inputs: the challenge is refuted, and the
    // journal names this checkpoint.
    let refuted = defence(&journal(&checkpoint, TransitionState::Finished, &[])).unwrap();
    assert_eq!(refuted.reason, DefenceReason::Refuted);
    assert_eq!(
        refuted.chain_commitment_digest,
        dispute_digest(&checkpoint).unwrap()
    );
    assert_eq!(refuted.challenged_stage, 0);

    // The same verdict over another program, or over other inputs, says
    // nothing about this stage: any trace diverges from a run it is not of.
    let mut another_program = journal(&checkpoint, TransitionState::Finished, &[]);
    another_program.program_commitment = vec![9; 32];
    assert_eq!(
        defence(&another_program),
        Err(DefenceRejection::Refused(DefenceRefusal::AnotherProgram))
    );
    let mut other_inputs = journal(&checkpoint, TransitionState::Finished, &[]);
    other_inputs.input_manifest_commitment = vec![9; 32];
    assert_eq!(
        defence(&other_inputs),
        Err(DefenceRejection::Refused(
            DefenceRefusal::AnotherInputManifest
        ))
    );
    // Nor does a window proven against another trace commitment.
    let mut another_trace = journal(&checkpoint, TransitionState::Finished, &[]);
    another_trace.refuted_trace_commitment = vec![9; 32];
    assert_eq!(
        defence(&another_trace),
        Err(DefenceRejection::Refused(
            DefenceRefusal::AnotherTraceCommitment
        ))
    );
    // Nor one verified against a guest other than the pinned one.
    assert_eq!(
        check_defence(
            &checkpoint,
            &challenge,
            &journal(&checkpoint, TransitionState::Finished, &[]),
            &[9; 32]
        ),
        Err(DefenceRejection::Refused(
            DefenceRefusal::TransitionImageIdMismatch
        ))
    );

    // An honest trace that ends in another output than the challenger
    // stated: a mislabel, which also dismisses the challenge.
    let committed = hex::decode(&checkpoint.output_sha256).unwrap();
    let mislabelled = defence(&journal(
        &checkpoint,
        TransitionState::TerminalMatch,
        &committed,
    ))
    .unwrap();
    assert_eq!(mislabelled.reason, DefenceReason::Mislabelled);
    assert_eq!(mislabelled.attested_output, Some(committed));
    // An honest trace that ends where the challenger said: the challenge
    // is true, and no defence exists.
    assert_eq!(
        defence(&journal(
            &checkpoint,
            TransitionState::TerminalMatch,
            &hex::decode(&stated).unwrap()
        )),
        Err(DefenceRejection::Refused(
            DefenceRefusal::OutputIsTheClaimed
        ))
    );

    // A challenge built for another checkpoint is not this one's to
    // answer, whatever the proof says.
    let other = Claim::honest().trace.checkpoints[2].clone();
    assert_eq!(
        check_defence(
            &other,
            &challenge,
            &journal(&other, TransitionState::Finished, &[]),
            &TRANSITION_ID
        ),
        Err(DefenceRejection::AnotherCheckpoint)
    );
}

#[test]
fn a_contract_takes_a_fault_proof_only_of_the_challenge_it_holds() {
    const AUTHORIZATION_ID: [u8; 32] = [0xa0; 32];
    let checkpoint = Claim::honest().trace.checkpoints[1].clone();
    let challenge = stage_challenge(&checkpoint, trace_header(), &hex::encode([0x13; 32])).unwrap();
    let check = |journal: &TransitionJournal| {
        check_fault_proof(&challenge, journal, &TRANSITION_ID, &AUTHORIZATION_ID)
    };
    let proven = || journal(&checkpoint, TransitionState::Finished, &[]);
    assert_eq!(check(&proven()), Ok(()));

    // The cheat this rule is for: a real proof that some other trace is
    // false, sent against this challenge.
    let mut another_trace = proven();
    another_trace.refuted_trace_commitment = vec![9; 32];
    assert_eq!(
        check(&another_trace),
        Err(FaultProofRejection::AnotherTraceCommitment)
    );
    // And a real proof that this trace is not a run of some other
    // program, or on other inputs, which is true of every trace.
    let mut another_program = proven();
    another_program.program_commitment = vec![9; 32];
    assert_eq!(
        check(&another_program),
        Err(FaultProofRejection::AnotherProgram)
    );
    let mut other_inputs = proven();
    other_inputs.input_manifest_commitment = vec![9; 32];
    assert_eq!(
        check(&other_inputs),
        Err(FaultProofRejection::AnotherInputManifest)
    );
    // A proof whose earlier steps or whose inputs were checked by a
    // program the contract does not pin.
    assert_eq!(
        check_fault_proof(&challenge, &proven(), &[9; 32], &AUTHORIZATION_ID),
        Err(FaultProofRejection::TransitionImageIdMismatch)
    );
    assert_eq!(
        check_fault_proof(&challenge, &proven(), &TRANSITION_ID, &[9; 32]),
        Err(FaultProofRejection::AuthorizationImageIdMismatch)
    );
    // A window that ended by matching the trace is not a fault proof,
    // whatever output it ended in.
    assert_eq!(
        check(&journal(
            &checkpoint,
            TransitionState::TerminalMatch,
            &[7; 32]
        )),
        Err(FaultProofRejection::NotAFaultVerdict)
    );
    // The verdict is asked first, as the contract asks it: a journal
    // short of one has nothing else read from it.
    let mut unfinished = journal(&checkpoint, TransitionState::TerminalMatch, &[]);
    unfinished.refuted_trace_commitment = vec![9; 32];
    assert_eq!(
        check(&unfinished),
        Err(FaultProofRejection::NotAFaultVerdict)
    );

    // Raster's own rule agrees wherever both speak.
    assert!(check_defence(&checkpoint, &challenge, &proven(), &TRANSITION_ID).is_ok());
    assert!(check_defence(&checkpoint, &challenge, &another_trace, &TRANSITION_ID).is_err());
    assert!(check_defence(&checkpoint, &challenge, &another_program, &TRANSITION_ID).is_err());
    assert!(check_defence(&checkpoint, &challenge, &other_inputs, &TRANSITION_ID).is_err());
}
