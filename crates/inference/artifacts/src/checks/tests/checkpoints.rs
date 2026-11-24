//! What a contract reads from a claim's checkpoints with no replay.

use super::*;

#[test]
fn a_challenged_stage_is_disputed_or_void_by_its_own_program_id() {
    let claim = Claim::honest();
    assert_eq!(
        claim
            .opened()
            .admit_stage_challenge(&pins().programs, 2, &claim.trace.checkpoints[2]),
        Ok(StageAdmission::Disputed {
            program_commitment: hex::encode(CHECK)
        })
    );
    assert_eq!(first_unapproved_stage(&pins().programs, &claim.trace), None);

    // A claimer that swapped a program in the middle of its chain, or
    // named none, and committed to that.
    for program in [hex::encode([3; 32]), String::new()] {
        let mut checkpoints = claim.trace.checkpoints.clone();
        checkpoints[1].program_commitment = program.clone();
        let swapped = Claim::of(checkpoints, claim.manifests.clone());
        assert_eq!(
            swapped.opened().admit_stage_challenge(
                &pins().programs,
                1,
                &swapped.trace.checkpoints[1]
            ),
            Ok(StageAdmission::UnapprovedProgram {
                program_commitment: program
            })
        );
        assert_eq!(
            first_unapproved_stage(&pins().programs, &swapped.trace),
            Some(1)
        );
        // The stages around it are still ordinary disputes.
        assert!(matches!(
            swapped.opened().admit_stage_challenge(
                &pins().programs,
                2,
                &swapped.trace.checkpoints[2]
            ),
            Ok(StageAdmission::Disputed { .. })
        ));
        // Its inputs are not defined, so a link challenge is refused: the
        // program challenge is the one that voids the claim.
        assert_eq!(
            swapped.link(1, "activations"),
            Err(ChallengeRefusal::UnapprovedStage { stage: 1 })
        );
    }
}

#[test]
fn a_challenge_must_present_the_list_and_the_checkpoint_the_claim_committed() {
    let claim = Claim::honest();
    let mut other = claim.list.clone();
    other.extend_from_slice(claim.list.split(|byte| *byte == b'\n').next().unwrap());
    other.push(b'\n');
    assert!(matches!(
        CheckpointList::open(&claim.submission, &other),
        Err(ChallengeRefusal::AnotherCheckpointList { .. })
    ));

    let opened = claim.opened();
    assert_eq!(opened.len(), 4);
    assert_eq!(
        opened.admit_stage_challenge(&pins().programs, 4, &claim.trace.checkpoints[2]),
        Err(ChallengeRefusal::UnknownStage {
            challenged: 4,
            stages: 4
        })
    );
    // A challenger cannot present a checkpoint of its own for a stage: a
    // program id the claimer did not commit to proves nothing.
    let mut forged = claim.trace.checkpoints[1].clone();
    forged.program_commitment = hex::encode([3; 32]);
    assert!(matches!(
        opened.admit_stage_challenge(&pins().programs, 1, &forged),
        Err(ChallengeRefusal::NotTheCheckpoint { stage: 1, .. })
    ));

    // Something that is not a hash list, and a list of nothing.
    assert_eq!(
        CheckpointList::open(&claim.submission, b"not hashes\n"),
        Err(ChallengeRefusal::MalformedCheckpointList)
    );
    assert_eq!(
        CheckpointList::open(&claim.submission, b""),
        Err(ChallengeRefusal::EmptyCheckpointList)
    );
    // The same root over another count is another list.
    let mut miscounted = claim.submission.clone();
    miscounted.checkpoint_count += 1;
    assert!(matches!(
        CheckpointList::open(&miscounted, &claim.list),
        Err(ChallengeRefusal::AnotherCheckpointList { .. })
    ));

    // What a contract does with the root alone: open one checkpoint
    // with its path.
    let hashes = crate::checkpoint::checkpoint_hashes(&claim.trace).unwrap();
    let root = &claim.submission.checkpoints_root;
    let count = claim.submission.checkpoint_count;
    for (index, checkpoint) in claim.trace.checkpoints.iter().enumerate() {
        let path = crate::checkpoint::checkpoint_opening(&hashes, index).unwrap();
        let opens = |index: u32, checkpoint: &Checkpoint, count: u32| {
            crate::checkpoint::verify_checkpoint_opening(root, count, index, checkpoint, &path)
        };
        assert!(opens(index as u32, checkpoint, count));
        assert!(!opens(index as u32, &forged, count));
        assert!(!opens(index as u32 ^ 1, checkpoint, count));
        assert!(!opens(index as u32, checkpoint, count + 1));
    }
    assert!(is_address(NO_ADDRESS));
    assert!(!is_address("0xABCDEF0000000000000000000000000000000000"));
    assert!(!is_address("0000000000000000000000000000000000000000"));
}

#[test]
fn every_chained_input_of_an_honest_claim_is_linked() {
    let claim = Claim::honest();
    assert_eq!(claim.link(1, "activations"), Ok(LinkCheck::Linked));
    assert_eq!(claim.link(2, "activations"), Ok(LinkCheck::Linked));
    // The first check has no check before it: the pinned empty edge.
    assert_eq!(claim.link(2, "prior"), Ok(LinkCheck::Linked));
    // Model data is held to the commitments the procedure pins.
    assert_eq!(claim.link(1, "layer"), Ok(LinkCheck::Linked));
}

/// The honest claim with the last check's inputs and producers replaced.
fn with_check(inputs: &[(&str, &str)], sources: &[(&str, u32)]) -> Claim {
    let honest = Claim::honest();
    let mut manifests = honest.manifests;
    manifests[2] = manifest(inputs);
    let mut checkpoints = honest.trace.checkpoints;
    checkpoints[2] = checkpoint("check_c0", CHECK, &manifests[2], sources, b"last");
    Claim::of(checkpoints, manifests)
}

#[test]
fn a_made_up_input_voids_the_claim() {
    let made_up = output_commitment(b"activations nobody computed");
    // Every stage still does what its own checkpoint says. Only the link
    // shows that the check ran on something the layer never produced.
    let claim = with_check(
        &[("activations", &made_up), ("prior", &empty_edge())],
        &[("activations", 1)],
    );
    assert_eq!(
        claim.link(2, "activations"),
        Ok(LinkCheck::Void(LinkFault::NotTheProducersOutput {
            producer: 1,
            loaded: made_up.clone(),
            produced: output_commitment(b"layered"),
        }))
    );

    // Passed off as an input from outside the claim: no producer named.
    let claim = with_check(&[("activations", &made_up), ("prior", &empty_edge())], &[]);
    assert_eq!(
        claim.link(2, "activations"),
        Ok(LinkCheck::Void(LinkFault::NoProducer { loaded: made_up }))
    );

    // An empty value where the program has none to accept.
    let claim = with_check(
        &[("activations", &empty_edge()), ("prior", &empty_edge())],
        &[],
    );
    assert!(matches!(
        claim.link(2, "activations"),
        Ok(LinkCheck::Void(LinkFault::NoProducer { .. }))
    ));

    // Left out of the input list altogether.
    let claim = with_check(&[("prior", &empty_edge())], &[("activations", 1)]);
    assert_eq!(
        claim.link(2, "activations"),
        Ok(LinkCheck::Void(LinkFault::InputMissing))
    );
}

#[test]
fn a_real_output_of_the_wrong_kind_or_from_later_voids_the_claim() {
    // The bind stage's real output, wired where a layer's belongs.
    let claim = with_check(
        &[
            ("activations", &output_commitment(b"bound")),
            ("prior", &empty_edge()),
        ],
        &[("activations", 0)],
    );
    assert_eq!(
        claim.link(2, "activations"),
        Ok(LinkCheck::Void(LinkFault::ProducerOfWrongKind {
            producer: 0,
            program_commitment: hex::encode(BIND),
        }))
    );

    // A stage cannot consume its own output, or a later one's.
    for producer in [2, 7] {
        let claim = with_check(
            &[
                ("activations", &output_commitment(b"last")),
                ("prior", &empty_edge()),
            ],
            &[("activations", producer)],
        );
        assert_eq!(
            claim.link(2, "activations"),
            Ok(LinkCheck::Void(LinkFault::ProducerNotEarlier { producer }))
        );
    }
}

#[test]
fn an_input_list_that_reads_two_ways_voids_the_claim() {
    let real = output_commitment(b"layered");
    let made_up = output_commitment(b"activations nobody computed");
    let entry = |commitment: &str| {
        format!(
                "{{\n    \"commitment\": \"{commitment}\",\n    \"encoding\": \"raster\",\n    \"type\": \"sha256\"\n  }}"
            )
    };
    // The same name twice: one reader takes the first, another the last.
    let twice = format!(
        "{{\n  \"activations\": {},\n  \"activations\": {},\n  \"prior\": {}\n}}",
        entry(&real),
        entry(&made_up),
        entry(&empty_edge())
    );
    // Honest content, another spelling.
    let respaced = String::from_utf8(manifest(&[
        ("activations", &real),
        ("prior", &empty_edge()),
    ]))
    .unwrap()
    .replace("\n  \"prior\"", "\n\n  \"prior\"");
    // Canonical, but with text a reader would have to unescape or decode:
    // a contract reads the list byte by byte and takes only plain text.
    let escaped = manifest(&[("activations", &real), ("pri\"r", &empty_edge())]);
    let beyond_ascii = manifest(&[("activations", &real), ("pri\u{f6}r", &empty_edge())]);
    for bytes in [
        twice.into_bytes(),
        respaced.into_bytes(),
        escaped,
        beyond_ascii,
    ] {
        let honest = Claim::honest();
        let mut manifests = honest.manifests;
        manifests[2] = bytes;
        let mut checkpoints = honest.trace.checkpoints;
        checkpoints[2] = checkpoint(
            "check_c0",
            CHECK,
            &manifests[2],
            &[("activations", 1)],
            b"last",
        );
        let claim = Claim::of(checkpoints, manifests);
        assert_eq!(
            claim.link(2, "activations"),
            Ok(LinkCheck::Void(LinkFault::UnreadableInputList))
        );
    }
}

#[test]
fn an_input_bound_to_fixed_data_must_name_a_pinned_commitment() {
    // The layer's weights are the model's: a commitment the
    // procedure pins.
    assert_eq!(Claim::honest().link(1, "layer"), Ok(LinkCheck::Linked));

    // The cheat: run the stage on other weights and state a clean output.
    // The stage's own check would catch it in an honest run, but a
    // challenger has no such weights to run the stage on, and a replay on
    // the real ones is not a run on this checkpoint's inputs. So the
    // checkpoint may name nothing here but a pinned commitment.
    let honest = Claim::honest();
    let mut manifests = honest.manifests;
    manifests[1] = manifest(&[
        ("activations", &output_commitment(b"bound")),
        ("layer", &hex::encode([0x66; 32])),
    ]);
    let mut checkpoints = honest.trace.checkpoints;
    checkpoints[1] = checkpoint(
        "range_p0_l0",
        LAYER,
        &manifests[1],
        &[("activations", 0)],
        b"layered",
    );
    let lying = Claim::of(checkpoints, manifests);
    assert_eq!(
        lying.link(1, "layer"),
        Ok(LinkCheck::Void(LinkFault::NotAPinnedValue {
            loaded: hex::encode([0x66; 32])
        }))
    );
    // The input beside it is still its producer's output.
    assert_eq!(lying.link(1, "activations"), Ok(LinkCheck::Linked));

    // A name the program does not list is no input of it.
    assert!(matches!(
        Claim::honest().link(0, "nothing"),
        Err(ChallengeRefusal::NotAnInput { .. })
    ));
}

#[test]
fn an_input_the_run_gives_must_be_one_the_claim_posted() {
    // The text is a value the claim posted.
    assert_eq!(Claim::honest().link(0, "text"), Ok(LinkCheck::Linked));

    // The cheat: post the real prompt, run a stage on other text, and
    // state a clean output. A challenger has only the hash of that text,
    // so it cannot run the stage on it, and its run on the posted text is
    // not a run on this checkpoint's inputs. So the checkpoint may name
    // nothing here but a posted value.
    let honest = Claim::honest();
    let mut manifests = honest.manifests;
    manifests[0] = manifest(&[("text", &hex::encode([0x77; 32]))]);
    let mut checkpoints = honest.trace.checkpoints;
    checkpoints[0] = checkpoint("prompt_bind_p0", BIND, &manifests[0], &[], b"bound");
    let lying = Claim::of(checkpoints, manifests);
    assert_eq!(
        lying.link(0, "text"),
        Ok(LinkCheck::Void(LinkFault::NotAPostedValue {
            loaded: hex::encode([0x77; 32])
        }))
    );

    // A claim that posted nothing holds no run input at all.
    let mut unposted = Claim::honest();
    unposted.submission.availability.clear();
    assert_eq!(
        unposted.link(0, "text"),
        Ok(LinkCheck::Void(LinkFault::NotAPostedValue {
            loaded: hex::encode([9; 32])
        }))
    );
}

#[test]
fn a_link_challenge_must_present_what_the_claim_committed() {
    let claim = Claim::honest();
    let opened = claim.opened();
    let check = &claim.trace.checkpoints[2];
    // Another stage's input list.
    assert!(matches!(
        opened.check_link(
            &pins().programs,
            2,
            check,
            &claim.manifests[1],
            "activations",
            Some(&claim.trace.checkpoints[1])
        ),
        Err(ChallengeRefusal::NotTheInputManifest { stage: 2, .. })
    ));
    // The named producer withheld, or another checkpoint in its place.
    assert_eq!(
        opened.check_link(
            &pins().programs,
            2,
            check,
            &claim.manifests[2],
            "activations",
            None
        ),
        Err(ChallengeRefusal::ProducerNotPresented {
            stage: 2,
            producer: 1
        })
    );
    assert!(matches!(
        opened.check_link(
            &pins().programs,
            2,
            check,
            &claim.manifests[2],
            "activations",
            Some(&claim.trace.checkpoints[0])
        ),
        Err(ChallengeRefusal::NotTheCheckpoint { stage: 1, .. })
    ));
}

#[test]
fn only_a_differing_output_is_settled_by_a_replay() {
    let recomputed = Claim::honest().trace.checkpoints[2].clone();
    let claimed = |change: fn(&mut Checkpoint)| {
        let mut claimed = Claim::honest().trace.checkpoints[2].clone();
        change(&mut claimed);
        settlement(&claimed, &recomputed)
    };
    assert_eq!(claimed(|_| {}), Settlement::Nothing);
    assert_eq!(
        claimed(|c| c.output_sha256 = hex::encode([3; 32])),
        Settlement::Replay
    );
    assert_eq!(
        claimed(|c| c.output_commitment = hex::encode([3; 32])),
        Settlement::Replay
    );
    assert_eq!(
        claimed(|c| c.input_commitment = hex::encode([3; 32])),
        Settlement::Inputs
    );
    assert_eq!(
        claimed(|c| {
            c.input_sources.insert(String::from("activations"), 0);
        }),
        Settlement::Inputs
    );
    // A claim that names another program is settled by the program check
    // even when its output differs too.
    assert_eq!(
        claimed(|c| {
            c.program_commitment = hex::encode([3; 32]);
            c.output_sha256 = hex::encode([3; 32]);
        }),
        Settlement::ProgramCheck
    );
    assert_eq!(
        claimed(|c| c.stage = String::from("check_c1")),
        Settlement::StageOrder
    );
}
