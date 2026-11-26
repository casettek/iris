//! What a contract checks of a claim when it arrives.

use super::*;

#[test]
fn an_honest_submission_opens_a_window() {
    let claim = Claim::honest();
    assert_eq!(
        check_submission(&pins().models[0], REQUIRED, &claim.submission, 0),
        Ok(())
    );
}

#[test]
fn one_equality_covers_the_model_and_every_parameter() {
    let submission = Claim::honest().submission;
    let mismatch = |pins: &ContractPins, required: RequiredParameters| {
        matches!(
            check_submission(&pins.models[0], required, &submission, 0),
            Err(SubmissionRejection::StatementMismatch { .. })
        )
    };

    // Another model: a consistent pin of other commitments.
    let mut other_model = pins();
    other_model.models[0].model_commitments[0].commitment = hex::encode([8; 32]);
    other_model.models[0].model_id = model_id_of(&other_model.models[0].model_commitments);
    assert!(mismatch(&other_model, REQUIRED));

    let mut other_range = pins();
    other_range.models[0].range_size = 256;
    assert!(mismatch(&other_range, REQUIRED));

    assert!(mismatch(
        &pins(),
        RequiredParameters {
            raw_prompt: false,
            ..REQUIRED
        }
    ));
    assert!(mismatch(
        &pins(),
        RequiredParameters {
            max_new_tokens: 9,
            ..REQUIRED
        }
    ));

    // The input is the claimer's to state, but the statement must be over it.
    let mut other_input = submission.clone();
    other_input.input_text_sha256 = hex::encode([6; 32]);
    assert!(matches!(
        check_submission(&pins().models[0], REQUIRED, &other_input, 0),
        Err(SubmissionRejection::StatementMismatch { .. })
    ));
}

#[test]
fn a_submission_is_rejected_for_its_output_or_its_verdict() {
    let submission = Claim::honest().submission;

    let mut other_text = submission.clone();
    other_text.inferences[0].output_text = String::from("ZK!");
    assert!(matches!(
        check_submission(&pins().models[0], REQUIRED, &other_text, 0),
        Err(SubmissionRejection::OutputMismatch { .. })
    ));

    let mut faulted = submission.clone();
    faulted.inferences[0].record.valid = false;
    assert_eq!(
        check_submission(&pins().models[0], REQUIRED, &faulted, 0),
        Err(SubmissionRejection::NotValid)
    );

    let mut malformed = submission;
    malformed.input_text_sha256 = String::from("prompt");
    assert_eq!(
        check_submission(&pins().models[0], REQUIRED, &malformed, 0),
        Err(SubmissionRejection::MalformedInputCommitment)
    );
}

#[test]
fn pins_whose_id_is_not_the_hash_of_their_commitments_reject_everything() {
    let submission = Claim::honest().submission;
    let mut pins = pins();
    pins.models[0].model_commitments[0].commitment = hex::encode([8; 32]);
    assert_eq!(
        check_submission(&pins.models[0], REQUIRED, &submission, 0),
        Err(SubmissionRejection::InconsistentPins)
    );
}

#[test]
fn a_submission_states_what_its_own_checkpoints_commit_to() {
    let claim = Claim::honest();
    let programs = pins().programs;
    let honest = stated_records(&claim);
    let check = |records: &crate::availability::StatedRecords| {
        check_stated_records(&programs, &claim.submission, &posted(records.clone()))
    };
    assert_eq!(check(&honest), Ok(()));

    // A proof that states nothing of this list.
    assert_eq!(
        check_stated_records(&programs, &claim.submission, &BlobContents::default()),
        Err(SubmissionRejection::RecordsNotStated)
    );
    let mut another_list = honest.clone();
    another_list.checkpoints_root = hex::encode([9; 32]);
    assert_eq!(
        check(&another_list),
        Err(SubmissionRejection::RecordsNotStated)
    );

    // The check committed a record the claimer did not submit.
    let mut other = honest.clone();
    other.checks[0].record.valid = false;
    assert_eq!(
        check(&other),
        Err(SubmissionRejection::AnotherStatedRecord { inference: 0 })
    );
    // The last stage committed another code, other forms, or a claim
    // record that is not the one submitted.
    let mut other = honest.clone();
    other.results.inferences[0].result.code += 1;
    assert_eq!(check(&other), Err(SubmissionRejection::AnotherStatedResult));
    let mut other = honest.clone();
    other.results.inferences[0].result.forms_sha256 = hex::encode([9; 32]);
    assert_eq!(check(&other), Err(SubmissionRejection::AnotherStatedResult));
    let mut other = honest.clone();
    other.results.inferences[0].record.valid = false;
    assert_eq!(check(&other), Err(SubmissionRejection::AnotherStatedResult));

    // The check is the one the last stage names as what it read: a
    // record stated for another position does not stand for it.
    let mut elsewhere = honest.clone();
    elsewhere.checks[0].checkpoint = 1;
    assert_eq!(
        check(&elsewhere),
        Err(SubmissionRejection::NoCheck { inference: 0 })
    );
    let mut unread = honest.clone();
    unread.checks.clear();
    assert_eq!(
        check(&unread),
        Err(SubmissionRejection::NoCheck { inference: 0 })
    );
}

#[test]
fn a_claim_that_does_not_end_in_its_checks_and_the_result_stage_is_refused() {
    let honest = Claim::honest();
    let programs = pins().programs;

    // A claim that stops at the check: its last stage is no result stage.
    let mut checkpoints = honest.list.checkpoints.clone();
    checkpoints.pop();
    let claim = Claim::of(checkpoints, honest.manifests[..3].to_vec());
    let mut stated = stated_records(&claim);
    stated.results = claim.submission.stated();
    assert_eq!(
        check_stated_records(&programs, &claim.submission, &posted(stated)),
        Err(SubmissionRejection::NotTheResultStage {
            expected: ProcedureKind::single().result_program(),
        })
    );

    // A result stage that read something that is no check.
    let mut checkpoints = honest.list.checkpoints.clone();
    checkpoints[3] = checkpoint(
        "result",
        RESULT,
        &honest.manifests[3],
        &[("check", 1)],
        b"result",
    );
    let claim = Claim::of(checkpoints, honest.manifests.clone());
    assert_eq!(
        check_stated_records(
            &programs,
            &claim.submission,
            &posted(stated_records(&claim))
        ),
        Err(SubmissionRejection::NotACheck { inference: 0 })
    );

    // A result stage that names no check at all.
    let mut checkpoints = honest.list.checkpoints.clone();
    checkpoints[3] = checkpoint("result", RESULT, &honest.manifests[3], &[], b"result");
    let claim = Claim::of(checkpoints, honest.manifests.clone());
    assert_eq!(
        check_stated_records(
            &programs,
            &claim.submission,
            &posted(stated_records(&claim))
        ),
        Err(SubmissionRejection::NoCheck { inference: 0 })
    );
}

#[test]
fn a_consensus_submission_is_two_inferences_of_one_input_and_their_agreement() {
    let (_, models) = consensus_pins();
    let claim = consensus_claim(" ZK\n");
    let submission = &claim.submission;
    let origin = no_source(forms());
    assert_eq!(
        check_procedure_kind(ProcedureKind::named("consensus").unwrap(), submission),
        Ok(())
    );
    for (inference, model) in models.iter().enumerate() {
        assert_eq!(
            check_submission(model, REQUIRED, submission, inference),
            Ok(())
        );
    }
    assert_eq!(check_origin(&origin, submission), Ok(()));
    assert_eq!(
        submission.consensus,
        Some(SubmittedConsensus {
            same_input: true,
            agree: true,
            code: 1
        })
    );

    // Each inference is held to its own model: the two cannot be swapped,
    // and one model cannot stand for both.
    assert!(matches!(
        check_submission(&models[0], REQUIRED, submission, 1),
        Err(SubmissionRejection::StatementMismatch { .. })
    ));
    assert!(matches!(
        check_submission(&models[1], REQUIRED, submission, 0),
        Err(SubmissionRejection::StatementMismatch { .. })
    ));
    // Both statements are over the one input the submission names.
    let mut other_input = submission.clone();
    other_input.input_text_sha256 = hex::encode([6; 32]);
    for (inference, model) in models.iter().enumerate() {
        assert!(matches!(
            check_submission(model, REQUIRED, &other_input, inference),
            Err(SubmissionRejection::StatementMismatch { .. })
        ));
    }
    // An invalid record in either inference is no claim.
    let mut faulted = submission.clone();
    faulted.inferences[1].record.valid = false;
    assert_eq!(check_submission(&models[0], REQUIRED, &faulted, 0), Ok(()));
    assert_eq!(
        check_submission(&models[1], REQUIRED, &faulted, 1),
        Err(SubmissionRejection::NotValid)
    );

    // The agreement is what the contract's forms give the two outputs.
    let mut overstated = submission.clone();
    overstated.consensus.as_mut().unwrap().code = 2;
    assert!(matches!(
        check_origin(&origin, &overstated),
        Err(SubmissionRejection::AnotherConsensus { .. })
    ));
    let mut denied = submission.clone();
    denied.consensus.as_mut().unwrap().agree = false;
    assert!(matches!(
        check_origin(&origin, &denied),
        Err(SubmissionRejection::AnotherConsensus { .. })
    ));
    let mut miscoded = submission.clone();
    miscoded.inferences[1].result.code = 2;
    assert_eq!(
        check_origin(&origin, &miscoded),
        Err(SubmissionRejection::AnotherResult {
            expected: 1,
            submitted: 2
        })
    );

    // Two models that disagree are still a claim, and it says so.
    let disagreeing = consensus_claim("no").submission;
    assert_eq!(check_origin(&origin, &disagreeing), Ok(()));
    assert_eq!(
        disagreeing.consensus,
        Some(SubmittedConsensus {
            same_input: true,
            agree: false,
            code: 0
        })
    );
    let mut claimed_agreement = disagreeing.clone();
    claimed_agreement.consensus = submission.consensus;
    assert!(matches!(
        check_origin(&origin, &claimed_agreement),
        Err(SubmissionRejection::AnotherConsensus { .. })
    ));

    // A claim holds what a claim of its procedure kind does.
    let shape = |submission: &ClaimSubmission, procedure_kind| {
        matches!(
            check_procedure_kind(procedure_kind, submission),
            Err(SubmissionRejection::AnotherShape { .. })
        )
    };
    let mut one = submission.clone();
    one.inferences.pop();
    assert!(shape(&one, ProcedureKind::named("consensus").unwrap()));
    let mut unstated = submission.clone();
    unstated.consensus = None;
    assert!(shape(&unstated, ProcedureKind::named("consensus").unwrap()));
    let mut single = Claim::honest().submission;
    assert_eq!(
        check_procedure_kind(ProcedureKind::single(), &single),
        Ok(())
    );
    single.consensus = submission.consensus;
    assert!(shape(&single, ProcedureKind::single()));
    assert_eq!(
        check_procedure_kind(ProcedureKind::single(), submission),
        Err(SubmissionRejection::AnotherProcedureKind {
            expected: ProcedureKind::single(),
            submitted: ProcedureKind::named("consensus").unwrap()
        })
    );
}

#[test]
fn a_consensus_claim_states_each_record_of_its_own_last_check() {
    let (programs, _) = consensus_pins();
    let claim = consensus_claim(" ZK\n");
    let submission = &claim.submission;
    let opened = claim.opened();
    let last = &claim.list.checkpoints[6];
    let honest = stated_records(&claim);
    let check = |records: &crate::availability::StatedRecords| {
        check_stated_records(&programs, submission, &posted(records.clone()))
    };
    assert_eq!(check(&honest), Ok(()));
    assert_eq!(
        honest
            .checks
            .iter()
            .map(|check| check.checkpoint)
            .collect::<Vec<_>>(),
        vec![2, 5]
    );

    // The other inference's record is not this check's.
    let mut swapped = honest.clone();
    let first = swapped.checks[0].record.clone();
    swapped.checks[0].record = swapped.checks[1].record.clone();
    swapped.checks[1].record = first;
    assert_eq!(
        check(&swapped),
        Err(SubmissionRejection::AnotherStatedRecord { inference: 0 })
    );
    // Nor is the other inference's check the one the result stage read
    // for this inference.
    let mut crossed = honest.clone();
    crossed.checks[0].checkpoint = 5;
    crossed.checks[1].checkpoint = 2;
    assert_eq!(
        check(&crossed),
        Err(SubmissionRejection::AnotherStatedRecord { inference: 0 })
    );

    // The result stage states both records, both results and the
    // agreement, and all of it is what was submitted.
    let mut other = honest.clone();
    other.results.consensus.as_mut().unwrap().agree = false;
    assert_eq!(check(&other), Err(SubmissionRejection::AnotherStatedResult));
    let mut other = honest.clone();
    other.results.inferences.swap(0, 1);
    assert_eq!(check(&other), Err(SubmissionRejection::AnotherStatedResult));
    let mut other = honest.clone();
    other.results.consensus = None;
    assert_eq!(check(&other), Err(SubmissionRejection::AnotherStatedResult));

    // Every stage names an approved program, and the result stage's two
    // checks are earlier checks' outputs.
    assert_eq!(first_unapproved_stage(&programs, &claim.list), None);
    for (input, producer) in [("first", 2), ("second", 5)] {
        assert_eq!(
            opened.check_link(
                &programs,
                6,
                last,
                &claim.manifests[6],
                input,
                Some(&claim.list.checkpoints[producer]),
            ),
            Ok(LinkCheck::Linked)
        );
    }
    // A consensus claim that ends in the single procedure kind's result program
    // is not one: that program is not in a consensus contract's set.
    assert!(!programs.approves(&hex::encode(RESULT)));
    // And a single contract's programs do not make this claim's last
    // stage a result stage.
    assert_eq!(
        check_stated_records(&pins().programs, submission, &posted(honest)),
        Err(SubmissionRejection::NotTheResultStage {
            expected: ProcedureKind::named("consensus").unwrap().result_program()
        })
    );
}

#[test]
fn a_submission_names_its_procedure_kind() {
    let submission = Claim::honest().submission;
    assert_eq!(
        check_procedure_kind(ProcedureKind::single(), &submission),
        Ok(())
    );
    let written = serde_json::to_value(&submission).unwrap();
    assert_eq!(written["procedure_kind"], "single");
    assert_eq!(written["version"], 8);

    // A submission that names no procedure kind, or one that does not exist, is
    // not a submission.
    let mut unnamed = written.clone();
    unnamed.as_object_mut().unwrap().remove("procedure_kind");
    assert!(serde_json::from_value::<ClaimSubmission>(unnamed).is_err());
    let mut unknown = written;
    unknown["procedure_kind"] = serde_json::json!("committee");
    assert!(serde_json::from_value::<ClaimSubmission>(unknown).is_err());

    // A contract accepts claims of one procedure kind.
    assert_eq!(
        check_procedure_kind(ProcedureKind::named("consensus").unwrap(), &submission),
        Err(SubmissionRejection::AnotherProcedureKind {
            expected: ProcedureKind::named("consensus").unwrap(),
            submitted: ProcedureKind::single(),
        })
    );
    // The label alone does not make a claim another procedure kind's: it holds
    // one inference, and its last stage runs the single procedure kind's result
    // program.
    let mut relabelled = submission.clone();
    relabelled.procedure_kind = ProcedureKind::named("consensus").unwrap();
    assert!(matches!(
        check_procedure_kind(ProcedureKind::named("consensus").unwrap(), &relabelled),
        Err(SubmissionRejection::AnotherShape { .. })
    ));
    let claim = Claim::honest();
    let mut stated = stated_records(&claim);
    stated.results = relabelled.stated();
    assert_eq!(
        check_stated_records(&pins().programs, &relabelled, &posted(stated)),
        Err(SubmissionRejection::NotTheResultStage {
            expected: ProcedureKind::named("consensus").unwrap().result_program()
        })
    );
}

#[test]
fn the_source_and_the_forms_are_held_to_what_the_contract_requires() {
    let plain = Claim::honest().submission;
    assert_eq!(plain.inferences[0].result.code, 1);
    let none = no_source(forms());
    assert_eq!(check_origin(&none, &plain), Ok(()));
    // Other forms, or the same in another order.
    let mut other = forms();
    other[0].text = String::from("nay");
    assert!(matches!(
        check_origin(&no_source(other), &plain),
        Err(SubmissionRejection::AnotherForms { .. })
    ));
    let mut reversed = forms();
    reversed.reverse();
    assert!(matches!(
        check_origin(&no_source(reversed), &plain),
        Err(SubmissionRejection::AnotherForms { .. })
    ));
    assert!(matches!(
        check_origin(&no_source(Vec::new()), &plain),
        Err(SubmissionRejection::AnotherForms { .. })
    ));
    // The right forms and a code they do not give this output. No stage
    // has to be replayed to refuse it.
    let mut miscoded = plain.clone();
    miscoded.inferences[0].result.code = 2;
    assert_eq!(
        check_origin(&none, &miscoded),
        Err(SubmissionRejection::AnotherResult {
            expected: 1,
            submitted: 2
        })
    );
    let mut unread = plain.clone();
    unread.inferences[0].result.code = 0;
    assert_eq!(
        check_origin(&none, &unread),
        Err(SubmissionRejection::AnotherResult {
            expected: 1,
            submitted: 0
        })
    );

    let record = source_core::SourceRecord {
        present: true,
        text_sha256: hex::encode([5; 32]),
        text_total: 2388,
        recipe_sha256: hex::encode([10; 32]),
        documents: 3,
        documents_sha256: hex::encode([11; 32]),
        signer: String::from("02ab"),
        fault: 0,
        fault_position: 0,
    };
    let mut sourced = plain.clone();
    sourced.source_sha256 = source_record_id(&record);
    sourced.source = Some(record.clone());
    assert_ne!(sourced.source_sha256, "00".repeat(32));
    let required = RequiredOrigin {
        source: Some(RequiredSource {
            signer: String::from("02ab"),
            recipe_sha256: Some(hex::encode([10; 32])),
        }),
        forms: forms(),
        relay_recipe_sha256: None,
    };
    assert_eq!(check_origin(&required, &sourced), Ok(()));

    // A contract that takes its prompt as given accepts no source, and
    // one that requires a source accepts no claim without its record.
    assert_eq!(
        check_origin(&none, &sourced),
        Err(SubmissionRejection::UnexpectedSource)
    );
    assert_eq!(
        check_origin(&required, &plain),
        Err(SubmissionRejection::MissingSource)
    );
    let mut digest_only = sourced.clone();
    digest_only.source = None;
    assert_eq!(
        check_origin(&required, &digest_only),
        Err(SubmissionRejection::MissingSource)
    );
    assert_eq!(
        check_origin(&none, &digest_only),
        Err(SubmissionRejection::UnexpectedSource)
    );

    // The record must be the one the statement's digest is over.
    let mut swapped = sourced.clone();
    swapped.source.as_mut().unwrap().documents_sha256 = hex::encode([12; 32]);
    assert!(matches!(
        check_origin(&required, &swapped),
        Err(SubmissionRejection::AnotherSourceRecord { .. })
    ));

    let with = |edit: &dyn Fn(&mut source_core::SourceRecord)| {
        let mut record = record.clone();
        edit(&mut record);
        let mut submission = sourced.clone();
        submission.source_sha256 = source_record_id(&record);
        submission.source = Some(record);
        check_origin(&required, &submission)
    };
    assert_eq!(
        with(&|record| record.fault = 3),
        Err(SubmissionRejection::SourceFault { fault: 3 })
    );
    assert!(matches!(
        with(&|record| record.signer = String::from("02cd")),
        Err(SubmissionRejection::AnotherSigner { .. })
    ));
    assert!(matches!(
        with(&|record| record.recipe_sha256 = hex::encode([13; 32])),
        Err(SubmissionRejection::AnotherRecipe { .. })
    ));
}

/// A relay claim of the fixture's two models: the second inference on a
/// prompt built around the first one's output, and the result of the second
/// alone. Returns the hash of its recipe beside it.
fn relay_submission() -> (ClaimSubmission, Vec<ModelPins>, String) {
    let (_, models) = consensus_pins();
    let mut submission = consensus_claim(" ZK\n").submission;
    submission.procedure_kind = ProcedureKind::named("relay").unwrap();
    submission.consensus = None;
    // The result stage does not read the first inference.
    submission.inferences[0].result = SubmittedResult::default();
    let first = submission.inferences[0].clone();
    let record = source_core::relay::run_relay_steps(
        &source_core::relay::relay_recipe("Was this right: ", "\nAnswer."),
        &first.output_text,
        &hex::decode(&first.record.statement_sha256).unwrap(),
        &hex::decode(&first.record.output_text_sha256).unwrap(),
        true,
    )
    .record;
    let output = "ZK";
    submission.inferences[1] = SubmittedInference {
        output_text: output.to_string(),
        record: SubmittedRecord {
            statement_sha256: hex::encode(
                expected_statement(
                    &models[1],
                    REQUIRED,
                    &record.text_sha256,
                    &source_record_id(&record),
                )
                .unwrap(),
            ),
            output_text_sha256: hex::encode(Sha256::digest(output.as_bytes())),
            valid: true,
        },
        result: no_source(forms()).result_of(output),
    };
    let recipe = record.recipe_sha256.clone();
    submission.relays = vec![record];
    (submission, models.to_vec(), recipe)
}

#[test]
fn a_relay_submission_holds_the_second_inference_to_the_first_ones_record() {
    let relay = ProcedureKind::named("relay").unwrap();
    let (submission, models, recipe) = relay_submission();
    let origin = RequiredOrigin {
        relay_recipe_sha256: Some(recipe),
        ..no_source(forms())
    };
    let checked = |submission: &ClaimSubmission| {
        check_procedure_kind(relay, submission)?;
        for (inference, model) in models.iter().enumerate() {
            check_submission(model, REQUIRED, submission, inference)?;
        }
        check_origin(&origin, submission)
    };
    assert_eq!(checked(&submission), Ok(()));
    // The two statements are over different inputs: the second is over the
    // relay's record.
    assert_ne!(
        submission.relays[0].text_sha256,
        submission.input_text_sha256
    );
    // The result stage states the second inference alone.
    assert_eq!(submission.stated().inferences.len(), 1);
    assert_eq!(
        submission.stated().inferences[0].record,
        submission.inferences[1].record
    );

    // Another first output, with a record that commits to it: the relay's
    // record names the output it was built on, and this is not it.
    let mut swapped = submission.clone();
    swapped.inferences[0].output_text = String::from("another answer");
    swapped.inferences[0].record.output_text_sha256 =
        hex::encode(Sha256::digest(b"another answer"));
    assert_eq!(
        checked(&swapped),
        Err(SubmissionRejection::AnotherRelayed { inference: 1 })
    );

    // Another relay record than the one the second statement names.
    let mut other_record = submission.clone();
    other_record.relays[0].text_total += 1;
    assert!(matches!(
        checked(&other_record),
        Err(SubmissionRejection::StatementMismatch { .. })
    ));
    // A relay record with a fault is no source for a claim.
    let mut faulted = submission.clone();
    faulted.relays[0].fault = source_core::relay::FAULT_RELAYED;
    assert_eq!(
        check_origin(&origin, &faulted),
        Err(SubmissionRejection::SourceFault {
            fault: source_core::relay::FAULT_RELAYED
        })
    );

    // A claim with no relay record, or one too many, is not a relay claim.
    for relays in [Vec::new(), vec![submission.relays[0].clone(); 2]] {
        let mut reshaped = submission.clone();
        reshaped.relays = relays;
        assert!(matches!(
            check_procedure_kind(relay, &reshaped),
            Err(SubmissionRejection::AnotherShape { .. })
        ));
    }

    // The recipe is the one the contract pins.
    let other_recipe = RequiredOrigin {
        relay_recipe_sha256: Some("ab".repeat(32)),
        ..no_source(forms())
    };
    assert!(matches!(
        check_origin(&other_recipe, &submission),
        Err(SubmissionRejection::AnotherRecipe { .. })
    ));

    // The first inference has no result: the result stage did not read it.
    let mut coded = submission.clone();
    coded.inferences[0].result = no_source(forms()).result_of("ZK");
    assert!(matches!(
        checked(&coded),
        Err(SubmissionRejection::AnotherForms { .. })
    ));
}
