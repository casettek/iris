//! The claim record: the three values a contract reads of a check's edge.
//!
//! The verdict has to be the whole of what rejects a claim, so every field
//! that can reject one is turned on here by itself.

use model_identity::pipeline_core::Statement;
use sequence_check::claim_record;
use sequence_check::input::{
    ClaimRecord, OutputRecord, CHECK_FAULT_UNSCORED, OUTPUT_FAULT_SHORT_WITHOUT_EOS,
};
use sha2::{Digest, Sha256};

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// The record of a run nothing went wrong in.
fn clean() -> OutputRecord {
    OutputRecord {
        model_fault: 0,
        generated_token_count: 2,
        generated_token_ids_sha256: hex(&[3; 32]),
        generated_text: String::from("ZK"),
        stop_reason: String::from("eos"),
        output_text_sha256: hex(&Sha256::digest(b"ZK")),
        output_fault: 0,
        input_text_sha256: hex(&[1; 32]),
        model_id: hex(&[2; 32]),
        raw_prompt: true,
        max_new_tokens: 8,
        range_size: 512,
        prompt_tokens: 5,
        prompt_complete: true,
        prompt_fault: 0,
        prompt_fault_position: 0,
    }
}

#[test]
fn a_clean_record_states_its_statement_its_text_hash_and_a_valid_verdict() {
    let claim = claim_record(&clean(), 0, 0);

    let statement = Statement {
        input_text: [1; 32],
        model: [2; 32],
        raw_prompt: true,
        max_new_tokens: 8,
        range_size: 512,
    };
    assert_eq!(claim.statement.as_slice(), statement.digest().as_slice());
    assert_eq!(
        claim.output_text_sha256.as_slice(),
        Sha256::digest(b"ZK").as_slice()
    );
    assert!(claim.valid);
}

#[test]
fn every_statement_field_moves_the_digest() {
    let digest = |change: fn(&mut OutputRecord)| {
        let mut record = clean();
        change(&mut record);
        claim_record(&record, 0, 0).statement.as_slice().to_vec()
    };
    let base = digest(|_| {});

    assert_ne!(digest(|r| r.input_text_sha256 = hex(&[9; 32])), base);
    assert_ne!(digest(|r| r.model_id = hex(&[9; 32])), base);
    assert_ne!(digest(|r| r.raw_prompt = false), base);
    assert_ne!(digest(|r| r.max_new_tokens = 9), base);
    assert_ne!(digest(|r| r.range_size = 256), base);
}

#[test]
fn each_way_a_run_can_be_wrong_clears_the_verdict() {
    let valid = |change: fn(&mut OutputRecord)| {
        let mut record = clean();
        change(&mut record);
        claim_record(&record, 0, 0).valid
    };

    assert!(valid(|_| {}));
    assert!(!valid(|r| r.model_fault = 1));
    assert!(!valid(|r| r.prompt_fault = 2));
    assert!(!valid(|r| r.prompt_complete = false));
    assert!(!valid(|r| r.output_fault = OUTPUT_FAULT_SHORT_WITHOUT_EOS));
    // The two the record does not hold: they are the edge's.
    assert!(!claim_record(&clean(), 1, 0).valid);
    assert!(!claim_record(&clean(), 0, CHECK_FAULT_UNSCORED).valid);
}

#[test]
fn an_id_that_is_not_a_hash_leaves_a_zero_statement_and_no_verdict() {
    for change in [
        (|r| r.model_id = String::from("gemma")) as fn(&mut OutputRecord),
        |r| r.input_text_sha256 = String::new(),
        // Uppercase is not the canonical spelling the chain binds.
        |r| r.model_id = "AB".repeat(32),
    ] {
        let mut record = clean();
        change(&mut record);
        let claim = claim_record(&record, 0, 0);
        assert_eq!(claim.statement.as_slice(), [0; 32].as_slice());
        assert!(!claim.valid);
    }
}

#[test]
fn an_edge_with_no_check_behind_it_holds_no_claim() {
    let empty = ClaimRecord::default();
    assert_eq!(empty.statement.as_slice(), [0; 32].as_slice());
    assert_eq!(empty.output_text_sha256.as_slice(), [0; 32].as_slice());
    assert!(!empty.valid);
}
