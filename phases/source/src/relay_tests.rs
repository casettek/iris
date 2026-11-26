use alloc::string::{String, ToString};
use alloc::vec::Vec;

use crate::relay::*;
use crate::source::{sha256, text_chunks, FAULT_NONE, FAULT_RECIPE, MAX_PART};

const STATEMENT: [u8; 32] = [7; 32];

fn relayed(recipe: &RelayRecipe, output: &str, valid: bool) -> RelayRun {
    run_relay_steps(
        recipe,
        output,
        &STATEMENT,
        &sha256(output.as_bytes()),
        valid,
    )
}

#[test]
fn the_record_is_of_the_text_before_the_output_and_the_text_after() {
    let recipe = relay_recipe("Is this answer right? Answer: ", "\nReply yes or no.");
    let run = relayed(&recipe, "Paris", true);
    assert_eq!(
        run.text,
        "Is this answer right? Answer: Paris\nReply yes or no."
    );
    let record = run.record;
    assert!(record.present);
    assert_eq!(record.fault, FAULT_NONE);
    assert_eq!(record.text_total, run.text.len() as u64);
    assert_eq!(record.text_sha256, crate::hex(&sha256(run.text.as_bytes())));
    assert_eq!(record.documents, 1);
    assert_eq!(record.signer, "");
}

#[test]
fn where_the_recipe_is_cut_changes_nothing() {
    let before = "x".repeat(700);
    let whole = relay_recipe(&before, "after");
    let cut = RelayRecipe {
        before: Vec::from([before[..3].to_string(), before[3..].to_string()]).into(),
        after: Vec::from(["af".to_string(), String::new(), "ter".to_string()]).into(),
    };
    assert_eq!(text_chunks(&before).len(), 3);
    assert_eq!(
        relayed(&whole, "out", true).record,
        relayed(&cut, "out", true).record
    );
    // And a recipe with nothing before or after is one too.
    let bare = relayed(&relay_recipe("", ""), "out", true);
    assert_eq!(bare.text, "out");
    assert_eq!(bare.record.fault, FAULT_NONE);
}

#[test]
fn every_value_the_step_reads_is_in_the_record() {
    let recipe = relay_recipe("before ", " after");
    let honest = relayed(&recipe, "out", true).record;
    // Text moved from one part to the other gives the same prompt and
    // another recipe.
    let moved = relayed(&relay_recipe("before", "  after"), " out", true);
    let same_text = relayed(&relay_recipe("before  ", "after"), "out ", true);
    assert_ne!(moved.record.recipe_sha256, honest.recipe_sha256);
    assert_ne!(same_text.record.recipe_sha256, honest.recipe_sha256);
    // Another output is another text and another claim record.
    let other = relayed(&recipe, "tuo", true).record;
    assert_ne!(other.text_sha256, honest.text_sha256);
    assert_ne!(other.documents_sha256, honest.documents_sha256);
    assert_eq!(other.recipe_sha256, honest.recipe_sha256);
    // Another statement over the same output is another claim record only.
    let restated = run_relay_steps(&recipe, "out", &[8; 32], &sha256(b"out"), true).record;
    assert_eq!(restated.text_sha256, honest.text_sha256);
    assert_ne!(restated.documents_sha256, honest.documents_sha256);
}

#[test]
fn a_record_that_is_not_valid_or_not_of_the_output_is_a_fault() {
    let recipe = relay_recipe("before ", " after");
    let honest = relayed(&recipe, "out", true).record;

    let invalid = relayed(&recipe, "out", false).record;
    assert_eq!(invalid.fault, FAULT_RELAYED);
    // The record still states the text and the recipe.
    assert_eq!(invalid.text_sha256, honest.text_sha256);
    assert_eq!(invalid.recipe_sha256, honest.recipe_sha256);

    // An output the claim record does not commit to.
    let other = run_relay_steps(&recipe, "out", &STATEMENT, &sha256(b"tuo"), true).record;
    assert_eq!(other.fault, FAULT_RELAYED);
    let short = run_relay_steps(&recipe, "out", &STATEMENT, &[], true).record;
    assert_eq!(short.fault, FAULT_RELAYED);
    assert_eq!(
        relay_fault_description(FAULT_RELAYED),
        "the inference whose output is relayed stated no valid claim record"
    );
    assert_eq!(
        relay_fault_description(FAULT_RECIPE),
        crate::fault_description(FAULT_RECIPE)
    );
}

#[test]
fn a_part_over_its_limit_is_a_fault_at_that_part() {
    let long = "y".repeat(MAX_PART + 1);
    let in_before = RelayRecipe {
        before: Vec::from([long.clone()]).into(),
        after: Vec::new().into(),
    };
    let record = relayed(&in_before, "out", true).record;
    assert_eq!((record.fault, record.fault_position), (FAULT_RECIPE, 0));
    let in_after = RelayRecipe {
        before: Vec::new().into(),
        after: Vec::from([long]).into(),
    };
    let record = relayed(&in_after, "out", true).record;
    assert_eq!((record.fault, record.fault_position), (FAULT_RECIPE, 1));
    // The first fault stands: a record not valid, then a long part after it.
    let record = relayed(&in_after, "out", false).record;
    assert_eq!(record.fault, FAULT_RELAYED);
}
