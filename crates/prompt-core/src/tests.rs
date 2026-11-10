use crate::*;
use alloc::format;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use sha2::{Digest, Sha256};

fn profile(specials: &str) -> TokenizerProfile {
    TokenizerProfile {
        specials: String::from(specials),
        ..TokenizerProfile::default()
    }
}

fn hash_in_chunks(text: &str, size: usize) -> String {
    let mut fold = begin_text_fold_state();
    for chunk in text.as_bytes().chunks(size) {
        fold = fold_text(fold, core::str::from_utf8(chunk).unwrap());
    }
    text_sha256_hex(&fold)
}

#[test]
fn the_running_hash_is_plain_sha256_wherever_the_text_is_cut() {
    for length in [0, 1, 55, 56, 63, 64, 65, 119, 120, 1000] {
        let text = "a".repeat(length);
        let expected = format!("{:x}", Sha256::digest(text.as_bytes()));
        for size in [1, 7, 64, 200, 4096] {
            assert_eq!(hash_in_chunks(&text, size), expected, "{length}/{size}");
        }
    }
}

fn atoms_of(text: &str, profile: &TokenizerProfile, size: usize) -> Vec<PromptAtom> {
    let plan = ScanPlan {
        total: text.len() as u64,
        keep_start: 0,
        keep_end: text.len() as u64,
        templated: false,
    };
    let mut atoms = Vec::new();
    let mut state = begin_scan_state(&plan);
    for chunk in text.as_bytes().chunks(size) {
        state = scan_chunk_into(
            state,
            core::str::from_utf8(chunk).unwrap(),
            &plan,
            profile,
            &mut |atom| atoms.push(atom),
        );
    }
    atoms
}

#[test]
fn a_special_token_stays_whole_across_chunk_cuts() {
    let profile = profile("<bos>\0<b>");
    let whole = atoms_of("x<bos>y<b>z", &profile, 1000);
    assert_eq!(
        whole.iter().map(|atom| atom.special).collect::<Vec<_>>(),
        vec![false, true, false, true, false]
    );
    for size in 1..8 {
        assert_eq!(atoms_of("x<bos>y<b>z", &profile, size), whole, "{size}");
    }
}

#[test]
fn the_template_trims_only_what_surrounds_the_text() {
    let mut fold = begin_text_fold_state();
    fold = fold_text(fold, "  a ");
    fold = fold_text(fold, "b\n ");
    let profile = TokenizerProfile {
        templated: true,
        ..TokenizerProfile::default()
    };
    let config = RunConfig::default();
    let plan = scan_plan_of(&fold, &config, &profile);
    assert_eq!((plan.keep_start, plan.keep_end), (2, 5));
    assert!(plan.templated);

    let raw = RunConfig {
        raw_prompt: true,
        ..RunConfig::default()
    };
    let plan = scan_plan_of(&fold, &raw, &profile);
    assert_eq!((plan.keep_start, plan.keep_end), (0, 7));
    assert!(!plan.templated);
}

#[test]
fn a_fallback_mask_answers_for_each_byte() {
    let mask = fallback_mask(|byte| byte == 0xC3 || byte == 0xA9 || byte == 0);
    let profile = TokenizerProfile {
        fallback_bytes: mask,
        ..TokenizerProfile::default()
    };
    assert_eq!(profile.fallback_bytes.len(), 64);
    for byte in 0..=255_u8 {
        assert_eq!(
            fallback_has(&profile, byte),
            byte == 0xC3 || byte == 0xA9 || byte == 0,
            "{byte}"
        );
    }
}

fn sliced(total: u64, start: u64, end: u64) -> (PromptContext, SliceState) {
    let context = PromptContext {
        next_piece: start,
        ..PromptContext::default()
    };
    let mut state = begin_slice_state(&context, end);
    for index in 0..total {
        observe_piece(
            &mut state,
            &BpePiece {
                text: format!("{index}"),
                segment: 0,
            },
        );
    }
    (context, state)
}

#[test]
fn a_slice_counts_its_pieces_and_keeps_the_two_either_side_of_its_end() {
    let (context, state) = sliced(4, 0, 2);
    assert_eq!(
        (state.start, state.end, state.position),
        (context.next_piece, 2, 4)
    );
    assert!(state.has_left && state.has_right);
    assert_eq!(state.left.text, "1");
    assert_eq!(state.right.text, "2");

    // A slice that ends with the prompt has nothing to its right.
    let (_, state) = sliced(4, 2, 4);
    assert_eq!(state.left.text, "3");
    assert!(!state.has_right);

    // One that ends past the prompt has neither.
    let (_, state) = sliced(4, 2, 9);
    assert!(!state.has_left && !state.has_right);
}

#[test]
fn the_first_fault_is_the_one_kept() {
    let mut context = PromptContext::default();
    record_fault(&mut context, FAULT_EMPTY_RANGE, 3);
    record_fault(&mut context, FAULT_RANGE_TOO_LARGE, 9);
    assert_eq!(
        (context.fault, context.fault_position),
        (FAULT_EMPTY_RANGE, 3)
    );
    assert_eq!(fault_description(4), "an unknown fault");
}

#[test]
fn prompt_join_rejects_same_model_different_statement_fields() {
    let bound = PromptContext {
        input_text_sha256: "input-a".into(),
        model_id: "model-a".into(),
        raw_prompt: false,
        max_new_tokens: 8,
        range_size: 4,
        ..Default::default()
    };
    let mut variants = [
        bound.clone(),
        bound.clone(),
        bound.clone(),
        bound.clone(),
        bound.clone(),
    ];
    variants[0].input_text_sha256 = "input-b".into();
    variants[1].model_id = "model-b".into();
    variants[2].raw_prompt = true;
    variants[3].max_new_tokens = 9;
    variants[4].range_size = 5;
    for previous in variants {
        assert_eq!(
            join_prompt_models(previous, &bound).fault,
            FAULT_CONTEXT_MISMATCH
        );
    }
}
#[test]
fn prompt_join_retains_faults_from_both_branches() {
    let mut bound = PromptContext::default();
    bound.fault = FAULT_NO_TEXT;
    bound.fault_position = 12;
    let joined = join_prompt_models(PromptContext::default(), &bound);
    assert_eq!((joined.fault, joined.fault_position), (FAULT_NO_TEXT, 12));
    let mut previous = PromptContext::default();
    previous.fault = FAULT_EMPTY_RANGE;
    previous.fault_position = 3;
    let joined = join_prompt_models(previous, &bound);
    assert_eq!(
        (joined.fault, joined.fault_position),
        (FAULT_EMPTY_RANGE, 3)
    );
}
