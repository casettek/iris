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
    let mut state = begin_scan_state(&plan, profile);
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
        template_suffix: String::from("<end>"),
        ..TokenizerProfile::default()
    };
    let plan = scan_plan_of(&fold, false, &profile);
    assert_eq!((plan.keep_start, plan.keep_end), (2, 5));
    assert!(plan.templated);

    let plan = scan_plan_of(&fold, true, &profile);
    assert_eq!((plan.keep_start, plan.keep_end), (0, 7));
    assert!(!plan.templated);

    // A model with no template keeps every byte of every prompt.
    let plan = scan_plan_of(&fold, false, &TokenizerProfile::default());
    assert_eq!((plan.keep_start, plan.keep_end), (0, 7));
    assert!(!plan.templated);
    // The claim an honest run makes is that plan and the text's hash.
    let claim = text_claim_of(&fold, false, &profile);
    assert_eq!((claim.total, claim.keep_start, claim.keep_end), (7, 2, 5));
    assert_eq!(claim.input_text_sha256, text_sha256_hex(&fold));
    assert_eq!(
        claimed_scan_plan(&claim, false, &profile),
        scan_plan_of(&fold, false, &profile)
    );
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

/// A range starting at piece `start` and marked to end at `end`, given
/// `total` pieces: the ones from `start` on.
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

    // A range that takes everything it is given has nothing to its right,
    // and its pieces are counted from where it starts.
    let (_, state) = sliced(2, 2, 4);
    assert_eq!((state.start, state.position), (2, 4));
    assert_eq!(state.left.text, "1");
    assert!(!state.has_right);

    // One given more than it takes leaves the rest to its right.
    let (_, state) = sliced(5, 2, 4);
    assert_eq!(state.position, 7);
    assert_eq!(
        (state.left.text.as_str(), state.right.text.as_str()),
        ("1", "2")
    );

    // One marked past what it is given has neither.
    let (_, state) = sliced(2, 2, 9);
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

// ---------------------------------------------------------------------------
// Binding the text one range at a time
// ---------------------------------------------------------------------------

/// A profile with a template, specials, space normalization and byte fallback.
fn chat_profile() -> TokenizerProfile {
    TokenizerProfile {
        template_prefix: String::from("<bos><|turn>user\n"),
        template_suffix: String::from("<turn|>\n<|turn>model\n"),
        normalize_spaces: true,
        byte_fallback: true,
        specials: String::from("<|turn>\0<turn|>\0<bos>"),
        fallback_bytes: fallback_mask(|_| true),
        ..TokenizerProfile::default()
    }
}

/// ASCII is in the vocabulary; anything else falls back to its bytes.
fn in_vocabulary(text: &str) -> bool {
    text == "▁" || text.is_ascii()
}

/// Surrounding whitespace to trim, a special in the text, a `<` that starts
/// none, and a character that becomes two pieces.
const TEXT: &str = "  Hi é<|turn>x <b y é\n  ";

fn chunks_of(text: &str, chars: usize) -> Vec<String> {
    let all: Vec<char> = text.chars().collect();
    all.chunks(chars)
        .map(|chunk| chunk.iter().collect())
        .collect()
}

fn claim_of(text: &str, raw: bool, profile: &TokenizerProfile) -> TextClaim {
    text_claim_of(&fold_text(begin_text_fold_state(), text), raw, profile)
}

/// Binds `ranges` of chunks one after another, as the chain's bind stages do.
fn bind_in_ranges(
    claim: &TextClaim,
    raw: bool,
    profile: &TokenizerProfile,
    ranges: &[&[String]],
) -> (Vec<BpePiece>, PromptContext, BindCarry) {
    let mut opened = BindOpen {
        context: PromptContext {
            raw_prompt: raw,
            ..PromptContext::default()
        },
        carry: begin_bind_carry(claim, raw, profile),
    };
    let mut pieces = Vec::new();
    for chunks in ranges {
        opened = bind_range(
            opened,
            chunks,
            profile,
            |text| (in_vocabulary(text), true),
            &mut |piece| pieces.push(piece),
        );
    }
    (pieces, opened.context, opened.carry)
}

#[test]
fn binding_in_ranges_gives_the_pieces_of_binding_whole_wherever_it_is_cut() {
    let profile = chat_profile();
    for raw in [false, true] {
        let claim = claim_of(TEXT, raw, &profile);
        let whole_text = [String::from(TEXT)];
        let (whole, context, carry) = bind_in_ranges(&claim, raw, &profile, &[&whole_text]);
        assert_eq!(context.fault, FAULT_NONE);
        assert!(carry.done);
        // The pieces are the rendered prompt's, by the unranged helper.
        let rendered = if raw {
            String::from(TEXT)
        } else {
            format!(
                "{}{}{}",
                profile.template_prefix,
                TEXT.trim(),
                profile.template_suffix
            )
        };
        assert_eq!(whole, split_rendered(&rendered, &profile, in_vocabulary));
        assert!(whole.iter().any(|piece| piece.text.starts_with("<0x")));

        let chunks = chunks_of(TEXT, 1);
        for first in 0..=chunks.len() {
            for second in first..=chunks.len() {
                let ranges: [&[String]; 3] =
                    [&chunks[..first], &chunks[first..second], &chunks[second..]];
                let (pieces, context, carry) = bind_in_ranges(&claim, raw, &profile, &ranges);
                assert_eq!(pieces, whole, "{raw} {first} {second}");
                // Any range may be empty but the first, which must hold text.
                let expected = if first == 0 {
                    FAULT_NO_TEXT
                } else {
                    FAULT_NONE
                };
                assert_eq!(context.fault, expected, "{raw} {first} {second}");
                assert!(carry.done);
            }
        }
        // Coarser chunks, cut anywhere, are the same text.
        for size in [2, 3, 5, 64] {
            let chunks = chunks_of(TEXT, size);
            for cut in 1..=chunks.len() {
                let ranges: [&[String]; 2] = [&chunks[..cut], &chunks[cut..]];
                let (pieces, context, _) = bind_in_ranges(&claim, raw, &profile, &ranges);
                assert_eq!((pieces, context.fault), (whole.clone(), FAULT_NONE));
            }
        }
    }
}

#[test]
fn a_range_holds_back_what_could_still_start_a_special_token() {
    let profile = chat_profile();
    let claim = claim_of("ab<|turn>", true, &profile);
    let chunks = chunks_of("ab<|turn>", 1);
    let begin = BindOpen {
        context: PromptContext {
            raw_prompt: true,
            ..PromptContext::default()
        },
        carry: begin_bind_carry(&claim, true, &profile),
    };
    let lookup = |text: &str| (in_vocabulary(text), true);
    let mut first = Vec::new();
    let opened = bind_range(begin, &chunks[..4], &profile, lookup, &mut |piece| {
        first.push(piece)
    });
    // Four characters in, none is out: any of them could still begin `<|turn>`.
    assert!(first.is_empty());
    assert_eq!(opened.carry.text.scan.pending, "ab<|");
    assert!(!opened.carry.done);
    let mut second = Vec::new();
    let BindOpen { context, carry } =
        bind_range(opened, &chunks[4..], &profile, lookup, &mut |piece| {
            second.push(piece)
        });
    assert_eq!(
        second
            .iter()
            .map(|piece| piece.text.as_str())
            .collect::<Vec<_>>(),
        ["a", "b", "<|turn>"]
    );
    assert!(carry.done && context.fault == FAULT_NONE);
}

#[test]
fn a_text_that_is_not_what_was_claimed_faults_where_it_ends() {
    let profile = chat_profile();
    let chunks = chunks_of(TEXT, 4);
    let cut = chunks.len() / 2;
    let bind = |claim: &TextClaim, ranges: &[&[String]]| {
        let (_, context, carry) = bind_in_ranges(claim, false, &profile, ranges);
        (context.fault, carry.done)
    };
    let honest = claim_of(TEXT, false, &profile);
    let both: [&[String]; 2] = [&chunks[..cut], &chunks[cut..]];
    assert_eq!(bind(&honest, &both), (FAULT_NONE, true));

    let with = |edit: fn(&mut TextClaim)| {
        let mut claim = honest.clone();
        edit(&mut claim);
        claim
    };
    for (case, claim) in [
        (
            "another hash",
            with(|c| c.input_text_sha256 = "00".repeat(32)),
        ),
        ("a trim that keeps more", with(|c| c.keep_start -= 1)),
        ("a trim that keeps less", with(|c| c.keep_end -= 1)),
    ] {
        assert_eq!(bind(&claim, &both), (FAULT_TEXT_CLAIM, false), "{case}");
    }
    // A length the text never reaches: no range sees the end, and none is done.
    assert_eq!(
        bind(&with(|c| c.total += 1), &both),
        (FAULT_NONE, false),
        "a longer text"
    );
    // A length that falls inside a chunk is never reached either.
    assert_eq!(
        bind(&with(|c| c.total -= 1), &both),
        (FAULT_NONE, false),
        "a length inside a chunk"
    );
    // A length that ends at a chunk: the scan ends there, and the fold has more.
    let prefix: String = chunks[..cut].concat();
    let short = with(|c| c.total = 1);
    let mut short = short;
    short.total = prefix.len() as u64;
    let whole: [&[String]; 1] = [&chunks[..]];
    assert_eq!(bind(&short, &whole), (FAULT_TEXT_CLAIM, false));
    assert_eq!(bind(&short, &both), (FAULT_TEXT_CLAIM, false));
    // A claim that is honestly about the first half makes the second half
    // text after the end.
    let half = claim_of(&prefix, false, &profile);
    assert_eq!(bind(&half, &both), (FAULT_TEXT_AFTER_END, true));
    let alone: [&[String]; 1] = [&chunks[..cut]];
    assert_eq!(bind(&half, &alone), (FAULT_NONE, true));
    // Ranges with no text may follow the end.
    let trailing: [&[String]; 3] = [&chunks[..cut], &chunks[cut..], &[]];
    assert_eq!(bind(&honest, &trailing), (FAULT_NONE, true));
    // No text at all is not an empty text.
    let none: [&[String]; 1] = [&[]];
    assert_eq!(bind(&honest, &none), (FAULT_NO_TEXT, false));
    let empty = [String::new()];
    let empty_ranges: [&[String]; 1] = [&empty];
    assert_eq!(
        bind(&claim_of("", false, &profile), &empty_ranges),
        (FAULT_NONE, true)
    );
}

#[test]
fn an_unproven_lookup_is_a_model_fault_and_changes_no_piece() {
    let profile = chat_profile();
    let claim = claim_of(TEXT, true, &profile);
    let open = || BindOpen {
        context: PromptContext {
            raw_prompt: true,
            ..PromptContext::default()
        },
        carry: begin_bind_carry(&claim, true, &profile),
    };
    let chunks = chunks_of(TEXT, 5);
    let run = |proven: fn(&str) -> bool| {
        let mut pieces = Vec::new();
        let closed = bind_range(
            open(),
            &chunks,
            &profile,
            |text| (in_vocabulary(text), proven(text)),
            &mut |piece| pieces.push(piece),
        );
        (pieces, closed.context.model.fault, closed.carry.done)
    };
    let (honest, fault, done) = run(|_| true);
    assert_eq!((fault, done), (0, true));
    let (pieces, fault, done) = run(|text| text != "x");
    assert_eq!(pieces, honest);
    assert_ne!(fault, 0);
    assert!(done);
}

#[test]
fn one_walk_of_a_bucket_finds_the_piece_and_proves_the_bucket() {
    use model_identity::sparse;
    let entries = [
        TokenEntry {
            token: String::from("a"),
            id: 7,
        },
        TokenEntry {
            token: String::from("b"),
            id: 9,
        },
    ];
    let walk = |piece: &str| {
        entries.iter().fold(begin_bucket_walk(), |walk, entry| {
            walk_bucket_entry(walk, entry, piece)
        })
    };
    assert_eq!((walk("b").found, walk("b").token_id), (true, 9));
    assert!(!walk("c").found);
    // The walk's hash is the bucket's, as the tokenizer's witness commits it.
    let bucket = VocabBucket {
        entries: raster::List::from(entries.to_vec()),
    };
    let other = VocabBucket::default();
    let witness = sparse::witness(&[bucket, other]).unwrap();
    let path = |index: usize| witness.paths.as_slice()[index].clone();
    assert!(bucket_walk_proven(&walk("b"), 0, &witness.header, &path(0)));
    assert!(bucket_walk_proven(&walk("c"), 0, &witness.header, &path(0)));
    assert!(!bucket_walk_proven(
        &walk("b"),
        1,
        &witness.header,
        &path(1)
    ));
    assert!(!bucket_walk_proven(
        &walk("b"),
        0,
        &witness.header,
        &path(1)
    ));
    assert!(bucket_walk_proven(
        &begin_bucket_walk(),
        1,
        &witness.header,
        &path(1)
    ));
}
