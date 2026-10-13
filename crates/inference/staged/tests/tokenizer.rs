use prompt_prepare::input::{BpePiece, BpePieces};
use serde_json::{json, Value};
use staged_infer::kernels::prompt_prepare::{
    best_merge, finalize_prompt, run_merge_batch, run_prompt_prepare_direct,
    PromptPrepareDirectInputs,
};
use std::path::Path;

fn root() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..")
}
fn pieces(text: &str, tokenizer: &Value) -> BpePieces {
    BpePieces {
        pieces: run_prep::split_prompt(text, tokenizer)
            .unwrap()
            .into_iter()
            .map(|p| BpePiece {
                text: p.text,
                segment: p.segment,
            })
            .collect::<Vec<_>>()
            .into(),
    }
}

fn check_reference(tokenizer_path: &Path, corpus_path: &Path) {
    let bytes = std::fs::read(tokenizer_path).unwrap();
    let tokenizer: Value = serde_json::from_slice(&bytes).unwrap();
    let corpus: Value = serde_json::from_slice(&std::fs::read(corpus_path).unwrap()).unwrap();
    use sha2::{Digest, Sha256};
    assert_eq!(
        format!("{:x}", Sha256::digest(&bytes)),
        corpus["tokenizer_sha256"]
    );
    assert_eq!(corpus["reference_version"], "0.22.2");
    let table = run_prep::prompt_tokenizer(&tokenizer).unwrap();
    for case in corpus["cases"].as_array().unwrap() {
        let initial = pieces(case["text"].as_str().unwrap(), &tokenizer);
        let direct = run_prompt_prepare_direct(PromptPrepareDirectInputs {
            tokenizer: &table,
            initial_pieces: &initial,
        })
        .unwrap();
        assert_eq!(
            serde_json::to_value(direct.token_ids.as_slice()).unwrap(),
            case["token_ids"],
            "{}",
            case["name"]
        );
        let count = run_prep::tokenizer_repeat_count(initial.pieces.len()).unwrap();
        let mut merged = initial;
        for _ in 0..=count {
            merged = run_merge_batch(PromptPrepareDirectInputs {
                tokenizer: &table,
                initial_pieces: &merged,
            })
            .unwrap();
        }
        let final_ids = finalize_prompt(PromptPrepareDirectInputs {
            tokenizer: &table,
            initial_pieces: &merged,
        })
        .unwrap();
        assert_eq!(direct, final_ids, "{}", case["name"]);
    }
}

#[test]
fn synthetic_reference_ids() {
    check_reference(
        &root().join("model-bundles/parity-gemma/tokenizer.json"),
        &root().join("tests/tokenizer/synthetic-reference.json"),
    );
}

#[test]
fn adversarial_reference_ids() {
    check_reference(
        &root().join("tests/tokenizer/ranked-tokenizer.json"),
        &root().join("tests/tokenizer/ranked-reference.json"),
    );
}

#[test]
fn unknown_handling_and_literal_space_segmentation_match_reference() {
    // Expectations independently checked with tokenizers 0.22.2.
    for (unknown, fuse, expected) in [
        (true, false, vec![0, 3, 3, 1]),
        (true, true, vec![0, 3, 1]),
        (false, false, vec![2]),
    ] {
        let tokenizer = json!({"model":{"type":"BPE","vocab":{"a":0,"b":1,"ab":2,"<unk>":3},"merges":[["a","b"]],"unk_token":if unknown {Some("<unk>")} else {None}, "fuse_unk":fuse}});
        let table = run_prep::prompt_tokenizer(&tokenizer).unwrap();
        let input = pieces("a🙂中b", &tokenizer);
        assert_eq!(
            run_prompt_prepare_direct(PromptPrepareDirectInputs {
                tokenizer: &table,
                initial_pieces: &input
            })
            .unwrap()
            .token_ids
            .as_slice(),
            expected
        );
    }
    let tokenizer = json!({"model":{"type":"BPE","vocab":{"a":0,"b":1," ":2,"a ":3," b":4},"merges":[[" ","b"],["a"," "]]}, "pre_tokenizer":{"type":"Split","pattern":{"String":" "},"behavior":"MergedWithPrevious","invert":false}});
    let table = run_prep::prompt_tokenizer(&tokenizer).unwrap();
    let input = pieces("a  b", &tokenizer);
    assert_eq!(
        run_prompt_prepare_direct(PromptPrepareDirectInputs {
            tokenizer: &table,
            initial_pieces: &input
        })
        .unwrap()
        .token_ids
        .as_slice(),
        &[3, 2, 1]
    );
}

#[test]
#[ignore = "requires BPE_REFERENCE_TOKENIZER pointing to the pinned local Gemma tokenizer"]
fn production_reference_ids() {
    let path = std::env::var("BPE_REFERENCE_TOKENIZER").expect("set BPE_REFERENCE_TOKENIZER");
    check_reference(
        Path::new(&path),
        &root().join("tests/tokenizer/gemma-reference.json"),
    );
}

#[test]
fn batch_boundaries_and_budget_rejection() {
    for merges_needed in [0_usize, 1, 7, 8, 9, 15, 16, 17] {
        let text = (0..=merges_needed)
            .map(|i| char::from(b'a' + i as u8))
            .collect::<String>();
        let mut vocab = serde_json::Map::new();
        for (i, c) in text.chars().enumerate() {
            vocab.insert(c.to_string(), json!(i));
        }
        let mut merges = vec![];
        for i in 2..=text.len() {
            vocab.insert(text[..i].to_string(), json!(100 + i));
            merges.push(json!([&text[..i - 1], &text[i - 1..i]]));
        }
        let tokenizer = json!({"model":{"type":"BPE","vocab":vocab,"merges":merges}});
        let table = run_prep::prompt_tokenizer(&tokenizer).unwrap();
        let mut merged = pieces(&text, &tokenizer);
        let before = best_merge(merged.pieces.as_slice(), &table).unwrap();
        if merges_needed > 0 {
            assert!(before.matched);
            assert!(finalize_prompt(PromptPrepareDirectInputs {
                tokenizer: &table,
                initial_pieces: &merged
            })
            .is_err());
        }
        let batches = run_prep::tokenizer_repeat_count(merged.pieces.len()).unwrap() + 1;
        for batch in 0..batches {
            merged = run_merge_batch(PromptPrepareDirectInputs {
                tokenizer: &table,
                initial_pieces: &merged,
            })
            .unwrap();
            assert_eq!(
                merged.pieces.len(),
                (merges_needed + 1)
                    .saturating_sub((batch as usize + 1) * 8)
                    .max(1)
            );
        }
        assert_eq!(merged.pieces.len(), 1);
        let again = run_merge_batch(PromptPrepareDirectInputs {
            tokenizer: &table,
            initial_pieces: &merged,
        })
        .unwrap();
        assert_eq!(again, merged);
        assert!(finalize_prompt(PromptPrepareDirectInputs {
            tokenizer: &table,
            initial_pieces: &merged
        })
        .is_ok());
    }
}

#[test]
fn special_segments_forbid_otherwise_eligible_merges() {
    let tokenizer =
        json!({"model":{"type":"BPE","vocab":{"a":0,"b":1,"ab":2},"merges":[["a","b"]]}});
    let table = run_prep::prompt_tokenizer(&tokenizer).unwrap();
    let input = BpePieces {
        pieces: vec![
            BpePiece {
                text: "a".into(),
                segment: 0,
            },
            BpePiece {
                text: "b".into(),
                segment: 1,
            },
        ]
        .into(),
    };
    assert!(!best_merge(input.pieces.as_slice(), &table).unwrap().matched);
    assert_eq!(
        run_prompt_prepare_direct(PromptPrepareDirectInputs {
            tokenizer: &table,
            initial_pieces: &input
        })
        .unwrap()
        .token_ids
        .as_slice(),
        &[0, 1]
    );
}

#[test]
fn repeat_count_is_checked_and_empty_inputs_need_only_seed() {
    for (pieces, count) in [(0, 0), (1, 0), (9, 0), (10, 1), (17, 1), (18, 2)] {
        assert_eq!(run_prep::tokenizer_repeat_count(pieces).unwrap(), count);
    }
    if usize::BITS > 32 {
        assert!(run_prep::tokenizer_repeat_count(usize::MAX).is_err());
    }
}
