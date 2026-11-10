use prompt_check::input::{BpePiece, BpePieces};
use serde_json::{json, Value};
use staged_infer::kernels::prompt_prepare::{
    best_merge, finalize_prompt, run_prompt_prepare_direct, PromptPrepareDirectInputs,
};
use std::path::Path;

fn root() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..")
}
fn pieces(text: &str, tokenizer: &Value) -> BpePieces {
    BpePieces::of_pieces(
        run_prep::split_prompt(text, tokenizer)
            .unwrap()
            .into_iter()
            .map(|p| BpePiece {
                text: p.text,
                segment: p.segment,
            })
            .collect(),
    )
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
        assert_eq!(direct.context.fault, 0, "{}", case["name"]);
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
fn pieces_that_could_still_merge_are_recorded_and_merging_them_clears_it() {
    let tokenizer =
        json!({"model":{"type":"BPE","vocab":{"a":0,"b":1,"ab":2},"merges":[["a","b"]]}});
    let table = run_prep::prompt_tokenizer(&tokenizer).unwrap();
    let unmerged = pieces("ab", &tokenizer);
    assert!(
        best_merge(unmerged.pieces.as_slice(), &table)
            .unwrap()
            .matched
    );
    let early = finalize_prompt(PromptPrepareDirectInputs {
        tokenizer: &table,
        initial_pieces: &unmerged,
    })
    .unwrap();
    assert_eq!(
        early.context.fault,
        prompt_check::input::FAULT_MERGES_INCOMPLETE
    );
    let merged = run_prompt_prepare_direct(PromptPrepareDirectInputs {
        tokenizer: &table,
        initial_pieces: &unmerged,
    })
    .unwrap();
    assert_eq!(merged.context.fault, 0);
    assert_eq!(merged.token_ids.as_slice(), &[2]);
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
        ..Default::default()
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
