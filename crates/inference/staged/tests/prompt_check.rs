//! The claimed-token check against the production Gemma merge table.
//! See `docs/proposals/claimed-prompt-tokens.md`.

use inference_kernels::kernels::prompt_check::rounds::{
    check_range_in_rounds, full_merge_scan_steps, marks_of,
};
use inference_kernels::kernels::prompt_check::{
    check_prompt, check_range, grouping_passes, merged_tokens, ClaimedToken,
};
use inference_kernels::kernels::prompt_prepare::best_merge;
use prompt_prepare::input::{split_rendered, BpePiece, PromptTokenizer, TokenizerProfile};
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

fn tokenizer_path() -> PathBuf {
    std::env::var("BPE_REFERENCE_TOKENIZER")
        .map(PathBuf::from)
        .unwrap_or_else(|_| root().join("../models/gemma-4-E2B-q8/tokenizer.json"))
}

/// Splits text the way `prompt-bind` does, by the same code.
fn pieces(text: &str, profile: &TokenizerProfile, vocab: &BTreeMap<String, u32>) -> Vec<BpePiece> {
    split_rendered(text, profile, |token| vocab.contains_key(token))
}

struct Rng(u64);
impl Rng {
    fn below(&mut self, bound: usize) -> usize {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 33) as usize) % bound
    }
}

/// Every way of cutting `len` pieces into consecutive groups.
fn groupings(len: usize) -> impl Iterator<Item = Vec<usize>> {
    (0..1_usize << (len - 1)).map(move |cuts| {
        let mut counts = Vec::new();
        let mut run = 1;
        for gap in 0..len - 1 {
            if cuts >> gap & 1 == 1 {
                counts.push(run);
                run = 1;
            } else {
                run += 1;
            }
        }
        counts.push(run);
        counts
    })
}

/// Every group joins to a vocabulary token inside one segment, and no two
/// neighbouring tokens have a rule: all a check of the final state would see.
fn spells_stable_tokens(
    table: &PromptTokenizer,
    vocab: &BTreeMap<String, u32>,
    pieces: &[BpePiece],
    counts: &[usize],
) -> bool {
    let mut tokens = Vec::new();
    let mut cursor = 0;
    for &len in counts {
        let group = &pieces[cursor..cursor + len];
        if group.iter().any(|piece| piece.segment != group[0].segment) {
            return false;
        }
        let text: String = group.iter().map(|piece| piece.text.as_str()).collect();
        if !vocab.contains_key(&text) {
            return false;
        }
        tokens.push(BpePiece {
            text,
            segment: group[0].segment,
        });
        cursor += len;
    }
    !best_merge(&tokens, table).unwrap().matched
}

#[test]
#[ignore = "requires the local Gemma tokenizer; run in release with --nocapture"]
fn claimed_tokens_on_the_production_vocabulary() {
    let tokenizer: Value =
        serde_json::from_slice(&std::fs::read(tokenizer_path()).expect("Gemma tokenizer.json"))
            .unwrap();
    let table = run_prep::prompt_tokenizer(&tokenizer).unwrap();
    let vocab = run_prep::vocab_map(&tokenizer["model"]).unwrap();
    let profile = run_prep::tokenizer_profile(&tokenizer).unwrap();

    // Text to build prompts from: ordinary vocabulary entries, spelled as a
    // user would type them. Joining entries end to end puts token boundaries
    // exactly where the merge table has the most alternatives.
    let words: Vec<String> = vocab
        .keys()
        .filter(|token| !token.starts_with('<') && !token.starts_with('['))
        .map(|token| token.replace('\u{2581}', " "))
        .collect();
    let mut rng = Rng(0x9e37_79b9);
    let text_of = |rng: &mut Rng, parts: usize, max_chars: usize| -> String {
        let mut text = String::new();
        for _ in 0..parts {
            let word = &words[rng.below(words.len())];
            if text.chars().count() + word.chars().count() <= max_chars {
                text.push_str(word);
            }
        }
        text
    };

    // 1. The reference corpus: the merge's tokens are the pinned ids, and pass.
    let corpus: Value = serde_json::from_slice(
        &std::fs::read(root().join("tests/tokenizer/gemma-reference.json")).unwrap(),
    )
    .unwrap();
    let mut corpus_cases = 0;
    for case in corpus["cases"].as_array().unwrap() {
        let pieces = pieces(case["text"].as_str().unwrap(), &profile, &vocab);
        let tokens = merged_tokens(&pieces, &table).unwrap();
        let ids: Vec<u32> = tokens.iter().map(|token| token.id).collect();
        assert_eq!(
            serde_json::to_value(&ids).unwrap(),
            case["token_ids"],
            "{}",
            case["name"]
        );
        assert_eq!(
            check_prompt(&table, &pieces, &tokens).unwrap(),
            None,
            "{}",
            case["name"]
        );
        corpus_cases += 1;
    }

    // 2. Agreement on longer built texts, whole and cut into two ranges at
    //    every token.
    let mut agreed = 0;
    let mut longest_token = 0;
    let mut token_lengths = BTreeMap::<usize, usize>::new();
    for _ in 0..3_000 {
        let parts = 2 + rng.below(10);
        let text = text_of(&mut rng, parts, 400);
        let pieces = pieces(&text, &profile, &vocab);
        if pieces.is_empty() {
            continue;
        }
        let Ok(tokens) = merged_tokens(&pieces, &table) else {
            continue;
        };
        assert_eq!(
            check_prompt(&table, &pieces, &tokens).unwrap(),
            None,
            "{text:?}"
        );
        let mut start = 0;
        for cut in 0..=tokens.len() {
            let (first, second) = tokens.split_at(cut);
            assert_eq!(check_range(&table, &pieces, 0, None, first).unwrap(), None);
            assert_eq!(
                check_range(&table, &pieces, start, first.last().copied(), second).unwrap(),
                None,
                "{text:?} cut {cut}"
            );
            start += second.first().map_or(0, |token| token.pieces as usize);
        }
        for token in &tokens {
            *token_lengths.entry(token.pieces as usize).or_default() += 1;
            longest_token = longest_token.max(token.pieces as usize);
        }
        agreed += 1;
    }

    // 3. Exhaustion on short built texts: of every grouping of the pieces,
    //    exactly the merge's passes.
    let (mut texts, mut examined, mut stable_but_wrong, mut wrong_id_refused) =
        (0, 0_u64, 0_u64, 0);
    while texts < 20_000 {
        let parts = 2 + rng.below(3);
        let text = text_of(&mut rng, parts, 12);
        let pieces = pieces(&text, &profile, &vocab);
        if pieces.len() < 2 || pieces.len() > 12 {
            continue;
        }
        let Ok(tokens) = merged_tokens(&pieces, &table) else {
            continue;
        };
        let truth: Vec<usize> = tokens.iter().map(|token| token.pieces as usize).collect();
        for counts in groupings(pieces.len()) {
            examined += 1;
            let passes = grouping_passes(&table, &pieces, &counts).unwrap();
            let staged =
                check_range_in_rounds(&table, &pieces, 0, None, &marks_of(&counts)).unwrap();
            assert_eq!(
                staged.fault.is_none(),
                passes,
                "staged rounds, text {text:?}, grouping {counts:?}"
            );
            assert_eq!(
                passes,
                counts == truth,
                "text {text:?}, grouping {counts:?}, merge {truth:?}"
            );
            if counts != truth && spells_stable_tokens(&table, &vocab, &pieces, &counts) {
                stable_but_wrong += 1;
            }
        }
        // The right grouping under one wrong id is refused too.
        let mut altered: Vec<ClaimedToken> = tokens.clone();
        let index = rng.below(altered.len());
        altered[index].id = altered[index].id.wrapping_add(1) % vocab.len() as u32;
        assert!(check_prompt(&table, &pieces, &altered).unwrap().is_some());
        wrong_id_refused += 1;
        texts += 1;
    }

    // 4. The work a stage would do on a long prompt, in ranges of 512 tokens,
    //    beside today's merge stages over the same ranges.
    let long = text_of(&mut rng, 2_000, 6_000);
    let long_pieces = pieces(&long, &profile, &vocab);
    let long_tokens = merged_tokens(&long_pieces, &table).unwrap();
    let (mut staged_steps, mut today_steps, mut start, mut previous) = (0_u64, 0_u64, 0, None);
    for range in long_tokens.chunks(512) {
        let counts: Vec<usize> = range.iter().map(|token| token.pieces as usize).collect();
        let len: usize = counts.iter().sum();
        let staged =
            check_range_in_rounds(&table, &long_pieces, start, previous, &marks_of(&counts))
                .unwrap();
        assert_eq!(staged.fault, None);
        assert_eq!(staged.texts.len(), range.len());
        staged_steps += staged.scan_steps;
        today_steps += full_merge_scan_steps(&long_pieces[start..start + len], &table).unwrap();
        start += len;
        previous = counts.last().copied();
    }

    let vocabulary_longest = vocab
        .keys()
        .map(|token| token.chars().count())
        .max()
        .unwrap();
    let over_16 = vocab
        .keys()
        .filter(|token| token.chars().count() > 16)
        .count();
    println!("reference corpus cases agreeing: {corpus_cases}");
    println!("built texts agreeing, whole and at every two-range cut: {agreed}");
    println!("short texts exhausted: {texts}; groupings examined: {examined}");
    println!(
        "  wrong groupings that spell stable vocabulary tokens, all refused: {stable_but_wrong}"
    );
    println!("  right grouping with one wrong id, refused: {wrong_id_refused}");
    println!("pieces per token in built texts: longest {longest_token}; counts {token_lengths:?}");
    println!(
        "long prompt: {} pieces, {} tokens, {} ranges; merge lookups today {today_steps}, as staged rounds {staged_steps} ({:.1}x fewer)",
        long_pieces.len(),
        long_tokens.len(),
        long_tokens.len().div_ceil(512),
        today_steps as f64 / staged_steps as f64
    );
    println!(
        "vocabulary: longest entry {vocabulary_longest} characters; {over_16} entries over 16"
    );
    assert!(
        stable_but_wrong > 0,
        "the exhaustion never met a plausible wrong grouping"
    );
}
