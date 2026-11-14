//! Mistral's "tekken" tokenizer, as `tokenizer.json` describes it.
//!
//! ```text
//!   text ─ special tokens out ─▶ plain text ─ split ─▶ words ─ bytes ─▶ pieces ─ merge ─▶ ids
//! ```
//!
//! - Special tokens are cut out of the text first, longest first.
//! - Plain text is split into words by one pattern (see [`split_words`]).
//! - A word's UTF-8 bytes are its pieces, each written as one character.
//! - A word that is itself a token is that token. Any other word has its
//!   lowest-ranked neighbouring pair merged, leftmost on a tie, until no pair
//!   has a rule. A rule's rank is its place in the file's merge list.
//!
//! This is the host's tokenizer for direct inference, and the one the prompt
//! stages are held to. They do not run this code: they cut words with
//! `prompt-core`'s scanner, a character at a time, and check the merge that a
//! claim states. The tests here compare the two.

use std::collections::HashMap;

use anyhow::{ensure, Context, Result};
use inference_artifacts::PreparedPiece;

use prompt_check::input::tekken::unicode_classes::{
    LOWER, MARK, MODIFIER, NUMBER, OTHER, OTHER_LETTER, RANGES, TITLE, UPPER,
};
use prompt_check::input::{fallback_mask, TokenizerProfile, SCANNER_WORDS};

/// The split's pattern. A bundle whose tokenizer names another is refused.
pub const PATTERN: &str = r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// What the template puts before and after the trimmed prompt.
pub const PROMPT_PREFIX: &str = "<s>[INST]";
pub const PROMPT_SUFFIX: &str = "[/INST]";

/// The character a byte is written as: itself where that prints, and one of
/// `U+0100..` in order otherwise.
pub fn byte_char(byte: u8) -> char {
    let mut next = 256_u32;
    for candidate in 0..=255_u8 {
        let prints = matches!(candidate, b'!'..=b'~' | 0xA1..=0xAC | 0xAE..=0xFF);
        if candidate == byte {
            return char::from_u32(if prints { u32::from(byte) } else { next })
                .expect("a code point below U+0200");
        }
        if !prints {
            next += 1;
        }
    }
    unreachable!("every byte is visited")
}

fn class_of(ch: char) -> u8 {
    let code = ch as u32;
    match RANGES.binary_search_by(|(first, last, _)| {
        if code < *first {
            std::cmp::Ordering::Greater
        } else if code > *last {
            std::cmp::Ordering::Less
        } else {
            std::cmp::Ordering::Equal
        }
    }) {
        Ok(index) => RANGES[index].2,
        Err(_) => OTHER,
    }
}

/// `\s`: the characters with the White_Space property.
fn is_space(ch: char) -> bool {
    matches!(
        ch,
        '\u{9}'..='\u{D}'
            | ' '
            | '\u{85}'
            | '\u{A0}'
            | '\u{1680}'
            | '\u{2000}'..='\u{200A}'
            | '\u{2028}'
            | '\u{2029}'
            | '\u{202F}'
            | '\u{205F}'
            | '\u{3000}'
    )
}

fn is_letter(class: u8) -> bool {
    matches!(class, UPPER | LOWER | TITLE | MODIFIER | OTHER_LETTER)
}

#[derive(Clone, Copy)]
struct Seen {
    ch: char,
    class: u8,
}

type Test = fn(Seen) -> bool;

/// `[^\r\n\p{L}\p{N}]`
fn lead(seen: Seen) -> bool {
    !matches!(seen.ch, '\r' | '\n') && !is_letter(seen.class) && seen.class != NUMBER
}
/// `[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]`
fn upper(seen: Seen) -> bool {
    matches!(seen.class, UPPER | TITLE | MODIFIER | OTHER_LETTER | MARK)
}
/// `[\p{Ll}\p{Lm}\p{Lo}\p{M}]`
fn lower(seen: Seen) -> bool {
    matches!(seen.class, LOWER | MODIFIER | OTHER_LETTER | MARK)
}
fn number(seen: Seen) -> bool {
    seen.class == NUMBER
}
fn one_space(seen: Seen) -> bool {
    seen.ch == ' '
}
/// `[^\s\p{L}\p{N}]`
fn symbol(seen: Seen) -> bool {
    !is_space(seen.ch) && !is_letter(seen.class) && seen.class != NUMBER
}
/// `[\r\n/]`
fn break_or_slash(seen: Seen) -> bool {
    matches!(seen.ch, '\r' | '\n' | '/')
}
fn space(seen: Seen) -> bool {
    is_space(seen.ch)
}
fn line_break(seen: Seen) -> bool {
    matches!(seen.ch, '\r' | '\n')
}

/// One repeated character class: `test{min,max}`, greedy.
struct Repeat {
    test: Test,
    min: usize,
    max: usize,
}

const fn repeat(test: Test, min: usize, max: usize) -> Repeat {
    Repeat { test, min, max }
}

const MANY: usize = usize::MAX;

/// The pattern's alternatives, in order. The flag is `(?!\S)`: the match must
/// end the text or be followed by a space.
const ALTERNATIVES: [(&[Repeat], bool); 7] = [
    (
        &[
            repeat(lead, 0, 1),
            repeat(upper, 0, MANY),
            repeat(lower, 1, MANY),
        ],
        false,
    ),
    (
        &[
            repeat(lead, 0, 1),
            repeat(upper, 1, MANY),
            repeat(lower, 0, MANY),
        ],
        false,
    ),
    (&[repeat(number, 1, 1)], false),
    (
        &[
            repeat(one_space, 0, 1),
            repeat(symbol, 1, MANY),
            repeat(break_or_slash, 0, MANY),
        ],
        false,
    ),
    (
        &[repeat(space, 0, MANY), repeat(line_break, 1, MANY)],
        false,
    ),
    (&[repeat(space, 1, MANY)], true),
    (&[repeat(space, 1, MANY)], false),
];

/// Where a match of `parts` from `at` ends: each part takes as many
/// characters as it can and gives them back one at a time while the rest
/// fails, as a backtracking matcher does.
fn match_from(text: &[Seen], at: usize, parts: &[Repeat], before_space: bool) -> Option<usize> {
    let Some((part, rest)) = parts.split_first() else {
        let blocked = before_space && text.get(at).is_some_and(|seen| !is_space(seen.ch));
        return (!blocked).then_some(at);
    };
    let mut taken = 0;
    while taken < part.max && text.get(at + taken).is_some_and(|seen| (part.test)(*seen)) {
        taken += 1;
    }
    loop {
        if taken < part.min {
            return None;
        }
        if let Some(end) = match_from(text, at + taken, rest, before_space) {
            return Some(end);
        }
        if taken == 0 {
            return None;
        }
        taken -= 1;
    }
}

/// Cuts plain text into words: at each place, the first alternative of
/// [`PATTERN`] that matches. Every character is in exactly one word.
pub fn split_words(text: &str) -> Vec<&str> {
    let seen: Vec<Seen> = text
        .chars()
        .map(|ch| Seen {
            ch,
            class: class_of(ch),
        })
        .collect();
    let offsets: Vec<usize> = text
        .char_indices()
        .map(|(offset, _)| offset)
        .chain([text.len()])
        .collect();
    let mut words = Vec::new();
    let mut at = 0;
    while at < seen.len() {
        let end = ALTERNATIVES
            .iter()
            .find_map(|(parts, before_space)| match_from(&seen, at, parts, *before_space))
            .filter(|end| *end > at)
            // No alternative leaves a character out; one that did would be
            // a word of its own.
            .unwrap_or(at + 1);
        words.push(&text[offsets[at]..offsets[end]]);
        at = end;
    }
    words
}

/// A `mistral-dense` bundle's tokenizer.
pub struct Tekken {
    vocab: HashMap<String, u32>,
    /// A pair's rank and what it merges to.
    merges: HashMap<(String, String), (u32, String)>,
    /// Special tokens, longest first.
    specials: Vec<String>,
    /// The longest token that is not a special one, in bytes.
    longest_token: u32,
}

impl Tekken {
    pub fn from_json(tokenizer: &serde_json::Value) -> Result<Self> {
        let model = tokenizer
            .get("model")
            .context("tokenizer.json has no model")?;
        ensure!(
            model.get("type").and_then(|v| v.as_str()) == Some("BPE"),
            "a tekken tokenizer is a BPE model"
        );
        ensure!(
            model.get("ignore_merges").and_then(|v| v.as_bool()) == Some(true),
            "a tekken tokenizer takes a whole word that is a token as that token"
        );
        for key in [
            "unk_token",
            "continuing_subword_prefix",
            "end_of_word_suffix",
            "dropout",
        ] {
            ensure!(
                model.get(key).is_none_or(|v| v.is_null()),
                "unsupported tekken {key}"
            );
        }
        for key in ["byte_fallback", "fuse_unk"] {
            ensure!(
                !model.get(key).and_then(|v| v.as_bool()).unwrap_or(false),
                "unsupported tekken {key}"
            );
        }
        for key in ["normalizer", "truncation", "padding"] {
            ensure!(
                tokenizer.get(key).is_none_or(|v| v.is_null()),
                "unsupported tokenizer {key}"
            );
        }
        let expected = serde_json::json!({
            "type": "Sequence",
            "pretokenizers": [
                {"type": "Split", "pattern": {"Regex": PATTERN}, "behavior": "Isolated", "invert": false},
                {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": false}
            ]
        });
        ensure!(
            tokenizer.get("pre_tokenizer") == Some(&expected),
            "the tokenizer's split is not the tekken pattern this build implements"
        );

        let vocab = crate::vocab_map(model)?;
        let added: std::collections::BTreeSet<&str> = tokenizer
            .get("added_tokens")
            .and_then(|v| v.as_array())
            .into_iter()
            .flatten()
            .filter_map(|token| token.get("content").and_then(|v| v.as_str()))
            .collect();
        // A token's text is bytes, each written as one character. The output
        // is decoded on that understanding.
        let mut longest_token = 0;
        for token in vocab.keys().filter(|token| !added.contains(token.as_str())) {
            ensure!(
                token
                    .chars()
                    .all(|ch| prompt_check::input::tekken::char_byte(ch).is_some()),
                "token {token:?} is not written as bytes"
            );
            longest_token = longest_token.max(token.chars().count());
        }
        for byte in 0..=255_u8 {
            ensure!(
                vocab.contains_key(byte_char(byte).to_string().as_str()),
                "byte {byte} has no token"
            );
        }
        let mut merges = HashMap::new();
        let entries = model
            .get("merges")
            .and_then(|v| v.as_array())
            .context("BPE merges must be an array")?;
        for (rank, entry) in entries.iter().enumerate() {
            let rule =
                crate::parse_merge(u32::try_from(rank).context("merge rank overflow")?, entry)
                    .context("malformed BPE merge")?;
            ensure!(
                vocab.contains_key(&rule.left)
                    && vocab.contains_key(&rule.right)
                    && vocab.contains_key(&rule.merged),
                "merge token absent from vocabulary"
            );
            ensure!(
                merges
                    .insert((rule.left, rule.right), (rule.rank, rule.merged))
                    .is_none(),
                "duplicate BPE pair"
            );
        }

        let mut specials = Vec::new();
        for token in tokenizer
            .get("added_tokens")
            .and_then(|v| v.as_array())
            .into_iter()
            .flatten()
        {
            for key in ["single_word", "lstrip", "rstrip", "normalized"] {
                ensure!(
                    !token.get(key).and_then(|v| v.as_bool()).unwrap_or(false),
                    "unsupported added-token {key}"
                );
            }
            ensure!(
                token.get("special").and_then(|v| v.as_bool()) == Some(true),
                "non-special added tokens are unsupported"
            );
            let text = token
                .get("content")
                .and_then(|v| v.as_str())
                .context("invalid added token")?;
            ensure!(
                !text.is_empty()
                    && vocab.get(text).map(|id| u64::from(*id))
                        == token.get("id").and_then(|v| v.as_u64()),
                "added token disagrees with vocabulary: {text}"
            );
            specials.push(text.to_string());
        }
        specials.sort_by(|a, b| b.len().cmp(&a.len()).then_with(|| a.cmp(b)));
        Ok(Self {
            vocab: vocab.into_iter().collect(),
            merges,
            specials,
            longest_token: u32::try_from(longest_token).context("token length overflow")?,
        })
    }

    /// The tokenizer's text handling, as the data the prompt stages commit to.
    pub fn profile(&self) -> TokenizerProfile {
        let template = |text: &str| {
            if self.templated() {
                text.to_string()
            } else {
                String::new()
            }
        };
        TokenizerProfile {
            template_prefix: template(PROMPT_PREFIX),
            template_suffix: template(PROMPT_SUFFIX),
            normalize_spaces: false,
            split_spaces: false,
            byte_fallback: false,
            fuse_unk: false,
            has_unk: false,
            unk_token: String::new(),
            specials: self.specials.join("\0"),
            fallback_bytes: fallback_mask(|_| false),
            scanner: SCANNER_WORDS,
            longest_token: self.longest_token,
        }
    }

    /// Whether the vocabulary has what the template is written with.
    pub fn templated(&self) -> bool {
        ["<s>", "[INST]", "[/INST]"]
            .iter()
            .all(|token| self.vocab.contains_key(*token))
    }

    /// The text's tokens: each one's string and segment, and its id.
    pub fn encode(&self, text: &str) -> Result<(Vec<PreparedPiece>, Vec<u32>)> {
        // The text as special tokens and the plain text between them.
        let mut parts: Vec<(bool, &str)> = Vec::new();
        let (mut plain_start, mut cursor) = (0, 0);
        while cursor < text.len() {
            let tail = &text[cursor..];
            match self
                .specials
                .iter()
                .find(|special| tail.starts_with(special.as_str()))
            {
                Some(special) => {
                    parts.push((false, &text[plain_start..cursor]));
                    parts.push((true, &text[cursor..cursor + special.len()]));
                    cursor += special.len();
                    plain_start = cursor;
                }
                None => cursor += tail.chars().next().map_or(1, char::len_utf8),
            }
        }
        parts.push((false, &text[plain_start..]));

        // Each word and each special token is a segment of its own.
        let mut tokens: Vec<(String, u32)> = Vec::new();
        let mut segment = 0_u32;
        for (special, part) in parts {
            if special {
                tokens.push((part.to_string(), segment));
                segment += 1;
                continue;
            }
            for word in split_words(part) {
                tokens.extend(
                    self.word_tokens(word)
                        .into_iter()
                        .map(|token| (token, segment)),
                );
                segment += 1;
            }
        }
        let mut pieces = Vec::with_capacity(tokens.len());
        let mut ids = Vec::with_capacity(tokens.len());
        for (token, segment) in tokens {
            ids.push(
                *self
                    .vocab
                    .get(&token)
                    .with_context(|| format!("token {token:?} is not in the vocabulary"))?,
            );
            pieces.push(PreparedPiece {
                text: token,
                segment,
            });
        }
        Ok((pieces, ids))
    }

    /// One word's tokens.
    fn word_tokens(&self, word: &str) -> Vec<String> {
        let mut pieces: Vec<String> = word
            .bytes()
            .map(|byte| byte_char(byte).to_string())
            .collect();
        if self.vocab.contains_key(pieces.concat().as_str()) {
            return vec![pieces.concat()];
        }
        loop {
            let mut best: Option<(u32, usize, &String)> = None;
            for (left, pair) in pieces.windows(2).enumerate() {
                // The clone is the map's key type; a word is short.
                if let Some((rank, merged)) = self.merges.get(&(pair[0].clone(), pair[1].clone())) {
                    if best.is_none_or(|(lowest, _, _)| *rank < lowest) {
                        best = Some((*rank, left, merged));
                    }
                }
            }
            let Some((_, left, merged)) = best else {
                return pieces;
            };
            pieces[left] = merged.clone();
            pieces.remove(left + 1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bytes_are_written_as_the_byte_level_characters() {
        assert_eq!(byte_char(b'a'), 'a');
        assert_eq!(byte_char(b' '), '\u{120}');
        assert_eq!(byte_char(b'\n'), '\u{10A}');
        assert_eq!(byte_char(0), '\u{100}');
        assert_eq!(byte_char(0xAD), '\u{143}');
        assert_eq!(byte_char(0xFF), '\u{FF}');
        let mut all: Vec<char> = (0..=255_u8).map(byte_char).collect();
        all.sort_unstable();
        all.dedup();
        assert_eq!(all.len(), 256);
    }

    #[test]
    fn words_follow_the_pattern() {
        let words = |text: &str| {
            split_words(text)
                .into_iter()
                .map(str::to_string)
                .collect::<Vec<_>>()
        };
        assert_eq!(words("Say ciao."), ["Say", " ciao", "."]);
        // A capital run then lower case is one word; a second capital starts another.
        assert_eq!(
            words("CamelCase HTTPServer"),
            ["Camel", "Case", " HTTPServer"]
        );
        assert_eq!(words("snake_case"), ["snake", "_case"]);
        // Digits come one at a time.
        assert_eq!(words("3.14 x12"), ["3", ".", "1", "4", " x", "1", "2"]);
        // A run of spaces before text leaves its last space to the word.
        assert_eq!(words("a   b"), ["a", "  ", " b"]);
        assert_eq!(words("a  "), ["a", "  "]);
        // Line breaks take the spaces before them.
        assert_eq!(words("a \n\nb"), ["a", " \n\n", "b"]);
        // Symbols take a leading space and trailing breaks or slashes.
        assert_eq!(words("x --/\ny"), ["x", " --/\n", "y"]);
        assert_eq!(words("don't"), ["don", "'t"]);
        assert_eq!(words("日本語のテキスト"), ["日本語のテキスト"]);
        assert_eq!(words(""), Vec::<String>::new());
        for text in ["tabs\tand  double", "“quotes”, é — 🙂", "A1b2 ÀÉ àé ǅ ʰ"] {
            assert_eq!(split_words(text).concat(), text);
        }
    }

    use prompt_check::input::tekken as scanner;
    use prompt_check::input::{
        begin_scan_state, scan_chunk_into, BpePiece, PromptAtom, ScanPlan, ATOM_FIRST_PART,
        ATOM_WHOLE,
    };

    fn words_profile(specials: &[&str], longest_token: u32) -> TokenizerProfile {
        TokenizerProfile {
            specials: specials.join("\0"),
            fallback_bytes: fallback_mask(|_| false),
            scanner: SCANNER_WORDS,
            longest_token,
            ..TokenizerProfile::default()
        }
    }

    /// The atoms the stages' scanner makes of `text` given `step` characters
    /// at a time, and whether it had to place text it could not hold.
    fn scanned(text: &str, profile: &TokenizerProfile, step: usize) -> (Vec<PromptAtom>, bool) {
        let plan = ScanPlan {
            total: text.len() as u64,
            keep_start: 0,
            keep_end: text.len() as u64,
            templated: false,
        };
        let chars: Vec<char> = text.chars().collect();
        let mut chunks: Vec<String> = chars
            .chunks(step)
            .map(|chunk| chunk.iter().collect())
            .collect();
        if chunks.is_empty() {
            chunks.push(String::new());
        }
        let mut state = begin_scan_state(&plan, profile);
        let mut atoms = Vec::new();
        for chunk in &chunks {
            state = scan_chunk_into(state, chunk, &plan, profile, &mut |atom| atoms.push(atom));
        }
        assert!(
            state.pending.is_empty() && state.word == 0 && state.held == 0,
            "the scanner kept text of {text:?}"
        );
        (atoms, state.overflowed)
    }

    /// The text an atom's characters are the bytes of.
    fn raw(atom: &PromptAtom) -> String {
        let bytes = atom
            .text
            .chars()
            .map(|ch| scanner::char_byte(ch).expect("a byte character"))
            .collect();
        String::from_utf8(bytes).expect("an atom ends between characters")
    }

    /// The words the atoms spell, with a word's parts joined, and each special
    /// token as it is.
    fn spelled(atoms: &[PromptAtom]) -> Vec<(String, bool)> {
        let mut words: Vec<(String, bool)> = Vec::new();
        for atom in atoms {
            if atom.special {
                words.push((atom.text.clone(), true));
            } else if atom.part == ATOM_WHOLE || atom.part == ATOM_FIRST_PART {
                words.push((raw(atom), false));
            } else {
                words.last_mut().expect("a part follows a part").0 += &raw(atom);
            }
        }
        words
    }

    fn pattern_words(text: &str) -> Vec<(String, bool)> {
        split_words(text)
            .into_iter()
            .map(|word| (word.to_string(), false))
            .collect()
    }

    /// One character of each kind the pattern tells apart: capital, lower
    /// case, a letter with no case, a combining mark, digit, line break,
    /// space, another blank, slash, and any other symbol.
    const KINDS: [char; 10] = ['A', 'a', '中', '\u{301}', '1', '\n', ' ', '\t', '/', '.'];

    fn every_string(longest: usize, mut check: impl FnMut(&str)) {
        let mut text = String::new();
        for length in 0..=longest {
            let mut places = vec![0_usize; length];
            loop {
                text.clear();
                text.extend(places.iter().map(|place| KINDS[*place]));
                check(&text);
                let mut at = 0;
                while at < length {
                    places[at] += 1;
                    if places[at] < KINDS.len() {
                        break;
                    }
                    places[at] = 0;
                    at += 1;
                }
                if at == length {
                    break;
                }
            }
        }
    }

    fn assert_scanner_matches_pattern(longest: usize) {
        let whole = words_profile(&[], 4096);
        let parts = words_profile(&[], 4);
        every_string(longest, |text| {
            let expected = pattern_words(text);
            for step in [1, 3, usize::MAX] {
                let (atoms, overflowed) = scanned(text, &whole, step);
                assert!(!overflowed);
                assert!(atoms.iter().all(|atom| atom.part == ATOM_WHOLE));
                assert_eq!(spelled(&atoms), expected, "{text:?} in steps of {step}");
            }
            // With a short longest token, long words come in parts.
            let (atoms, _) = scanned(text, &parts, 2);
            assert_eq!(spelled(&atoms), expected, "{text:?} in parts");
            for atom in &atoms {
                assert!(raw(atom).len() <= 4, "{text:?} has an atom over the limit");
            }
            for (word, _) in &expected {
                let whole_atoms = atoms
                    .iter()
                    .filter(|atom| atom.part == ATOM_WHOLE && raw(atom) == *word)
                    .count();
                assert!(word.len() <= 4 || whole_atoms == 0, "{text:?}: {word:?} is whole");
            }
        });
    }

    #[test]
    fn the_stage_scanner_cuts_words_as_the_pattern_does() {
        assert_scanner_matches_pattern(5);
        for text in [
            "Say ciao.",
            "CamelCase HTTPServer snake_case 3.14 x12",
            "a \r\n\r\n  b\u{A0}\u{2028}c",
            "“quotes”, é — 🙂 A1b2 ÀÉ àé ǅ ʰ e\u{301}E\u{301}COLE ",
            "x --/\ny\t\t.z //\n\n/ q",
        ] {
            for step in [1, 2, 7, usize::MAX] {
                let (atoms, overflowed) = scanned(text, &words_profile(&[], 76), step);
                assert!(!overflowed);
                assert_eq!(spelled(&atoms), pattern_words(text), "{text:?}");
            }
        }
    }

    #[test]
    #[ignore = "eleven million strings: run with --release"]
    fn the_stage_scanner_cuts_longer_strings_as_the_pattern_does() {
        assert_scanner_matches_pattern(7);
    }

    #[test]
    fn the_scanner_agrees_on_the_byte_characters() {
        for byte in 0..=255_u8 {
            assert_eq!(scanner::byte_char(byte), byte_char(byte));
            assert_eq!(scanner::char_byte(byte_char(byte)), Some(byte));
        }
        assert_eq!(scanner::char_byte('\u{144}'), None);
        assert_eq!(scanner::char_byte(' '), None);
    }

    #[test]
    fn special_tokens_end_the_plain_text_around_them() {
        let profile = words_profile(&["[/INST]", "[INST]", "<s>"], 76);
        let plain = "Say  [x] <s\n ";
        let text = format!("<s>[INST]{plain}[/INST]");
        let mut expected = vec![("<s>".to_string(), true), ("[INST]".to_string(), true)];
        expected.extend(pattern_words(plain));
        expected.push(("[/INST]".to_string(), true));
        for step in [1, 2, 5, usize::MAX] {
            let (atoms, overflowed) = scanned(&text, &profile, step);
            assert!(!overflowed);
            assert_eq!(spelled(&atoms), expected, "in steps of {step}");
        }
    }

    #[test]
    fn a_run_too_long_to_hold_is_reported() {
        let profile = words_profile(&[], 76);
        let most = scanner::HOLD_BYTES as usize;
        // Blanks after a line break, and capitals after a letter with no case.
        for (lead, run, ends) in [("a\n", " ", ["b", "\nb"]), ("中", "A", [" ", "d"])] {
            for end in ends {
                let fits = format!("{lead}{}{end}", run.repeat(most));
                let (atoms, overflowed) = scanned(&fits, &profile, 256);
                assert!(!overflowed);
                assert_eq!(spelled(&atoms), pattern_words(&fits));

                let over = format!("{lead}{}{end}", run.repeat(most + 1));
                let (atoms, overflowed) = scanned(&over, &profile, 256);
                assert!(overflowed);
                // Every byte still comes out, once and in order.
                let all: String = spelled(&atoms).into_iter().map(|(word, _)| word).collect();
                assert_eq!(all, over);
            }
        }
    }

    #[test]
    fn a_word_is_one_piece_when_it_is_a_token_and_its_bytes_otherwise() {
        let profile = words_profile(&["<s>"], 6);
        let known = ["Say", "\u{120}ciao"];
        let pieces = prompt_check::input::split_rendered("<s>Say ciao. abcdefgh", &profile, |text| {
            known.contains(&text)
        });
        let shown: Vec<(&str, u32)> = pieces
            .iter()
            .map(|piece| (piece.text.as_str(), piece.segment))
            .collect();
        assert_eq!(
            shown,
            [
                ("<s>", 1),
                ("Say", 3),
                ("\u{120}ciao", 4),
                (".", 5),
                // Longer than the longest token: bytes, in one segment.
                ("\u{120}", 6),
                ("a", 6),
                ("b", 6),
                ("c", 6),
                ("d", 6),
                ("e", 6),
                ("f", 6),
                ("g", 6),
                ("h", 6),
            ]
        );
    }

    /// The ids the prompt stages would accept: their scanner's pieces, merged.
    fn staged_ids(
        text: &str,
        tokenizer: &serde_json::Value,
        committed: &crate::PromptTokenizer,
    ) -> Vec<u32> {
        let pieces: Vec<BpePiece> = crate::split_prompt(text, tokenizer)
            .unwrap()
            .into_iter()
            .map(|piece| BpePiece {
                text: piece.text,
                segment: piece.segment,
            })
            .collect();
        inference_kernels::kernels::prompt_check::merged_tokens(&pieces, committed)
            .unwrap()
            .into_iter()
            .map(|token| token.id)
            .collect()
    }

    /// Token ids against a reference implementation's, for texts it tokenized.
    ///
    /// `TEKKEN_TOKENIZER` is the bundle's `tokenizer.json`. `TEKKEN_REFERENCE`
    /// is a file of JSON lines, each `{"file": <text file>, "prompt_ids": [...]}`,
    /// where the ids are of `<s>` followed by the file's text.
    #[test]
    #[ignore = "needs TEKKEN_TOKENIZER and TEKKEN_REFERENCE"]
    fn ids_match_a_reference_tokenizer() {
        let tokenizer: serde_json::Value = serde_json::from_slice(
            &std::fs::read(std::env::var("TEKKEN_TOKENIZER").unwrap()).unwrap(),
        )
        .unwrap();
        let tekken = Tekken::from_json(&tokenizer).unwrap();
        let committed = crate::prompt_tokenizer(&tokenizer).unwrap();
        let reference = std::fs::read(std::env::var("TEKKEN_REFERENCE").unwrap()).unwrap();
        let mut checked = 0;
        let mut wrong = Vec::new();
        for line in String::from_utf8_lossy(&reference).lines() {
            let case: serde_json::Value = serde_json::from_str(line).unwrap();
            let file = case["file"].as_str().unwrap();
            let expected: Vec<u32> = case["prompt_ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|id| id.as_u64().unwrap() as u32)
                .collect();
            let text = format!("<s>{}", std::fs::read_to_string(file).unwrap());
            let (_, ids) = tekken.encode(&text).unwrap();
            checked += 1;
            if staged_ids(&text, &tokenizer, &committed) != ids {
                wrong.push(format!("{file}: the prompt stages' pieces merge to other ids"));
            }
            if ids != expected {
                let at = ids
                    .iter()
                    .zip(&expected)
                    .take_while(|(a, b)| a == b)
                    .count();
                wrong.push(format!(
                    "{file}: first difference at token {at}: got {:?}, expected {:?}",
                    &ids[at..ids.len().min(at + 6)],
                    &expected[at..expected.len().min(at + 6)]
                ));
            }
        }
        assert!(checked > 0, "the reference has no cases");
        assert!(
            wrong.is_empty(),
            "{} of {checked} differ:\n{}",
            wrong.len(),
            wrong.join("\n")
        );
    }
}
