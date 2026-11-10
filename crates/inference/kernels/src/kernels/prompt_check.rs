//! Checks claimed prompt tokens against the ranked merge without replaying it
//! over the whole prompt.
//!
//! Native prototype for `docs/proposals/claimed-prompt-tokens.md`. No stage
//! uses it yet. It exists so the rule can be tested against real merge tables
//! before a stage is authored.
//!
//! The rule: every claimed token is merged together with its left neighbour
//! in the segment, on those two tokens' pieces alone, and the result must be
//! exactly the two tokens, split where the claim says. A token with no left
//! neighbour in its segment is merged alone and must become one piece.

use anyhow::{ensure, Result};
use prompt_check::input::{vocab_bucket_of, BpePiece, PromptTokenizer};

use super::prompt_prepare::best_merge;

/// One claimed token: its id, and how many of `prompt-bind`'s pieces it spans.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClaimedToken {
    pub id: u32,
    pub pieces: u32,
}

/// Why a claim was refused, and at which token of the range.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenFault {
    /// The token has no pieces, runs past the prompt, or spans two segments.
    DoesNotFit { token: usize },
    /// Merging the token with its left neighbour, or alone, does not give it.
    NotTheMerge { token: usize },
    /// The merge gives a piece that is not the vocabulary entry for the id.
    WrongId { token: usize },
}

/// A group of consecutive pieces: where it starts and how many it holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Group {
    start: usize,
    len: usize,
}

/// Runs the ranked merge over `pieces` alone and returns each resulting
/// piece's text with the number of original pieces it was made from.
pub fn merge_with_spans(
    pieces: &[BpePiece],
    tokenizer: &PromptTokenizer,
) -> Result<Vec<(String, usize)>> {
    let mut current = pieces.to_vec();
    let mut spans = vec![1_usize; pieces.len()];
    loop {
        let best = best_merge(&current, tokenizer)?;
        if !best.matched {
            break;
        }
        let left = best.left as usize;
        current[left].text = best.merged;
        current.remove(left + 1);
        spans[left] += spans.remove(left + 1);
    }
    Ok(current
        .into_iter()
        .map(|piece| piece.text)
        .zip(spans)
        .collect())
}

fn vocab_id(tokenizer: &PromptTokenizer, text: &str) -> Result<Option<u32>> {
    ensure!(
        tokenizer.vocab_bucket_count > 0
            && tokenizer.vocab_bucket_count as usize == tokenizer.vocab_buckets.len(),
        "invalid vocabulary bucket count"
    );
    let index = vocab_bucket_of(text, tokenizer.vocab_bucket_count) as usize;
    Ok(tokenizer.vocab_buckets[index]
        .entries
        .iter()
        .find(|entry| entry.token == text)
        .map(|entry| entry.id))
}

/// The tokens the full merge produces for `pieces`: what run preparation
/// would supply as the claim.
pub fn merged_tokens(
    pieces: &[BpePiece],
    tokenizer: &PromptTokenizer,
) -> Result<Vec<ClaimedToken>> {
    let mut tokens = Vec::new();
    let mut start = 0;
    while start < pieces.len() {
        let segment = pieces[start].segment;
        let end = start
            + pieces[start..]
                .iter()
                .take_while(|piece| piece.segment == segment)
                .count();
        for (text, span) in merge_with_spans(&pieces[start..end], tokenizer)? {
            let id = vocab_id(tokenizer, &text)?.ok_or_else(|| {
                anyhow::anyhow!("merged piece is absent from vocabulary: {text:?}")
            })?;
            tokens.push(ClaimedToken {
                id,
                pieces: u32::try_from(span)?,
            });
        }
        start = end;
    }
    Ok(tokens)
}

/// The texts of the groups `counts` describes, if every group passes the rule.
///
/// `previous` is the piece count of the token just before `start`, when the
/// range does not begin the prompt. That token is not re-checked here; it is
/// only the left neighbour of the range's first token.
fn check_groups(
    tokenizer: &PromptTokenizer,
    pieces: &[BpePiece],
    start: usize,
    previous: Option<usize>,
    counts: &[usize],
) -> Result<std::result::Result<Vec<String>, TokenFault>> {
    let mut left = match previous {
        Some(len) if len >= 1 && len <= start => Some(Group {
            start: start - len,
            len,
        }),
        Some(_) => return Ok(Err(TokenFault::DoesNotFit { token: 0 })),
        None => None,
    };
    let mut texts = Vec::with_capacity(counts.len());
    let mut cursor = start;
    for (token, &len) in counts.iter().enumerate() {
        let fits = len >= 1
            && cursor
                .checked_add(len)
                .is_some_and(|end| end <= pieces.len());
        if !fits {
            return Ok(Err(TokenFault::DoesNotFit { token }));
        }
        let group = Group { start: cursor, len };
        let segment = pieces[cursor].segment;
        if pieces[cursor..cursor + len]
            .iter()
            .any(|piece| piece.segment != segment)
        {
            return Ok(Err(TokenFault::DoesNotFit { token }));
        }
        // A neighbour is a token whose last piece is in this token's segment.
        let neighbour = left.filter(|left| pieces[left.start + left.len - 1].segment == segment);
        let text = match neighbour {
            // Merged with its left neighbour: exactly the two tokens, split
            // where the claim says.
            Some(left) => {
                let merged =
                    merge_with_spans(&pieces[left.start..group.start + group.len], tokenizer)?;
                match merged.as_slice() {
                    [(_, first), (text, second)] if *first == left.len && *second == group.len => {
                        text.clone()
                    }
                    _ => return Ok(Err(TokenFault::NotTheMerge { token })),
                }
            }
            // No left neighbour in the segment: alone it must become one piece.
            None => {
                let merged =
                    merge_with_spans(&pieces[group.start..group.start + group.len], tokenizer)?;
                match merged.as_slice() {
                    [(text, _)] => text.clone(),
                    _ => return Ok(Err(TokenFault::NotTheMerge { token })),
                }
            }
        };
        texts.push(text);
        left = Some(group);
        cursor += len;
    }
    Ok(Ok(texts))
}

/// Checks one range of claimed tokens. `start` is the first piece the range
/// covers and `previous` the last token of the range before it, if any.
/// Returns the first fault, or `None` when the tokens are the merge's result
/// for the pieces they cover.
pub fn check_range(
    tokenizer: &PromptTokenizer,
    pieces: &[BpePiece],
    start: usize,
    previous: Option<ClaimedToken>,
    tokens: &[ClaimedToken],
) -> Result<Option<TokenFault>> {
    let counts = tokens
        .iter()
        .map(|token| token.pieces as usize)
        .collect::<Vec<_>>();
    let texts = match check_groups(
        tokenizer,
        pieces,
        start,
        previous.map(|token| token.pieces as usize),
        &counts,
    )? {
        Ok(texts) => texts,
        Err(fault) => return Ok(Some(fault)),
    };
    for (token, (claimed, text)) in tokens.iter().zip(&texts).enumerate() {
        if vocab_id(tokenizer, text)? != Some(claimed.id) {
            return Ok(Some(TokenFault::WrongId { token }));
        }
    }
    Ok(None)
}

/// Checks tokens claimed for the whole prompt: they must also cover every piece.
pub fn check_prompt(
    tokenizer: &PromptTokenizer,
    pieces: &[BpePiece],
    tokens: &[ClaimedToken],
) -> Result<Option<TokenFault>> {
    if let Some(fault) = check_range(tokenizer, pieces, 0, None, tokens)? {
        return Ok(Some(fault));
    }
    let covered: usize = tokens.iter().map(|token| token.pieces as usize).sum();
    Ok((covered != pieces.len()).then_some(TokenFault::DoesNotFit {
        token: tokens.len(),
    }))
}

/// Whether grouping `pieces` by `counts` passes the rule, ignoring ids.
/// For tests that enumerate every grouping.
pub fn grouping_passes(
    tokenizer: &PromptTokenizer,
    pieces: &[BpePiece],
    counts: &[usize],
) -> Result<bool> {
    Ok(counts.iter().sum::<usize>() == pieces.len()
        && check_groups(tokenizer, pieces, 0, None, counts)?.is_ok())
}

/// The check as a stage would run it: a fixed number of rounds, each merging
/// at most one pair in every window, with no loop that depends on the data.
///
/// The claim marks which pieces start a token. Tokens are numbered in order,
/// and two passes pair them up differently: one pairs tokens (0,1), (2,3), …
/// and the other (1,2), (3,4), …, so every neighbouring pair shares a window
/// in exactly one pass. A window is merged on its own, which is the pair run
/// of `check_range`. A merge that joins pieces of two tokens, a token left in
/// more than one piece, or a window still merging after the last round is a
/// fault.
pub mod rounds {
    use super::*;

    /// Merge rounds per pass. A window of two tokens needs two merges fewer
    /// than it has pieces, so this covers tokens of up to 33 pieces each.
    pub const ROUNDS: usize = 64;

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct Outcome {
        pub fault: Option<TokenFault>,
        /// The range's tokens as text, when there is no fault.
        pub texts: Vec<String>,
        /// Pieces visited by merge scans over both passes: one authenticated
        /// merge-table lookup each in the stage.
        pub scan_steps: u64,
    }

    /// `marks[i]` is non-zero when piece `start + i` starts a token; the first
    /// piece of a range always does. `previous` is the piece count of the
    /// token just before `start`, as for `check_range`.
    pub fn check_range_in_rounds(
        tokenizer: &PromptTokenizer,
        pieces: &[BpePiece],
        start: usize,
        previous: Option<usize>,
        marks: &[u32],
    ) -> Result<Outcome> {
        let mut fault = None;
        let lead = match previous {
            Some(len) if len >= 1 && len <= start => len,
            Some(_) => {
                fault = Some(TokenFault::DoesNotFit { token: 0 });
                0
            }
            None => 0,
        };
        let end = start.saturating_add(marks.len()).min(pieces.len());
        if start.saturating_add(marks.len()) > pieces.len() {
            fault = fault.or(Some(TokenFault::DoesNotFit { token: 0 }));
        }
        // Number the tokens. The previous range's last token, when present,
        // is token 0 and only serves as the first token's left neighbour.
        let from = start.min(pieces.len());
        // The previous token matters only as a left neighbour, so it is left
        // out when the range starts a new segment.
        let neighbours = from > 0
            && from < pieces.len()
            && !marks.is_empty()
            && pieces[from - 1].segment == pieces[from].segment;
        let lead = if neighbours { lead.min(from) } else { 0 };
        let first = usize::from(lead > 0);
        let mut list = pieces[from - lead..from].to_vec();
        let mut tokens = vec![0_usize; list.len()];
        let mut token = first;
        for (offset, piece) in pieces[start.min(end)..end].iter().enumerate() {
            if offset > 0 && marks[offset] != 0 {
                token += 1;
            }
            list.push(piece.clone());
            tokens.push(token);
        }
        for index in 1..list.len() {
            if tokens[index] == tokens[index - 1] && list[index].segment != list[index - 1].segment
            {
                fault = fault.or(Some(TokenFault::DoesNotFit {
                    token: tokens[index].saturating_sub(first),
                }));
            }
        }

        let mut scan_steps = 0;
        let mut texts = Vec::new();
        for pass in 0..2 {
            let mut current = list.clone();
            let mut owner = tokens.clone();
            for round in 0..=ROUNDS {
                scan_steps += current.len() as u64;
                // The best pair of every window, found in one scan.
                let mut merges = Vec::new();
                let mut run_start = 0;
                while run_start < current.len() {
                    let window = (owner[run_start] + pass) / 2;
                    let run_end = run_start
                        + owner[run_start..]
                            .iter()
                            .take_while(|token| (**token + pass) / 2 == window)
                            .count();
                    let best = best_merge(&current[run_start..run_end], tokenizer)?;
                    if best.matched {
                        merges.push((run_start + best.left as usize, best.merged));
                    }
                    run_start = run_end;
                }
                if round == ROUNDS {
                    // The convergence check after the last round.
                    if let Some((left, _)) = merges.first() {
                        fault = fault.or(Some(TokenFault::NotTheMerge {
                            token: owner[*left].saturating_sub(first),
                        }));
                    }
                    break;
                }
                if merges.is_empty() {
                    // The stage still runs its remaining rounds and the
                    // convergence check over the unchanged list.
                    scan_steps += ((ROUNDS - round) * current.len()) as u64;
                    break;
                }
                for (left, merged) in merges.into_iter().rev() {
                    if owner[left] != owner[left + 1] {
                        fault = fault.or(Some(TokenFault::NotTheMerge {
                            token: owner[left + 1].saturating_sub(first),
                        }));
                    }
                    current[left].text = merged;
                    current.remove(left + 1);
                    owner.remove(left + 1);
                }
            }
            // Every token must have become exactly one piece.
            for index in 1..owner.len() {
                if owner[index] == owner[index - 1] {
                    fault = fault.or(Some(TokenFault::NotTheMerge {
                        token: owner[index].saturating_sub(first),
                    }));
                }
            }
            if pass == 0 {
                texts = current
                    .iter()
                    .zip(&owner)
                    .filter(|(_, token)| **token >= first)
                    .map(|(piece, _)| piece.text.clone())
                    .collect();
            }
        }
        if fault.is_some() {
            texts.clear();
        }
        Ok(Outcome {
            fault,
            texts,
            scan_steps,
        })
    }

    /// The marks for a list of per-token piece counts.
    pub fn marks_of(counts: &[usize]) -> Vec<u32> {
        counts
            .iter()
            .flat_map(|count| (0..*count).map(|offset| u32::from(offset == 0)))
            .collect()
    }

    /// Pieces visited by merge scans had the stages this check replaced
    /// tokenized `pieces` as one range: every round scanned the whole range,
    /// rounds came in batches of 64, and a last stage scanned once more.
    pub fn full_merge_scan_steps(pieces: &[BpePiece], tokenizer: &PromptTokenizer) -> Result<u64> {
        let mut current = pieces.to_vec();
        let mut steps = 0_u64;
        let mut rounds = 0_u64;
        loop {
            steps += current.len() as u64;
            rounds += 1;
            let best = best_merge(&current, tokenizer)?;
            if !best.matched {
                break;
            }
            let left = best.left as usize;
            current[left].text = best.merged;
            current.remove(left + 1);
        }
        let padded = rounds.div_ceil(ROUNDS as u64) * ROUNDS as u64;
        Ok(steps + (padded - rounds + 1) * current.len() as u64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prompt_check::input::{merge_bucket_of, BpeMerge, MergeBucket, TokenEntry, VocabBucket};
    use std::collections::BTreeSet;

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

    /// A tokenizer over `letters` with the given `(left, right)` rules, ranked
    /// in the order given.
    fn tokenizer(letters: &[&str], rules: &[(String, String)]) -> PromptTokenizer {
        const BUCKETS: u32 = 8;
        let mut vocab: Vec<String> = letters.iter().map(|letter| letter.to_string()).collect();
        let mut merges = vec![MergeBucket::default(); BUCKETS as usize];
        for (rank, (left, right)) in rules.iter().enumerate() {
            let merged = format!("{left}{right}");
            if !vocab.contains(&merged) {
                vocab.push(merged.clone());
            }
            let index = merge_bucket_of(left, right, BUCKETS) as usize;
            let mut bucket = merges[index].rules.as_slice().to_vec();
            bucket.push(BpeMerge {
                rank: rank as u32,
                left: left.clone(),
                right: right.clone(),
                merged,
            });
            merges[index].rules = bucket.into();
        }
        let mut vocab_buckets = vec![VocabBucket::default(); BUCKETS as usize];
        for (id, token) in vocab.into_iter().enumerate() {
            let index = vocab_bucket_of(&token, BUCKETS) as usize;
            let mut bucket = vocab_buckets[index].entries.as_slice().to_vec();
            bucket.push(TokenEntry {
                token,
                id: id as u32,
            });
            vocab_buckets[index].entries = bucket.into();
        }
        PromptTokenizer {
            vocab_bucket_count: BUCKETS,
            merge_bucket_count: BUCKETS,
            vocab_buckets: vocab_buckets.into(),
            merge_buckets: merges.into(),
            ..Default::default()
        }
    }

    /// A random merge table. Each rule joins two tokens that already exist, as
    /// training would produce. With `shuffled`, ranks are then assigned in a
    /// random order, so a rule may outrank the rules that build its parts.
    fn random_rules(
        rng: &mut Rng,
        letters: &[&str],
        count: usize,
        shuffled: bool,
    ) -> Vec<(String, String)> {
        let mut tokens: Vec<String> = letters.iter().map(|letter| letter.to_string()).collect();
        let mut seen = BTreeSet::new();
        let mut rules = Vec::new();
        let mut attempts = 0;
        while rules.len() < count && attempts < count * 20 {
            attempts += 1;
            let left = tokens[rng.below(tokens.len())].clone();
            let right = tokens[rng.below(tokens.len())].clone();
            if left.len() + right.len() > 5 || !seen.insert((left.clone(), right.clone())) {
                continue;
            }
            let merged = format!("{left}{right}");
            if !tokens.contains(&merged) {
                tokens.push(merged);
            }
            rules.push((left, right));
        }
        if shuffled {
            for index in (1..rules.len()).rev() {
                rules.swap(index, rng.below(index + 1));
            }
        }
        rules
    }

    fn pieces_of(text: &str, segment_break: Option<usize>) -> Vec<BpePiece> {
        text.chars()
            .enumerate()
            .map(|(index, letter)| BpePiece {
                text: letter.to_string(),
                segment: u32::from(segment_break.is_some_and(|at| index >= at)),
            })
            .collect()
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

    fn strings(letters: &[&str], max_len: usize) -> Vec<String> {
        let mut all = Vec::new();
        let mut frontier = vec![String::new()];
        for _ in 0..max_len {
            frontier = frontier
                .iter()
                .flat_map(|prefix| {
                    letters
                        .iter()
                        .map(move |letter| format!("{prefix}{letter}"))
                })
                .collect();
            all.extend(frontier.iter().cloned());
        }
        all
    }

    #[test]
    fn a_grouping_that_spells_the_text_but_is_not_the_merge_is_refused() {
        let rules = [
            ("a".to_string(), "b".to_string()),
            ("b".to_string(), "c".to_string()),
        ];
        let tokenizer = tokenizer(&["a", "b", "c"], &rules);
        let pieces = pieces_of("abc", None);
        let id = |text: &str| vocab_id(&tokenizer, text).unwrap().unwrap();
        // The merge joins `a b` first, so the result is [ab, c].
        let merged = merged_tokens(&pieces, &tokenizer).unwrap();
        assert_eq!(
            merged,
            [
                ClaimedToken {
                    id: id("ab"),
                    pieces: 2
                },
                ClaimedToken {
                    id: id("c"),
                    pieces: 1
                }
            ]
        );
        assert_eq!(check_prompt(&tokenizer, &pieces, &merged).unwrap(), None);
        // [a, bc] spells the text, uses vocabulary tokens and has no pair left
        // to merge. It is still not the merge's result.
        let other = [
            ClaimedToken {
                id: id("a"),
                pieces: 1,
            },
            ClaimedToken {
                id: id("bc"),
                pieces: 2,
            },
        ];
        assert_eq!(
            check_prompt(&tokenizer, &pieces, &other).unwrap(),
            Some(TokenFault::NotTheMerge { token: 1 })
        );
        // The right grouping under a wrong id, and tokens that stop short.
        let wrong_id = [
            merged[0],
            ClaimedToken {
                id: id("b"),
                pieces: 1,
            },
        ];
        assert_eq!(
            check_prompt(&tokenizer, &pieces, &wrong_id).unwrap(),
            Some(TokenFault::WrongId { token: 1 })
        );
        assert_eq!(
            check_prompt(&tokenizer, &pieces, &merged[..1]).unwrap(),
            Some(TokenFault::DoesNotFit { token: 1 })
        );
    }

    /// For every string up to `max_len` over the letters and every grouping of
    /// it, the grouping passes exactly when it is the full merge's. Returns
    /// how many groupings were examined and how many wrong ones had no pair
    /// left to merge, the kind a weaker check would accept.
    fn exhaust(tokenizer: &PromptTokenizer, letters: &[&str], max_len: usize) -> (usize, usize) {
        let mut examined = 0;
        let mut stable_but_wrong = 0;
        for text in strings(letters, max_len) {
            for segment_break in [None, Some(text.len() / 2)] {
                if segment_break == Some(0) {
                    continue;
                }
                let pieces = pieces_of(&text, segment_break);
                let truth = merged_tokens(&pieces, tokenizer)
                    .unwrap()
                    .iter()
                    .map(|token| token.pieces as usize)
                    .collect::<Vec<_>>();
                for counts in groupings(pieces.len()) {
                    examined += 1;
                    let passes = grouping_passes(tokenizer, &pieces, &counts).unwrap();
                    assert_eq!(
                        passes,
                        counts == truth,
                        "text {text:?}, break {segment_break:?}, grouping {counts:?}, merge {truth:?}"
                    );
                    if counts != truth && spells_stable_tokens(tokenizer, &pieces, &counts) {
                        stable_but_wrong += 1;
                    }
                }
            }
        }
        (examined, stable_but_wrong)
    }

    /// Every group joins to a vocabulary token inside one segment, and no two
    /// neighbouring tokens have a rule: all a check of the final state sees.
    fn spells_stable_tokens(
        tokenizer: &PromptTokenizer,
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
            if vocab_id(tokenizer, &text).unwrap().is_none() {
                return false;
            }
            tokens.push(BpePiece {
                text,
                segment: group[0].segment,
            });
            cursor += len;
        }
        !best_merge(&tokens, tokenizer).unwrap().matched
    }

    fn exhaust_random_tables(tables: usize, max_len: usize) -> (usize, usize) {
        let letters = ["a", "b", "c"];
        let mut rng = Rng(0x5eed);
        let mut totals = (0, 0);
        for table in 0..tables {
            let count = 2 + rng.below(14);
            let rules = random_rules(&mut rng, &letters, count, table % 2 == 1);
            let (examined, stable) = exhaust(&tokenizer(&letters, &rules), &letters, max_len);
            totals = (totals.0 + examined, totals.1 + stable);
        }
        totals
    }

    #[test]
    fn exactly_the_merge_grouping_passes_on_small_random_tables() {
        let (examined, stable_but_wrong) = exhaust_random_tables(12, 5);
        assert!(examined > 10_000 && stable_but_wrong > 0);
    }

    #[test]
    #[ignore = "exhaustive; run in release: cargo test --release -p inference-kernels -- --ignored exhaust"]
    fn exactly_the_merge_grouping_passes_exhaustive() {
        let (examined, stable_but_wrong) = exhaust_random_tables(400, 7);
        println!(
            "groupings examined: {examined}; wrong but stable, and refused: {stable_but_wrong}"
        );
        assert!(stable_but_wrong > 0);
    }

    #[test]
    fn ranges_cut_anywhere_agree_with_the_whole_prompt() {
        let letters = ["a", "b", "c"];
        let mut rng = Rng(0xca7);
        for table in 0..20 {
            let rules = random_rules(&mut rng, &letters, 10, table % 2 == 1);
            let tokenizer = tokenizer(&letters, &rules);
            for text in strings(&letters, 5) {
                let pieces = pieces_of(&text, Some(text.len() / 2).filter(|at| *at > 0));
                let truth = merged_tokens(&pieces, &tokenizer).unwrap();
                // The merge's tokens pass range by range wherever the cut falls.
                for cut in 0..=truth.len() {
                    let (first, second) = truth.split_at(cut);
                    let start: usize = first.iter().map(|token| token.pieces as usize).sum();
                    assert_eq!(
                        check_range(&tokenizer, &pieces, 0, None, first).unwrap(),
                        None
                    );
                    assert_eq!(
                        check_range(&tokenizer, &pieces, start, first.last().copied(), second)
                            .unwrap(),
                        None,
                        "text {text:?}, cut {cut}"
                    );
                }
                // A wrong grouping fails in one of its two ranges wherever the
                // cut falls, because the first token of the second range is
                // checked against the last token of the first.
                for counts in groupings(pieces.len()) {
                    if counts
                        .iter()
                        .copied()
                        .eq(truth.iter().map(|token| token.pieces as usize))
                    {
                        continue;
                    }
                    for cut in 0..=counts.len() {
                        let (first, second) = counts.split_at(cut);
                        let start: usize = first.iter().sum();
                        let first_ok = check_groups(&tokenizer, &pieces, 0, None, first)
                            .unwrap()
                            .is_ok();
                        let second_ok =
                            check_groups(&tokenizer, &pieces, start, first.last().copied(), second)
                                .unwrap()
                                .is_ok();
                        assert!(
                            !(first_ok && second_ok),
                            "text {text:?}, grouping {counts:?}, cut {cut}"
                        );
                    }
                }
            }
        }
    }

    /// The rounds a stage would run accept and refuse exactly what the pair
    /// check does, for a whole prompt and for a second range at every cut.
    #[test]
    fn the_staged_rounds_agree_with_the_pair_check() {
        let letters = ["a", "b", "c"];
        let mut rng = Rng(0xbeef);
        for table in 0..30 {
            let count = 2 + rng.below(14);
            let rules = random_rules(&mut rng, &letters, count, table % 2 == 1);
            let tokenizer = tokenizer(&letters, &rules);
            for text in strings(&letters, 5) {
                for segment_break in [None, Some(text.len() / 2)] {
                    if segment_break == Some(0) {
                        continue;
                    }
                    let pieces = pieces_of(&text, segment_break);
                    for counts in groupings(pieces.len()) {
                        let mut ranges = vec![(0, None, &counts[..])];
                        for cut in 1..counts.len() {
                            let (first, second) = counts.split_at(cut);
                            ranges.push((first.iter().sum(), first.last().copied(), second));
                        }
                        for (start, previous, range) in ranges {
                            let expected =
                                check_groups(&tokenizer, &pieces, start, previous, range).unwrap();
                            let staged = rounds::check_range_in_rounds(
                                &tokenizer,
                                &pieces,
                                start,
                                previous,
                                &rounds::marks_of(range),
                            )
                            .unwrap();
                            assert_eq!(
                                staged.fault.is_none(),
                                expected.is_ok(),
                                "text {text:?}, break {segment_break:?}, grouping {counts:?}, from {start}"
                            );
                            if let Ok(texts) = expected {
                                assert_eq!(staged.texts, texts);
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn a_window_still_merging_after_the_last_round_is_refused() {
        // Longer rules rank first, so a run of `a` grows from the left into
        // one token of up to 40 letters. Two such tokens side by side need 78
        // merges, more than the rounds allow.
        let rules: Vec<(String, String)> = (1..40)
            .rev()
            .map(|len| ("a".repeat(len), "a".to_string()))
            .collect();
        let tokenizer = tokenizer(&["a", "b"], &rules);
        let pieces = pieces_of(&format!("{}b{}", "a".repeat(40), "a".repeat(40)), None);
        let counts = [40, 1, 40];
        assert!(check_groups(&tokenizer, &pieces, 0, None, &counts)
            .unwrap()
            .is_ok());
        let staged =
            rounds::check_range_in_rounds(&tokenizer, &pieces, 0, None, &rounds::marks_of(&counts))
                .unwrap();
        // 40 + 1 pieces need 39 merges and fit; the bound is on a window of
        // two long tokens, which this grouping does not contain.
        assert_eq!(staged.fault, None);
        let pieces = pieces_of(&"a".repeat(80), None);
        let long = [40, 40];
        let staged =
            rounds::check_range_in_rounds(&tokenizer, &pieces, 0, None, &rounds::marks_of(&long))
                .unwrap();
        assert!(staged.fault.is_some());
    }

    /// The stage's native twin, run range by range with each range handing
    /// its last token to the next, ends without a fault exactly when the
    /// marks are the merge's, and then publishes the merge's ids.
    #[test]
    fn the_stage_twin_accepts_exactly_the_merge_across_ranges() {
        use crate::kernels::prompt_check_stage::{marks_for, run_prompt_check, Trusted};
        use prompt_check::input::PromptState;

        let letters = ["a", "b", "c"];
        let root = raster::BytesPage::__from_parts(0, 0, Vec::new());
        let mut rng = Rng(0x57a9e);
        for table in 0..30 {
            let count = 2 + rng.below(14);
            let rules = random_rules(&mut rng, &letters, count, table % 2 == 1);
            let tokenizer = tokenizer(&letters, &rules);
            for text in strings(&letters, 5) {
                for segment_break in [None, Some(text.len() / 2)] {
                    if segment_break == Some(0) {
                        continue;
                    }
                    let pieces = pieces_of(&text, segment_break);
                    let merged = merged_tokens(&pieces, &tokenizer).unwrap();
                    let bound = PromptState::of_pieces(pieces.clone());
                    for counts in groupings(pieces.len()) {
                        let passes = grouping_passes(&tokenizer, &pieces, &counts).unwrap();
                        // One range, then every way of cutting it in two.
                        for cut in 1..=counts.len() {
                            let (first, second) = counts.split_at(cut);
                            let mut state = run_prompt_check(
                                &tokenizer,
                                &root,
                                &bound,
                                &bound,
                                &marks_for(first.iter().copied()),
                                &mut Trusted,
                            )
                            .unwrap();
                            let mut ids = state.token_ids.as_slice().to_vec();
                            if !second.is_empty() {
                                state = run_prompt_check(
                                    &tokenizer,
                                    &root,
                                    &bound,
                                    &state,
                                    &marks_for(second.iter().copied()),
                                    &mut Trusted,
                                )
                                .unwrap();
                                ids.extend(state.token_ids.iter().copied());
                            }
                            let label = format!(
                                "text {text:?}, break {segment_break:?}, grouping {counts:?}, cut {cut}"
                            );
                            assert_eq!(state.context.fault == 0, passes, "{label}");
                            assert!(state.context.complete, "{label}");
                            if passes {
                                let expected: Vec<u32> =
                                    merged.iter().map(|token| token.id).collect();
                                assert_eq!(ids, expected, "{label}");
                                assert_eq!(state.context.prompt_tokens as usize, merged.len());
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn the_stage_twin_records_what_is_wrong_with_a_range() {
        use crate::kernels::prompt_check_stage::{run_prompt_check, Trusted};
        use prompt_check::input::{
            PromptState, FAULT_CUT_BEYOND_TEXT, FAULT_EMPTY_RANGE, FAULT_RANGE_TOO_LARGE,
        };
        use prompt_check::input::{FAULT_NOT_THE_MERGE, FAULT_TOKEN_SPLIT};

        let rules = [
            ("a".to_string(), "b".to_string()),
            ("b".to_string(), "c".to_string()),
        ];
        let tokenizer = tokenizer(&["a", "b", "c"], &rules);
        let root = raster::BytesPage::__from_parts(0, 0, Vec::new());
        let check = |pieces: &[BpePiece], marks: &[u32], range_size: u32| {
            let mut bound = PromptState::of_pieces(pieces.to_vec());
            bound.context.range_size = range_size;
            run_prompt_check(&tokenizer, &root, &bound, &bound, marks, &mut Trusted)
                .unwrap()
                .context
        };
        let abc = pieces_of("abc", None);
        // The merge's own marks: [ab, c].
        let good = check(&abc, &[1, 0, 1], 0);
        assert_eq!(
            (good.fault, good.prompt_tokens, good.complete),
            (0, 2, true)
        );
        // [a, bc] spells the text with vocabulary tokens; token 2 is refused.
        let other = check(&abc, &[1, 1, 0], 0);
        assert_eq!(
            (other.fault, other.fault_position),
            (FAULT_NOT_THE_MERGE, 1)
        );
        // No marks, marks past the prompt, a token across two segments, and
        // more tokens than the range size.
        assert_eq!(check(&abc, &[], 0).fault, FAULT_EMPTY_RANGE);
        let beyond = check(&abc, &[1, 0, 1, 1], 0);
        assert_eq!(
            (beyond.fault, beyond.fault_position),
            (FAULT_CUT_BEYOND_TEXT, 4)
        );
        let split = check(&pieces_of("abc", Some(1)), &[1, 0, 1], 0);
        assert_eq!((split.fault, split.fault_position), (FAULT_TOKEN_SPLIT, 1));
        assert_eq!(check(&abc, &[1, 0, 1], 1).fault, FAULT_RANGE_TOO_LARGE);
    }
}
