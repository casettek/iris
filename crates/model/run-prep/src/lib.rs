use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

use anyhow::{ensure, Context, Result};
use inference_artifacts::{InferenceRunSpec, ModelManifest, PreparedPiece, PreparedPrompt};
use input_embedding::input::ClaimedOutput;
pub use output_finalize::input::{output_fault_description, OUTPUT_FAULT_SHORT_WITHOUT_EOS};
use output_finalize::input::{DecoderTable, DecoderToken};
use prompt_prepare::input::{
    boundary_bucket_of, byte_token, fallback_mask, merge_bucket_of, split_rendered,
    vocab_bucket_of, BoundaryBucket, BoundaryPair, BpeMerge, MergeBucket, PromptText,
    TokenizerProfile, PROMPT_PREFIX, PROMPT_SUFFIX,
};
pub use prompt_prepare::input::{
    fault_description, RangeCut, RunConfig, FAULT_CUT_BEYOND_TEXT, FAULT_EMPTY_RANGE, FAULT_NONE,
    FAULT_UNSAFE_CUT,
};
use prompt_prepare::input::{BpePiece, BpePieces, PromptTokenizer, TokenEntry, VocabBucket};
use raster::List;
pub use sequence_check::input::{
    check_fault_description, CHECK_FAULT_NO_CLAIMED_TOKENS, CHECK_FAULT_UNSCORED,
};

const BOS_TOKEN: &str = "<bos>";
const TURN_OPEN: &str = "<|turn>";
const TURN_CLOSE: &str = "<turn|>";
const NEWLINE_TOKEN: &str = "\n";

pub fn prepare_prompt(
    tokenizer: &serde_json::Value,
    spec_dir: &Path,
    spec: &InferenceRunSpec,
    model: &ModelManifest,
) -> Result<PreparedPrompt> {
    let model_json = tokenizer_model(tokenizer)?;
    let vocab = vocab_map(model_json)?;
    let prompt = resolve_prompt(spec_dir, spec)?;
    let templated = !spec.raw_prompt && supports_gemma_turns(&vocab);
    let rendered_prompt = if templated {
        render_gemma_turns(&prompt)
    } else {
        prompt.clone()
    };
    let initial_pieces = split_prompt(&rendered_prompt, tokenizer)?;

    Ok(PreparedPrompt {
        resolved_prompt: prompt,
        rendered_prompt,
        initial_pieces,
        eos_token_ids: model.eos_token_ids.clone(),
    })
}

pub fn prompt_tokenizer(tokenizer: &serde_json::Value) -> Result<PromptTokenizer> {
    validate_tokenizer_profile(tokenizer)?;
    let model = tokenizer_model(tokenizer)?;
    let mut vocab: Vec<TokenEntry> = vocab_map(model)?
        .into_iter()
        .map(|(token, id)| TokenEntry { token, id })
        .collect();
    vocab.sort_by_key(|entry| entry.id);
    let vocab_ids = vocab_map(model)?;
    let mut merges = Vec::new();
    let mut pairs = BTreeSet::new();
    if let Some(entries) = model.get("merges").and_then(|v| v.as_array()) {
        for (rank, entry) in entries.iter().enumerate() {
            let rule = parse_merge(u32::try_from(rank).context("merge rank overflow")?, entry)
                .context("malformed BPE merge")?;
            ensure!(
                vocab_ids.contains_key(&rule.left)
                    && vocab_ids.contains_key(&rule.right)
                    && vocab_ids.contains_key(&rule.merged),
                "merge token absent from vocabulary"
            );
            ensure!(
                pairs.insert((rule.left.clone(), rule.right.clone())),
                "duplicate BPE pair"
            );
            merges.push(rule);
        }
    }
    let (vocab_bucket_count, vocab_buckets) = bucket_vocab(vocab);
    let (merge_bucket_count, merge_buckets) = bucket_merges(merges);
    let (boundary_bucket_count, boundary_buckets) = bucket_boundaries(vocabulary_pairs(tokenizer)?);
    Ok(PromptTokenizer {
        vocab_bucket_count,
        merge_bucket_count,
        boundary_bucket_count,
        profile: tokenizer_profile(tokenizer)?,
        vocab_buckets: List::from(vocab_buckets),
        merge_buckets: List::from(merge_buckets),
        boundary_buckets: List::from(boundary_buckets),
    })
}

/// The tokenizer's text handling, as the data the prompt stages commit to.
pub fn tokenizer_profile(tokenizer: &serde_json::Value) -> Result<TokenizerProfile> {
    let model = tokenizer_model(tokenizer)?;
    let vocab = vocab_map(model)?;
    let present = |key: &str| {
        !tokenizer
            .get(key)
            .unwrap_or(&serde_json::Value::Null)
            .is_null()
    };
    let flag = |key: &str| model.get(key).and_then(|v| v.as_bool()).unwrap_or(false);
    let unk = model.get("unk_token").and_then(|v| v.as_str());
    Ok(TokenizerProfile {
        templated: supports_gemma_turns(&vocab),
        normalize_spaces: present("normalizer"),
        split_spaces: present("pre_tokenizer"),
        byte_fallback: flag("byte_fallback"),
        fuse_unk: flag("fuse_unk"),
        has_unk: unk.is_some(),
        unk_token: unk.unwrap_or_default().to_string(),
        specials: special_tokens(tokenizer).join("\0"),
        fallback_bytes: fallback_mask(|byte| vocab.contains_key(&byte_token(byte))),
    })
}

pub fn decoder_table(tokenizer: &serde_json::Value, eos_ids: &[u32]) -> Result<DecoderTable> {
    let model = tokenizer_model(tokenizer)?;
    let vocab_map = vocab_map(model)?;
    let special_ids: BTreeSet<u32> = tokenizer
        .get("added_tokens")
        .and_then(serde_json::Value::as_array)
        .into_iter()
        .flatten()
        .filter(|entry| {
            entry
                .get("special")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false)
        })
        .filter_map(|entry| entry.get("id").and_then(serde_json::Value::as_u64))
        .map(|id| id as u32)
        .collect();
    let max_token_id = vocab_map.values().copied().max().unwrap_or(0);
    let mut tokens = vec![DecoderToken::default(); max_token_id as usize + 1];
    for (token, id) in vocab_map {
        tokens[id as usize] = DecoderToken {
            token,
            special: special_ids.contains(&id),
            terminal: eos_ids.contains(&id),
        };
    }
    Ok(DecoderTable {
        tokens: List::from(tokens),
    })
}

/// One run-specific external: its data file, index and commitment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunFile {
    pub path: std::path::PathBuf,
    pub index_path: std::path::PathBuf,
    pub commitment: String,
}

fn write_run_file<T: serde::Serialize>(value: &T, dir: &Path, name: &str) -> Result<RunFile> {
    fs::create_dir_all(dir).with_context(|| format!("failed to create {}", dir.display()))?;
    let path = dir.join(format!("{name}.rastered"));
    let index_path = dir.join(format!("{name}.rindex"));
    let commitment = raster::write_raster_files(value, &path, &index_path)
        .map_err(|error| anyhow::anyhow!("{error}"))?;
    Ok(RunFile {
        path,
        index_path,
        commitment,
    })
}

/// Every pair of adjacent characters inside some vocabulary token.
///
/// A merge always produces a vocabulary token, and a merge across the
/// boundary between two pieces produces one containing the left piece's last
/// character followed by the right piece's first. So a boundary whose pair is
/// not in this set can never be crossed, and the pieces on either side
/// tokenize independently to the same ids as the whole prompt.
pub fn vocabulary_pairs(tokenizer: &serde_json::Value) -> Result<BTreeSet<(char, char)>> {
    let vocab = vocab_map(tokenizer_model(tokenizer)?)?;
    let mut pairs = BTreeSet::new();
    for token in vocab.keys() {
        let chars = token.chars().collect::<Vec<_>>();
        pairs.extend(chars.windows(2).map(|pair| (pair[0], pair[1])));
    }
    Ok(pairs)
}

fn can_cut(left: &PreparedPiece, right: &PreparedPiece, pairs: &BTreeSet<(char, char)>) -> bool {
    if left.segment != right.segment {
        return true;
    }
    match (left.text.chars().last(), right.text.chars().next()) {
        (Some(last), Some(first)) => !pairs.contains(&(last, first)),
        _ => true,
    }
}

/// One prompt range: pieces between two boundaries no merge can cross, and
/// the number of tokens they merge into.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PromptRange {
    pub pieces: Vec<PreparedPiece>,
    pub tokens: usize,
}

/// Cuts the prompt's pieces into ranges of at most `range_size` tokens, each
/// ending at the last boundary no merge can cross.
///
/// Each stretch between two such boundaries tokenizes on its own to the same
/// ids as the whole prompt, so its token count is exact and ranges pack whole
/// stretches. Fails if one stretch alone is longer than `range_size` tokens.
pub fn prompt_ranges(
    pieces: &[PreparedPiece],
    range_size: usize,
    pairs: &BTreeSet<(char, char)>,
    tokenizer: &PromptTokenizer,
) -> Result<Vec<PromptRange>> {
    ensure!(
        !pieces.is_empty(),
        "a teacher-forced claim needs a non-empty prompt"
    );
    ensure!(range_size >= 2, "range size must be at least 2");
    let empty = || PromptRange {
        pieces: Vec::new(),
        tokens: 0,
    };
    let mut ranges = Vec::new();
    let mut current = empty();
    let mut start = 0;
    while start < pieces.len() {
        let end = (start + 1..pieces.len())
            .find(|&cut| can_cut(&pieces[cut - 1], &pieces[cut], pairs))
            .unwrap_or(pieces.len());
        let stretch = &pieces[start..end];
        let tokens = merged_len(stretch, tokenizer)?;
        ensure!(
            tokens <= range_size,
            "pieces {start}..{end} merge into {tokens} tokens with no boundary a merge cannot \
             cross; reimport with a --range-size of at least {tokens}"
        );
        if current.tokens + tokens > range_size {
            ranges.push(std::mem::replace(&mut current, empty()));
        }
        current.pieces.extend_from_slice(stretch);
        current.tokens += tokens;
        start = end;
    }
    ranges.push(current);
    Ok(ranges)
}

fn merged_len(pieces: &[PreparedPiece], tokenizer: &PromptTokenizer) -> Result<usize> {
    let pieces = pieces
        .iter()
        .map(|piece| BpePiece {
            text: piece.text.clone(),
            segment: piece.segment,
        })
        .collect();
    Ok(inference_kernels::kernels::prompt_prepare::merge_fully(pieces, tokenizer)?.len())
}

/// Ordinary tokenizer repeats per prompt range after the seed batch: enough
/// for the range needing the most merges, and at least one so the template's
/// batch fan always exists.
pub fn range_tokenizer_repeat_count(ranges: &[PromptRange]) -> Result<u32> {
    let merges = ranges
        .iter()
        .map(|range| range.pieces.len() - range.tokens)
        .max()
        .unwrap_or(0);
    let batches = merges.div_ceil(MERGES_PER_STAGE).max(2);
    u32::try_from(batches - 1).context("tokenizer repeat count exceeds u32")
}

/// Where each range ends, as an index into the prompt's pieces.
pub fn range_cuts(ranges: &[PromptRange]) -> Vec<RangeCut> {
    let mut end = 0;
    ranges
        .iter()
        .map(|range| {
            end += range.pieces.len() as u64;
            RangeCut { end }
        })
        .collect()
}

/// Writes each range's cut as `cut_{j}`.
pub fn write_range_cuts(ranges: &[PromptRange], dir: &Path) -> Result<Vec<RunFile>> {
    write_cuts(&range_cuts(ranges), dir)
}

/// Writes these cuts as `cut_{j}`.
pub fn write_cuts(cuts: &[RangeCut], dir: &Path) -> Result<Vec<RunFile>> {
    cuts.iter()
        .enumerate()
        .map(|(index, cut)| write_run_file(cut, dir, &format!("cut_{index}")))
        .collect()
}

/// The most bytes of prompt text in one chunk. `prompt-bind` writes up to one
/// atom per byte, at about 144 draft bytes an atom, and Raster lets a tile write
/// 64 KiB per chunk it consumes. A longer chunk stops the stage under Raster,
/// though not natively.
const PROMPT_CHUNK_BYTES: usize = 256;

/// The original prompt in chunks cut at character boundaries, each at most
/// [`PROMPT_CHUNK_BYTES`]. There is always at least one chunk, so the empty
/// prompt is still committed text.
pub fn prompt_text(prompt: &str) -> PromptText {
    let mut chunks = Vec::new();
    let mut rest = prompt;
    while rest.len() > PROMPT_CHUNK_BYTES {
        let mut end = PROMPT_CHUNK_BYTES;
        while !rest.is_char_boundary(end) {
            end -= 1;
        }
        chunks.push(rest[..end].to_string());
        rest = &rest[end..];
    }
    chunks.push(rest.to_string());
    PromptText {
        chunks: List::from(chunks),
    }
}

/// Writes the original prompt as `text`.
pub fn write_prompt_text(prompt: &str, dir: &Path) -> Result<RunFile> {
    write_run_file(&prompt_text(prompt), dir, "text")
}

/// Writes the run's configuration as `config`.
pub fn write_run_config(config: &RunConfig, dir: &Path) -> Result<RunFile> {
    write_run_file(config, dir, "config")
}

/// A claim's output ids cut into ranges of at most `range_size`.
pub fn claimed_ranges(token_ids: &[u32], range_size: usize) -> Result<Vec<ClaimedOutput>> {
    ensure!(
        !token_ids.is_empty(),
        "a teacher-forced claim needs at least one output token"
    );
    ensure!(range_size >= 1, "range size must be positive");
    Ok(token_ids
        .chunks(range_size)
        .map(|chunk| ClaimedOutput {
            token_ids: List::from(chunk.to_vec()),
        })
        .collect())
}

/// Writes each claimed range as `claimed_{k}`.
pub fn write_claimed_ranges(
    token_ids: &[u32],
    range_size: usize,
    dir: &Path,
) -> Result<Vec<RunFile>> {
    claimed_ranges(token_ids, range_size)?
        .iter()
        .enumerate()
        .map(|(index, range)| write_run_file(range, dir, &format!("claimed_{index}")))
        .collect()
}

/// The commitments `write_claimed_ranges` would bind for these ids.
pub fn claimed_range_commitments(token_ids: &[u32], range_size: usize) -> Result<Vec<String>> {
    claimed_ranges(token_ids, range_size)?
        .iter()
        .map(|range| {
            let (_, _, commitment) =
                raster::encode_raster_value(range).map_err(|error| anyhow::anyhow!("{error}"))?;
            Ok(commitment)
        })
        .collect()
}

/// A claim's output must fill its token budget unless it ends at EOS: the
/// chain never stops early, so a shorter claim would be a truncated answer.
pub fn validate_claimed_length(token_ids: &[u32], tokens: u32, eos_ids: &[u32]) -> Result<()> {
    let ends_at_eos = token_ids.last().is_some_and(|id| eos_ids.contains(id));
    ensure!(
        !token_ids.is_empty() && token_ids.len() <= tokens as usize,
        "claimed output has {} tokens; the run allows 1 to {tokens}",
        token_ids.len()
    );
    ensure!(
        token_ids.len() == tokens as usize || ends_at_eos,
        "claimed output has {} of {tokens} tokens and does not end at EOS",
        token_ids.len()
    );
    ensure!(
        token_ids[..token_ids.len() - 1]
            .iter()
            .all(|id| !eos_ids.contains(id)),
        "claimed output continues past EOS"
    );
    Ok(())
}

pub fn render_gemma_turns(prompt: &str) -> String {
    format!("{PROMPT_PREFIX}{}{PROMPT_SUFFIX}", prompt.trim())
}

pub fn supports_gemma_turns(vocab: &BTreeMap<String, u32>) -> bool {
    [BOS_TOKEN, TURN_OPEN, TURN_CLOSE, NEWLINE_TOKEN]
        .iter()
        .all(|token| vocab.contains_key(*token))
}

/// Merge rounds per `prompt-merge` stage. MUST match its `main`.
pub const MERGES_PER_STAGE: usize = 64;

/// The number of ordinary repeats after the mandatory seed batch.
pub fn tokenizer_repeat_count(piece_count: usize) -> Result<u32> {
    let merges = piece_count.saturating_sub(1);
    let batches = merges.div_ceil(MERGES_PER_STAGE).max(1);
    u32::try_from(batches - 1).context("tokenizer repeat count exceeds u32")
}

/// Splits rendered text the way `prompt-bind` does, by the same code.
/// Unsupported transformations fail explicitly instead of silently approximating.
pub fn split_prompt(prompt: &str, tokenizer: &serde_json::Value) -> Result<Vec<PreparedPiece>> {
    validate_tokenizer_profile(tokenizer)?;
    let vocab = vocab_map(tokenizer_model(tokenizer)?)?;
    let profile = tokenizer_profile(tokenizer)?;
    Ok(
        split_rendered(prompt, &profile, |text| vocab.contains_key(text))
            .into_iter()
            .map(|piece| PreparedPiece {
                text: piece.text,
                segment: piece.segment,
            })
            .collect(),
    )
}

pub fn validate_tokenizer_profile(tokenizer: &serde_json::Value) -> Result<()> {
    let model = tokenizer_model(tokenizer)?;
    ensure!(
        model.get("type").and_then(|v| v.as_str()) == Some("BPE"),
        "only BPE tokenizer models are supported"
    );
    for key in ["truncation", "padding"] {
        ensure!(
            tokenizer.get(key).is_none_or(|v| v.is_null()),
            "unsupported tokenizer {key}"
        );
    }
    ensure!(
        model.get("merges").is_some_and(|v| v.is_array()),
        "BPE merges must be an array"
    );
    ensure!(
        tokenizer
            .get("added_tokens")
            .is_none_or(|v| v.is_null() || v.is_array()),
        "added_tokens must be an array"
    );
    for key in ["continuing_subword_prefix", "end_of_word_suffix"] {
        ensure!(
            model
                .get(key)
                .is_none_or(|v| v.is_null() || v.as_str() == Some("")),
            "unsupported BPE {key}"
        );
    }
    ensure!(
        model
            .get("dropout")
            .is_none_or(|v| v.is_null() || v.as_f64() == Some(0.0)),
        "BPE dropout is unsupported"
    );
    ensure!(
        !model
            .get("ignore_merges")
            .and_then(|v| v.as_bool())
            .unwrap_or(false),
        "ignore_merges is unsupported"
    );
    for (key, expected) in [
        (
            "normalizer",
            serde_json::json!({"type":"Replace","pattern":{"String":" "},"content":"▁"}),
        ),
        (
            "pre_tokenizer",
            serde_json::json!({"type":"Split","pattern":{"String":" "},"behavior":"MergedWithPrevious","invert":false}),
        ),
        (
            "post_processor",
            serde_json::json!({"type":"TemplateProcessing","single":[{"Sequence":{"id":"A","type_id":0}}],"pair":[{"Sequence":{"id":"A","type_id":0}},{"Sequence":{"id":"B","type_id":1}}],"special_tokens":{}}),
        ),
    ] {
        ensure!(
            tokenizer
                .get(key)
                .is_none_or(|v| v.is_null() || *v == expected),
            "unsupported tokenizer {key}"
        );
    }
    let vocab = vocab_map(model)?;
    if let Some(unknown) = model.get("unk_token").filter(|v| !v.is_null()) {
        ensure!(
            unknown.as_str().is_some_and(|s| vocab.contains_key(s)),
            "unknown token absent from vocabulary"
        );
    }
    if let Some(tokens) = tokenizer.get("added_tokens").and_then(|v| v.as_array()) {
        for token in tokens {
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
                    && vocab.get(text).map(|id| *id as u64)
                        == token.get("id").and_then(|v| v.as_u64()),
                "added token disagrees with vocabulary: {text}"
            );
        }
    }
    Ok(())
}

pub fn vocab_map(model: &serde_json::Value) -> Result<BTreeMap<String, u32>> {
    model
        .get("vocab")
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| anyhow::anyhow!("tokenizer.json has no vocab"))?
        .iter()
        .map(|(token, id)| {
            Ok((
                token.clone(),
                id.as_u64()
                    .ok_or_else(|| anyhow::anyhow!("token '{token}' has non-integer id"))?
                    .try_into()
                    .context("token id exceeds u32")?,
            ))
        })
        .collect()
}

pub fn special_tokens(tokenizer: &serde_json::Value) -> Vec<String> {
    let mut tokens: Vec<String> = tokenizer
        .get("added_tokens")
        .and_then(serde_json::Value::as_array)
        .into_iter()
        .flatten()
        .filter(|entry| {
            entry
                .get("special")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false)
        })
        .filter_map(|entry| entry.get("content").and_then(serde_json::Value::as_str))
        .map(str::to_string)
        .collect();
    tokens.sort_by(|a, b| b.len().cmp(&a.len()).then_with(|| a.cmp(b)));
    tokens
}

fn resolve_prompt(spec_dir: &Path, spec: &InferenceRunSpec) -> Result<String> {
    if let Some(prompt) = spec.prompt.as_ref() {
        return Ok(prompt.clone());
    }
    let prompt_file = spec
        .prompt_file
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("run spec must set `prompt` or `prompt_file`"))?;
    let path = if prompt_file.is_absolute() {
        prompt_file.clone()
    } else {
        spec_dir.join(prompt_file)
    };
    fs::read_to_string(&path)
        .with_context(|| format!("failed to read prompt file {}", path.display()))
}

fn tokenizer_model(tokenizer: &serde_json::Value) -> Result<&serde_json::Value> {
    tokenizer
        .get("model")
        .ok_or_else(|| anyhow::anyhow!("tokenizer.json has no model"))
}

fn bucket_vocab(vocab: Vec<TokenEntry>) -> (u32, Vec<VocabBucket>) {
    let count = bucket_count_for(vocab.len());
    let mut buckets: Vec<Vec<TokenEntry>> = vec![Vec::new(); count as usize];
    for entry in vocab {
        buckets[vocab_bucket_of(&entry.token, count) as usize].push(entry);
    }
    (
        count,
        buckets
            .into_iter()
            .map(|entries| VocabBucket {
                entries: List::from(entries),
            })
            .collect(),
    )
}

fn bucket_merges(merges: Vec<BpeMerge>) -> (u32, Vec<MergeBucket>) {
    let count = bucket_count_for(merges.len());
    let mut buckets: Vec<Vec<BpeMerge>> = vec![Vec::new(); count as usize];
    for rule in merges {
        buckets[merge_bucket_of(&rule.left, &rule.right, count) as usize].push(rule);
    }
    for bucket in &mut buckets {
        bucket.sort_by_key(|rule| rule.rank);
    }
    (
        count,
        buckets
            .into_iter()
            .map(|rules| MergeBucket {
                rules: List::from(rules),
            })
            .collect(),
    )
}

fn bucket_boundaries(pairs: BTreeSet<(char, char)>) -> (u32, Vec<BoundaryBucket>) {
    let count = bucket_count_for(pairs.len());
    let mut buckets: Vec<Vec<BoundaryPair>> = vec![Vec::new(); count as usize];
    for (left, right) in pairs {
        let (left, right) = (left as u32, right as u32);
        buckets[boundary_bucket_of(left, right, count) as usize].push(BoundaryPair { left, right });
    }
    (
        count,
        buckets
            .into_iter()
            .map(|pairs| BoundaryBucket {
                pairs: List::from(pairs),
            })
            .collect(),
    )
}

fn bucket_count_for(len: usize) -> u32 {
    (len / 4).max(1) as u32
}

fn parse_merge(rank: u32, entry: &serde_json::Value) -> Option<BpeMerge> {
    let (left, right) = match entry {
        serde_json::Value::String(text) => {
            let mut parts = text.splitn(2, ' ');
            (parts.next()?.to_string(), parts.next()?.to_string())
        }
        serde_json::Value::Array(pair) if pair.len() == 2 => {
            (pair[0].as_str()?.to_string(), pair[1].as_str()?.to_string())
        }
        _ => return None,
    };
    let merged = format!("{left}{right}");
    Some(BpeMerge {
        rank,
        left,
        right,
        merged,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use inference_artifacts::{DirectInferBundle, DirectInferShape};
    use std::path::PathBuf;

    const ONE: i32 = 1 << 16;

    #[test]
    fn gemma_turn_rendering_matches_current_shape() {
        assert_eq!(
            render_gemma_turns(" hello raster "),
            "<bos><|turn>user\nhello raster<turn|>\n<|turn>model\n"
        );
    }

    #[test]
    fn raw_prompt_bypasses_turn_rendering() {
        let tokenizer = gemma_tokenizer();
        let spec = run_spec("h", true);
        let prepared =
            prepare_prompt(&tokenizer, Path::new("."), &spec, &model_manifest()).unwrap();

        assert_eq!(prepared.rendered_prompt, "h");
        assert_eq!(
            prepared
                .initial_pieces
                .iter()
                .map(|p| p.text.as_str())
                .collect::<Vec<_>>(),
            vec!["h"]
        );
    }

    #[test]
    fn special_tokens_stay_whole_in_initial_pieces() {
        let tokenizer = gemma_tokenizer();
        let spec = run_spec("hello", false);
        let prepared =
            prepare_prompt(&tokenizer, Path::new("."), &spec, &model_manifest()).unwrap();

        assert!(prepared.initial_pieces.iter().any(|p| p.text == "<bos>"));
        assert!(prepared.initial_pieces.iter().any(|p| p.text == "<|turn>"));
    }

    #[test]
    fn byte_fallback_emits_hex_pieces_for_unknown_utf8_bytes() {
        let tokenizer = serde_json::json!({
            "model": { "type": "BPE", "vocab": {"<0xC3>": 1, "<0xA9>": 2}, "merges": [], "byte_fallback": true },
            "added_tokens": []
        });
        assert_eq!(
            split_prompt("é", &tokenizer)
                .unwrap()
                .iter()
                .map(|p| p.text.as_str())
                .collect::<Vec<_>>(),
            vec!["<0xC3>", "<0xA9>"]
        );
    }

    #[test]
    fn prompt_ranges_count_tokens_not_pieces() {
        let tokenizer = serde_json::json!({
            "model": { "type": "BPE", "vocab": {"a": 0, "b": 1, "c": 2, "ab": 3}, "merges": ["a b"] },
            "added_tokens": []
        });
        let pieces = "abcabcabc"
            .chars()
            .map(|c| PreparedPiece {
                text: c.to_string(),
                segment: 0,
            })
            .collect::<Vec<_>>();
        let ranges = prompt_ranges(
            &pieces,
            4,
            &vocabulary_pairs(&tokenizer).unwrap(),
            &prompt_tokenizer(&tokenizer).unwrap(),
        )
        .unwrap();
        let texts = ranges
            .iter()
            .map(|range| {
                range
                    .pieces
                    .iter()
                    .map(|p| p.text.as_str())
                    .collect::<String>()
            })
            .collect::<Vec<_>>();
        // "ab" merges, so the first range holds six pieces but four tokens.
        assert_eq!(texts, ["abcabc", "abc"]);
        assert_eq!(ranges.iter().map(|r| r.tokens).collect::<Vec<_>>(), [4, 2]);
        assert_eq!(range_tokenizer_repeat_count(&ranges).unwrap(), 1);
    }

    #[test]
    fn prompt_ranges_reject_an_uncuttable_stretch_over_the_range_size() {
        // "ba" is in the vocabulary, so no boundary in "ababab" is safe.
        let tokenizer = serde_json::json!({
            "model": { "type": "BPE", "vocab": {"a": 0, "b": 1, "ab": 2, "ba": 3}, "merges": ["a b"] },
            "added_tokens": []
        });
        let pieces = "ababab"
            .chars()
            .map(|c| PreparedPiece {
                text: c.to_string(),
                segment: 0,
            })
            .collect::<Vec<_>>();
        let pairs = vocabulary_pairs(&tokenizer).unwrap();
        let prompt = prompt_tokenizer(&tokenizer).unwrap();
        assert!(prompt_ranges(&pieces, 2, &pairs, &prompt).is_err());
        assert_eq!(prompt_ranges(&pieces, 3, &pairs, &prompt).unwrap().len(), 1);
    }

    #[test]
    fn range_cuts_are_cumulative_piece_counts() {
        let range = |pieces: usize| PromptRange {
            pieces: vec![
                PreparedPiece {
                    text: String::from("a"),
                    segment: 0,
                };
                pieces
            ],
            tokens: 1,
        };
        let ends = range_cuts(&[range(3), range(1), range(4)])
            .iter()
            .map(|cut| cut.end)
            .collect::<Vec<_>>();
        assert_eq!(ends, [3, 4, 8]);
    }

    #[test]
    fn prompt_text_cuts_at_character_boundaries_and_keeps_every_byte() {
        let empty = prompt_text("");
        assert_eq!(empty.chunks.iter().collect::<Vec<_>>(), [""]);
        // A two-byte character straddles every chunk limit.
        let text = "é".repeat(PROMPT_CHUNK_BYTES);
        let chunks = prompt_text(&text);
        assert!(chunks.chunks.len() > 1);
        assert!(chunks
            .chunks
            .iter()
            .all(|chunk| chunk.len() <= PROMPT_CHUNK_BYTES));
        assert_eq!(chunks.chunks.iter().cloned().collect::<String>(), text);
    }

    #[test]
    fn the_bind_stage_splits_the_prompt_as_run_preparation_does() {
        use inference_kernels::kernels::prompt_prepare::run_prompt_bind;
        let tokenizer = serde_json::json!({
            "model": {
                "type": "BPE",
                "vocab": {
                    "<bos>": 0, "<|turn>": 1, "<turn|>": 2, "\n": 3, "u": 4, "s": 5, "e": 6,
                    "r": 7, "m": 8, "o": 9, "d": 10, "l": 11, "h": 12, "▁": 13, "<0xC3>": 14,
                    "<0xA9>": 15, "<unk>": 16
                },
                "merges": [],
                "byte_fallback": true,
                "unk_token": "<unk>",
                "fuse_unk": true
            },
            "normalizer": {"type":"Replace","pattern":{"String":" "},"content":"▁"},
            "added_tokens": [
                { "id": 0, "content": "<bos>", "special": true },
                { "id": 1, "content": "<|turn>", "special": true },
                { "id": 2, "content": "<turn|>", "special": true }
            ]
        });
        let prompt_tokenizer = prompt_tokenizer(&tokenizer).unwrap();
        for (prompt, raw_prompt) in [
            ("  he é<turn|>hold\u{4e16}\u{754c}  ", false),
            ("  he é<turn|>hold\u{4e16}\u{754c}  ", true),
            ("   ", false),
            ("", false),
            ("", true),
        ] {
            let spec = run_spec(prompt, raw_prompt);
            let prepared =
                prepare_prompt(&tokenizer, Path::new("."), &spec, &model_manifest()).unwrap();
            let bound = run_prompt_bind(
                &prompt_tokenizer,
                &prompt_text(prompt),
                &RunConfig {
                    raw_prompt,
                    ..RunConfig::default()
                },
            )
            .unwrap();
            let pieces = bound
                .pieces
                .iter()
                .map(|piece| PreparedPiece {
                    text: piece.text.clone(),
                    segment: piece.segment,
                })
                .collect::<Vec<_>>();
            assert_eq!(
                pieces, prepared.initial_pieces,
                "{prompt:?} raw {raw_prompt}"
            );
            assert_eq!(bound.context.fault, 0);
        }
    }

    #[test]
    fn prompt_seed_records_a_bad_cut_and_still_finishes() {
        use inference_kernels::kernels::prompt_prepare::{run_prompt_bind, run_prompt_seed};
        use prompt_prepare::input::{FAULT_CUT_BEYOND_TEXT, FAULT_EMPTY_RANGE, FAULT_UNSAFE_CUT};
        // "ab" is a vocabulary token, so a cut between a and b could be merged across.
        let tokenizer = serde_json::json!({
            "model": { "type": "BPE", "vocab": {"a": 0, "b": 1, "c": 2, "ab": 3}, "merges": ["a b"] },
            "added_tokens": []
        });
        let tokenizer = prompt_tokenizer(&tokenizer).unwrap();
        let config = RunConfig {
            raw_prompt: true,
            ..RunConfig::default()
        };
        let bound = run_prompt_bind(&tokenizer, &prompt_text("abcab"), &config).unwrap();
        let seed = |previous: &prompt_prepare::input::PromptContext, end| {
            run_prompt_seed(&tokenizer, &bound, previous, &RangeCut { end }).unwrap()
        };

        let first = seed(&bound.context, 3);
        assert_eq!(first.context.fault, 0);
        assert_eq!(
            first
                .pieces
                .iter()
                .map(|p| p.text.as_str())
                .collect::<Vec<_>>(),
            ["ab", "c"]
        );
        assert!(!first.context.complete);
        let last = seed(&first.context, 5);
        assert_eq!(last.context.fault, 0);
        assert!(last.context.complete);

        assert_eq!(seed(&bound.context, 1).context.fault, FAULT_UNSAFE_CUT);
        assert_eq!(seed(&first.context, 3).context.fault, FAULT_EMPTY_RANGE);
        assert_eq!(seed(&first.context, 9).context.fault, FAULT_CUT_BEYOND_TEXT);
    }

    #[test]
    fn the_context_binds_the_original_text_and_says_when_a_range_is_left_out() {
        use inference_kernels::kernels::prompt_prepare::{run_prompt_bind, run_prompt_seed};
        use sha2::{Digest, Sha256};
        let tokenizer = serde_json::json!({
            "model": { "type": "BPE", "vocab": {"a": 0, "b": 1, "c": 2}, "merges": [] },
            "added_tokens": []
        });
        let tokenizer = prompt_tokenizer(&tokenizer).unwrap();
        let config = RunConfig {
            raw_prompt: true,
            model_id: String::from("model"),
            ..RunConfig::default()
        };
        let bind = |text: &str| run_prompt_bind(&tokenizer, &prompt_text(text), &config).unwrap();
        let bound = bind("abc");
        assert_eq!(
            bound.context.input_text_sha256,
            format!("{:x}", Sha256::digest(b"abc"))
        );
        assert_ne!(
            bound.context.input_text_sha256,
            bind("abd").context.input_text_sha256
        );
        assert_eq!(bound.context.model_id, "model");

        // Stopping early covers less than the whole prompt.
        let early =
            run_prompt_seed(&tokenizer, &bound, &bound.context, &RangeCut { end: 2 }).unwrap();
        assert_eq!(early.context.fault, 0);
        assert!(!early.context.complete);
        // A range cannot begin anywhere but where the last one ended.
        let next =
            run_prompt_seed(&tokenizer, &bound, &early.context, &RangeCut { end: 3 }).unwrap();
        assert_eq!(
            next.pieces
                .iter()
                .map(|p| p.text.as_str())
                .collect::<Vec<_>>(),
            ["c"]
        );
        assert!(next.context.complete);
    }

    #[test]
    fn prompt_prepare_records_unfinished_merges_and_still_finishes() {
        use inference_kernels::kernels::prompt_prepare::{
            finalize_prompt, PromptPrepareDirectInputs,
        };
        use prompt_prepare::input::FAULT_MERGES_INCOMPLETE;
        let tokenizer = serde_json::json!({
            "model": { "type": "BPE", "vocab": {"a": 0, "b": 1, "ab": 2}, "merges": ["a b"] },
            "added_tokens": []
        });
        let tokenizer = prompt_tokenizer(&tokenizer).unwrap();
        let pieces = ["a", "b"]
            .map(|text| BpePiece {
                text: text.to_string(),
                segment: 0,
            })
            .to_vec();
        let output = finalize_prompt(PromptPrepareDirectInputs {
            tokenizer: &tokenizer,
            initial_pieces: &BpePieces::of_pieces(pieces),
        })
        .unwrap();
        assert_eq!(output.context.fault, FAULT_MERGES_INCOMPLETE);
        assert_eq!(output.token_ids.iter().copied().collect::<Vec<_>>(), [0, 1]);
    }

    fn run_spec(prompt: &str, raw_prompt: bool) -> InferenceRunSpec {
        InferenceRunSpec {
            model_manifest: PathBuf::from("model-artifacts/manifest.json"),
            prompt: Some(prompt.to_string()),
            prompt_file: None,
            raw_prompt,
            tokens: 1,
        }
    }

    fn model_manifest() -> ModelManifest {
        ModelManifest {
            version: inference_artifacts::MODEL_MANIFEST_VERSION,
            model_id: inference_artifacts::model_id_of(&[]),
            model_commitments: Vec::new(),
            range_size: 512,
            bundle: DirectInferBundle {
                model_detwgt_path: PathBuf::from("model.detwgt"),
                model_detwgt_sha256: String::new(),
                config_path: PathBuf::from("config.json"),
                config_sha256: String::new(),
                tokenizer_path: PathBuf::from("tokenizer.json"),
                tokenizer_sha256: String::new(),
            },
            shape: DirectInferShape {
                hidden_size: 1,
                num_hidden_layers: 0,
                num_attention_heads: 1,
                num_key_value_heads: 1,
                head_dim: 1,
                global_head_dim: 1,
                vocab_size: 1,
                hidden_size_per_layer_input: 1,
                sliding_window: 0,
                layer_types: Vec::new(),
                num_kv_shared_layers: 0,
                norm_eps: 0,
                rope_base_sliding: 10_000_i64 << 32,
                rope_base_full: 1_000_000_i64 << 32,
                full_partial_rotary_factor_q16: ONE,
                embedding_scale: ONE,
                ple_embedding_scale: ONE,
                ple_projection_scalar: ONE,
                ple_input_scale: ONE,
                final_logit_softcap: 0,
            },
            eos_token_ids: vec![1, 2],
            provenance: None,
        }
    }

    fn gemma_tokenizer() -> serde_json::Value {
        serde_json::json!({
            "model": {
                "type": "BPE",
                "vocab": {
                    "<bos>": 0,
                    "<|turn>": 1,
                    "<turn|>": 2,
                    "\n": 3,
                    "hello": 4,
                    "user": 5,
                    "model": 6,
                    "h": 7, "e": 8, "l": 9, "o": 10, "u": 11, "s": 12, "r": 13, "m": 14, "d": 15
                },
                "merges": []
            },
            "added_tokens": [
                { "id": 0, "content": "<bos>", "special": true },
                { "id": 1, "content": "<|turn>", "special": true },
                { "id": 2, "content": "<turn|>", "special": true }
            ]
        })
    }
}
