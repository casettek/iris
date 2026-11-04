use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

use anyhow::{ensure, Context, Result};
use inference_artifacts::{InferenceRunSpec, ModelManifest, PreparedPiece, PreparedPrompt};
use output_finalize::input::{DecoderTable, DecoderToken};
use prompt_prepare::input::{merge_bucket_of, vocab_bucket_of, BpeMerge, MergeBucket};
use prompt_prepare::input::{BpePiece, BpePieces, PromptTokenizer, TokenEntry, VocabBucket};
use raster::List;
use input_embedding::input::ClaimedOutput;

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
    Ok(PromptTokenizer {
        vocab_bucket_count,
        merge_bucket_count,
        vocab_buckets: List::from(vocab_buckets),
        merge_buckets: List::from(merge_buckets),
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

pub fn write_initial_pieces(
    prompt: &PreparedPrompt,
    dir: &Path,
) -> Result<(std::path::PathBuf, std::path::PathBuf, String)> {
    fs::create_dir_all(dir).with_context(|| format!("failed to create {}", dir.display()))?;
    let pieces = BpePieces {
        pieces: List::from(
            prompt
                .initial_pieces
                .iter()
                .map(|p| BpePiece {
                    text: p.text.clone(),
                    segment: p.segment,
                })
                .collect::<Vec<_>>(),
        ),
    };
    let data_path = dir.join("initial_pieces.rastered");
    let index_path = dir.join("initial_pieces.rindex");
    let commitment = raster::write_raster_files(&pieces, &data_path, &index_path)
        .map_err(|error| anyhow::anyhow!("{error}"))?;
    Ok((data_path, index_path, commitment))
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

/// Cuts the prompt's pieces into ranges of at most `range_size` pieces, each
/// ending at the last boundary no merge can cross.
///
/// Fails rather than overflow a range: a range's tokenizer batches are sized
/// for `range_size` pieces, so a longer one would not finish merging.
pub fn prompt_ranges(
    pieces: &[PreparedPiece],
    range_size: usize,
    pairs: &BTreeSet<(char, char)>,
) -> Result<Vec<Vec<PreparedPiece>>> {
    ensure!(!pieces.is_empty(), "a teacher-forced claim needs a non-empty prompt");
    ensure!(range_size >= 2, "range size must be at least 2");
    let mut ranges = Vec::new();
    let mut start = 0;
    while start < pieces.len() {
        let limit = (start + range_size).min(pieces.len());
        let end = if limit == pieces.len() {
            limit
        } else {
            (start + 1..=limit)
                .rev()
                .find(|&cut| can_cut(&pieces[cut - 1], &pieces[cut], pairs))
                .with_context(|| {
                    format!(
                        "pieces {start}..{limit} contain no boundary a merge cannot cross; \
                         reimport with a larger --range-size"
                    )
                })?
        };
        ranges.push(pieces[start..end].to_vec());
        start = end;
    }
    Ok(ranges)
}

fn range_pieces(range: &[PreparedPiece]) -> BpePieces {
    BpePieces {
        pieces: List::from(
            range
                .iter()
                .map(|piece| BpePiece {
                    text: piece.text.clone(),
                    segment: piece.segment,
                })
                .collect::<Vec<_>>(),
        ),
    }
}

/// The commitments `write_prompt_ranges` would bind for these ranges.
pub fn prompt_range_commitments(ranges: &[Vec<PreparedPiece>]) -> Result<Vec<String>> {
    ranges
        .iter()
        .map(|range| {
            let (_, _, commitment) = raster::encode_raster_value(&range_pieces(range))
                .map_err(|error| anyhow::anyhow!("{error}"))?;
            Ok(commitment)
        })
        .collect()
}

/// Writes each prompt range's pieces as `pieces_{j}`.
pub fn write_prompt_ranges(ranges: &[Vec<PreparedPiece>], dir: &Path) -> Result<Vec<RunFile>> {
    ranges
        .iter()
        .enumerate()
        .map(|(index, range)| write_run_file(&range_pieces(range), dir, &format!("pieces_{index}")))
        .collect()
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
pub fn write_claimed_ranges(token_ids: &[u32], range_size: usize, dir: &Path) -> Result<Vec<RunFile>> {
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
    format!(
        "{BOS_TOKEN}{TURN_OPEN}user\n{}{TURN_CLOSE}\n{TURN_OPEN}model\n",
        prompt.trim()
    )
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

/// Supported tokenizer profile: Gemma's literal space normalization/splitting,
/// atomic added tokens, character pieces, byte fallback and fused unknowns.
/// Unsupported transformations fail explicitly instead of silently approximating.
pub fn split_prompt(prompt: &str, tokenizer: &serde_json::Value) -> Result<Vec<PreparedPiece>> {
    let model = tokenizer_model(tokenizer)?;
    validate_tokenizer_profile(tokenizer)?;
    let vocab = vocab_map(model)?;
    let specials = special_tokens(tokenizer);
    let normalize_spaces = !tokenizer
        .get("normalizer")
        .unwrap_or(&serde_json::Value::Null)
        .is_null();
    let split_spaces = !tokenizer
        .get("pre_tokenizer")
        .unwrap_or(&serde_json::Value::Null)
        .is_null();
    let byte_fallback = model
        .get("byte_fallback")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let fuse_unk = model
        .get("fuse_unk")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let unk = model.get("unk_token").and_then(|v| v.as_str());
    let mut pieces = Vec::<PreparedPiece>::new();
    let mut rest = prompt;
    let mut segment = 0_u32;
    let mut previous_unknown = false;
    while !rest.is_empty() {
        if let Some(special) = specials
            .iter()
            .find(|token| rest.starts_with(token.as_str()))
        {
            segment = segment.checked_add(1).context("piece segment overflow")?;
            pieces.push(PreparedPiece {
                text: special.clone(),
                segment,
            });
            segment = segment.checked_add(1).context("piece segment overflow")?;
            rest = &rest[special.len()..];
            previous_unknown = false;
            continue;
        }
        let ch = rest.chars().next().expect("nonempty prompt");
        rest = &rest[ch.len_utf8()..];
        let text = if normalize_spaces && ch == ' ' {
            "▁".to_string()
        } else {
            ch.to_string()
        };
        if vocab.contains_key(&text) {
            pieces.push(PreparedPiece {
                text: text.clone(),
                segment,
            });
            previous_unknown = false;
        } else {
            let bytes = text
                .as_bytes()
                .iter()
                .map(|b| format!("<0x{b:02X}>"))
                .collect::<Vec<_>>();
            if byte_fallback && bytes.iter().all(|s| vocab.contains_key(s)) {
                pieces.extend(
                    bytes
                        .into_iter()
                        .map(|text| PreparedPiece { text, segment }),
                );
                previous_unknown = false;
            } else if let Some(unknown) = unk {
                if !fuse_unk || !previous_unknown {
                    pieces.push(PreparedPiece {
                        text: unknown.to_string(),
                        segment,
                    });
                }
                previous_unknown = true;
            } else {
                // BPE without an unknown token drops characters outside its
                // vocabulary when byte fallback cannot represent them.
                previous_unknown = false;
            }
        }
        // Split(MergedWithPrevious) runs after normalization. In production
        // spaces have become ▁, so it introduces no new boundaries.
        if split_spaces && text == " " {
            segment = segment.checked_add(1).context("piece segment overflow")?;
            previous_unknown = false;
        }
    }
    Ok(pieces)
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

    fn run_spec(prompt: &str, raw_prompt: bool) -> InferenceRunSpec {
        InferenceRunSpec {
            model_manifest: PathBuf::from("model-artifacts/manifest.json"),
            prompt: Some(prompt.to_string()),
            prompt_file: None,
            raw_prompt,
            tokens: 1,
            chain: Default::default(),
        }
    }

    fn model_manifest() -> ModelManifest {
        ModelManifest {
            version: 2,
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
