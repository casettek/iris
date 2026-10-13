use std::any::Any;
use std::collections::{BTreeMap, VecDeque};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::{bail, Context, Result};
pub use inference_kernels::MaterializationCacheKey;
use raster::List;
use serde::Deserialize;

use decode_embed::input as decode_embed_input;
use decode_select_token::input as decode_input;
use input_embedding::input as embedding_input;
use output_finalize::input as output_input;
use prefill_finalize::input as finalize_input;
use prefill_prepare_aux::input as aux_input;
use prefill_range::input as range_input;
use prompt_prepare::input as prompt_input;

#[derive(Clone, Debug)]
pub enum CachedStageValue {
    PromptTokenization(prompt_input::PromptTokenization),
    BpePieces(prompt_input::BpePieces),
    EmbeddedActivations(embedding_input::ActivationSequence),
    PleLayerInputs(aux_input::PleLayerInputs),
    RangeActivations(range_input::ActivationSequence),
    PrefillLogits(finalize_input::PrefillLogits),
    DecodeEdge(decode_input::DecodeEdge),
    DecodeActivations(decode_embed_input::ActivationSequence),
    GeneratedOutput(output_input::GeneratedOutput),
}

#[derive(Clone, Debug)]
pub struct CachedStageOutput {
    pub structural_commitment: Vec<u8>,
    pub value: Arc<CachedStageValue>,
}

#[derive(Default)]
pub struct StageOutputCache {
    outputs: BTreeMap<String, CachedStageOutput>,
}

pub type CachedInputs = BTreeMap<String, Arc<CachedStageValue>>;

#[derive(Debug, Deserialize)]
struct InputDocumentEntry {
    path: PathBuf,
    #[serde(default)]
    index_path: Option<PathBuf>,
}

#[derive(Debug, Deserialize)]
struct InputManifestEntry {
    commitment: String,
}

pub struct MaterializationCache {
    state: Mutex<MaterializationCacheState>,
    capacity: usize,
}

#[derive(Default)]
struct MaterializationCacheState {
    entries: BTreeMap<MaterializationCacheKey, Arc<dyn Any + Send + Sync>>,
    order: VecDeque<MaterializationCacheKey>,
}

impl Default for MaterializationCache {
    fn default() -> Self {
        let capacity = std::env::var("STAGED_INFER_MATERIALIZATION_CACHE_ENTRIES")
            .or_else(|_| std::env::var("DIRECT_NATIVE_MATERIALIZATION_CACHE_ENTRIES"))
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(2);
        Self {
            state: Mutex::new(MaterializationCacheState::default()),
            capacity,
        }
    }
}

impl MaterializationCache {
    pub fn get_or_insert_with<T>(
        &self,
        key: Option<MaterializationCacheKey>,
        load: impl FnOnce() -> T,
    ) -> Result<Arc<T>>
    where
        T: Send + Sync + 'static,
    {
        let Some(key) = key else {
            return Ok(Arc::new(load()));
        };

        if self.capacity == 0 {
            return Ok(Arc::new(load()));
        }

        if let Some(existing) = self.state.lock().unwrap().entries.get(&key).cloned() {
            return existing.downcast::<T>().map_err(|_| {
                anyhow::anyhow!(
                    "cached materialization `{}` for `{}` had the wrong type",
                    key.param,
                    key.type_name
                )
            });
        }

        let value: Arc<T> = Arc::new(load());
        let mut state = self.state.lock().unwrap();
        if let Some(existing) = state.entries.get(&key).cloned() {
            return existing.downcast::<T>().map_err(|_| {
                anyhow::anyhow!(
                    "cached materialization `{}` for `{}` had the wrong type",
                    key.param,
                    key.type_name
                )
            });
        }
        if state.entries.len() >= self.capacity {
            if let Some(evicted) = state.order.pop_front() {
                state.entries.remove(&evicted);
            }
        }
        state.order.push_back(key.clone());
        state.entries.insert(key, value.clone());
        Ok(value)
    }
}

pub fn materialize_with_cache<T>(
    cache: Option<&MaterializationCache>,
    key: Option<MaterializationCacheKey>,
    load: impl FnOnce() -> T,
) -> Result<Arc<T>>
where
    T: Send + Sync + 'static,
{
    match cache {
        Some(cache) => cache.get_or_insert_with(key, load),
        None => Ok(Arc::new(load())),
    }
}

pub fn materialization_key_from_stage_files<T>(
    input_path: &Path,
    input_manifest_path: &Path,
    param: &str,
) -> Result<Option<MaterializationCacheKey>>
where
    T: 'static,
{
    let input_entries = read_input_entries(input_path)?;
    let manifest_entries = read_manifest_entries(input_manifest_path)?;
    let Some(input) = input_entries.get(param) else {
        return Ok(None);
    };
    let Some(manifest) = manifest_entries.get(param) else {
        return Ok(None);
    };
    Ok(Some(MaterializationCacheKey {
        param: param.to_string(),
        path: input.path.clone(),
        index_path: input
            .index_path
            .clone()
            .unwrap_or_else(|| input.path.with_extension("rindex")),
        commitment: manifest.commitment.clone(),
        type_name: std::any::type_name::<T>(),
    }))
}

fn read_input_entries(path: &Path) -> Result<BTreeMap<String, InputDocumentEntry>> {
    let bytes = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
    serde_json::from_slice(&bytes).with_context(|| format!("failed to decode {}", path.display()))
}

fn read_manifest_entries(path: &Path) -> Result<BTreeMap<String, InputManifestEntry>> {
    let bytes = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
    serde_json::from_slice(&bytes).with_context(|| format!("failed to decode {}", path.display()))
}

impl StageOutputCache {
    pub fn insert(
        &mut self,
        stage: impl Into<String>,
        structural_commitment: Vec<u8>,
        value: CachedStageValue,
    ) -> Arc<CachedStageValue> {
        let value = Arc::new(value);
        self.outputs.insert(
            stage.into(),
            CachedStageOutput {
                structural_commitment,
                value: Arc::clone(&value),
            },
        );
        value
    }

    pub fn get(
        &self,
        stage: &str,
        expected_structural_commitment: &[u8],
    ) -> Result<Option<Arc<CachedStageValue>>> {
        let Some(output) = self.outputs.get(stage) else {
            return Ok(None);
        };
        if output.structural_commitment != expected_structural_commitment {
            bail!("cached staged-infer output for stage `{stage}` has stale commitment");
        }
        Ok(Some(Arc::clone(&output.value)))
    }
}

impl CachedStageValue {
    pub fn as_bpe_pieces(&self) -> Result<prompt_input::BpePieces> {
        match self {
            Self::BpePieces(value) => Ok(value.clone()),
            _ => bail!("cached value is not BPE pieces"),
        }
    }

    pub fn as_embedding_prompt(&self) -> Result<embedding_input::PromptTokenization> {
        match self {
            Self::PromptTokenization(value) => Ok(embedding_input::PromptTokenization {
                token_ids: value.token_ids.clone(),
            }),
            _ => bail!("cached value is not a prompt tokenization"),
        }
    }

    pub fn as_aux_activation_sequence(&self) -> Result<aux_input::ActivationSequence> {
        match self {
            Self::EmbeddedActivations(value) => Ok(aux_activation_from_embedding(value)),
            Self::DecodeActivations(value) => Ok(aux_activation_from_decode_embed(value)),
            Self::RangeActivations(value) => Ok(aux_activation_from_range(value)),
            _ => bail!("cached value is not an activation sequence"),
        }
    }

    pub fn as_range_activation_sequence(&self) -> Result<range_input::ActivationSequence> {
        match self {
            Self::EmbeddedActivations(value) => Ok(range_activation_from_embedding(value)),
            Self::DecodeActivations(value) => Ok(range_activation_from_decode_embed(value)),
            Self::RangeActivations(value) => Ok(value.clone()),
            _ => bail!("cached value is not an activation sequence"),
        }
    }

    pub fn as_range_ple_inputs(&self) -> Result<range_input::PleLayerInputs> {
        match self {
            Self::PleLayerInputs(value) => Ok(range_ple_from_aux(value)),
            _ => bail!("cached value is not PLE layer inputs"),
        }
    }

    pub fn as_finalize_activation_sequence(&self) -> Result<finalize_input::ActivationSequence> {
        match self {
            Self::EmbeddedActivations(value) => Ok(finalize_activation_from_embedding(value)),
            Self::DecodeActivations(value) => Ok(finalize_activation_from_decode_embed(value)),
            Self::RangeActivations(value) => Ok(finalize_activation_from_range(value)),
            _ => bail!("cached value is not an activation sequence"),
        }
    }

    pub fn as_decode_logits(&self) -> Result<decode_input::PrefillLogits> {
        match self {
            Self::PrefillLogits(value) => Ok(decode_logits_from_finalize(value)),
            _ => bail!("cached value is not prefill logits"),
        }
    }

    pub fn as_decode_edge(&self) -> Result<decode_input::DecodeEdge> {
        match self {
            Self::DecodeEdge(value) => Ok(value.clone()),
            _ => bail!("cached value is not a decode edge"),
        }
    }

    pub fn as_decode_embed_edge(&self) -> Result<decode_embed_input::DecodeEdge> {
        match self {
            Self::DecodeEdge(value) => Ok(decode_embed_edge_from_decode(value)),
            _ => bail!("cached value is not a decode edge"),
        }
    }

    pub fn as_output_edge(&self) -> Result<output_input::DecodeEdge> {
        match self {
            Self::DecodeEdge(value) => Ok(output_edge_from_decode(value)),
            _ => bail!("cached value is not a decode edge"),
        }
    }
}

fn map_list<T, U>(source: &List<T>, mut convert: impl FnMut(&T) -> U) -> List<U> {
    List::from(
        source
            .iter()
            .map(|value| convert(value))
            .collect::<Vec<_>>(),
    )
}

fn aux_activation_from_embedding(
    source: &embedding_input::ActivationSequence,
) -> aux_input::ActivationSequence {
    aux_input::ActivationSequence {
        rows: map_list(&source.rows, |row| aux_input::ActivationRow {
            token_id: row.token_id,
            values: row.values.clone(),
        }),
        errors: source.errors.clone(),
        kv: map_list(&source.kv, |row| aux_input::KeyRow {
            position: row.position,
            k: row.k.clone(),
            v: row.v.clone(),
        }),
        start_position: source.start_position,
    }
}

fn aux_activation_from_range(
    source: &range_input::ActivationSequence,
) -> aux_input::ActivationSequence {
    aux_input::ActivationSequence {
        rows: map_list(&source.rows, |row| aux_input::ActivationRow {
            token_id: row.token_id,
            values: row.values.clone(),
        }),
        errors: source.errors.clone(),
        kv: map_list(&source.kv, |row| aux_input::KeyRow {
            position: row.position,
            k: row.k.clone(),
            v: row.v.clone(),
        }),
        start_position: source.start_position,
    }
}

fn aux_activation_from_decode_embed(
    source: &decode_embed_input::ActivationSequence,
) -> aux_input::ActivationSequence {
    aux_input::ActivationSequence {
        rows: map_list(&source.rows, |row| aux_input::ActivationRow {
            token_id: row.token_id,
            values: row.values.clone(),
        }),
        errors: source.errors.clone(),
        kv: map_list(&source.kv, |row| aux_input::KeyRow {
            position: row.position,
            k: row.k.clone(),
            v: row.v.clone(),
        }),
        start_position: source.start_position,
    }
}

fn range_activation_from_embedding(
    source: &embedding_input::ActivationSequence,
) -> range_input::ActivationSequence {
    range_input::ActivationSequence {
        rows: map_list(&source.rows, |row| range_input::ActivationRow {
            token_id: row.token_id,
            values: row.values.clone(),
        }),
        errors: source.errors.clone(),
        kv: map_list(&source.kv, |row| range_input::KeyRow {
            position: row.position,
            k: row.k.clone(),
            v: row.v.clone(),
        }),
        start_position: source.start_position,
    }
}

fn range_activation_from_decode_embed(
    source: &decode_embed_input::ActivationSequence,
) -> range_input::ActivationSequence {
    range_input::ActivationSequence {
        rows: map_list(&source.rows, |row| range_input::ActivationRow {
            token_id: row.token_id,
            values: row.values.clone(),
        }),
        errors: source.errors.clone(),
        kv: map_list(&source.kv, |row| range_input::KeyRow {
            position: row.position,
            k: row.k.clone(),
            v: row.v.clone(),
        }),
        start_position: source.start_position,
    }
}

fn finalize_activation_from_embedding(
    source: &embedding_input::ActivationSequence,
) -> finalize_input::ActivationSequence {
    finalize_input::ActivationSequence {
        rows: map_list(&source.rows, |row| finalize_input::ActivationRow {
            token_id: row.token_id,
            values: row.values.clone(),
        }),
        errors: source.errors.clone(),
        kv: map_list(&source.kv, |row| finalize_input::KeyRow {
            position: row.position,
            k: row.k.clone(),
            v: row.v.clone(),
        }),
        start_position: source.start_position,
    }
}

fn finalize_activation_from_decode_embed(
    source: &decode_embed_input::ActivationSequence,
) -> finalize_input::ActivationSequence {
    finalize_input::ActivationSequence {
        rows: map_list(&source.rows, |row| finalize_input::ActivationRow {
            token_id: row.token_id,
            values: row.values.clone(),
        }),
        errors: source.errors.clone(),
        kv: map_list(&source.kv, |row| finalize_input::KeyRow {
            position: row.position,
            k: row.k.clone(),
            v: row.v.clone(),
        }),
        start_position: source.start_position,
    }
}

fn finalize_activation_from_range(
    source: &range_input::ActivationSequence,
) -> finalize_input::ActivationSequence {
    finalize_input::ActivationSequence {
        rows: map_list(&source.rows, |row| finalize_input::ActivationRow {
            token_id: row.token_id,
            values: row.values.clone(),
        }),
        errors: source.errors.clone(),
        kv: map_list(&source.kv, |row| finalize_input::KeyRow {
            position: row.position,
            k: row.k.clone(),
            v: row.v.clone(),
        }),
        start_position: source.start_position,
    }
}

fn range_ple_from_aux(source: &aux_input::PleLayerInputs) -> range_input::PleLayerInputs {
    range_input::PleLayerInputs {
        layer_idx: source.layer_idx,
        rows: map_list(&source.rows, |row| range_input::PleRow {
            values: row.values.clone(),
        }),
        errors: source.errors.clone(),
    }
}

fn decode_logits_from_finalize(
    source: &finalize_input::PrefillLogits,
) -> decode_input::PrefillLogits {
    decode_input::PrefillLogits {
        decode_position: source.decode_position,
        logits: map_list(&source.logits, |entry| decode_input::LogitEntry {
            token_id: entry.token_id,
            value: entry.value,
        }),
        errors: source.errors.clone(),
    }
}

fn decode_embed_edge_from_decode(
    source: &decode_input::DecodeEdge,
) -> decode_embed_input::DecodeEdge {
    decode_embed_input::DecodeEdge {
        has_selected: source.has_selected,
        decode_position: source.decode_position,
        token_id: source.token_id,
        value: source.value,
        generated_token_ids: source.generated_token_ids.clone(),
    }
}

fn output_edge_from_decode(source: &decode_input::DecodeEdge) -> output_input::DecodeEdge {
    output_input::DecodeEdge {
        has_selected: source.has_selected,
        decode_position: source.decode_position,
        token_id: source.token_id,
        value: source.value,
        generated_token_ids: source.generated_token_ids.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn materialization_cache_reuses_matching_key() {
        let cache = MaterializationCache::default();
        let key = MaterializationCacheKey {
            param: String::from("embedding"),
            path: PathBuf::from("embedding.rastered"),
            index_path: PathBuf::from("embedding.rindex"),
            commitment: String::from("abc"),
            type_name: std::any::type_name::<String>(),
        };

        let first =
            materialize_with_cache(Some(&cache), Some(key.clone()), || String::from("loaded"))
                .unwrap();
        let second =
            materialize_with_cache(Some(&cache), Some(key), || String::from("reloaded")).unwrap();

        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!(second.as_str(), "loaded");
    }

    #[test]
    fn materialization_cache_misses_when_commitment_changes() {
        let cache = MaterializationCache::default();
        let base_key = MaterializationCacheKey {
            param: String::from("embedding"),
            path: PathBuf::from("embedding.rastered"),
            index_path: PathBuf::from("embedding.rindex"),
            commitment: String::from("abc"),
            type_name: std::any::type_name::<String>(),
        };
        let changed_key = MaterializationCacheKey {
            commitment: String::from("def"),
            ..base_key.clone()
        };

        let first =
            materialize_with_cache(Some(&cache), Some(base_key), || String::from("first")).unwrap();
        let second =
            materialize_with_cache(Some(&cache), Some(changed_key), || String::from("second"))
                .unwrap();

        assert!(!Arc::ptr_eq(&first, &second));
        assert_eq!(first.as_str(), "first");
        assert_eq!(second.as_str(), "second");
    }

    #[test]
    fn materialization_key_uses_stage_files_and_type_name() {
        let base = temp_dir("materialization-key");
        fs::create_dir_all(&base).unwrap();
        let input = base.join("input.json");
        let manifest = base.join("input_manifest.json");
        fs::write(
            &input,
            r#"{"embedding":{"path":"/tmp/embedding.rastered","index_path":"/tmp/embedding.rindex","load_preference":"mmap"}}"#,
        )
        .unwrap();
        fs::write(
            &manifest,
            r#"{"embedding":{"type":"sha256","encoding":"raster","commitment":"abc"}}"#,
        )
        .unwrap();

        let key = materialization_key_from_stage_files::<String>(&input, &manifest, "embedding")
            .unwrap()
            .unwrap();

        assert_eq!(key.param, "embedding");
        assert_eq!(key.path, PathBuf::from("/tmp/embedding.rastered"));
        assert_eq!(key.index_path, PathBuf::from("/tmp/embedding.rindex"));
        assert_eq!(key.commitment, "abc");
        assert_eq!(key.type_name, std::any::type_name::<String>());

        fs::remove_dir_all(base).unwrap();
    }

    #[test]
    fn cache_rejects_stale_commitment() {
        let mut cache = StageOutputCache::default();
        cache.insert(
            "producer",
            vec![0xde, 0xad],
            CachedStageValue::PromptTokenization(prompt_input::PromptTokenization {
                token_ids: List::from(vec![1, 2, 3]),
            }),
        );

        let error = cache.get("producer", &[0xbe, 0xef]).unwrap_err();

        assert!(error.to_string().contains("stale commitment"));
    }

    #[test]
    fn stage_output_cache_returns_shared_values() {
        let mut cache = StageOutputCache::default();
        cache.insert(
            "producer",
            vec![0xde, 0xad],
            CachedStageValue::PromptTokenization(prompt_input::PromptTokenization {
                token_ids: List::from(vec![1, 2, 3]),
            }),
        );

        let first = cache.get("producer", &[0xde, 0xad]).unwrap().unwrap();
        let second = cache.get("producer", &[0xde, 0xad]).unwrap().unwrap();

        assert!(Arc::ptr_eq(&first, &second));
    }

    #[test]
    fn cached_prompt_converts_to_embedding_input() {
        let cached = CachedStageValue::PromptTokenization(prompt_input::PromptTokenization {
            token_ids: List::from(vec![7, 8]),
        });

        let prompt = cached.as_embedding_prompt().unwrap();

        assert_eq!(prompt.token_ids.as_slice(), &[7, 8]);
    }

    fn temp_dir(label: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "staged-infer-cache-{label}-{}-{nanos}",
            std::process::id()
        ))
    }
}
