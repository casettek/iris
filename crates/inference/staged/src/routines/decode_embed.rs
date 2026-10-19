use crate::{
    artifact_io::{with_main_sequence_scope, with_stage_sequence_scope},
    cache::{
        materialization_key_from_stage_files, materialize_with_cache, CachedInputs,
        MaterializationCache, MaterializationCacheKey,
    },
    embedding::{self, SelectedEmbedding, StaticEmbedding},
    verified_weights::{input_documents_from_args, raster_inputs},
};
use anyhow::Result;
use decode_embed::input::{
    pack_i32s, ActivationRow, ActivationSequence, DecodeEdge, EmbeddingTable,
};
use inference_kernels::kernels::decode_embed::{run_decode_embed_direct, DecodeEmbedDirectInputs};
use raster::List;
use std::{path::Path, sync::Arc};

pub struct Inputs {
    pub selected: DecodeEdge,
    embedding: Weights,
}
enum Weights {
    Eager(Arc<EmbeddingTable>),
    Selective(SelectedEmbedding),
}
impl Inputs {
    /// Already-materialized comparison path; indexed Raster inputs use selective loading.
    pub fn eager(selected: DecodeEdge, embedding: Arc<EmbeddingTable>) -> Self {
        Self {
            selected,
            embedding: Weights::Eager(embedding),
        }
    }
}

pub fn load_inputs_from_args() -> Result<Inputs> {
    let (input, manifest) = input_documents_from_args();
    with_main_sequence_scope(|| {
        load_inputs_from_initialized_runtime(
            &CachedInputs::new(),
            None,
            None,
            input.as_deref(),
            manifest.as_deref(),
        )
    })
}
pub fn load_inputs_from_paths(
    input: &Path,
    input_manifest: &Path,
    cached_inputs: &CachedInputs,
) -> Result<Inputs> {
    load_inputs_from_paths_with_cache(input, input_manifest, cached_inputs, None)
}
pub fn load_inputs_from_paths_with_cache(
    input: &Path,
    input_manifest: &Path,
    cached_inputs: &CachedInputs,
    materialization_cache: Option<&MaterializationCache>,
) -> Result<Inputs> {
    let key = materialization_key_from_stage_files::<StaticEmbedding>(
        input,
        input_manifest,
        "embedding",
    )?;
    with_stage_sequence_scope(input, input_manifest, || {
        load_inputs_from_initialized_runtime(
            cached_inputs,
            materialization_cache,
            key,
            input.to_str(),
            input_manifest.to_str(),
        )
    })
}
pub fn run_direct(inputs: &Inputs) -> Result<ActivationSequence> {
    match &inputs.embedding {
        Weights::Eager(embedding) => run_decode_embed_direct(DecodeEmbedDirectInputs {
            selected: &inputs.selected,
            embedding,
        }),
        Weights::Selective(embedding) => {
            let rows = vec![ActivationRow {
                token_id: inputs.selected.token_id,
                values: pack_i32s(&embedding.row(inputs.selected.token_id)?),
            }];
            Ok(ActivationSequence {
                rows: List::from(rows),
                errors: List::new(),
                kv: List::new(),
                start_position: inputs.selected.decode_position,
            })
        }
    }
}
fn load_inputs_from_initialized_runtime(
    cached_inputs: &CachedInputs,
    materialization_cache: Option<&MaterializationCache>,
    key: Option<MaterializationCacheKey>,
    input: Option<&str>,
    manifest: Option<&str>,
) -> Result<Inputs> {
    let binding = raster::start_program(&[
        raster::entry_argument_spec::<DecodeEdge>("selected"),
        raster::entry_argument_spec::<EmbeddingTable>("embedding"),
    ])?;
    let selected = match cached_inputs.get("selected") {
        Some(value) => value.as_decode_embed_edge()?,
        None => raster::materialize_auth_return(raster::entry_argument_auth_ref::<DecodeEdge>(
            binding.reference.clone(),
            "selected",
        )),
    };
    anyhow::ensure!(
        selected.has_selected,
        "decode-embed received the empty decode edge with no selected token"
    );
    let embedding = if raster_inputs(&binding) {
        Weights::Selective(embedding::load(
            binding,
            input,
            manifest,
            std::iter::once(selected.token_id),
            materialization_cache,
            key,
        )?)
    } else {
        let key = key.map(|mut key| {
            key.type_name = std::any::type_name::<EmbeddingTable>();
            key
        });
        // Preserve postcard compatibility; never use this to recover from invalid Raster.
        Weights::Eager(materialize_with_cache(materialization_cache, key, || {
            raster::materialize_auth_return(raster::entry_argument_auth_ref::<EmbeddingTable>(
                binding.reference,
                "embedding",
            ))
        })?)
    };
    Ok(Inputs {
        selected,
        embedding,
    })
}
