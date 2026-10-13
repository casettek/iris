use std::path::Path;
use std::sync::Arc;

use anyhow::Result;
use decode_embed::input::{ActivationSequence, DecodeEdge, EmbeddingTable};

use crate::artifact_io::{with_main_sequence_scope, with_stage_sequence_scope};
use crate::cache::{
    materialization_key_from_stage_files, materialize_with_cache, CachedInputs,
    MaterializationCache, MaterializationCacheKey,
};
use inference_kernels::kernels::decode_embed::{run_decode_embed_direct, DecodeEmbedDirectInputs};

pub struct Inputs {
    pub selected: DecodeEdge,
    pub embedding: Arc<EmbeddingTable>,
}

pub fn load_inputs_from_args() -> Result<Inputs> {
    with_main_sequence_scope(|| {
        load_inputs_from_initialized_runtime(&CachedInputs::new(), None, None)
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
    let embedding_key =
        materialization_key_from_stage_files::<EmbeddingTable>(input, input_manifest, "embedding")?;
    with_stage_sequence_scope(input, input_manifest, || {
        load_inputs_from_initialized_runtime(cached_inputs, materialization_cache, embedding_key)
    })
}

pub fn run_direct(inputs: &Inputs) -> Result<ActivationSequence> {
    run_decode_embed_direct(DecodeEmbedDirectInputs {
        selected: &inputs.selected,
        embedding: &inputs.embedding,
    })
}

fn load_inputs_from_initialized_runtime(
    cached_inputs: &CachedInputs,
    materialization_cache: Option<&MaterializationCache>,
    embedding_key: Option<MaterializationCacheKey>,
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
    let embedding = materialize_with_cache(materialization_cache, embedding_key, || {
        raster::materialize_auth_return(raster::entry_argument_auth_ref::<EmbeddingTable>(
            binding.reference,
            "embedding",
        ))
    })?;
    Ok(Inputs {
        selected,
        embedding,
    })
}
