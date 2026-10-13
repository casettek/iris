use std::path::Path;
use std::sync::Arc;

use anyhow::Result;
use input_embedding::input::{ActivationSequence, EmbeddingTable, PromptTokenization};

use crate::artifact_io::{with_main_sequence_scope, with_stage_sequence_scope};
use crate::cache::{
    materialization_key_from_stage_files, materialize_with_cache, CachedInputs,
    MaterializationCache, MaterializationCacheKey,
};
use inference_kernels::kernels::input_embedding::{
    run_input_embedding_direct, InputEmbeddingDirectInputs,
};

pub struct Inputs {
    pub prompt: PromptTokenization,
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
    run_input_embedding_direct(InputEmbeddingDirectInputs {
        prompt: &inputs.prompt,
        embedding: &inputs.embedding,
    })
}

fn load_inputs_from_initialized_runtime(
    cached_inputs: &CachedInputs,
    materialization_cache: Option<&MaterializationCache>,
    embedding_key: Option<MaterializationCacheKey>,
) -> Result<Inputs> {
    let binding = raster::start_program(&[
        raster::entry_argument_spec::<PromptTokenization>("prompt"),
        raster::entry_argument_spec::<EmbeddingTable>("embedding"),
    ])?;
    let prompt = match cached_inputs.get("prompt") {
        Some(value) => value.as_embedding_prompt()?,
        None => raster::materialize_auth_return(raster::entry_argument_auth_ref::<
            PromptTokenization,
        >(binding.reference.clone(), "prompt")),
    };
    let embedding = materialize_with_cache(materialization_cache, embedding_key, || {
        raster::materialize_auth_return(raster::entry_argument_auth_ref::<EmbeddingTable>(
            binding.reference,
            "embedding",
        ))
    })?;
    Ok(Inputs { prompt, embedding })
}
