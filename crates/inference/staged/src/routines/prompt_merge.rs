use std::path::Path;
use std::sync::Arc;

use anyhow::Result;
use prompt_prepare::input::{BpePieces, PromptTokenizer};

use crate::artifact_io::{with_main_sequence_scope, with_stage_sequence_scope};
use crate::cache::{
    materialization_key_from_stage_files, materialize_with_cache, CachedInputs,
    MaterializationCache, MaterializationCacheKey,
};
use inference_kernels::kernels::prompt_prepare::{run_merge_batch, PromptPrepareDirectInputs};

pub struct Inputs {
    pub witness: prompt_prepare::input::TokenizerWitness,
    pub tokenizer: Arc<PromptTokenizer>,
    pub initial_pieces: BpePieces,
}

pub fn load_inputs_from_args() -> Result<Inputs> {
    with_main_sequence_scope(load_inputs_from_initialized_runtime)
}

pub fn load_inputs_from_paths(
    input: &Path,
    input_manifest: &Path,
    cached_inputs: &CachedInputs,
) -> Result<Inputs> {
    load_inputs_from_paths_with_cache(input, input_manifest, cached_inputs, None)
}

/// The tokenizer is identical for every merge round and range of a run, so a
/// run-owned cache keeps one verified copy instead of re-materializing it.
pub fn load_inputs_from_paths_with_cache(
    input: &Path,
    input_manifest: &Path,
    cached_inputs: &CachedInputs,
    materialization_cache: Option<&MaterializationCache>,
) -> Result<Inputs> {
    let key = materialization_key_from_stage_files::<PromptTokenizer>(
        input,
        input_manifest,
        "tokenizer",
    )?;
    with_stage_sequence_scope(input, input_manifest, || {
        load_inputs_with_cache(cached_inputs, materialization_cache, key)
    })
}

pub fn run_direct(inputs: &Inputs) -> Result<BpePieces> {
    let mut output = run_merge_batch(PromptPrepareDirectInputs {
        tokenizer: &inputs.tokenizer,
        initial_pieces: &inputs.initial_pieces,
    })?;
    output.context.model = model_identity::stream::verify(
        output.context.model,
        &crate::prompt_identity::root(&inputs.tokenizer, &inputs.witness),
        model_identity::Role::Tokenizer,
    );
    output
        .context
        .model
        .require(output.context.model.model_id == output.context.model_id);
    output.context.model.require(crate::prompt_identity::batch(
        &inputs.tokenizer,
        &inputs.witness,
        inputs.initial_pieces.pieces.as_slice(),
    ));
    Ok(output)
}

fn load_inputs_from_initialized_runtime() -> Result<Inputs> {
    load_inputs_with_cache(&CachedInputs::new(), None, None)
}

fn load_inputs_with_cache(
    cached: &CachedInputs,
    materialization_cache: Option<&MaterializationCache>,
    key: Option<MaterializationCacheKey>,
) -> Result<Inputs> {
    let binding = raster::start_program(&[
        raster::entry_argument_spec::<PromptTokenizer>("tokenizer"),
        raster::entry_argument_spec::<prompt_prepare::input::TokenizerWitness>("witness"),
        raster::entry_argument_spec::<BpePieces>("initial_pieces"),
    ])?;
    let witness = raster::materialize_auth_return(raster::entry_argument_auth_ref::<
        prompt_prepare::input::TokenizerWitness,
    >(binding.reference.clone(), "witness"));
    let tokenizer = materialize_with_cache(materialization_cache, key, || {
        raster::materialize_auth_return(raster::entry_argument_auth_ref::<PromptTokenizer>(
            binding.reference.clone(),
            "tokenizer",
        ))
    })?;
    let initial_pieces = match cached.get("initial_pieces") {
        Some(value) => value.as_bpe_pieces()?,
        None => raster::materialize_auth_return(raster::entry_argument_auth_ref::<BpePieces>(
            binding.reference,
            "initial_pieces",
        )),
    };
    Ok(Inputs {
        witness,
        tokenizer,
        initial_pieces,
    })
}
