use std::path::Path;
use std::sync::Arc;

use anyhow::Result;
use prompt_prepare::input::{PromptState, PromptText, PromptTokenizer, RunConfig};

use crate::artifact_io::{with_main_sequence_scope, with_stage_sequence_scope};
use crate::cache::{
    materialization_key_from_stage_files, materialize_with_cache, MaterializationCache,
    MaterializationCacheKey,
};
use inference_kernels::kernels::prompt_prepare::run_prompt_bind;

pub struct Inputs {
    pub tokenizer: Arc<PromptTokenizer>,
    pub text: PromptText,
    pub config: RunConfig,
}

pub fn load_inputs_from_args() -> Result<Inputs> {
    with_main_sequence_scope(|| load_inputs_with_cache(None, None))
}

pub fn load_inputs_from_paths_with_cache(
    input: &Path,
    input_manifest: &Path,
    materialization_cache: Option<&MaterializationCache>,
) -> Result<Inputs> {
    let key = materialization_key_from_stage_files::<PromptTokenizer>(
        input,
        input_manifest,
        "tokenizer",
    )?;
    with_stage_sequence_scope(input, input_manifest, || {
        load_inputs_with_cache(materialization_cache, key)
    })
}

pub fn run_direct(inputs: &Inputs) -> Result<PromptState> {
    run_prompt_bind(&inputs.tokenizer, &inputs.text, &inputs.config)
}

fn load_inputs_with_cache(
    materialization_cache: Option<&MaterializationCache>,
    key: Option<MaterializationCacheKey>,
) -> Result<Inputs> {
    let binding = raster::start_program(&[
        raster::entry_argument_spec::<PromptTokenizer>("tokenizer"),
        raster::entry_argument_spec::<PromptText>("text"),
        raster::entry_argument_spec::<RunConfig>("config"),
    ])?;
    let tokenizer = materialize_with_cache(materialization_cache, key, || {
        raster::materialize_auth_return(raster::entry_argument_auth_ref::<PromptTokenizer>(
            binding.reference.clone(),
            "tokenizer",
        ))
    })?;
    let text = raster::materialize_auth_return(raster::entry_argument_auth_ref::<PromptText>(
        binding.reference.clone(),
        "text",
    ));
    let config = raster::materialize_auth_return(raster::entry_argument_auth_ref::<RunConfig>(
        binding.reference,
        "config",
    ));
    Ok(Inputs {
        tokenizer,
        text,
        config,
    })
}
