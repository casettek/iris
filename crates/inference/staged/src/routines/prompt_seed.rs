use std::path::Path;
use std::sync::Arc;

use anyhow::Result;
use prompt_prepare::input::{PromptState, PromptTokenizer, RangeCut};

use crate::artifact_io::{with_main_sequence_scope, with_stage_sequence_scope};
use crate::cache::{
    materialization_key_from_stage_files, materialize_with_cache, CachedInputs,
    MaterializationCache, MaterializationCacheKey,
};
use inference_kernels::kernels::prompt_prepare::run_prompt_seed;

pub struct Inputs {
    pub witness: prompt_prepare::input::TokenizerWitness,
    pub tokenizer: Arc<PromptTokenizer>,
    pub bound: PromptState,
    pub previous: PromptState,
    pub cut: RangeCut,
}

pub fn load_inputs_from_args() -> Result<Inputs> {
    with_main_sequence_scope(|| load_inputs_with_cache(&CachedInputs::new(), None, None))
}

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

pub fn run_direct(inputs: &Inputs) -> Result<PromptState> {
    let mut output = run_prompt_seed(
        &inputs.tokenizer,
        &inputs.bound,
        &inputs.previous.context,
        &inputs.cut,
    )?;
    output.context.model = model_identity::stream::verify(
        output.context.model,
        &crate::prompt_identity::root(&inputs.tokenizer, &inputs.witness),
        model_identity::Role::Tokenizer,
    );
    output
        .context
        .model
        .require(output.context.model.model_id == output.context.model_id);
    output.context.model.require(crate::prompt_identity::seed(
        &inputs.tokenizer,
        &inputs.witness,
        &inputs.bound,
        &inputs.previous.context,
        &inputs.cut,
    ));
    Ok(output)
}

fn load_inputs_with_cache(
    cached: &CachedInputs,
    materialization_cache: Option<&MaterializationCache>,
    key: Option<MaterializationCacheKey>,
) -> Result<Inputs> {
    let binding = raster::start_program(&[
        raster::entry_argument_spec::<PromptTokenizer>("tokenizer"),
        raster::entry_argument_spec::<prompt_prepare::input::TokenizerWitness>("witness"),
        raster::entry_argument_spec::<PromptState>("bound"),
        raster::entry_argument_spec::<PromptState>("previous"),
        raster::entry_argument_spec::<RangeCut>("cut"),
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
    let bound = match cached.get("bound") {
        Some(value) => value.as_prompt_state()?,
        None => raster::materialize_auth_return(raster::entry_argument_auth_ref::<PromptState>(
            binding.reference.clone(),
            "bound",
        )),
    };
    let previous = match cached.get("previous") {
        Some(value) => value.as_prompt_state()?,
        None => raster::materialize_auth_return(raster::entry_argument_auth_ref::<PromptState>(
            binding.reference.clone(),
            "previous",
        )),
    };
    let cut = raster::materialize_auth_return(raster::entry_argument_auth_ref::<RangeCut>(
        binding.reference,
        "cut",
    ));
    Ok(Inputs {
        witness,
        tokenizer,
        bound,
        previous,
        cut,
    })
}
