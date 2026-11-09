use std::path::Path;
use std::sync::Arc;

use anyhow::Result;
use prompt_prepare::input::{
    bind_run_schedule, PromptState, PromptText, PromptTokenizer, RunConfig, RunSchedule,
};

use crate::artifact_io::{with_main_sequence_scope, with_stage_sequence_scope};
use crate::cache::{
    materialization_key_from_stage_files, materialize_with_cache, MaterializationCache,
    MaterializationCacheKey,
};
use inference_kernels::kernels::prompt_prepare::run_prompt_bind;

pub struct Inputs {
    pub witness: prompt_prepare::input::TokenizerWitness,
    pub tokenizer: Arc<PromptTokenizer>,
    pub text: PromptText,
    pub config: RunConfig,
    pub descriptor: raster::BytesPage,
    pub schedule: RunSchedule,
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
    let mut output = run_prompt_bind(&inputs.tokenizer, &inputs.text, &inputs.config)?;
    output.context.model =
        model_identity::ModelContext::bind(inputs.descriptor.as_slice(), &inputs.config.model_id);
    output.context.model = model_identity::stream::verify(
        output.context.model,
        &crate::prompt_identity::root(&inputs.tokenizer, &inputs.witness),
        model_identity::Role::Tokenizer,
    );
    output.context.model.require(crate::prompt_identity::bind(
        &inputs.tokenizer,
        &inputs.witness,
        &inputs.text,
        &inputs.config,
    ));
    output.context.model.bind_statement(
        &output.context.input_text_sha256,
        output.context.raw_prompt,
        output.context.max_new_tokens,
        output.context.range_size,
    );
    output.context = bind_run_schedule(output.context, &inputs.schedule);
    Ok(output)
}

fn load_inputs_with_cache(
    materialization_cache: Option<&MaterializationCache>,
    key: Option<MaterializationCacheKey>,
) -> Result<Inputs> {
    let binding = raster::start_program(&[
        raster::entry_argument_spec::<PromptTokenizer>("tokenizer"),
        raster::entry_argument_spec::<prompt_prepare::input::TokenizerWitness>("witness"),
        raster::entry_argument_spec::<PromptText>("text"),
        raster::entry_argument_spec::<RunConfig>("config"),
        raster::entry_argument_spec::<raster::BytesPage>("descriptor"),
        raster::entry_argument_spec::<RunSchedule>("schedule"),
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
    let text = raster::materialize_auth_return(raster::entry_argument_auth_ref::<PromptText>(
        binding.reference.clone(),
        "text",
    ));
    let config = raster::materialize_auth_return(raster::entry_argument_auth_ref::<RunConfig>(
        binding.reference.clone(),
        "config",
    ));
    let descriptor = raster::materialize_auth_return(raster::entry_argument_auth_ref::<
        raster::BytesPage,
    >(binding.reference.clone(), "descriptor"));
    let schedule = raster::materialize_auth_return(raster::entry_argument_auth_ref::<RunSchedule>(
        binding.reference,
        "schedule",
    ));
    Ok(Inputs {
        witness,
        descriptor,
        schedule,
        tokenizer,
        text,
        config,
    })
}
