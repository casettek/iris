use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use anyhow::Result;
use prompt_check::input::{
    PromptState, PromptText, PromptTokenizer, RunConfig, RunSchedule, TextClaim, TokenizerWitness,
};

use crate::artifact_io::{with_main_sequence_scope, with_stage_sequence_scope};
use crate::cache::{
    materialization_key_from_stage_files, materialize_with_cache, CachedInputs,
    MaterializationCache, MaterializationCacheKey,
};
use crate::routines::prompt_check::WitnessProofs;
use inference_kernels::kernels::prompt_prepare::{run_prompt_bind, PromptBindInputs};

pub struct Inputs {
    pub witness: Arc<TokenizerWitness>,
    pub tokenizer: Arc<PromptTokenizer>,
    /// This range's slice of the original text.
    pub text: PromptText,
    pub config: RunConfig,
    pub descriptor: raster::BytesPage,
    pub schedule: RunSchedule,
    pub claim: TextClaim,
    /// The bind of the range before, or the empty state.
    pub previous: PromptState,
}

pub fn load_inputs_from_args() -> Result<Inputs> {
    with_main_sequence_scope(|| load_inputs_with_cache(&CachedInputs::new(), None, None, None))
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
    // The witness is the same committed input for every prompt stage of a run.
    let witness_key =
        materialization_key_from_stage_files::<TokenizerWitness>(input, input_manifest, "witness")?;
    with_stage_sequence_scope(input, input_manifest, || {
        load_inputs_with_cache(cached_inputs, materialization_cache, key, witness_key)
    })
}

pub fn run_direct(inputs: &Inputs) -> Result<PromptState> {
    let mut proofs = WitnessProofs {
        tokenizer: &inputs.tokenizer,
        witness: &inputs.witness,
        merges: HashMap::new(),
        vocab: HashMap::new(),
    };
    run_prompt_bind(
        PromptBindInputs {
            tokenizer: &inputs.tokenizer,
            root: &crate::prompt_identity::root(&inputs.tokenizer, &inputs.witness),
            text: &inputs.text,
            config: &inputs.config,
            descriptor: inputs.descriptor.as_slice(),
            schedule: &inputs.schedule,
            claim: &inputs.claim,
            previous: &inputs.previous,
        },
        &mut proofs,
    )
}

fn load_inputs_with_cache(
    cached: &CachedInputs,
    materialization_cache: Option<&MaterializationCache>,
    key: Option<MaterializationCacheKey>,
    witness_key: Option<MaterializationCacheKey>,
) -> Result<Inputs> {
    let binding = raster::start_program(&[
        raster::entry_argument_spec::<PromptTokenizer>("tokenizer"),
        raster::entry_argument_spec::<TokenizerWitness>("witness"),
        raster::entry_argument_spec::<PromptText>("text"),
        raster::entry_argument_spec::<RunConfig>("config"),
        raster::entry_argument_spec::<raster::BytesPage>("descriptor"),
        raster::entry_argument_spec::<RunSchedule>("schedule"),
        raster::entry_argument_spec::<TextClaim>("claim"),
        raster::entry_argument_spec::<PromptState>("previous"),
    ])?;
    let witness = materialize_with_cache(materialization_cache, witness_key, || {
        raster::materialize_auth_return(raster::entry_argument_auth_ref::<TokenizerWitness>(
            binding.reference.clone(),
            "witness",
        ))
    })?;
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
        binding.reference.clone(),
        "schedule",
    ));
    let claim = raster::materialize_auth_return(raster::entry_argument_auth_ref::<TextClaim>(
        binding.reference.clone(),
        "claim",
    ));
    let previous = match cached.get("previous") {
        Some(value) => value.as_prompt_state()?,
        None => raster::materialize_auth_return(raster::entry_argument_auth_ref::<PromptState>(
            binding.reference,
            "previous",
        )),
    };
    Ok(Inputs {
        witness,
        descriptor,
        schedule,
        claim,
        previous,
        tokenizer,
        text,
        config,
    })
}
