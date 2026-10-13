use std::path::Path;

use anyhow::Result;
use prompt_prepare::input::{BpePieces, PromptTokenization, PromptTokenizer};

use crate::artifact_io::{with_main_sequence_scope, with_stage_sequence_scope};
use crate::cache::CachedInputs;
use inference_kernels::kernels::prompt_prepare::{finalize_prompt, PromptPrepareDirectInputs};

pub struct Inputs {
    pub tokenizer: PromptTokenizer,
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
    with_stage_sequence_scope(input, input_manifest, || load_inputs(cached_inputs))
}

pub fn run_direct(inputs: &Inputs) -> Result<PromptTokenization> {
    finalize_prompt(PromptPrepareDirectInputs {
        tokenizer: &inputs.tokenizer,
        initial_pieces: &inputs.initial_pieces,
    })
}

fn load_inputs_from_initialized_runtime() -> Result<Inputs> {
    load_inputs(&CachedInputs::new())
}

fn load_inputs(cached: &CachedInputs) -> Result<Inputs> {
    let binding = raster::start_program(&[
        raster::entry_argument_spec::<PromptTokenizer>("tokenizer"),
        raster::entry_argument_spec::<BpePieces>("merged_pieces"),
    ])?;
    let tokenizer = raster::materialize_auth_return(raster::entry_argument_auth_ref::<
        PromptTokenizer,
    >(binding.reference.clone(), "tokenizer"));
    let initial_pieces = match cached.get("merged_pieces") {
        Some(value) => value.as_bpe_pieces()?,
        None => raster::materialize_auth_return(raster::entry_argument_auth_ref::<BpePieces>(
            binding.reference,
            "merged_pieces",
        )),
    };
    Ok(Inputs {
        tokenizer,
        initial_pieces,
    })
}
