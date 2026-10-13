use std::path::Path;

use anyhow::Result;
use prompt_prepare::input::{BpePieces, PromptTokenizer};

use crate::artifact_io::{with_main_sequence_scope, with_stage_sequence_scope};
use crate::cache::CachedInputs;
use inference_kernels::kernels::prompt_prepare::{run_merge_batch, PromptPrepareDirectInputs};

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

pub fn run_direct(inputs: &Inputs) -> Result<BpePieces> {
    run_merge_batch(PromptPrepareDirectInputs {
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
        raster::entry_argument_spec::<BpePieces>("initial_pieces"),
    ])?;
    let tokenizer = raster::materialize_auth_return(raster::entry_argument_auth_ref::<
        PromptTokenizer,
    >(binding.reference.clone(), "tokenizer"));
    let initial_pieces = match cached.get("initial_pieces") {
        Some(value) => value.as_bpe_pieces()?,
        None => raster::materialize_auth_return(raster::entry_argument_auth_ref::<BpePieces>(
            binding.reference,
            "initial_pieces",
        )),
    };
    Ok(Inputs {
        tokenizer,
        initial_pieces,
    })
}
