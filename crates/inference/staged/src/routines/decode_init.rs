use std::path::Path;

use anyhow::Result;
use decode_select_token::input::DecodeEdge;

use crate::artifact_io::{with_main_sequence_scope, with_stage_sequence_scope};
use crate::cache::CachedInputs;
use inference_kernels::kernels::decode_init::{run_decode_init_direct, DecodeInitDirectInputs};

pub struct Inputs;

pub fn load_inputs_from_args() -> Result<Inputs> {
    with_main_sequence_scope(|| Ok(Inputs))
}

pub fn load_inputs_from_paths(
    input: &Path,
    input_manifest: &Path,
    _cached_inputs: &CachedInputs,
) -> Result<Inputs> {
    with_stage_sequence_scope(input, input_manifest, || Ok(Inputs))
}

pub fn run_direct(_inputs: &Inputs) -> Result<DecodeEdge> {
    run_decode_init_direct(DecodeInitDirectInputs)
}
