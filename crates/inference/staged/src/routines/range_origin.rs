use std::path::Path;

use anyhow::Result;
use range_origin::input::ActivationSequence;

use crate::artifact_io::{with_main_sequence_scope, with_stage_sequence_scope};
use crate::cache::CachedInputs;
use inference_kernels::kernels::range_origin::run_range_origin_direct;

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

pub fn run_direct(_inputs: &Inputs) -> Result<ActivationSequence> {
    run_range_origin_direct()
}
