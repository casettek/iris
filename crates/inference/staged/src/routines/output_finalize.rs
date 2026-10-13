use std::path::Path;

use anyhow::Result;
use output_finalize::input::{DecodeEdge, DecoderTable, GeneratedOutput};

use crate::artifact_io::{with_main_sequence_scope, with_stage_sequence_scope};
use crate::cache::CachedInputs;
use inference_kernels::kernels::output_finalize::{
    run_output_finalize_direct, OutputFinalizeDirectInputs,
};

pub struct Inputs {
    pub edge: DecodeEdge,
    pub decoder: DecoderTable,
}

pub fn load_inputs_from_args() -> Result<Inputs> {
    with_main_sequence_scope(|| load_inputs_from_initialized_runtime(&CachedInputs::new()))
}

pub fn load_inputs_from_paths(
    input: &Path,
    input_manifest: &Path,
    cached_inputs: &CachedInputs,
) -> Result<Inputs> {
    with_stage_sequence_scope(input, input_manifest, || {
        load_inputs_from_initialized_runtime(cached_inputs)
    })
}

pub fn run_direct(inputs: &Inputs) -> Result<GeneratedOutput> {
    run_output_finalize_direct(OutputFinalizeDirectInputs {
        edge: &inputs.edge,
        decoder: &inputs.decoder,
    })
}

fn load_inputs_from_initialized_runtime(cached_inputs: &CachedInputs) -> Result<Inputs> {
    let binding = raster::start_program(&[
        raster::entry_argument_spec::<DecodeEdge>("edge"),
        raster::entry_argument_spec::<DecoderTable>("decoder"),
    ])?;
    let edge = match cached_inputs.get("edge") {
        Some(value) => value.as_output_edge()?,
        None => raster::materialize_auth_return(raster::entry_argument_auth_ref::<DecodeEdge>(
            binding.reference.clone(),
            "edge",
        )),
    };
    let decoder = raster::materialize_auth_return(raster::entry_argument_auth_ref::<DecoderTable>(
        binding.reference,
        "decoder",
    ));
    Ok(Inputs { edge, decoder })
}
