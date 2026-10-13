use std::path::Path;

use anyhow::Result;
use decode_select_token::input::{DecodeEdge, PrefillLogits};

use crate::artifact_io::{with_main_sequence_scope, with_stage_sequence_scope};
use crate::cache::CachedInputs;
use inference_kernels::kernels::decode_select_token::{
    run_decode_select_token_direct, DecodeSelectTokenDirectInputs,
};

pub struct Inputs {
    pub logits: PrefillLogits,
    pub prior: DecodeEdge,
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

pub fn run_direct(inputs: &Inputs) -> Result<DecodeEdge> {
    run_decode_select_token_direct(DecodeSelectTokenDirectInputs {
        logits: &inputs.logits,
        prior: &inputs.prior,
    })
}

fn load_inputs_from_initialized_runtime(cached_inputs: &CachedInputs) -> Result<Inputs> {
    let binding = raster::start_program(&[
        raster::entry_argument_spec::<PrefillLogits>("logits"),
        raster::entry_argument_spec::<DecodeEdge>("prior"),
    ])?;
    let logits = match cached_inputs.get("logits") {
        Some(value) => value.as_decode_logits()?,
        None => raster::materialize_auth_return(raster::entry_argument_auth_ref::<PrefillLogits>(
            binding.reference.clone(),
            "logits",
        )),
    };
    let prior = match cached_inputs.get("prior") {
        Some(value) => value.as_decode_edge()?,
        None => raster::materialize_auth_return(raster::entry_argument_auth_ref::<DecodeEdge>(
            binding.reference,
            "prior",
        )),
    };
    Ok(Inputs { logits, prior })
}
