use std::path::Path;

use anyhow::Result;
use raster_runtime::OutputArtifact;
use serde::Serialize;

#[derive(Debug)]
pub struct EncodedArtifact {
    pub data: Vec<u8>,
    pub index: Vec<u8>,
    pub structural_commitment: String,
}

pub fn with_main_sequence_scope<T>(load: impl FnOnce() -> Result<T>) -> Result<T> {
    raster::init();
    let _scope = SequenceScope::enter("main");
    load()
}

pub fn with_stage_sequence_scope<T>(
    input: &Path,
    input_manifest: &Path,
    load: impl FnOnce() -> Result<T>,
) -> Result<T> {
    raster::init();
    raster_runtime::install_file_source_resolver(input, input_manifest)?;
    let _scope = SequenceScope::enter("main");
    load()
}

struct SequenceScope;

impl SequenceScope {
    fn enter(sequence_id: &str) -> Self {
        raster_runtime::enter_sequence_scope(sequence_id);
        Self
    }
}

impl Drop for SequenceScope {
    fn drop(&mut self) {
        raster_runtime::exit_sequence_scope();
    }
}

pub fn encode_output<T: Serialize>(value: &T) -> Result<EncodedArtifact> {
    let (data, index, structural_commitment) = raster::encode_raster_value(value)?;
    Ok(EncodedArtifact {
        data,
        index,
        structural_commitment,
    })
}

pub fn write_output<T: Serialize>(value: &T) -> Result<OutputArtifact> {
    raster_runtime::write_program_output_artifact(value)?
        .ok_or_else(|| anyhow::anyhow!("{} is not set", raster_runtime::OUTPUT_DIR_ENV))
}
