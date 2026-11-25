use std::path::Path;

use anyhow::Result;
use source_core::{run_source_steps, Recipe, SignerKey, SourceDocuments, SourceRecord};

use crate::artifact_io::{with_main_sequence_scope, with_stage_sequence_scope};
use crate::cache::{CachedInputs, MaterializationCache};

/// The `prompt-source` stage's inputs. The types are `source-core`'s: the
/// stage program compiles the same source file, and a chain links by
/// structure.
pub struct Inputs {
    /// The key attestations are checked against.
    pub signer: SignerKey,
    pub recipe: Recipe,
    /// The run's documents, with the attestation over each.
    pub documents: SourceDocuments,
}

pub fn load_inputs_from_args() -> Result<Inputs> {
    with_main_sequence_scope(load_inputs)
}

pub fn load_inputs_from_paths_with_cache(
    input: &Path,
    input_manifest: &Path,
    _cached_inputs: &CachedInputs,
    _materialization_cache: Option<&MaterializationCache>,
) -> Result<Inputs> {
    with_stage_sequence_scope(input, input_manifest, load_inputs)
}

/// The stage's steps, in its order, over the same functions its tiles call.
pub fn run_direct(inputs: &Inputs) -> Result<SourceRecord> {
    Ok(run_source_steps(&inputs.signer, &inputs.recipe, &inputs.documents).record)
}

fn load_inputs() -> Result<Inputs> {
    let binding = raster::start_program(&[
        raster::entry_argument_spec::<SignerKey>("signer"),
        raster::entry_argument_spec::<Recipe>("recipe"),
        raster::entry_argument_spec::<SourceDocuments>("documents"),
    ])?;
    let signer = raster::materialize_auth_return(raster::entry_argument_auth_ref::<SignerKey>(
        binding.reference.clone(),
        "signer",
    ));
    let recipe = raster::materialize_auth_return(raster::entry_argument_auth_ref::<Recipe>(
        binding.reference.clone(),
        "recipe",
    ));
    let documents = raster::materialize_auth_return(raster::entry_argument_auth_ref::<
        SourceDocuments,
    >(binding.reference, "documents"));
    Ok(Inputs {
        signer,
        recipe,
        documents,
    })
}
