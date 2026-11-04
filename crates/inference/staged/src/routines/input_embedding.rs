use crate::{
    artifact_io::{with_main_sequence_scope, with_stage_sequence_scope},
    cache::{
        materialization_key_from_stage_files, materialize_with_cache, CachedInputs,
        MaterializationCache, MaterializationCacheKey,
    },
    embedding::{self, SelectedEmbedding, StaticEmbedding},
    verified_weights::{input_documents_from_args, raster_inputs},
};
use anyhow::Result;
use inference_kernels::kernels::input_embedding::{
    run_input_embedding_direct, InputEmbeddingDirectInputs,
};
use input_embedding::input::{
    pack_i32s, ActivationRow, ActivationSequence, ClaimedOutput, EmbeddingTable, PromptTokenization,
};
use raster::List;
use std::{path::Path, sync::Arc};

pub struct Inputs {
    pub prompt: PromptTokenization,
    /// Embedded after the prompt; empty in the autoregressive chain.
    pub claimed: ClaimedOutput,
    /// Where this range starts: right after the previous range's rows.
    pub start_position: u32,
    embedding: Weights,
}
enum Weights {
    Eager(Arc<EmbeddingTable>),
    Selective(SelectedEmbedding),
}
impl Inputs {
    /// Already-materialized comparison path; indexed Raster inputs use selective loading.
    pub fn eager(prompt: PromptTokenization, embedding: Arc<EmbeddingTable>) -> Self {
        Self {
            prompt,
            claimed: ClaimedOutput::default(),
            start_position: 0,
            embedding: Weights::Eager(embedding),
        }
    }

    /// `prompt ‖ claimed`: the ids this stage embeds, in row order.
    pub fn token_ids(&self) -> impl Iterator<Item = u32> + '_ {
        self.prompt
            .token_ids
            .iter()
            .chain(self.claimed.token_ids.iter())
            .copied()
    }
}

pub fn load_inputs_from_args() -> Result<Inputs> {
    let (input, manifest) = input_documents_from_args();
    with_main_sequence_scope(|| {
        load_inputs_from_initialized_runtime(
            &CachedInputs::new(),
            None,
            None,
            input.as_deref(),
            manifest.as_deref(),
        )
    })
}
pub fn load_inputs_from_paths(
    input: &Path,
    input_manifest: &Path,
    cached_inputs: &CachedInputs,
) -> Result<Inputs> {
    load_inputs_from_paths_with_cache(input, input_manifest, cached_inputs, None)
}
pub fn load_inputs_from_paths_with_cache(
    input: &Path,
    input_manifest: &Path,
    cached_inputs: &CachedInputs,
    materialization_cache: Option<&MaterializationCache>,
) -> Result<Inputs> {
    let key = materialization_key_from_stage_files::<StaticEmbedding>(
        input,
        input_manifest,
        "embedding",
    )?;
    with_stage_sequence_scope(input, input_manifest, || {
        load_inputs_from_initialized_runtime(
            cached_inputs,
            materialization_cache,
            key,
            input.to_str(),
            input_manifest.to_str(),
        )
    })
}
pub fn run_direct(inputs: &Inputs) -> Result<ActivationSequence> {
    let mut output = match &inputs.embedding {
        Weights::Eager(embedding) => run_input_embedding_direct(InputEmbeddingDirectInputs {
            prompt: &PromptTokenization {
                token_ids: List::from(inputs.token_ids().collect::<Vec<_>>()),
            },
            embedding,
        })?,
        Weights::Selective(embedding) => {
            let rows = inputs
                .token_ids()
                .map(|token_id| {
                    Ok(ActivationRow {
                        token_id,
                        values: pack_i32s(&embedding.row(token_id)?),
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            ActivationSequence {
                rows: List::from(rows),
                errors: List::new(),
                kv: List::new(),
                start_position: 0,
            }
        }
    };
    output.start_position = inputs.start_position;
    Ok(output)
}
fn load_inputs_from_initialized_runtime(
    cached_inputs: &CachedInputs,
    materialization_cache: Option<&MaterializationCache>,
    key: Option<MaterializationCacheKey>,
    input: Option<&str>,
    manifest: Option<&str>,
) -> Result<Inputs> {
    let binding = raster::start_program(&[
        raster::entry_argument_spec::<PromptTokenization>("prompt"),
        raster::entry_argument_spec::<ClaimedOutput>("claimed"),
        raster::entry_argument_spec::<ActivationSequence>("previous"),
        raster::entry_argument_spec::<EmbeddingTable>("embedding"),
    ])?;
    let prompt = match cached_inputs.get("prompt") {
        Some(value) => value.as_embedding_prompt()?,
        None => raster::materialize_auth_return(raster::entry_argument_auth_ref::<
            PromptTokenization,
        >(binding.reference.clone(), "prompt")),
    };
    let claimed = raster::materialize_auth_return(
        raster::entry_argument_auth_ref::<ClaimedOutput>(binding.reference.clone(), "claimed"),
    );
    let previous = match cached_inputs.get("previous") {
        Some(value) => value.as_embedding_previous()?,
        None => raster::materialize_auth_return(raster::entry_argument_auth_ref::<
            ActivationSequence,
        >(binding.reference.clone(), "previous")),
    };
    let start_position = previous
        .start_position
        .checked_add(previous.rows.len() as u32)
        .ok_or_else(|| anyhow::anyhow!("range start position overflows u32"))?;

    let embedding = if raster_inputs(&binding) {
        Weights::Selective(embedding::load(
            binding,
            input,
            manifest,
            prompt
                .token_ids
                .iter()
                .chain(claimed.token_ids.iter())
                .copied(),
            materialization_cache,
            key,
        )?)
    } else {
        let key = key.map(|mut key| {
            key.type_name = std::any::type_name::<EmbeddingTable>();
            key
        });
        // Preserve postcard compatibility; never use this to recover from invalid Raster.
        Weights::Eager(materialize_with_cache(None, key, || {
            raster::materialize_auth_return(raster::entry_argument_auth_ref::<EmbeddingTable>(
                binding.reference,
                "embedding",
            ))
        })?)
    };
    Ok(Inputs {
        prompt,
        claimed,
        start_position,
        embedding,
    })
}
