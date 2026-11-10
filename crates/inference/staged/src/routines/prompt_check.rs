use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use anyhow::Result;
use prompt_check::check::TokenMarks;
use prompt_prepare::input::{PromptState, PromptTokenizer, TokenizerWitness};

use crate::artifact_io::{with_main_sequence_scope, with_stage_sequence_scope};
use crate::cache::{
    materialization_key_from_stage_files, materialize_with_cache, CachedInputs,
    MaterializationCache, MaterializationCacheKey,
};
use inference_kernels::kernels::prompt_check_stage::{run_prompt_check, BucketProofs};

pub struct Inputs {
    pub witness: Arc<TokenizerWitness>,
    pub tokenizer: Arc<PromptTokenizer>,
    pub bound: PromptState,
    pub previous: PromptState,
    pub marks: TokenMarks,
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

/// Checks each tokenizer bucket the stage reads against the witness, once.
/// A bucket's check depends only on the tokenizer, the witness and its index.
struct WitnessProofs<'a> {
    tokenizer: &'a PromptTokenizer,
    witness: &'a TokenizerWitness,
    merges: HashMap<u32, bool>,
    vocab: HashMap<u32, bool>,
}

impl BucketProofs for WitnessProofs<'_> {
    fn merge_bucket(&mut self, index: u32) -> bool {
        let (tokenizer, witness) = (self.tokenizer, self.witness);
        *self
            .merges
            .entry(index)
            .or_insert_with(|| crate::prompt_identity::merge_bucket(tokenizer, witness, index))
    }
    fn vocab_bucket(&mut self, index: u32) -> bool {
        let (tokenizer, witness) = (self.tokenizer, self.witness);
        *self
            .vocab
            .entry(index)
            .or_insert_with(|| crate::prompt_identity::vocab_bucket(tokenizer, witness, index))
    }
}

pub fn run_direct(inputs: &Inputs) -> Result<PromptState> {
    let mut proofs = WitnessProofs {
        tokenizer: &inputs.tokenizer,
        witness: &inputs.witness,
        merges: HashMap::new(),
        vocab: HashMap::new(),
    };
    run_prompt_check(
        &inputs.tokenizer,
        &crate::prompt_identity::root(&inputs.tokenizer, &inputs.witness),
        &inputs.bound,
        &inputs.previous,
        inputs.marks.starts.as_slice(),
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
        raster::entry_argument_spec::<PromptState>("bound"),
        raster::entry_argument_spec::<PromptState>("previous"),
        raster::entry_argument_spec::<TokenMarks>("marks"),
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
    let bound = match cached.get("bound") {
        Some(value) => value.as_prompt_state()?,
        None => raster::materialize_auth_return(raster::entry_argument_auth_ref::<PromptState>(
            binding.reference.clone(),
            "bound",
        )),
    };
    let previous = match cached.get("previous") {
        Some(value) => value.as_prompt_state()?,
        None => raster::materialize_auth_return(raster::entry_argument_auth_ref::<PromptState>(
            binding.reference.clone(),
            "previous",
        )),
    };
    let marks = raster::materialize_auth_return(raster::entry_argument_auth_ref::<TokenMarks>(
        binding.reference,
        "marks",
    ));
    Ok(Inputs {
        witness,
        tokenizer,
        bound,
        previous,
        marks,
    })
}
