use std::path::Path;

use anyhow::Result;
use claim_result::input::DecodeEdge;
use source_phase::relay::{run_relay_steps, RelayRecipe};
use source_phase::SourceRecord;

use crate::artifact_io::{with_main_sequence_scope, with_stage_sequence_scope};
use crate::cache::{CachedInputs, MaterializationCache};

/// The `prompt-relay` stage's inputs. The recipe's type is `source-phase`'s,
/// which the stage program compiles from the same source file, and the edge
/// is read through the copy of its type the result stages read it by.
pub struct Inputs {
    /// The last check of the inference whose output is relayed.
    pub check: DecodeEdge,
    /// The text before and after that output in the prompt it makes.
    pub recipe: RelayRecipe,
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
pub fn run_native(inputs: &Inputs) -> Result<SourceRecord> {
    let claim = &inputs.check.claim;
    Ok(run_relay_steps(
        &inputs.recipe,
        &inputs.check.record.generated_text,
        claim.statement.as_slice(),
        claim.output_text_sha256.as_slice(),
        claim.valid,
    )
    .record)
}

fn load_inputs() -> Result<Inputs> {
    let binding = raster::start_program(&[
        raster::entry_argument_spec::<DecodeEdge>("check"),
        raster::entry_argument_spec::<RelayRecipe>("recipe"),
    ])?;
    let check = raster::materialize_auth_return(raster::entry_argument_auth_ref::<DecodeEdge>(
        binding.reference.clone(),
        "check",
    ));
    let recipe = raster::materialize_auth_return(raster::entry_argument_auth_ref::<RelayRecipe>(
        binding.reference,
        "recipe",
    ));
    Ok(Inputs { check, recipe })
}

/// This stage as the runner runs it.
pub struct Stage;

impl crate::segments::NativeStage for Stage {
    type Inputs = Inputs;
    type Output = SourceRecord;
    const PROJECT: &'static str = "prompt-relay";
    const NAME: &'static str = "prompt_relay";

    fn load_from_args() -> Result<Inputs> {
        load_inputs_from_args()
    }

    fn load_from_paths(
        input: &std::path::Path,
        input_manifest: &std::path::Path,
        cached_inputs: &crate::cache::CachedInputs,
        materializations: Option<&crate::cache::MaterializationCache>,
    ) -> Result<Inputs> {
        load_inputs_from_paths_with_cache(input, input_manifest, cached_inputs, materializations)
    }

    fn run(
        inputs: &Inputs,
        _caches: crate::segments::NativeStageCaches<'_>,
    ) -> Result<SourceRecord> {
        run_native(inputs)
    }

    fn cached(output: SourceRecord) -> crate::cache::CachedStageValue {
        crate::cache::CachedStageValue::SourceRecord(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};
    use source_phase::relay::{relay_recipe, FAULT_RELAYED};

    /// The stage reads the earlier inference's last check as that check
    /// writes it, and the record it states is the one the first bind of the
    /// later inference reads.
    #[test]
    fn the_record_is_of_the_output_the_check_wrote_and_reads_as_a_source_record() {
        raster::init();
        let output = " Paris\n";
        let mut written = sequence_check::input::DecodeEdge::default();
        written.record.generated_text = String::from(output);
        written.claim.valid = true;
        written.claim.statement = raster::BytesPage::__from_parts(0, 0, vec![0x5a; 32]);
        written.claim.output_text_sha256 =
            raster::BytesPage::__from_parts(0, 0, Sha256::digest(output.as_bytes()).to_vec());
        let read = |edge: &sequence_check::input::DecodeEdge| -> DecodeEdge {
            serde_json::from_value(serde_json::to_value(edge).unwrap()).unwrap()
        };

        let recipe = relay_recipe("Is this right:", "Answer yes or no.");
        let record = run_native(&Inputs {
            check: read(&written),
            recipe: recipe.clone(),
        })
        .unwrap();
        assert!(record.present);
        assert_eq!(record.fault, 0);
        let text = format!("Is this right:{output}Answer yes or no.");
        assert_eq!(record.text_total, text.len() as u64);
        assert_eq!(
            record.text_sha256,
            hex::encode(Sha256::digest(text.as_bytes()))
        );

        // The bind reads the record through the prompt stages' own type,
        // which must commit as this one does.
        let bound: prompt_check::input::SourceRecord =
            serde_json::from_value(serde_json::to_value(&record).unwrap()).unwrap();
        assert_eq!(
            raster::encode_raster_value(&bound).unwrap(),
            raster::encode_raster_value(&record).unwrap()
        );

        // An earlier inference with no valid record is a fault in the
        // relay's, which the bind turns into a fault of the later prompt.
        written.claim.valid = false;
        let record = run_native(&Inputs {
            check: read(&written),
            recipe,
        })
        .unwrap();
        assert_eq!(record.fault, FAULT_RELAYED);
    }
}
