use std::path::Path;

use anyhow::Result;
use claim_result::forms::{close_result, run_forms, ResultForms, ResultRecord};
use claim_result::input::DecodeEdge;

use crate::artifact_io::{with_main_sequence_scope, with_stage_sequence_scope};
use crate::cache::{CachedInputs, MaterializationCache};

/// The `claim-result` stage's inputs.
pub struct Inputs {
    /// The last check's output.
    pub check: DecodeEdge,
    /// The texts the run's output is compared with.
    pub forms: ResultForms,
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
pub fn run_native(inputs: &Inputs) -> Result<ResultRecord> {
    let claim = &inputs.check.claim;
    Ok(close_result(
        run_forms(
            inputs.forms.forms.iter(),
            &inputs.check.record.generated_text,
        ),
        claim.statement.as_slice(),
        claim.output_text_sha256.as_slice(),
        claim.valid,
    ))
}

fn load_inputs() -> Result<Inputs> {
    let binding = raster::start_program(&[
        raster::entry_argument_spec::<DecodeEdge>("check"),
        raster::entry_argument_spec::<ResultForms>("forms"),
    ])?;
    let check = raster::materialize_auth_return(raster::entry_argument_auth_ref::<DecodeEdge>(
        binding.reference.clone(),
        "check",
    ));
    let forms = raster::materialize_auth_return(raster::entry_argument_auth_ref::<ResultForms>(
        binding.reference,
        "forms",
    ));
    Ok(Inputs { check, forms })
}

/// This stage as the runner runs it.
pub struct Stage;

impl crate::segments::NativeStage for Stage {
    type Inputs = Inputs;
    type Output = ResultRecord;
    const PROJECT: &'static str = "claim-result";
    const NAME: &'static str = "claim_result";

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
    ) -> Result<ResultRecord> {
        run_native(inputs)
    }

    fn cached(output: ResultRecord) -> crate::cache::CachedStageValue {
        crate::cache::CachedStageValue::ResultRecord(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The stage reads the last check output through its own copy of the
    /// edge type. The copy must commit and decode as the check type does,
    /// or the stage would be given another value than the check produced.
    #[test]
    fn the_stage_reads_an_edge_as_the_check_writes_it() {
        raster::init();
        let mut written = sequence_check::input::DecodeEdge {
            has_selected: true,
            decode_position: 7,
            token_id: 4_211,
            value: -19,
            claim_mismatches: 2,
            first_claim_mismatch_position: 5,
            check_fault: 3,
            check_fault_position: 6,
            generated_token_ids: vec![4_211, 9, 0].into(),
            ..Default::default()
        };
        written.record.generated_text = String::from(" yes\n");
        written.claim.valid = true;
        written.claim.statement = raster::BytesPage::__from_parts(0, 0, vec![0x5a; 32]);

        let read: DecodeEdge =
            serde_json::from_value(serde_json::to_value(&written).unwrap()).unwrap();
        assert_eq!(
            serde_json::to_value(&read).unwrap(),
            serde_json::to_value(&written).unwrap()
        );
        assert_eq!(
            raster::encode_raster_value(&read).unwrap(),
            raster::encode_raster_value(&written).unwrap()
        );

        let forms = ResultForms {
            forms: vec![
                claim_result::forms::Form {
                    text: String::from("no"),
                    code: 2,
                },
                claim_result::forms::Form {
                    text: String::from("yes"),
                    code: 1,
                },
            ]
            .into(),
        };
        let record = run_native(&Inputs { check: read, forms }).unwrap();
        assert_eq!(record.code, 1);
        assert!(record.valid);
        assert_eq!(record.statement.as_slice(), &[0x5a; 32]);
    }
}
