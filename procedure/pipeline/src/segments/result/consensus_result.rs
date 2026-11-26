use std::path::Path;

use anyhow::Result;
use consensus_result::consensus::{
    close_consensus, run_consensus, same_input, Claimed, ConsensusRecord,
};
use consensus_result::forms::ResultForms;
use consensus_result::input::DecodeEdge;

use crate::artifact_io::{with_main_sequence_scope, with_stage_sequence_scope};
use crate::cache::{CachedInputs, MaterializationCache};

/// The `consensus-result` stage's inputs.
pub struct Inputs {
    /// The first inference's last check's output.
    pub first: DecodeEdge,
    /// The second inference's.
    pub second: DecodeEdge,
    /// The texts the run's outputs are compared with.
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
pub fn run_native(inputs: &Inputs) -> Result<ConsensusRecord> {
    let (first, second) = (&inputs.first, &inputs.second);
    let matched = run_consensus(
        inputs.forms.forms.iter(),
        &first.record.generated_text,
        &second.record.generated_text,
        same_input(
            &first.record.input_text_sha256,
            &first.record.source_sha256,
            &second.record.input_text_sha256,
            &second.record.source_sha256,
        ),
    );
    let claimed = |edge: &'_ DecodeEdge| -> (Vec<u8>, Vec<u8>, bool) {
        (
            edge.claim.statement.as_slice().to_vec(),
            edge.claim.output_text_sha256.as_slice().to_vec(),
            edge.claim.valid,
        )
    };
    let (first, second) = (claimed(first), claimed(second));
    Ok(close_consensus(
        matched,
        Claimed {
            statement: &first.0,
            output_text_sha256: &first.1,
            valid: first.2,
        },
        Claimed {
            statement: &second.0,
            output_text_sha256: &second.1,
            valid: second.2,
        },
    ))
}

fn load_inputs() -> Result<Inputs> {
    let binding = raster::start_program(&[
        raster::entry_argument_spec::<DecodeEdge>("first"),
        raster::entry_argument_spec::<DecodeEdge>("second"),
        raster::entry_argument_spec::<ResultForms>("forms"),
    ])?;
    let first = raster::materialize_auth_return(raster::entry_argument_auth_ref::<DecodeEdge>(
        binding.reference.clone(),
        "first",
    ));
    let second = raster::materialize_auth_return(raster::entry_argument_auth_ref::<DecodeEdge>(
        binding.reference.clone(),
        "second",
    ));
    let forms = raster::materialize_auth_return(raster::entry_argument_auth_ref::<ResultForms>(
        binding.reference,
        "forms",
    ));
    Ok(Inputs {
        first,
        second,
        forms,
    })
}

/// This stage as the runner runs it.
pub struct Stage;

impl crate::segments::NativeStage for Stage {
    type Inputs = Inputs;
    type Output = ConsensusRecord;
    const PROJECT: &'static str = "consensus-result";
    const NAME: &'static str = "consensus_result";

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
    ) -> Result<ConsensusRecord> {
        run_native(inputs)
    }

    fn cached(output: ConsensusRecord) -> crate::cache::CachedStageValue {
        crate::cache::CachedStageValue::ConsensusRecord(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The stage reads two check outputs through its own copy of the edge
    /// type, and states what the native run computes from the same texts.
    #[test]
    fn the_stage_states_whether_two_checks_agree() {
        raster::init();
        let edge = |text: &str, valid: bool| -> DecodeEdge {
            let mut written = sequence_check::input::DecodeEdge::default();
            written.record.generated_text = String::from(text);
            written.record.input_text_sha256 = "ab".repeat(32);
            written.record.source_sha256 = "00".repeat(32);
            written.claim.valid = valid;
            let read: DecodeEdge =
                serde_json::from_value(serde_json::to_value(&written).unwrap()).unwrap();
            assert_eq!(
                raster::encode_raster_value(&read).unwrap(),
                raster::encode_raster_value(&written).unwrap()
            );
            read
        };
        let forms = || ResultForms {
            forms: vec![
                consensus_result::forms::Form {
                    text: String::from("no"),
                    code: 2,
                },
                consensus_result::forms::Form {
                    text: String::from("yes"),
                    code: 1,
                },
            ]
            .into(),
        };
        let run = |first: DecodeEdge, second: DecodeEdge| {
            run_native(&Inputs {
                first,
                second,
                forms: forms(),
            })
            .unwrap()
        };
        let record = run(edge("yes", true), edge(" yes\n", true));
        assert!(record.agree && record.same_input);
        assert_eq!(
            (record.code, record.first.code, record.second.code),
            (1, 1, 1)
        );

        let record = run(edge("yes", true), edge("no", true));
        assert_eq!(
            (record.agree, record.code, record.second.code),
            (false, 0, 2)
        );
        assert!(!run(edge("yes", true), edge("yes", false)).agree);
        let mut other_input = edge("yes", true);
        other_input.record.input_text_sha256 = "cd".repeat(32);
        let record = run(edge("yes", true), other_input);
        assert_eq!((record.agree, record.same_input), (false, false));
    }
}
