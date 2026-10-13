use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use decode_select_token::input::{LogitEntry, PrefillLogits};
use output_finalize::input::{DecoderTable, DecoderToken};
use raster::List;
use serde::Serialize;
use sha2::{Digest, Sha256};
use staged_infer::chain_runner::StagedExecutionBackend;
use staged_infer::{
    CheckpointedInferenceConfig, CheckpointedInferenceExecutor, UncheckpointedInferenceConfig,
    UncheckpointedInferenceExecutor,
};

#[test]
fn infer_final_result_matches_checkpointed_claim_path() {
    let base = temp_dir("infer-parity");
    fs::create_dir_all(&base).unwrap();

    let logits = PrefillLogits {
        decode_position: 1,
        logits: List::from(vec![
            LogitEntry {
                token_id: 0,
                value: 10,
            },
            LogitEntry {
                token_id: 1,
                value: 20,
            },
        ]),
        errors: List::new(),
    };
    let decoder = DecoderTable {
        tokens: List::from(vec![
            DecoderToken {
                token: String::from("<pad>"),
                special: true,
                terminal: false,
            },
            DecoderToken {
                token: String::from("Hi"),
                special: false,
                terminal: false,
            },
        ]),
    };

    let logits_commitment = write_external(&base, "logits", &logits);
    let decoder_commitment = write_external(&base, "decoder", &decoder);
    write_manifest(&base, &logits_commitment, &decoder_commitment);

    let infer_report = UncheckpointedInferenceExecutor
        .run_with_report(UncheckpointedInferenceConfig {
            base_dir: base.clone(),
            manifest_path: base.join("Raster.toml"),
        })
        .unwrap();
    let infer_result = infer_report.result;
    assert!(
        !base.join("target").exists(),
        "infer should not write checkpoint artifacts under the imported workspace"
    );
    assert_eq!(infer_report.timings.stages.len(), 3);
    assert!(infer_report.timings.aux_waves.is_empty());

    let checkpointed = CheckpointedInferenceExecutor
        .run(CheckpointedInferenceConfig {
            base_dir: base.clone(),
            manifest_path: base.join("Raster.toml"),
            current_exe: PathBuf::from("unused-for-in-process"),
            staged_backend: StagedExecutionBackend::InProcess,
            parity_policy: staged_infer::ParityPolicy::Skip,
        })
        .unwrap();
    let checkpointed_result = checkpointed
        .final_result
        .expect("checkpointed decode chain should end in GeneratedOutput");

    assert_eq!(infer_result, checkpointed_result);
    assert_eq!(infer_result.generated_token_count, 1);
    assert_eq!(infer_result.generated_token_ids, vec![1]);
    assert_eq!(infer_result.generated_text, "Hi");
    assert_eq!(infer_result.stop_reason, "max_new_tokens");
    assert_eq!(
        infer_result.generated_token_ids_sha256,
        format!("{:x}", Sha256::digest(b"[1]"))
    );
    assert!(checkpointed
        .chain_dir
        .join("output_finalize")
        .join("output.bin")
        .is_file());

    fs::remove_dir_all(base).unwrap();
}

fn write_external<T: Serialize>(base: &Path, name: &str, value: &T) -> String {
    raster::init();
    let (data, index, commitment) = raster::encode_raster_value(value).unwrap();
    fs::write(base.join(format!("{name}.rastered")), data).unwrap();
    fs::write(base.join(format!("{name}.rindex")), index).unwrap();
    commitment
}

fn write_manifest(base: &Path, logits_commitment: &str, decoder_commitment: &str) {
    fs::write(
        base.join("Raster.toml"),
        format!(
            r#"[chain]
name = "infer-parity"
version = "0.1.0"

[[chain.stage]]
name = "decode_init"
project = "raster-stages/decode-init"

[[chain.stage]]
name = "decode_select_token"
project = "raster-stages/decode-select-token"
inputs.logits = {{ external = {{ path = "logits.rastered", index_path = "logits.rindex", commitment = "{logits_commitment}" }} }}
inputs.prior = {{ from = "decode_init" }}

[[chain.stage]]
name = "output_finalize"
project = "raster-stages/output-finalize"
inputs.edge = {{ from = "decode_select_token" }}
inputs.decoder = {{ external = {{ path = "decoder.rastered", index_path = "decoder.rindex", commitment = "{decoder_commitment}" }} }}
"#
        ),
    )
    .unwrap();
}

fn temp_dir(label: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!(
        "staged-infer-{label}-{}-{nanos}",
        std::process::id()
    ))
}
