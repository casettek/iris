pub mod challenge;
pub mod checkpoint;
pub mod claim;
pub mod direct;
pub mod inference;
pub mod io;
pub mod manifest;
pub mod run;
pub mod replay_profile;

pub use challenge::{
    read_challenge_bundle, write_challenge_artifacts, ChallengeBundle, ChallengeSourcePath,
    ChallengeTrace, Divergence, DivergenceReason, ReplayPackage, CHALLENGE_BUNDLE_JSON,
    CHALLENGE_TRACE_JSON, DIVERGENCE_JSON, REPLAY_PACKAGE_JSON,
};
pub use checkpoint::{
    build_checkpoint_trace, checkpoint_hash, checkpoint_hashes, read_checkpoint_from_stage_dir,
    read_checkpoint_hashes, read_checkpoint_trace, read_stage_order, write_checkpoint_hashes,
    write_checkpoint_hashes_artifact, write_checkpoint_trace_artifact, Checkpoint, CheckpointTrace,
    CHECKPOINTS_TXT, CHECKPOINT_HASHES_TXT, CHECKPOINT_TRACE_JSON, EXECUTION_TIMES_JSON,
};
pub use claim::{
    read_claim_bundle, write_claim_artifacts, write_claim_artifacts_with_prepared_run, ClaimBundle,
    ClaimEndpoint, CLAIM_BUNDLE_JSON,
};
pub use direct::{
    file_sha256, verify_file_sha256, DirectInferBundle, DirectInferProvenance, DirectInferShape,
    ModelManifest, MODEL_ARTIFACTS_DIR, MODEL_MANIFEST_JSON,
};
pub use inference::{
    InferAuxWaveTiming, InferStageTiming, InferenceResult, InferenceRunReport, InferenceTimings,
};
pub use io::{read_json, write_json};
pub use manifest::resolve_raster_manifest_paths;
pub use run::{
    read_run_spec, InferenceRunSpec, PreparedPiece, PreparedPrompt, PreparedRun,
    INFERENCE_RUN_SPEC_TOML, PREPARED_RUN_JSON, PREPARED_RUN_VERSION,
};

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn claim_artifacts_summarize_chain_outputs() {
        let base = temp_dir("claim-artifacts");
        fs::create_dir_all(&base).unwrap();
        write_stage(&base, "stage_a", "aaa", b"stage-a");
        write_stage(&base, "stage_b", "bbb", b"stage-b");
        fs::write(
            base.join(EXECUTION_TIMES_JSON),
            r#"{"version":2,"stages":[{"name":"stage_b","exec_duration_ns":20},{"name":"stage_a","exec_duration_ns":10}],"total_exec_duration_ns":30}"#,
        )
        .unwrap();

        let manifest_path = base.join("Raster.toml");
        fs::write(&manifest_path, "[chain]\nname = \"test\"\n").unwrap();
        let (trace_path, hashes_path, bundle_path) =
            write_claim_artifacts(&base, &manifest_path).unwrap();
        assert_eq!(trace_path.file_name().unwrap(), CHECKPOINT_TRACE_JSON);
        assert_eq!(hashes_path.file_name().unwrap(), CHECKPOINT_HASHES_TXT);
        assert_eq!(bundle_path.file_name().unwrap(), CLAIM_BUNDLE_JSON);

        let trace: CheckpointTrace =
            serde_json::from_slice(&fs::read(&trace_path).unwrap()).unwrap();
        let bundle: ClaimBundle = serde_json::from_slice(&fs::read(&bundle_path).unwrap()).unwrap();

        assert_eq!(trace.checkpoints[0].stage, "stage_b");
        assert_eq!(trace.checkpoints[0].output_commitment, "bbb");
        assert_eq!(trace.checkpoints[1].stage, "stage_a");
        assert_eq!(checkpoint_hashes(&trace).unwrap().len(), 2);
        assert_eq!(fs::read_to_string(&hashes_path).unwrap().lines().count(), 2);
        assert!(trace_path.is_file());
        assert!(hashes_path.is_file());
        let input_commitment = format!("{:x}", Sha256::digest(b"input-manifest"));
        assert_eq!(bundle.input.stage, "stage_b");
        assert_eq!(bundle.input.commitment, input_commitment);
        assert_eq!(
            bundle.output,
            ClaimEndpoint {
                stage: String::from("stage_a"),
                commitment: String::from("aaa")
            }
        );

        fs::remove_dir_all(base).unwrap();
    }

    #[test]
    fn checkpoint_trace_without_execution_times_uses_stage_name_order() {
        let base = temp_dir("claim-artifacts-fallback-order");
        fs::create_dir_all(&base).unwrap();
        write_stage(&base, "stage_b", "bbb", b"stage-b");
        write_stage(&base, "stage_a", "aaa", b"stage-a");
        fs::write(base.join("not-a-stage.txt"), b"ignored").unwrap();
        fs::create_dir_all(base.join("scratch")).unwrap();

        let manifest_path = base.join("Raster.toml");
        fs::write(&manifest_path, "[chain]\nname = \"test\"\n").unwrap();
        let trace = build_checkpoint_trace(&base, &manifest_path).unwrap();

        assert_eq!(
            trace
                .checkpoints
                .iter()
                .map(|checkpoint| checkpoint.stage.as_str())
                .collect::<Vec<_>>(),
            ["stage_a", "stage_b"]
        );

        fs::remove_dir_all(base).unwrap();
    }

    #[test]
    fn raster_chain_commitment_preserves_order_without_timing_report() {
        use raster_core::chain::{ChainCommitment, ChainShape, StageCheckpoint};
        let base = temp_dir("chain-commitment-order");
        fs::create_dir_all(&base).unwrap();
        for name in ["prompt_merge_seed", "prompt_merge_b0", "prompt_prepare"] {
            write_stage(&base, name, "abc", name.as_bytes());
        }
        let expected = vec![
            "prompt_merge_seed".to_string(),
            "prompt_merge_b0".to_string(),
            "prompt_prepare".to_string(),
        ];
        let chain = ChainCommitment {
            stages: expected
                .iter()
                .map(|name| StageCheckpoint {
                    name: name.clone(),
                    program_commitment: vec![],
                    input_manifest_commitment: vec![],
                    input_bindings: Default::default(),
                    output_payload_commitment: vec![],
                    output_structural_commitment: vec![],
                })
                .collect(),
            shape: ChainShape {
                spec_digest: vec![],
                repeats: vec![],
            },
        };
        fs::write(
            base.join("chain-commitment"),
            postcard::to_allocvec(&chain).unwrap(),
        )
        .unwrap();
        assert!(!base.join(EXECUTION_TIMES_JSON).exists());
        assert_eq!(read_stage_order(&base).unwrap(), Some(expected.clone()));
        let trace = build_checkpoint_trace(&base, &base.join("Raster.toml")).unwrap();
        assert_eq!(
            trace
                .checkpoints
                .iter()
                .map(|c| c.stage.clone())
                .collect::<Vec<_>>(),
            expected
        );

        // A present but damaged commitment must not be hidden by legacy timing data.
        fs::write(
            base.join(EXECUTION_TIMES_JSON),
            r#"{"stages":[{"name":"prompt_prepare"}]}"#,
        )
        .unwrap();
        fs::write(base.join("chain-commitment"), b"invalid").unwrap();
        assert!(read_stage_order(&base)
            .unwrap_err()
            .to_string()
            .contains("chain-commitment"));
        fs::remove_dir_all(base).unwrap();
    }

    #[test]
    fn checkpoint_hashes_round_trip_from_text_file() {
        let base = temp_dir("checkpoint-hashes");
        fs::create_dir_all(&base).unwrap();
        let path = base.join(CHECKPOINT_HASHES_TXT);
        let hashes = vec![
            String::from("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"),
            String::from("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"),
        ];

        write_checkpoint_hashes(&path, &hashes).unwrap();

        assert_eq!(read_checkpoint_hashes(&path).unwrap(), hashes);

        fs::remove_dir_all(base).unwrap();
    }

    #[test]
    fn checkpoint_hashes_reject_blank_lines() {
        let base = temp_dir("checkpoint-hashes-blank");
        fs::create_dir_all(&base).unwrap();
        let path = base.join(CHECKPOINT_HASHES_TXT);
        fs::write(
            &path,
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n\n",
        )
        .unwrap();

        assert!(read_checkpoint_hashes(&path)
            .unwrap_err()
            .to_string()
            .contains("invalid checkpoint hash on line 2"));

        fs::remove_dir_all(base).unwrap();
    }

    #[test]
    fn claim_artifacts_round_trip_as_json() {
        let trace = CheckpointTrace {
            checkpoints: vec![Checkpoint {
                stage: String::from("output_finalize"),
                input_commitment: String::from("input"),
                output_commitment: String::from("abc"),
                output_sha256: String::from("deadbeef"),
            }],
        };
        let bundle = ClaimBundle {
            version: 1,
            checkpoint_trace_path: PathBuf::from("checkpoint_trace.json"),
            input: ClaimEndpoint {
                stage: String::from("prompt_prepare"),
                commitment: String::from("input"),
            },
            output: ClaimEndpoint {
                stage: String::from("output_finalize"),
                commitment: String::from("abc"),
            },
            prepared_run_path: None,
        };

        assert_eq!(
            serde_json::from_slice::<CheckpointTrace>(&serde_json::to_vec(&trace).unwrap())
                .unwrap(),
            trace
        );
        assert_eq!(
            serde_json::from_slice::<ClaimBundle>(&serde_json::to_vec(&bundle).unwrap()).unwrap(),
            bundle
        );
    }

    #[test]
    fn model_manifest_round_trips_without_run_prompt() {
        let manifest = ModelManifest {
            version: 2,
            bundle: DirectInferBundle {
                model_detwgt_path: PathBuf::from("../model.detwgt"),
                model_detwgt_sha256: String::from("model-sha"),
                config_path: PathBuf::from("../config.json"),
                config_sha256: String::from("config-sha"),
                tokenizer_path: PathBuf::from("../tokenizer.json"),
                tokenizer_sha256: String::from("tokenizer-sha"),
            },
            shape: DirectInferShape {
                hidden_size: 4,
                num_hidden_layers: 1,
                num_attention_heads: 2,
                num_key_value_heads: 1,
                head_dim: 2,
                global_head_dim: 2,
                vocab_size: 8,
                hidden_size_per_layer_input: 2,
                sliding_window: 16,
                layer_types: vec![String::from("sliding_attention")],
                num_kv_shared_layers: 0,
                norm_eps: 0,
                rope_base_sliding: 10_000_i64 << 32,
                rope_base_full: 1_000_000_i64 << 32,
                full_partial_rotary_factor_q16: 1 << 16,
                embedding_scale: 1 << 16,
                ple_embedding_scale: 1 << 16,
                ple_projection_scalar: 1 << 16,
                ple_input_scale: 1 << 16,
                final_logit_softcap: 0,
            },
            eos_token_ids: vec![1, 2],
            provenance: Some(DirectInferProvenance {
                raster_manifest_path: PathBuf::from("../Raster.toml"),
                raster_manifest_sha256: String::from("raster-sha"),
            }),
        };

        assert_eq!(
            serde_json::from_slice::<ModelManifest>(&serde_json::to_vec(&manifest).unwrap())
                .unwrap(),
            manifest
        );
    }

    #[test]
    fn run_spec_requires_exactly_one_prompt_source() {
        let both = InferenceRunSpec {
            model_manifest: PathBuf::from("model-artifacts/manifest.json"),
            prompt: Some(String::from("hello")),
            prompt_file: Some(PathBuf::from("prompt.txt")),
            raw_prompt: false,
            tokens: 1,
        };
        assert!(both
            .validate()
            .unwrap_err()
            .to_string()
            .contains("not both"));

        let neither = InferenceRunSpec {
            model_manifest: PathBuf::from("model-artifacts/manifest.json"),
            prompt: None,
            prompt_file: None,
            raw_prompt: false,
            tokens: 1,
        };
        assert!(neither
            .validate()
            .unwrap_err()
            .to_string()
            .contains("prompt"));
    }

    #[test]
    fn prepared_run_metadata_round_trips_as_json() {
        let prepared = PreparedRun {
            version: PREPARED_RUN_VERSION,
            run_spec_path: PathBuf::from("inference.toml"),
            model_manifest_path: PathBuf::from("model-artifacts/manifest.json"),
            model_manifest_sha256: String::from("model-sha"),
            prompt: PreparedPrompt {
                resolved_prompt: String::from("hello"),
                rendered_prompt: String::from("hello"),
                initial_pieces: vec![PreparedPiece {
                    text: String::from("hello"),
                    segment: 0,
                }],
                eos_token_ids: vec![1, 2],
            },
            tokens: 2,
            run_manifest_path: Some(PathBuf::from("target/runs/run/Raster.toml")),
            run_manifest_sha256: Some(String::from("run-sha")),
        };

        assert_eq!(
            serde_json::from_slice::<PreparedRun>(&serde_json::to_vec(&prepared).unwrap()).unwrap(),
            prepared
        );
    }

    #[test]
    fn challenge_artifacts_summarize_raster_replay() {
        let base = temp_dir("challenge-artifacts");
        let (divergence, replay_package) = matching_challenge_fixture(&base);

        let (divergence_path, replay_package_path, challenge_trace_path, bundle_path) =
            write_challenge_artifacts(
                &base.join("challenge"),
                ChallengeSourcePath::Trace(Path::new("claimed.json")),
                Path::new("recomputed.json"),
                &divergence,
                &replay_package,
            )
            .unwrap();

        assert_eq!(divergence_path.file_name().unwrap(), DIVERGENCE_JSON);
        assert_eq!(
            replay_package_path.file_name().unwrap(),
            REPLAY_PACKAGE_JSON
        );
        assert_eq!(
            challenge_trace_path.file_name().unwrap(),
            CHALLENGE_TRACE_JSON
        );
        assert_eq!(bundle_path.file_name().unwrap(), CHALLENGE_BUNDLE_JSON);

        let bundle: ChallengeBundle =
            serde_json::from_slice(&fs::read(&bundle_path).unwrap()).unwrap();
        assert_eq!(bundle.stage, "stage_a");
        assert_eq!(
            bundle.source_trace_path,
            Some(PathBuf::from("claimed.json"))
        );
        assert_eq!(bundle.source_checkpoint_hashes_path, None);
        assert_eq!(bundle.raster_commit_path, replay_package.commit_path);

        let challenge_trace: ChallengeTrace =
            serde_json::from_slice(&fs::read(&challenge_trace_path).unwrap()).unwrap();
        assert_eq!(
            challenge_trace.raster_output_commitment,
            "raster-commitment"
        );

        fs::remove_dir_all(base).unwrap();
    }

    #[test]
    fn challenge_artifacts_accept_matching_replay_with_hash_claim() {
        let base = temp_dir("challenge-hash-parity");
        let (mut divergence, replay_package) = matching_challenge_fixture(&base);
        use_hash_claim(&mut divergence);

        let (_, _, _, bundle_path) = write_challenge_artifacts(
            &base.join("challenge"),
            ChallengeSourcePath::CheckpointHashes(Path::new("checkpoints.txt")),
            &divergence.recomputed_trace_path,
            &divergence,
            &replay_package,
        )
        .unwrap();

        let replay =
            read_checkpoint_from_stage_dir(&replay_package.stage, &replay_package.stage_dir)
                .unwrap();
        assert_eq!(
            checkpoint_hash(&replay).unwrap(),
            divergence.recomputed_hash.unwrap()
        );
        let bundle = read_challenge_bundle(&bundle_path).unwrap();
        assert_eq!(
            bundle.source_checkpoint_hashes_path,
            Some(PathBuf::from("checkpoints.txt"))
        );
        fs::remove_dir_all(base).unwrap();
    }

    #[test]
    fn challenge_artifacts_reject_replay_that_agrees_with_claimant() {
        for hash_claim in [false, true] {
            let base = temp_dir("challenge-replay-agrees-with-claimant");
            let (mut divergence, replay_package) = matching_challenge_fixture(&base);
            write_stage(
                &replay_package.replay_run_dir,
                &replay_package.stage,
                "claimed",
                b"claimed-output",
            );
            let replay =
                read_checkpoint_from_stage_dir(&replay_package.stage, &replay_package.stage_dir)
                    .unwrap();
            assert_eq!(divergence.claimed.as_ref(), Some(&replay));
            let source = if hash_claim {
                use_hash_claim(&mut divergence);
                assert_eq!(
                    divergence.claimed_hash.as_ref(),
                    Some(&checkpoint_hash(&replay).unwrap())
                );
                ChallengeSourcePath::CheckpointHashes(Path::new("checkpoints.txt"))
            } else {
                ChallengeSourcePath::Trace(Path::new("claimed.json"))
            };

            let error = write_challenge_artifacts(
                &base.join("challenge"),
                source,
                &divergence.recomputed_trace_path,
                &divergence,
                &replay_package,
            )
            .unwrap_err()
            .to_string();

            assert!(error.contains("native/Raster checkpoint parity mismatch"));
            assert!(error.contains("stage `stage_a`"));
            assert!(error.contains("output_commitment differs"));
            assert!(error.contains("native `raster-commitment`, Raster `claimed`"));
            assert_no_challenge_artifacts(&base.join("challenge"));
            // Keep replay evidence available to diagnose the parity failure.
            assert_eq!(
                fs::read(&replay_package.output_path).unwrap(),
                b"claimed-output"
            );
            fs::remove_dir_all(base).unwrap();
        }
    }

    #[test]
    fn challenge_artifacts_reject_each_checkpoint_field_mismatch() {
        for field in [
            "stage",
            "input_commitment",
            "output_commitment",
            "output_sha256",
        ] {
            let base = temp_dir("challenge-checkpoint-mismatch");
            let (divergence, mut replay_package) = matching_challenge_fixture(&base);
            match field {
                "stage" => replay_package.stage = String::from("other_stage"),
                "input_commitment" => {
                    fs::write(&replay_package.input_manifest_path, b"different-input").unwrap();
                }
                "output_commitment" => {
                    fs::write(
                        &replay_package.output_manifest_path,
                        r#"{"output":{"commitment":"different-commitment"}}"#,
                    )
                    .unwrap();
                }
                "output_sha256" => {
                    fs::write(&replay_package.output_path, b"different-output").unwrap();
                }
                _ => unreachable!(),
            }

            let error = write_challenge_artifacts(
                &base.join("challenge"),
                ChallengeSourcePath::Trace(Path::new("claimed.json")),
                &divergence.recomputed_trace_path,
                &divergence,
                &replay_package,
            )
            .unwrap_err()
            .to_string();

            assert!(error.contains("native/Raster checkpoint parity mismatch"));
            assert!(error.contains(&format!("{field} differs")), "{error}");
            assert_no_challenge_artifacts(&base.join("challenge"));
            fs::remove_dir_all(base).unwrap();
        }
    }

    #[test]
    fn challenge_artifacts_require_recomputed_checkpoint() {
        let base = temp_dir("challenge-missing-recomputed-checkpoint");
        let (mut divergence, replay_package) = matching_challenge_fixture(&base);
        divergence.recomputed = None;

        let error = write_challenge_artifacts(
            &base.join("challenge"),
            ChallengeSourcePath::Trace(Path::new("claimed.json")),
            &divergence.recomputed_trace_path,
            &divergence,
            &replay_package,
        )
        .unwrap_err()
        .to_string();

        assert!(error.contains("missing recomputed checkpoint"));
        assert_no_challenge_artifacts(&base.join("challenge"));
        fs::remove_dir_all(base).unwrap();
    }

    fn matching_challenge_fixture(base: &Path) -> (Divergence, ReplayPackage) {
        let native_run_dir = base.join("native");
        let replay_run_dir = base.join("challenge").join("raster-replay");
        for run_dir in [&native_run_dir, &replay_run_dir] {
            write_stage(run_dir, "stage_a", "raster-commitment", b"raster-output");
        }
        let recomputed =
            read_checkpoint_from_stage_dir("stage_a", &native_run_dir.join("stage_a")).unwrap();
        let claimed = Checkpoint {
            output_commitment: String::from("claimed"),
            output_sha256: format!("{:x}", Sha256::digest(b"claimed-output")),
            ..recomputed.clone()
        };
        let divergence = Divergence {
            version: 1,
            checkpoint_index: 0,
            stage: String::from("stage_a"),
            reason: DivergenceReason::OutputCommitment,
            claimed_trace_path: Some(PathBuf::from("claimed.json")),
            claimed_checkpoint_hashes_path: None,
            recomputed_trace_path: PathBuf::from("recomputed.json"),
            claimed_hash: None,
            recomputed_hash: None,
            claimed: Some(claimed),
            recomputed: Some(recomputed),
        };
        let stage_dir = replay_run_dir.join("stage_a");
        fs::write(stage_dir.join("input.json"), b"{}").unwrap();
        fs::write(stage_dir.join("commit.bin"), b"commit").unwrap();
        let replay_package = ReplayPackage {
            version: 1,
            stage: String::from("stage_a"),
            replay_run_dir,
            stage_dir: stage_dir.clone(),
            input_path: stage_dir.join("input.json"),
            input_manifest_path: stage_dir.join("input_manifest.json"),
            output_path: stage_dir.join("output.bin"),
            output_index_path: stage_dir.join("output.rindex"),
            output_manifest_path: stage_dir.join("output_manifest.json"),
            commit_path: stage_dir.join("commit.bin"),
        };
        (divergence, replay_package)
    }

    fn use_hash_claim(divergence: &mut Divergence) {
        divergence.reason = DivergenceReason::CheckpointHash;
        divergence.claimed_trace_path = None;
        divergence.claimed_checkpoint_hashes_path = Some(PathBuf::from("checkpoints.txt"));
        divergence.claimed_hash =
            Some(checkpoint_hash(&divergence.claimed.take().unwrap()).unwrap());
        divergence.recomputed_hash =
            Some(checkpoint_hash(divergence.recomputed.as_ref().unwrap()).unwrap());
    }

    fn assert_no_challenge_artifacts(challenge_dir: &Path) {
        for filename in [
            DIVERGENCE_JSON,
            REPLAY_PACKAGE_JSON,
            CHALLENGE_TRACE_JSON,
            CHALLENGE_BUNDLE_JSON,
        ] {
            assert!(!challenge_dir.join(filename).exists(), "wrote {filename}");
        }
    }

    fn write_stage(base: &Path, stage: &str, commitment: &str, output: &[u8]) {
        let stage_dir = base.join(stage);
        fs::create_dir_all(&stage_dir).unwrap();
        fs::write(stage_dir.join("input_manifest.json"), b"input-manifest").unwrap();
        fs::write(stage_dir.join("output.bin"), output).unwrap();
        fs::write(stage_dir.join("output.rindex"), b"index").unwrap();
        fs::write(
            stage_dir.join("output_manifest.json"),
            format!(
                r#"{{"output":{{"type":"sha256","encoding":"raster","commitment":"{commitment}"}}}}"#
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
            "iris-cli-{label}-{}-{nanos}",
            std::process::id()
        ))
    }
}
