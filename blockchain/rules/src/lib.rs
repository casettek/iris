//! What a claim, a challenge and a contract hold, and the contract's rules in
//! Rust: the submissions, the pins, data availability and the registry.

pub mod availability;
pub mod challenge;
pub mod checks;
pub mod claim;
pub mod registry;

pub use challenge::{
    read_challenge_bundle, write_challenge_artifacts, ChallengeBundle, ChallengeSourcePath,
    ChallengeTrace, ReplayPackage, CHALLENGE_BUNDLE_JSON, CHALLENGE_TRACE_JSON,
    REPLAY_PACKAGE_JSON,
};
pub use claim::{
    read_claim_bundle, write_claim_artifacts, write_claim_artifacts_with_prepared_run, ClaimBundle,
    ClaimEndpoint, CLAIM_BUNDLE_JSON,
};

#[cfg(test)]
mod tests {
    use super::*;
    use procedure_definition::*;
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
        let (list_path, hashes_path, bundle_path) =
            write_claim_artifacts(&base, &manifest_path).unwrap();
        assert_eq!(list_path.file_name().unwrap(), CHECKPOINT_LIST_JSON);
        assert_eq!(hashes_path.file_name().unwrap(), CHECKPOINT_HASHES_TXT);
        assert_eq!(bundle_path.file_name().unwrap(), CLAIM_BUNDLE_JSON);

        let list: CheckpointList = serde_json::from_slice(&fs::read(&list_path).unwrap()).unwrap();
        let bundle: ClaimBundle = serde_json::from_slice(&fs::read(&bundle_path).unwrap()).unwrap();

        assert_eq!(list.checkpoints[0].stage, "stage_b");
        assert_eq!(list.checkpoints[0].output_commitment, "bbb");
        assert_eq!(list.checkpoints[1].stage, "stage_a");
        assert_eq!(checkpoint_hashes(&list).unwrap().len(), 2);
        assert_eq!(fs::read_to_string(&hashes_path).unwrap().lines().count(), 2);
        assert!(list_path.is_file());
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
    fn claim_artifacts_round_trip_as_json() {
        let list = CheckpointList {
            checkpoints: vec![Checkpoint {
                stage: String::from("check_c0"),
                program_commitment: String::from("program"),
                input_sources: Default::default(),
                input_commitment: String::from("input"),
                output_commitment: String::from("abc"),
                output_sha256: String::from("deadbeef"),
            }],
        };
        let bundle = ClaimBundle {
            version: 1,
            checkpoint_list_path: PathBuf::from("checkpoint_list.json"),
            input: ClaimEndpoint {
                stage: String::from("prompt_bind"),
                commitment: String::from("input"),
            },
            output: ClaimEndpoint {
                stage: String::from("check_c0"),
                commitment: String::from("abc"),
            },
            prepared_run_path: None,
        };

        assert_eq!(
            serde_json::from_slice::<CheckpointList>(&serde_json::to_vec(&list).unwrap()).unwrap(),
            list
        );
        assert_eq!(
            serde_json::from_slice::<ClaimBundle>(&serde_json::to_vec(&bundle).unwrap()).unwrap(),
            bundle
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
        assert_eq!(bundle.source_list_path, Some(PathBuf::from("claimed.json")));
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
            &divergence.recomputed_list_path,
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
                &divergence.recomputed_list_path,
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
                &divergence.recomputed_list_path,
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
            &divergence.recomputed_list_path,
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
            version: 2,
            checkpoint_index: 0,
            stage: String::from("stage_a"),
            reason: DivergenceReason::OutputCommitment,
            claimed_list_path: Some(PathBuf::from("claimed.json")),
            claimed_checkpoint_hashes_path: None,
            recomputed_list_path: PathBuf::from("recomputed.json"),
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
        divergence.claimed_list_path = None;
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
        std::env::temp_dir().join(format!("iris-cli-{label}-{}-{nanos}", std::process::id()))
    }
}
