pub mod availability;
pub mod challenge;
pub mod checkpoint;
pub mod checks;
pub mod claim;
pub mod direct;
pub mod encoding;
pub mod inference;
pub mod io;
pub mod manifest;
pub mod model_identity;
pub mod procedure_kind;
pub mod registry;
pub mod replay_profile;
pub mod run;
pub mod specification;

pub use challenge::{
    read_challenge_bundle, write_challenge_artifacts, ChallengeBundle, ChallengeSourcePath,
    ChallengeTrace, Divergence, DivergenceReason, ReplayPackage, CHALLENGE_BUNDLE_JSON,
    CHALLENGE_TRACE_JSON, DIVERGENCE_JSON, REPLAY_PACKAGE_JSON,
};
pub use checkpoint::{
    build_checkpoint_list, checkpoint_hash, checkpoint_hashes, checkpoint_opening,
    checkpoints_root, read_checkpoint_from_stage_dir, read_checkpoint_hashes, read_checkpoint_list,
    read_stage_order, verify_checkpoint_opening, write_checkpoint_hashes,
    write_checkpoint_hashes_artifact, write_checkpoint_list_artifact, write_stage_program,
    Checkpoint, CheckpointList, CHECKPOINTS_TXT, CHECKPOINT_HASHES_TXT, CHECKPOINT_LIST_JSON,
    EXECUTION_TIMES_JSON, STAGE_PROGRAM_JSON,
};
pub use claim::{
    read_claim_bundle, write_claim_artifacts, write_claim_artifacts_with_prepared_run, ClaimBundle,
    ClaimEndpoint, CLAIM_BUNDLE_JSON,
};
pub use direct::{
    file_sha256, model_id_of, verify_file_sha256, DirectInferBundle, DirectInferProvenance,
    DirectInferShape, FamilyShape, MistralShape, ModelCommitment, ModelFamily, ModelManifest,
    MODEL_ARTIFACTS_DIR, MODEL_MANIFEST_JSON, MODEL_MANIFEST_VERSION,
};
pub use inference::{
    ConsensusOutcome, InferAuxWaveTiming, InferStageTiming, InferenceResult, InferenceRunReport,
    InferenceTimings,
};
pub use io::{read_json, write_json};
pub use manifest::resolve_raster_manifest_paths;
pub use procedure_kind::ProcedureKind;
pub use run::{
    PreparedInference, PreparedPiece, PreparedPrompt, PreparedRun, PreparedSourceInputs,
    RunFileRef, PREPARED_RUN_JSON, PREPARED_RUN_VERSION,
};
pub use specification::{
    read_case, read_specification, Case, CaseDocument, PathStep, Specification,
    SpecificationDocument, SpecificationField, SpecificationForm, SpecificationModel,
    SpecificationSource, CASE_TOML, SPECIFICATION_TOML,
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
    fn checkpoint_list_without_execution_times_uses_stage_name_order() {
        let base = temp_dir("claim-artifacts-fallback-order");
        fs::create_dir_all(&base).unwrap();
        write_stage(&base, "stage_b", "bbb", b"stage-b");
        write_stage(&base, "stage_a", "aaa", b"stage-a");
        fs::write(base.join("not-a-stage.txt"), b"ignored").unwrap();
        fs::create_dir_all(base.join("scratch")).unwrap();

        let manifest_path = base.join("Raster.toml");
        fs::write(&manifest_path, "[chain]\nname = \"test\"\n").unwrap();
        let list = build_checkpoint_list(&base, &manifest_path).unwrap();

        assert_eq!(
            list.checkpoints
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
        for name in ["prompt_bind", "prompt_check_p0", "embed_p0"] {
            write_stage(&base, name, "abc", name.as_bytes());
        }
        let expected = vec![
            "prompt_bind".to_string(),
            "prompt_check_p0".to_string(),
            "embed_p0".to_string(),
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
        let list = build_checkpoint_list(&base, &base.join("Raster.toml")).unwrap();
        assert_eq!(
            list.checkpoints
                .iter()
                .map(|c| c.stage.clone())
                .collect::<Vec<_>>(),
            expected
        );

        // A present but damaged commitment must not be hidden by legacy timing data.
        fs::write(
            base.join(EXECUTION_TIMES_JSON),
            r#"{"stages":[{"name":"embed_p0"}]}"#,
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
    fn the_model_id_covers_every_role_and_commitment_in_order() {
        let entry = |role: &str, commitment: &str| ModelCommitment {
            role: role.to_string(),
            commitment: commitment.to_string(),
        };
        let id = model_id_of(&[entry("embedding", "a"), entry("head", "b")]);
        assert_eq!(
            id,
            model_id_of(&[entry("embedding", "a"), entry("head", "b")])
        );
        for other in [
            vec![entry("embedding", "x"), entry("head", "b")],
            vec![entry("head", "a"), entry("embedding", "b")],
            vec![entry("head", "b"), entry("embedding", "a")],
            vec![entry("embedding", "a")],
            vec![entry("embeddinga", ""), entry("head", "b")],
        ] {
            assert_ne!(id, model_id_of(&other), "{other:?}");
        }
    }

    #[test]
    fn a_model_id_that_does_not_match_its_commitments_is_refused() {
        let dir = std::env::temp_dir().join(format!(
            "iris-model-id-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        for file in ["model.detwgt", "config.json", "tokenizer.json"] {
            std::fs::write(dir.join(file), file).unwrap();
        }
        let hash = |file: &str| file_sha256(&dir.join(file)).unwrap();
        let commitments = vec![ModelCommitment {
            role: String::from("embedding"),
            commitment: String::from("abc"),
        }];
        let mut value = serde_json::json!({
            "version": MODEL_MANIFEST_VERSION,
            "model_id": model_id_of(&commitments),
            "model_commitments": commitments,
            "bundle": {
                "model_detwgt_path": "model.detwgt", "model_detwgt_sha256": hash("model.detwgt"),
                "config_path": "config.json", "config_sha256": hash("config.json"),
                "tokenizer_path": "tokenizer.json", "tokenizer_sha256": hash("tokenizer.json"),
            },
            "shape": {
                "hidden_size": 2, "num_hidden_layers": 0, "num_attention_heads": 1,
                "num_key_value_heads": 1, "head_dim": 2, "global_head_dim": 2,
                "vocab_size": 1, "hidden_size_per_layer_input": 1, "sliding_window": 0,
                "layer_types": [], "num_kv_shared_layers": 0, "norm_eps": 0,
                "rope_base_sliding": 0, "rope_base_full": 0, "full_partial_rotary_factor_q16": 0,
                "embedding_scale": 0, "ple_embedding_scale": 0, "ple_projection_scalar": 0,
                "ple_input_scale": 0, "final_logit_softcap": 0
            },
            "eos_token_ids": [],
            "range_size": 4,
        });
        let path = dir.join("manifest.json");
        write_json(&path, &value).unwrap();
        // A manifest written before families were recorded is the first family's.
        let loaded = ModelManifest::load_verified(&path).unwrap();
        assert_eq!(loaded.family().unwrap(), ModelFamily::GemmaDense);
        assert_eq!(
            serde_json::to_value(&loaded).unwrap()["family"],
            serde_json::json!("gemma-dense")
        );
        // One this build has no layer for is refused before anything runs.
        value["family"] = serde_json::json!("qwen");
        write_json(&path, &value).unwrap();
        let error = ModelManifest::load_verified(&path).unwrap_err().to_string();
        assert!(
            error.contains("model family `qwen` is not supported"),
            "{error}"
        );
        assert!(error.contains("gemma-dense"), "{error}");
        value["family"] = serde_json::json!("gemma-dense");
        value["model_commitments"][0]["commitment"] = serde_json::json!("other");
        write_json(&path, &value).unwrap();
        assert!(ModelManifest::load_verified(&path)
            .unwrap_err()
            .to_string()
            .contains("model id does not match"));
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn model_manifest_round_trips_without_run_prompt() {
        let manifest = ModelManifest {
            version: MODEL_MANIFEST_VERSION,
            family: ModelFamily::GemmaDense.name().to_string(),
            model_id: model_id_of(&[]),
            model_commitments: Vec::new(),
            range_size: 512,
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
            }
            .into(),
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
    fn a_case_gives_its_prompt_one_way() {
        let specification = Specification {
            procedure_kind: Default::default(),
            raw_prompt: false,
            max_new_tokens: 1,
            models: vec![SpecificationModel {
                manifest: PathBuf::from("model-artifacts/manifest.json"),
            }],
            source: None,
            forms: Vec::new(),
        };
        let both = Case {
            prompt: Some(String::from("hello")),
            prompt_file: Some(PathBuf::from("prompt.txt")),
            documents: Vec::new(),
        };
        assert!(both
            .validate_for(&specification)
            .unwrap_err()
            .to_string()
            .contains("not both"));

        let neither = Case::default();
        assert!(neither
            .validate_for(&specification)
            .unwrap_err()
            .to_string()
            .contains("prompt"));
        assert!(Case::of_prompt(String::from("hello"))
            .validate_for(&specification)
            .is_ok());
    }

    const MODEL: &str = "[[model]]\nmanifest = \"m.json\"\n";

    #[test]
    fn a_run_spec_is_refused_by_what_replaced_its_keys() {
        let base = temp_dir("specification-run-spec");
        fs::create_dir_all(&base).unwrap();
        let path = base.join("specification.toml");
        fs::write(&path, MODEL).unwrap();
        assert_eq!(read_specification(&path).unwrap().max_new_tokens, 1);
        for (old, replacement) in [
            ("model_manifest = \"m.json\"\n", "`[[model]]` table"),
            ("tokens = 3\n", "`max_new_tokens`"),
            ("prompt = \"hi\"\n", "in the case file"),
            ("prompt_file = \"p.txt\"\n", "in the case file"),
            ("pipeline = \"consensus\"\n", "`procedure_kind`"),
            ("chain = \"teacher-forced\"\n", "only staged chain"),
        ] {
            fs::write(&path, format!("{old}{MODEL}")).unwrap();
            let error = format!("{:#}", read_specification(&path).unwrap_err());
            assert!(
                error.contains("is a run spec from before") && error.contains(replacement),
                "{error}"
            );
        }
        for (old, replacement) in [
            ("recipe = \"recipe.json\"\n", "`[[source.field]]`"),
            ("signer = \"signer.json\"\n", "`[source.signer]`"),
            ("documents = []\n", "in the case file"),
        ] {
            fs::write(&path, format!("{MODEL}[source]\n{old}")).unwrap();
            let error = format!("{:#}", read_specification(&path).unwrap_err());
            assert!(
                error.contains("is a run spec from before") && error.contains(replacement),
                "{error}"
            );
        }
        // A key nothing ever read is refused too, by the reader itself.
        fs::write(&path, format!("extra = \"x\"\n{MODEL}")).unwrap();
        let error = format!("{:#}", read_specification(&path).unwrap_err());
        assert!(error.contains("unknown field `extra`"), "{error}");
        fs::remove_dir_all(base).unwrap();
    }

    #[test]
    fn a_specification_names_its_procedure_kind_or_is_of_the_single_one() {
        let base = temp_dir("specification-procedure-kind");
        fs::create_dir_all(&base).unwrap();
        let path = base.join("specification.toml");
        fs::write(&path, MODEL).unwrap();
        assert_eq!(
            read_specification(&path).unwrap().procedure_kind,
            ProcedureKind::Single
        );
        fs::write(&path, format!("procedure_kind = \"single\"\n{MODEL}")).unwrap();
        assert_eq!(
            read_specification(&path).unwrap().procedure_kind,
            ProcedureKind::Single
        );
        // A name that is no procedure kind must not silently be the single one.
        fs::write(&path, format!("procedure_kind = \"committee\"\n{MODEL}")).unwrap();
        let error = format!("{:#}", read_specification(&path).unwrap_err());
        assert!(
            error.contains("committee") && error.contains("single") && error.contains("consensus"),
            "{error}"
        );
        fs::remove_dir_all(base).unwrap();
    }

    #[test]
    fn a_specification_names_as_many_models_as_its_procedure_kind_runs() {
        let base = temp_dir("specification-consensus");
        fs::create_dir_all(&base).unwrap();
        let path = base.join("specification.toml");
        let models = "[[model]]\nmanifest = \"a/manifest.json\"\n[[model]]\nmanifest = \"b/manifest.json\"\n";
        fs::write(
            &path,
            format!("procedure_kind = \"consensus\"\nmax_new_tokens = 4\n{models}"),
        )
        .unwrap();
        let specification = read_specification(&path).unwrap();
        assert_eq!(specification.procedure_kind, ProcedureKind::Consensus);
        assert_eq!(specification.max_new_tokens, 4);
        assert_eq!(
            specification.model_manifests(),
            [Path::new("a/manifest.json"), Path::new("b/manifest.json")]
        );
        assert_eq!(
            specification.model_manifest_of(1).unwrap(),
            Path::new("b/manifest.json")
        );
        assert!(specification.model_manifest_of(2).is_err());

        // No model, one model or three models: each is refused, and the
        // refusal says how the models are named.
        let one = "[[model]]\nmanifest = \"a/manifest.json\"\n";
        for body in [String::new(), one.to_string(), format!("{models}{one}")] {
            fs::write(&path, format!("procedure_kind = \"consensus\"\n{body}")).unwrap();
            let error = format!("{:#}", read_specification(&path).unwrap_err());
            assert!(error.contains("[[model]]"), "{error}");
        }
        // And a single procedure takes one.
        for body in [String::new(), models.to_string()] {
            fs::write(&path, body).unwrap();
            let error = format!("{:#}", read_specification(&path).unwrap_err());
            assert!(error.contains("[[model]]"), "{error}");
        }
        fs::remove_dir_all(base).unwrap();
    }

    const SOURCE: &str = "[source]\nbase = \"Is mag 5 or more?\\n\"\ntail = \"\\n\"\n\n[source.signer]\nscheme = 1\nkey = \"02ab\"\n\n[[source.document]]\nserver = \"example.org\"\ntarget = \"/event.json\"\n\n[[source.field]]\nlead = \"link: \"\npath = [\"products\", 1, \"0\", \"url\"]\n";

    #[test]
    fn a_specification_takes_a_source_and_its_forms() {
        let base = temp_dir("specification-source");
        fs::create_dir_all(&base).unwrap();
        let path = base.join("specification.toml");
        let form = "\n[[form]]\ntext = \"yes\"\ncode = 1\n";
        fs::write(&path, format!("max_new_tokens = 4\n{MODEL}{SOURCE}{form}")).unwrap();
        let specification = read_specification(&path).unwrap();
        let source = specification.source.clone().unwrap();
        assert_eq!(source.signer.key, "02ab");
        assert_eq!(source.documents.len(), 1);
        // A bare number is an index and a quoted one a key, and the source
        // step reads both as steps of one path.
        assert_eq!(
            source.fields[0].path,
            [
                PathStep::Key(String::from("products")),
                PathStep::Index(1),
                PathStep::Key(String::from("0")),
                PathStep::Key(String::from("url")),
            ]
        );
        let recipe = source.recipe();
        assert_eq!(recipe.fields[0].path, "products\n1\n0\nurl");
        assert_eq!(recipe.fields[0].document, 0);
        assert_eq!(recipe.tail, "\n");
        assert_eq!(specification.forms()[0].code, 1);

        // What is written is read back as the same values: `claim fetch`
        // writes a case this way, and the gate its specifications.
        let written = toml::to_string(&specification).unwrap();
        assert_eq!(
            toml::from_str::<Specification>(&written).unwrap(),
            specification
        );

        // Forms are read with a source's prompt.
        fs::write(&path, format!("{MODEL}{form}")).unwrap();
        let error = format!("{:#}", read_specification(&path).unwrap_err());
        assert!(error.contains("only with a `[source]`"), "{error}");

        // A key the source does not have is refused.
        fs::write(
            &path,
            format!("{MODEL}{SOURCE}").replace("tail = ", "notary = \"x\"\ntail = "),
        )
        .unwrap();
        assert!(read_specification(&path).is_err());
        fs::remove_dir_all(base).unwrap();
    }

    #[test]
    fn a_case_is_held_to_its_specification() {
        let base = temp_dir("case");
        fs::create_dir_all(&base).unwrap();
        let path = base.join("specification.toml");
        fs::write(&path, MODEL).unwrap();
        let plain = read_specification(&path).unwrap();
        fs::write(&path, format!("{MODEL}{SOURCE}")).unwrap();
        let sourced = read_specification(&path).unwrap();

        let path = base.join("case.toml");
        fs::write(&path, "prompt = \"hi\"\n").unwrap();
        let prompt = read_case(&path).unwrap();
        assert_eq!(prompt.prompt.as_deref(), Some("hi"));
        assert!(prompt.validate_for(&plain).is_ok());
        let error = format!("{:#}", prompt.validate_for(&sourced).unwrap_err());
        assert!(error.contains("builds the prompt"), "{error}");

        fs::write(
            &path,
            "[[document]]\nfile = \"event.bin\"\nattestation = \"event.json\"\n",
        )
        .unwrap();
        let documents = read_case(&path).unwrap();
        assert_eq!(
            documents.documents,
            [CaseDocument {
                file: PathBuf::from("event.bin"),
                attestation: PathBuf::from("event.json"),
            }]
        );
        assert!(documents.validate_for(&sourced).is_ok());
        let error = format!("{:#}", documents.validate_for(&plain).unwrap_err());
        assert!(error.contains("no `[source]`"), "{error}");
        assert_eq!(
            toml::from_str::<Case>(&toml::to_string(&documents).unwrap()).unwrap(),
            documents
        );

        // A prompt beside documents, and a key of the specification's.
        fs::write(
            &path,
            "prompt = \"hi\"\n[[document]]\nfile = \"a\"\nattestation = \"b\"\n",
        )
        .unwrap();
        assert!(read_case(&path).is_err());
        fs::write(&path, "prompt = \"hi\"\nmax_new_tokens = 3\n").unwrap();
        let error = format!("{:#}", read_case(&path).unwrap_err());
        assert!(error.contains("unknown field `max_new_tokens`"), "{error}");
        fs::remove_dir_all(base).unwrap();
    }

    #[test]
    fn a_case_is_read_back_as_written_whatever_its_text() {
        // `claim fetch` writes the text a claim posted into a case, and a
        // challenger reads it back: no byte of it may change on the way.
        for text in [
            "line\r\nline",
            "\r",
            "a\u{0}b\u{7f}\u{1b}",
            "quotes \" \x27 \"\"\" \x27\x27\x27",
            "\u{fffd}\u{10ffff} \u{e9}",
            " lead and trail \n",
            "\n\nstart",
            "tab\tback\\slash",
        ] {
            let case = Case::of_prompt(text.to_string());
            let written = toml::to_string(&case).unwrap();
            assert_eq!(
                toml::from_str::<Case>(&written).unwrap(),
                case,
                "{text:?} written as {written:?}"
            );
        }
    }

    #[test]
    fn prepared_run_metadata_round_trips_as_json() {
        let prepared = PreparedRun {
            version: PREPARED_RUN_VERSION,
            procedure_kind: ProcedureKind::Single,
            specification_path: PathBuf::from("specification.toml"),
            case_path: Some(PathBuf::from("case.toml")),
            inferences: vec![PreparedInference {
                model_manifest_path: PathBuf::from("model-artifacts/manifest.json"),
                model_manifest_sha256: String::from("model-sha"),
                model_id: String::from("model-id"),
                prompt: PreparedPrompt {
                    resolved_prompt: String::from("hello"),
                    rendered_prompt: String::from("hello"),
                    initial_pieces: vec![PreparedPiece {
                        text: String::from("hello"),
                        segment: 0,
                    }],
                    eos_token_ids: vec![1, 2],
                    source: None,
                    forms: Vec::new(),
                },
            }],
            raw_prompt: false,
            max_new_tokens: 2,
            run_manifest_path: Some(PathBuf::from("target/runs/run/Raster.toml")),
            run_manifest_sha256: Some(String::from("run-sha")),
            source_inputs: None,
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
