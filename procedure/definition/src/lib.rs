//! What a procedure is, and the record of a run of one: the specification
//! and the case, the procedure kinds, a run's inputs, its checkpoints, and
//! where two runs differ.

pub mod checkpoint;
pub mod divergence;
pub mod encoding;
pub mod inference;
pub mod procedure_kind;
pub mod replay_profile;
pub mod run;
pub mod run_inputs;
pub mod segment_kind;
pub mod shape;
pub mod specification;

pub use checkpoint::{
    build_checkpoint_list, checkpoint_hash, checkpoint_hashes, checkpoint_opening,
    checkpoints_root, read_checkpoint_from_stage_dir, read_checkpoint_hashes, read_checkpoint_list,
    read_stage_order, verify_checkpoint_opening, write_checkpoint_hashes,
    write_checkpoint_hashes_artifact, write_checkpoint_list_artifact, write_stage_program,
    Checkpoint, CheckpointList, CHECKPOINTS_TXT, CHECKPOINT_HASHES_TXT, CHECKPOINT_LIST_JSON,
    EXECUTION_TIMES_JSON, STAGE_PROGRAM_JSON,
};
pub use divergence::{Divergence, DivergenceReason, DIVERGENCE_JSON};
pub use inference::{
    ConsensusOutcome, InferAuxWaveTiming, InferStageTiming, InferenceResult, InferenceRunReport,
    InferenceTimings,
};
pub use model_manifest::io::{read_json, write_json};
pub use procedure_kind::ProcedureKind;
pub use run::{
    PreparedInference, PreparedPiece, PreparedPrompt, PreparedRun, PreparedSourceInputs,
    RunFileRef, PREPARED_RUN_JSON, PREPARED_RUN_VERSION,
};
pub use segment_kind::{ResultKind, SegmentKind};
pub use shape::Shape;
pub use specification::{
    read_case, read_specification, Case, CaseDocument, PathStep, Specification,
    SpecificationDocument, SpecificationField, SpecificationForm, SpecificationModel,
    SpecificationSource, CASE_TOML, SPECIFICATION_TOML,
};

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::time::{SystemTime, UNIX_EPOCH};

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
            ProcedureKind::single()
        );
        fs::write(&path, format!("procedure_kind = \"single\"\n{MODEL}")).unwrap();
        assert_eq!(
            read_specification(&path).unwrap().procedure_kind,
            ProcedureKind::single()
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
        assert_eq!(
            specification.procedure_kind,
            ProcedureKind::named("consensus").unwrap()
        );
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
            procedure_kind: ProcedureKind::single(),
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
