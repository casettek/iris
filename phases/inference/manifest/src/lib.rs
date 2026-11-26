//! A model as an import leaves it: the model manifest and its families, the
//! descriptor a stage reads, and the pipeline template's paths.

pub mod io;
pub mod manifest;
pub mod model_identity;
pub mod native_run;

pub use io::{read_json, write_json};
pub use manifest::resolve_raster_manifest_paths;
pub use native_run::{
    file_sha256, model_id_of, verify_file_sha256, FamilyShape, MistralShape, ModelCommitment,
    ModelFamily, ModelManifest, NativeInferBundle, NativeInferProvenance, NativeInferShape,
    MODEL_ARTIFACTS_DIR, MODEL_MANIFEST_JSON, MODEL_MANIFEST_VERSION,
};

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

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
            "iwebs-model-id-{}-{:?}",
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
            bundle: NativeInferBundle {
                model_detwgt_path: PathBuf::from("../model.detwgt"),
                model_detwgt_sha256: String::from("model-sha"),
                config_path: PathBuf::from("../config.json"),
                config_sha256: String::from("config-sha"),
                tokenizer_path: PathBuf::from("../tokenizer.json"),
                tokenizer_sha256: String::from("tokenizer-sha"),
            },
            shape: NativeInferShape {
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
            provenance: Some(NativeInferProvenance {
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
}
