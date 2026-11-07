//! Authored Raster sequence versus native finalization on identical hostile inputs.
use model_identity::ModelDescriptor;
use output_finalize::input::{DecodeEdge, DecoderTable, DecoderToken};
use output_finalize::prompt::{PromptContext, PromptState};
use raster::{BytesPage, List};
use serde::Serialize;
use serde_json::{json, Value};
use staged_infer::{cache::CachedInputs, routines};
use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

fn page(bytes: Vec<u8>) -> BytesPage {
    BytesPage::__from_parts(0, 0, bytes)
}

fn write<T: Serialize>(
    dir: &Path,
    name: &str,
    value: &T,
    inputs: &mut Value,
    manifest: &mut Value,
) {
    raster::init();
    let data = dir.join(format!("{name}.rastered"));
    let index = dir.join(format!("{name}.rindex"));
    let root = raster::write_raster_files(value, &data, &index).unwrap();
    inputs[name] = json!({"path":data, "index_path":index, "load_preference":"mmap"});
    manifest[name] = json!({"type":"sha256", "encoding":"raster", "commitment":root});
}

#[test]
#[ignore = "requires built output-finalize binary; set STAGED_INFER_REPLAY_BIN_DIR"]
fn same_substituted_inputs_finish_with_matching_model_faults() {
    let bin = PathBuf::from(
        std::env::var_os("STAGED_INFER_REPLAY_BIN_DIR").expect("build Raster binary first"),
    );
    let base = std::env::temp_dir().join(format!("iris-model-sequence-{}", std::process::id()));
    fs::create_dir_all(&base).unwrap();
    let honest = DecoderTable {
        tokens: List::from(vec![
            DecoderToken {
                token: "hello".into(),
                special: false,
                terminal: false,
            },
            DecoderToken {
                token: "<hidden>".into(),
                special: true,
                terminal: false,
            },
            DecoderToken {
                token: "<eos>".into(),
                special: true,
                terminal: true,
            },
        ]),
    };
    let d = ModelDescriptor {
        version: 1,
        tokenizer: [1; 32],
        decoder: raster::core::tree::value_root(&honest).unwrap(),
        embedding: [2; 32],
        head: [3; 32],
        auxiliary: vec![[4; 32]],
        transformer: vec![[5; 32]],
    };
    let id = hex::encode(d.model_id().unwrap());
    let descriptor = page(d.encode().unwrap());
    let mut edge = DecodeEdge {
        has_selected: true,
        generated_token_ids: vec![0, 1, 2].into(),
        ..Default::default()
    };
    let model = model_identity::ModelContext::bind(descriptor.as_slice(), &id);
    edge.model = model.clone();
    edge.model.producer = model_identity::context::DECODE;
    edge.model.completed_layers = edge.model.layers;
    let prompt = PromptState {
        context: PromptContext {
            model,
            model_id: id.clone(),
            max_new_tokens: 3,
            complete: true,
            ..Default::default()
        },
        ..Default::default()
    };
    let mut cache = CachedInputs::new();
    // Retain the real predecessor values while changing supplied model witnesses.
    cache.insert(
        "prompt".into(),
        std::sync::Arc::new(staged_infer::cache::CachedStageValue::PromptTokenization(
            serde_json::from_value(serde_json::to_value(&prompt).unwrap()).unwrap(),
        )),
    );
    cache.insert(
        "edge".into(),
        std::sync::Arc::new(staged_infer::cache::CachedStageValue::DecodeEdge(
            serde_json::from_value(serde_json::to_value(&edge).unwrap()).unwrap(),
        )),
    );
    for (name, fault) in [
        ("honest", 0),
        ("hidden-text", 1),
        ("changed-eos", 1),
        ("wrong-proof", 1),
        ("wrong-descriptor", 1),
    ] {
        let dir = base.join(name);
        fs::create_dir_all(&dir).unwrap();
        let mut tokens = honest.tokens.iter().cloned().collect::<Vec<_>>();
        if name == "hidden-text" {
            tokens[1].token = "secret substitution".into();
        }
        if name == "changed-eos" {
            tokens[2].terminal = false;
        }
        let table = DecoderTable {
            tokens: tokens.into(),
        };
        let witness = if name == "wrong-proof" {
            let mut changed = honest.tokens.iter().cloned().collect::<Vec<_>>();
            changed[0].token = "another model".into();
            output_finalize::decoder_witness(&DecoderTable {
                tokens: changed.into(),
            })
            .unwrap()
        } else {
            output_finalize::decoder_witness(&honest).unwrap()
        };
        let descriptor = if name == "wrong-descriptor" {
            let mut changed = d.clone();
            changed.transformer[0][0] ^= 1;
            page(changed.encode().unwrap())
        } else {
            descriptor.clone()
        };
        let mut inputs = json!({});
        let mut manifest = json!({});
        write(&dir, "edge", &edge, &mut inputs, &mut manifest);
        // Recommit the changed artifact honestly: this must reach Iris's check.
        write(&dir, "decoder", &table, &mut inputs, &mut manifest);
        write(&dir, "prompt", &prompt, &mut inputs, &mut manifest);
        write(&dir, "descriptor", &descriptor, &mut inputs, &mut manifest);
        write(&dir, "witness", &witness, &mut inputs, &mut manifest);
        fs::write(dir.join("input.json"), serde_json::to_vec(&inputs).unwrap()).unwrap();
        fs::write(
            dir.join("input_manifest.json"),
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();
        let loaded = routines::output_finalize::load_inputs_from_paths(
            &dir.join("input.json"),
            &dir.join("input_manifest.json"),
            &cache,
        )
        .unwrap();
        let expected = routines::output_finalize::run_direct(&loaded).unwrap();
        let uncached = routines::output_finalize::load_inputs_from_paths(
            &dir.join("input.json"),
            &dir.join("input_manifest.json"),
            &CachedInputs::new(),
        )
        .unwrap();
        assert_eq!(
            expected,
            routines::output_finalize::run_direct(&uncached).unwrap(),
            "{name}"
        );
        assert_eq!(expected.model_fault, fault, "{name}");
        assert_eq!(expected.model_id, id);
        if name == "hidden-text" {
            assert_eq!(expected.generated_text, "hello");
        }
        let dest = dir.join("raster");
        fs::create_dir_all(&dest).unwrap();
        let execution = Command::new(bin.join("output-finalize"))
            .arg("--input")
            .arg(dir.join("input.json"))
            .arg("--input-manifest")
            .arg(dir.join("input_manifest.json"))
            .env("RASTER_AUTH", "1")
            .env("RASTER_OUTPUT_DIR", &dest)
            .env("RASTER_TRACE_PATH", dest.join("trace.json"))
            .env("RASTER_TRACE_FORMAT", "json")
            .output()
            .unwrap();
        assert!(
            execution.status.success(),
            "{name}: {}",
            String::from_utf8_lossy(&execution.stderr)
        );
        let (bytes, _, _) = raster::encode_raster_value(&expected).unwrap();
        assert_eq!(bytes, fs::read(dest.join("output.bin")).unwrap(), "{name}");
    }
    fs::remove_dir_all(base).unwrap();
}
