//! What a contract's pins are read from.

use super::*;

#[test]
fn pins_are_of_one_procedure_kind_over_its_models_in_order() {
    let manifest = |pins: &ModelPins| -> ModelManifest {
        serde_json::from_value(serde_json::json!({
            "version": model_manifest::MODEL_MANIFEST_VERSION,
            "model_id": pins.model_id,
            "model_commitments": pins.model_commitments,
            "range_size": pins.range_size,
            "bundle": {
                "model_detwgt_path": "m", "model_detwgt_sha256": "",
                "config_path": "c", "config_sha256": "",
                "tokenizer_path": "t", "tokenizer_sha256": ""
            },
            "shape": {
                "hidden_size": 2, "num_hidden_layers": 0, "num_attention_heads": 1,
                "num_key_value_heads": 1, "head_dim": 2, "global_head_dim": 2,
                "vocab_size": 1, "hidden_size_per_layer_input": 1, "sliding_window": 0,
                "layer_types": [], "num_kv_shared_layers": 0, "norm_eps": 0,
                "rope_base_sliding": 0, "rope_base_full": 0,
                "full_partial_rotary_factor_q16": 0, "embedding_scale": 0,
                "ple_embedding_scale": 0, "ple_projection_scalar": 0,
                "ple_input_scale": 0, "final_logit_softcap": 0
            },
            "eos_token_ids": []
        }))
        .unwrap()
    };
    let single = pins();
    let first = manifest(&single.models[0]);
    assert_eq!(single.mismatch(ProcedureKind::Single, &[&first]), None);
    // The file says which procedure kind it is of and which models, in order.
    let written = serde_json::to_value(&single).unwrap();
    assert_eq!(written["version"], 6);
    assert_eq!(written["procedure_kind"], "single");
    assert_eq!(written["models"].as_array().unwrap().len(), 1);
    assert!(written["programs"]["stage_programs"].is_array());
    assert_eq!(
        serde_json::from_value::<ContractPins>(written.clone()).unwrap(),
        single
    );
    // A file of the version before, which named one `model`, is not read
    // as this one.
    let mut older = written.clone();
    let fields = older.as_object_mut().unwrap();
    let models = fields.remove("models").unwrap();
    fields.insert("model".into(), models[0].clone());
    assert!(serde_json::from_value::<ContractPins>(older).is_err());

    let (programs, models) = consensus_pins();
    let consensus = ContractPins {
        version: CONTRACT_PINS_VERSION,
        procedure_kind: ProcedureKind::Consensus,
        programs,
        models: models.to_vec(),
    };
    let second = manifest(&models[1]);
    assert_eq!(
        consensus.mismatch(ProcedureKind::Consensus, &[&first, &second]),
        None
    );
    // Another procedure kind, the models the other way round, one model, or a
    // file of another version: not this contract's pins.
    for (procedure_kind, models) in [
        (ProcedureKind::Single, vec![&first, &second]),
        (ProcedureKind::Consensus, vec![&second, &first]),
        (ProcedureKind::Consensus, vec![&first]),
        (ProcedureKind::Consensus, vec![&first, &first]),
    ] {
        assert!(consensus.mismatch(procedure_kind, &models).is_some());
    }
    assert!(single
        .mismatch(ProcedureKind::Consensus, &[&first])
        .is_some());
    let mut other_range = manifest(&single.models[0]);
    other_range.range_size = 256;
    assert!(single
        .mismatch(ProcedureKind::Single, &[&other_range])
        .is_some());
    let mut old = single.clone();
    old.version = 3;
    assert!(old.mismatch(ProcedureKind::Single, &[&first]).is_some());
}

#[test]
fn a_templates_run_inputs_are_what_a_run_writes() {
    let template = r#"[chain]
name = "iris"

[chain.input.prompt_text]
index = "j"
path = "/repo/target/iris/run-prompt-placeholder/text_{j}.rastered"
commitments = []

[chain.input.transformer_layer]
index = "l"
path = "/repo/model/layer_{l}.rastered"
commitments = []

[[chain.stage]]
name = "source"
project = "raster-stages/prompt-source"
inputs.recipe = { run = "source_recipe" }

[[chain.stage]]
name = "prompt_bind_p0"
project = "raster-stages/prompt-bind"
inputs.text = { input = "prompt_text_0" }
inputs.config = { external = { path = "/repo/target/iris/run-prompt-placeholder/config.rastered", index_path = "x", commitment = "0" } }
inputs.tokenizer = { external = { path = "/repo/model/tokenizer.rastered", index_path = "x", commitment = "t" } }
inputs.source = { from = "source" }

[[chain.repeat]]
name = "layers"
index = "l"
start = 0
count = 2

  [[chain.repeat.stage]]
name = "range_p0_l{l}"
project = "raster-stages/prefill-range"
inputs.layer = { input = "transformer_layer_{l}" }
"#;
    let run_inputs = chain_run_inputs(template).unwrap();
    let of = |project: &str| {
        run_inputs
            .get(Path::new(project))
            .map(|names| names.iter().map(String::as_str).collect::<Vec<_>>())
    };
    assert_eq!(of("raster-stages/prompt-source"), Some(vec!["recipe"]));
    assert_eq!(
        of("raster-stages/prompt-bind"),
        Some(vec!["config", "text"])
    );
    assert_eq!(of("raster-stages/prefill-range"), None);
}

#[test]
fn the_wiring_is_read_from_the_chain_template() {
    let template = r#"
[chain]
name = "iris"

[[chain.stage]]
name = "prompt_bind_p0"
project = "/repo/raster-stages/prompt-bind"
inputs.tokenizer = { external = { path = "t", commitment = "aa" } }
inputs.text = { input = "prompt_text_0" }
inputs.previous = { external = { path = "e", commitment = "e0" } }

[[chain.repeat]]
name = "prompt"
index = "j"
start = 1
count = 0

  [[chain.repeat.stage]]
name = "prompt_bind_p{j}"
project = "/repo/raster-stages/prompt-bind"
inputs.tokenizer = { external = { path = "t", commitment = "aa" } }
inputs.text = { input = "prompt_text_{j}" }
inputs.previous = { from = "prompt_bind_p{j-1}", first = "prompt_bind_p0" }

[[chain.repeat]]
name = "layers"
index = "l"
count = 2

  [[chain.repeat.stage]]
name = "range_p{j}_l{l}"
index = "j"
count = 0
project = "/repo/raster-stages/prefill-range"
inputs.activations = { from = "range_p{j}_l{l-1}", first = "prompt_bind_p{j}" }
inputs.prior_kv = { external = { path = "k", commitment = "e1" } }
inputs.layer = { input = "transformer_layer_{l}" }

  [[chain.repeat.stage]]
name = "range_c{k}_l{l}"
index = "k"
count = 0
project = "/repo/raster-stages/prefill-range"
inputs.activations = { from = "range_c{k}_l{l-1}", first = "prompt_bind_p@LAST_PROMPT_RANGE@" }
inputs.prior_kv = { from = "range_c{k-1}_l{l}", first = "range_p@LAST_PROMPT_RANGE@_l{l}" }
inputs.layer = { input = "transformer_layer_{l}" }
"#;
    let wiring = chain_wiring(template).unwrap();
    let input = |name: &str, producers: &[&str], empty_values: &[&str]| ChainedInput {
        name: name.to_string(),
        producers: producers.iter().map(|p| p.to_string()).collect(),
        empty_values: empty_values.iter().map(|v| v.to_string()).collect(),
    };
    assert_eq!(
        wiring,
        vec![
            (
                PathBuf::from("/repo/raster-stages/prompt-bind"),
                // Model data and run inputs are not chained; the first
                // bind's fixed `previous` is the input's empty value.
                vec![input("previous", &["prompt-bind"], &["e0"])],
            ),
            (
                PathBuf::from("/repo/raster-stages/prefill-range"),
                vec![
                    input("activations", &["prefill-range", "prompt-bind"], &[]),
                    input("prior_kv", &["prefill-range"], &["e1"]),
                ],
            ),
        ]
    );

    assert!(chain_wiring("[chain]\n[[chain.stage]]\nname = \"a\"\nproject = \"p\"\ninputs.x = { from = \"nowhere\" }\n").is_err());
}

#[test]
fn two_inferences_of_one_chain_are_wired_to_their_own_programs() {
    // Two inferences whose layer stages have one name and different
    // programs: each takes its layers' outputs from its own.
    let segment = |label: &str, layer: &str| {
        format!(
            r#"
[[chain.stage]]
name = "{label}prompt_bind_p0"
project = "/repo/raster-stages/prompt-bind"
inputs.previous = {{ external = {{ path = "e", commitment = "e0" }} }}

[[chain.repeat]]
name = "{label}layers"
index = "l"
count = 2

  [[chain.repeat.stage]]
name = "{label}range_p0_l{{l}}"
project = "/repo/raster-stages/{layer}"
inputs.activations = {{ from = "{label}range_p0_l{{l-1}}", first = "{label}prompt_bind_p0" }}

[[chain.stage]]
name = "{label}check_c0"
project = "/repo/raster-stages/sequence-check"
inputs.activations = {{ from = "{label}range_p0_l1" }}
"#
        )
    };
    let result = r#"
[[chain.stage]]
name = "result"
project = "/repo/raster-stages/consensus-result"
inputs.first = { from = "m0_check_c@m0_LAST_CLAIMED_RANGE@" }
inputs.second = { from = "m1_check_c@m1_LAST_CLAIMED_RANGE@" }
"#;
    let template = format!(
        "[chain]\nname = \"iris\"\n{}{}{result}",
        segment("m0_", "prefill-range"),
        segment("m1_", "layer-mistral")
    );
    let wiring = chain_wiring(&template).unwrap();
    let producers = |project: &str, input: &str| -> Vec<String> {
        wiring
            .iter()
            .find(|(path, _)| path.ends_with(project))
            .and_then(|(_, inputs)| inputs.iter().find(|chained| chained.name == input))
            .map(|chained| chained.producers.iter().cloned().collect())
            .unwrap_or_default()
    };
    assert_eq!(
        producers("prefill-range", "activations"),
        ["prefill-range", "prompt-bind"]
    );
    assert_eq!(
        producers("layer-mistral", "activations"),
        ["layer-mistral", "prompt-bind"]
    );
    // The check is one program for both, fed by either family's layers.
    assert_eq!(
        producers("sequence-check", "activations"),
        ["layer-mistral", "prefill-range"]
    );
    assert_eq!(producers("consensus-result", "first"), ["sequence-check"]);
    assert_eq!(producers("consensus-result", "second"), ["sequence-check"]);
}
