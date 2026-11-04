//! The local three-path regression gate, plus the teacher-forced leg that
//! checks the one-pass chain against the autoregressive one position by
//! position. Every worker owns its runtime state.
mod compare;

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use anyhow::{ensure, Context, Result};
use direct_infer::{DirectInferenceConfig, DirectInferenceExecutor};
use inference_artifacts::{file_sha256, read_json, write_json, InferenceResult, ModelManifest};
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use staged_infer::chain_runner::StagedExecutionBackend;

use crate::claim::{build_claim, prepare_claim_run, ClaimBuildOptions};
use compare::{
    compare, read_boundaries, staged_names, teacher_forced_names, Boundary, RangedShape,
};

#[derive(Debug, Deserialize)]
struct Case {
    name: String,
    prompt: String,
    raw_prompt: bool,
    tokens: u32,
    input_token_ids: Vec<u32>,
}

pub fn run() -> Result<()> {
    ensure!(
        !cfg!(debug_assertions),
        "test-parity requires a release build; run `just test-parity`"
    );
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../..")
        .canonicalize()?;
    let run_dir = root.join("target/parity").join(format!(
        "{}-{}",
        SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos(),
        std::process::id()
    ));
    fs::create_dir_all(&run_dir)?;
    println!("Three-path parity artifacts: {}", run_dir.display());
    let started = Instant::now();
    let mut report = json!({"version": 1, "mode": "host_no_auth", "passed": false, "cases": []});
    let result = run_suite(&root, &run_dir, &mut report);
    report["elapsed_seconds"] = json!(started.elapsed().as_secs_f64());
    report["passed"] = json!(result.is_ok());
    if let Err(error) = &result {
        report["error"] = json!(format!("{error:#}"));
    }
    write_json(&run_dir.join("report.json"), &report)?;
    result
        .with_context(|| format!("parity failed; artifacts and report: {}", run_dir.display()))?;
    println!(
        "PASS: 3 cases, 3 independent paths plus teacher-forced, exact parity ({:.2}s total).",
        started.elapsed().as_secs_f64()
    );
    Ok(())
}

fn run_suite(root: &Path, run_dir: &Path, report: &mut Value) -> Result<()> {
    let model_bundle = root.join("model-bundles/parity-gemma");
    let checksums: BTreeMap<String, String> = read_json(&model_bundle.join("checksums.json"))?;
    ensure!(
        checksums.keys().cloned().collect::<BTreeSet<_>>()
            == [
                "model.detwgt",
                "config.json",
                "tokenizer.json",
                "cases.json"
            ]
            .map(str::to_string)
            .into_iter()
            .collect(),
        "invalid model bundle checksum inventory"
    );
    for (name, expected) in &checksums {
        ensure!(
            file_sha256(&model_bundle.join(name))? == *expected,
            "model bundle checksum mismatch: {name}"
        );
    }
    report["model_bundle_sha256"] = json!(checksums);
    let exe = std::env::current_exe()?;
    let raster_root = root
        .join("../raster")
        .canonicalize()
        .context("the sibling Raster checkout is required")?;
    let tools_dir = root.join("target/parity-tools");
    let build_started = Instant::now();
    println!("Building release binaries and checking arithmetic vectors...");
    let mut build = command("cargo", root);
    build
        .args([
            "build",
            "--release",
            "--locked",
            "-p",
            "raster-cli",
            "--manifest-path",
        ])
        .arg(raster_root.join("Cargo.toml"))
        .arg("--target-dir")
        .arg(&tools_dir);
    logged(build, &run_dir.join("build-raster.log"))?;
    let raster = tools_dir.join("release/cargo-raster");
    let mut build = command("cargo", root);
    build.args(["build", "--release", "--locked", "--workspace", "--bins"]);
    logged(build, &run_dir.join("build-inference.log"))?;
    let mut vectors = command("cargo", root);
    vectors.args([
        "test",
        "--release",
        "--locked",
        "-p",
        "det-num",
        "--test",
        "golden_vectors",
    ]);
    logged(vectors, &run_dir.join("arithmetic.log"))?;
    report["build_and_arithmetic_seconds"] = json!(build_started.elapsed().as_secs_f64());
    report["executables"] = json!({
        "inference": {"path": exe, "sha256": file_sha256(&exe)?},
        "raster": {"path": raster, "sha256": file_sha256(&raster)?},
        "inference_lock_sha256": file_sha256(&root.join("Cargo.lock"))?,
        "raster_lock_sha256": file_sha256(&raster_root.join("Cargo.lock"))?
    });
    report["build_configuration"] =
        json!({"profile": "release", "incremental": true, "proof_guests": false});

    let import_started = Instant::now();
    let artifact_root = run_dir.join("model-artifacts");
    let mut import = command(&exe, root);
    import
        .args(["model", "import", "--model"])
        .arg(&model_bundle)
        .arg("--artifact-root")
        .arg(&artifact_root)
        .args(["--model-id", "parity-gemma", "--no-stage-fixtures"])
        // Small enough that every case's prompt and 8-token outputs split into
        // several teacher-forced ranges.
        .args(["--range-size", "6"]);
    logged(import, &run_dir.join("import.log"))?;
    report["import_seconds"] = json!(import_started.elapsed().as_secs_f64());
    let model_dir = artifact_root.join("parity-gemma");
    let model_path = model_dir.join("manifest.json");
    let model = ModelManifest::load_verified(&model_path)?;
    validate_model_bundle(&model_bundle, &model_dir, &model)?;
    let cases: Vec<Case> = read_json(&model_bundle.join("cases.json"))?;
    ensure!(
        cases.len() == 3
            && cases
                .iter()
                .map(|c| c.name.as_str())
                .eq(["short", "near-window", "over-window"]),
        "invalid case inventory"
    );
    ensure!(
        cases[0].raw_prompt && cases[0].tokens == 1 && cases[0].input_token_ids.len() < 16,
        "invalid short case"
    );
    ensure!(
        cases[1].raw_prompt && cases[1].tokens == 8 && cases[1].input_token_ids.len() == 15,
        "invalid near-window case"
    );
    ensure!(
        !cases[2].raw_prompt && cases[2].tokens == 8 && cases[2].input_token_ids.len() > 16,
        "invalid over-window case"
    );
    let mut selected = BTreeSet::new();
    let mut logit_vectors = BTreeSet::new();
    for case in cases {
        println!(
            "Running {} ({} input tokens → {} tokens)...",
            case.name,
            case.input_token_ids.len(),
            case.tokens
        );
        let case_dir = run_dir.join(&case.name);
        fs::create_dir(&case_dir)?;
        let mut case_report = json!({"case": case.name, "input_tokens": case.input_token_ids.len(), "generated_tokens": case.tokens, "passed": false});
        let result = run_case(
            root,
            &case_dir,
            &case,
            &model_path,
            &model,
            &exe,
            &raster,
            &mut case_report,
            &mut selected,
            &mut logit_vectors,
        );
        case_report["passed"] = json!(result.is_ok());
        if let Err(error) = &result {
            case_report["error"] = json!(format!("{error:#}"));
        }
        report["cases"].as_array_mut().unwrap().push(case_report);
        write_json(&run_dir.join("report.json"), report)?;
        result.with_context(|| format!("case {}", case.name))?;
    }
    ensure!(
        selected.len() > 1,
        "degenerate model bundle: every selection produced the same token"
    );
    ensure!(
        logit_vectors.len() > 3,
        "degenerate model bundle: logit vectors do not vary across decode steps"
    );
    report["distinct_selected_tokens"] = json!(selected.len());
    report["distinct_logit_vectors"] = json!(logit_vectors.len());
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_case(
    root: &Path,
    dir: &Path,
    case: &Case,
    model_path: &Path,
    model: &ModelManifest,
    exe: &Path,
    raster: &Path,
    report: &mut Value,
    selected: &mut BTreeSet<u32>,
    logit_vectors: &mut BTreeSet<String>,
) -> Result<()> {
    let started = Instant::now();
    let spec = dir.join("inference.toml");
    // JSON basic strings are valid TOML strings for these pinned corpus values.
    fs::write(
        &spec,
        format!(
            "model_manifest = {}\nprompt = {}\nraw_prompt = {}\ntokens = {}\n",
            serde_json::to_string(model_path)?,
            serde_json::to_string(&case.prompt)?,
            case.raw_prompt,
            case.tokens
        ),
    )?;
    let prepared = prepare_claim_run(&ClaimBuildOptions {
        base_dir: dir.to_path_buf(),
        run_spec_path: spec.clone(),
        current_exe: exe.to_path_buf(),
        staged_backend: StagedExecutionBackend::InProcess,
        claimed_token_ids: None,
    })?;
    let raster_dir = dir.join("raster");
    fs::create_dir(&raster_dir)?;
    let mut run = command(raster, root);
    run.args(["raster", "chain", "run"])
        .arg(&prepared.manifest_path)
        .args(["--no-auth", "--run"])
        .arg(&raster_dir);
    let duration = logged(run, &dir.join("raster.log")).context("Raster-staged execution")?;
    report["raster_seconds"] = json!(duration);

    for (name, direct) in [("native-staged", false), ("native-direct", true)] {
        let mut run = command(exe, root);
        run.args(["parity-worker", "--run"])
            .arg(&spec)
            .arg("--output")
            .arg(dir.join(name));
        if direct {
            run.arg("--direct");
        }
        let duration = logged(run, &dir.join(format!("{name}.log")))
            .with_context(|| format!("{name} execution"))?;
        report[format!("{name}_seconds")] = json!(duration);
    }
    let native_info: Value = read_json(&dir.join("native-staged/result.json"))?;
    let native_dir = PathBuf::from(
        native_info["chain_dir"]
            .as_str()
            .context("missing native chain directory")?,
    );
    let names = staged_names(
        model.shape.num_hidden_layers,
        case.tokens,
        run_prep::tokenizer_repeat_count(prepared.prepared_run.prompt.initial_pieces.len())?,
    );
    let direct_names =
        direct_infer::diagnostics::boundary_names(model.shape.num_hidden_layers, case.tokens);
    let raster_outputs =
        read_boundaries(&raster_dir, &names, false).context("Raster-staged artifacts")?;
    let native = read_boundaries(&native_dir, &names, false).context("native-staged artifacts")?;
    let direct = read_boundaries(&dir.join("native-direct"), &direct_names, true)
        .context("native-direct artifacts")?;
    for name in &names {
        compare(name, &raster_outputs[name], &native[name])
            .context("Raster-staged ↔ native-staged")?;
    }
    // Use staged semantic order for failure localization; never infer which
    // comparisons are required from the intersection of actual outputs.
    for name in names.iter().filter(|name| direct_names.contains(name)) {
        compare(name, &native[name], &direct[name]).context("native-staged ↔ native-direct")?;
    }
    for (name, outputs, trailing) in [
        ("Raster-staged", &raster_outputs, true),
        ("native-staged", &native, true),
        ("native-direct", &direct, false),
    ] {
        validate_case(case, model, outputs, trailing)
            .with_context(|| format!("{name} coverage"))?;
    }
    validate_donor_bindings(&raster_dir, &raster_outputs, case.tokens)?;
    validate_donor_bindings(&native_dir, &native, case.tokens)?;
    let final_value = &native["output_finalize"].value;
    ensure!(
        native_info["result"] == *final_value,
        "native-staged returned result differs from output artifact"
    );
    let direct_result: Value = read_json(&dir.join("native-direct/result.json"))?;
    ensure!(
        direct_result == *final_value,
        "native-direct returned result differs from output artifact"
    );
    for token in final_value["generated_token_ids"]
        .as_array()
        .context("missing generated tokens")?
    {
        selected.insert(u32::try_from(token.as_u64().context("invalid token")?)?);
    }
    for step in 0..case.tokens {
        let stage = logits_stage(step);
        logit_vectors.insert(serde_json::to_string(&native[&stage].value["logits"])?);
    }
    report["raster_native_boundaries"] = json!(names.len());
    report["native_direct_boundaries"] = json!(direct_names.len());
    report["logit_vectors_compared"] = json!(case.tokens);
    report["result"] = final_value.clone();
    let teacher_forced =
        run_teacher_forced_leg(root, dir, case, model_path, model, exe, raster, &native_dir)
            .context("teacher-forced leg")?;
    report["teacher_forced"] = teacher_forced;
    report["elapsed_seconds"] = json!(started.elapsed().as_secs_f64());
    println!(
        "  PASS: {} staged checkpoints, {} direct boundaries, {} complete logit vectors, {} teacher-forced checkpoints in {} prompt + {} claimed ranges ({:.2}s)",
        names.len(),
        direct_names.len(),
        case.tokens,
        report["teacher_forced"]["checkpoints"],
        report["teacher_forced"]["prompt_ranges"],
        report["teacher_forced"]["claimed_ranges"],
        started.elapsed().as_secs_f64()
    );
    Ok(())
}

/// The fourth leg: a teacher-forced pass over `prompt ‖ direct output`,
/// through Raster and native-staged, checked against the autoregressive
/// chain position by position.
///
/// Causal attention means row `i` of a teacher-forced layer has seen exactly
/// what the autoregressive path saw at position `i`, so every activation, PLE
/// row and KV entry must be bit-identical: prefill rows for the prompt, decode
/// step `t`'s single row for position `prompt_len + t`. The check stage must
/// then reproduce the final decode edge, and `output_finalize` the same
/// output. Any difference would make an honest claimer convict itself.
#[allow(clippy::too_many_arguments)]
fn run_teacher_forced_leg(
    root: &Path,
    dir: &Path,
    case: &Case,
    model_path: &Path,
    model: &ModelManifest,
    exe: &Path,
    raster: &Path,
    autoregressive_dir: &Path,
) -> Result<Value> {
    let tf_dir = dir.join("teacher-forced");
    fs::create_dir(&tf_dir)?;
    let spec = tf_dir.join("inference.toml");
    fs::write(
        &spec,
        format!(
            "model_manifest = {}\nprompt = {}\nraw_prompt = {}\ntokens = {}\nchain = \"teacher-forced\"\n",
            serde_json::to_string(model_path)?,
            serde_json::to_string(&case.prompt)?,
            case.raw_prompt,
            case.tokens
        ),
    )?;
    // Preparation discovers the claimed ids with direct decode.
    let prepared = prepare_claim_run(&ClaimBuildOptions {
        base_dir: tf_dir.clone(),
        run_spec_path: spec.clone(),
        current_exe: exe.to_path_buf(),
        staged_backend: StagedExecutionBackend::InProcess,
        claimed_token_ids: None,
    })?;
    let claimed = prepared
        .prepared_run
        .claimed_token_ids
        .clone()
        .context("teacher-forced preparation recorded no claimed ids")?;
    let raster_dir = tf_dir.join("raster");
    fs::create_dir(&raster_dir)?;
    let mut run = command(raster, root);
    run.args(["raster", "chain", "run"])
        .arg(&prepared.manifest_path)
        .args(["--no-auth", "--run"])
        .arg(&raster_dir);
    logged(run, &tf_dir.join("raster.log")).context("teacher-forced Raster-staged execution")?;
    let mut run = command(exe, root);
    run.args(["parity-worker", "--run"])
        .arg(&spec)
        .arg("--output")
        .arg(tf_dir.join("native-staged"));
    logged(run, &tf_dir.join("native-staged.log"))
        .context("teacher-forced native-staged execution")?;
    let native_info: Value = read_json(&tf_dir.join("native-staged/result.json"))?;
    let native_dir = PathBuf::from(
        native_info["chain_dir"]
            .as_str()
            .context("missing teacher-forced native chain directory")?,
    );

    let order = inference_artifacts::read_stage_order(&native_dir)?
        .context("teacher-forced native chain recorded no stage inventory")?;
    let shape = RangedShape::from_names(&order);
    ensure!(
        shape.layers == model.shape.num_hidden_layers,
        "teacher-forced chain has {} layers per range, model has {}",
        shape.layers,
        model.shape.num_hidden_layers
    );
    let names = teacher_forced_names(shape);
    let raster_outputs = read_boundaries(&raster_dir, &names, false)
        .context("teacher-forced Raster-staged artifacts")?;
    let native = read_boundaries(&native_dir, &names, false)
        .context("teacher-forced native-staged artifacts")?;
    for name in &names {
        compare(name, &raster_outputs[name], &native[name])
            .context("teacher-forced Raster-staged ↔ native-staged")?;
    }

    ensure!(
        claimed.len() == case.tokens as usize,
        "direct decode produced {} of {} tokens; the pinned corpus fills its budget",
        claimed.len(),
        case.tokens
    );
    let mut comparison = compare_teacher_forced(&native_dir, autoregressive_dir)?;
    ensure!(
        comparison["prompt_token_ids"] == json!(case.input_token_ids),
        "teacher-forced prompt ids differ from the pinned ids"
    );
    ensure!(
        comparison["generated_token_ids"] == json!(claimed),
        "teacher-forced output differs from the claimed ids"
    );
    let checkpoints = inference_artifacts::read_checkpoint_hashes(
        &native_dir.join(inference_artifacts::CHECKPOINT_HASHES_TXT),
    )?;
    comparison["checkpoints"] = json!(names.len());
    comparison["prompt_ranges"] = json!(shape.prompt_ranges);
    comparison["claimed_ranges"] = json!(shape.claimed_ranges);
    comparison["final_checkpoint_sha256"] = json!(checkpoints.last());
    Ok(comparison)
}

/// Compares a ranged teacher-forced native-staged chain with the
/// autoregressive one for the same prompt and output, bit for bit and
/// position by position.
///
/// The prompt ranges' token ids must concatenate to the autoregressive
/// tokenization. Stitching each layer's ranges together, row `i` of every
/// embedding, PLE and layer stage must equal the autoregressive prefill row
/// `i` (`i < prompt_len`) or decode step `i - prompt_len`'s single row, and
/// every K/V entry a range publishes — inherited or its own — the
/// autoregressive entry at the same position. `output_finalize` must be
/// identical, and the last range's check the final decode edge.
///
/// Stages are read one layer at a time, so real-model chains with hundreds of
/// rows fit in memory.
pub(crate) fn compare_teacher_forced(tf_dir: &Path, ar_dir: &Path) -> Result<Value> {
    let load = |dir: &Path, stage: &str| -> Result<Boundary> {
        let boundary = compare::read_boundary(&dir.join(stage))
            .with_context(|| format!("{stage} in {}", dir.display()))?;
        compare::validate_errors(stage, &boundary.value)?;
        Ok(boundary)
    };
    let order = |dir: &Path| -> Result<Vec<String>> {
        inference_artifacts::read_stage_order(dir)?
            .with_context(|| format!("{}: missing recorded stage inventory", dir.display()))
    };
    let tf_names = order(tf_dir)?;
    let shape = RangedShape::from_names(&tf_names);
    ensure!(
        tf_names == teacher_forced_names(shape),
        "{}: not a ranged teacher-forced chain",
        tf_dir.display()
    );
    let ar_names = order(ar_dir)?;
    let repeats = ar_names
        .iter()
        .filter(|name| name.starts_with("prompt_merge_b"))
        .count() as u32;

    let prompt_ids = load(ar_dir, "prompt_prepare")?.value["token_ids"].clone();
    let generated = load(ar_dir, "output_finalize")?.value["generated_token_ids"].clone();
    let prompt_len = prompt_ids.as_array().context("missing prompt ids")?.len();
    let tokens = generated.as_array().context("missing generated ids")?.len();
    ensure!(tokens > 0, "autoregressive chain generated no tokens");
    // `range_origin` holds no computation; chains built before it existed are
    // still valid references.
    let without_origin = |names: Vec<String>| {
        names
            .into_iter()
            .filter(|name| name != "range_origin")
            .collect::<Vec<_>>()
    };
    ensure!(
        without_origin(ar_names) == without_origin(staged_names(shape.layers, tokens as u32, repeats)),
        "{}: not the autoregressive chain for {} layers and {tokens} tokens",
        ar_dir.display(),
        shape.layers
    );

    let mut ranged_prompt = Vec::new();
    for j in 0..shape.prompt_ranges {
        let ids = load(tf_dir, &format!("prompt_prepare_p{j}"))?.value["token_ids"].clone();
        ranged_prompt.extend(ids.as_array().cloned().unwrap_or_default());
    }
    ensure!(
        json!(ranged_prompt) == prompt_ids,
        "prompt ranges tokenize differently from the whole prompt"
    );
    compare(
        "output_finalize",
        &load(tf_dir, "output_finalize")?,
        &load(ar_dir, "output_finalize")?,
    )
    .context("teacher-forced ↔ autoregressive")?;
    let last_check = format!("check_c{}", shape.claimed_ranges - 1);
    let last_select = format!("decode_select_t{}", tokens - 1);
    compare(
        &last_check,
        &load(tf_dir, &last_check)?,
        &load(ar_dir, &last_select)?,
    )
    .with_context(|| format!("teacher-forced {last_check} ↔ autoregressive {last_select}"))?;

    // The same family of stages, ranged and autoregressive.
    let ranges = (0..shape.prompt_ranges)
        .map(|j| format!("p{j}"))
        .chain((0..shape.claimed_ranges).map(|k| format!("c{k}")))
        .collect::<Vec<_>>();
    let mut families = vec![(
        ranges.iter().map(|r| format!("embed_{r}")).collect::<Vec<_>>(),
        String::from("input_embedding"),
        (0..tokens).map(|t| format!("decode_embed_t{t}")).collect::<Vec<_>>(),
    )];
    for layer in 0..shape.layers {
        families.push((
            ranges.iter().map(|r| format!("aux_{r}_l{layer}")).collect(),
            format!("prefill_prepare_aux_l{layer}"),
            (0..tokens).map(|t| format!("decode_aux_t{t}_l{layer}")).collect(),
        ));
        families.push((
            ranges.iter().map(|r| format!("range_{r}_l{layer}")).collect(),
            format!("prefill_range_l{layer}"),
            (0..tokens).map(|t| format!("decode_range_t{t}_l{layer}")).collect(),
        ));
    }
    let mut rows_compared = 0;
    let mut keys_compared = 0;
    for (ranged, prefill, decode) in &families {
        // Position `prompt_len + t` comes from decode step `t`, which embeds
        // the token selected at step `t`.
        let prefill_boundary = load(ar_dir, prefill)?;
        let decode_boundaries = decode
            .iter()
            .map(|stage| load(ar_dir, stage))
            .collect::<Result<Vec<_>>>()?;
        let source = |position: usize| {
            if position < prompt_len {
                (&prefill_boundary, position)
            } else {
                (&decode_boundaries[position - prompt_len], 0)
            }
        };
        let mut position = 0;
        for stage in ranged {
            let teacher = load(tf_dir, stage)?;
            // PLE inputs carry no position; activations must start where the
            // previous range ended.
            if let Some(start) = teacher.value.get("start_position") {
                ensure!(
                    *start == json!(position),
                    "{stage}: starts at {start}, expected {position}"
                );
            }
            let rows = teacher.value["rows"]
                .as_array()
                .with_context(|| format!("{stage}: missing rows"))?;
            for row in rows {
                ensure!(
                    position < prompt_len + tokens,
                    "{stage}: more rows than prompt ‖ output positions"
                );
                let (source, index) = source(position);
                ensure!(
                    source.value["rows"][index] == *row,
                    "{stage}: position {position} differs from the autoregressive row"
                );
                position += 1;
                rows_compared += 1;
            }
            if !prefill.starts_with("prefill_range_") {
                continue;
            }
            for key in teacher.value["kv"]
                .as_array()
                .with_context(|| format!("{stage}: missing KV"))?
            {
                let key_position = key["position"]
                    .as_u64()
                    .with_context(|| format!("{stage}: KV entry without a position"))?
                    as usize;
                ensure!(
                    key_position < position,
                    "{stage}: publishes K/V for position {key_position}, not yet reached"
                );
                let expected = source(key_position).0.value["kv"]
                    .as_array()
                    .with_context(|| format!("{stage}: autoregressive source has no KV"))?
                    .iter()
                    .find(|entry| entry["position"] == key["position"])
                    .with_context(|| {
                        format!("{stage}: no autoregressive K/V entry for position {key_position}")
                    })?;
                ensure!(
                    expected == key,
                    "{stage}: K/V at position {key_position} differs from the autoregressive entry"
                );
                keys_compared += 1;
            }
        }
        ensure!(
            position == prompt_len + tokens,
            "{prefill}: ranges cover {position} positions, expected {}",
            prompt_len + tokens
        );
    }
    Ok(json!({
        "layers": shape.layers,
        "prompt_ranges": shape.prompt_ranges,
        "claimed_ranges": shape.claimed_ranges,
        "prompt_token_ids": prompt_ids,
        "generated_token_ids": generated,
        "rows_compared": rows_compared,
        "kv_entries_compared": keys_compared,
    }))
}

fn logits_stage(step: u32) -> String {
    if step == 0 {
        "prefill_finalize".into()
    } else {
        format!("decode_finalize_t{}", step - 1)
    }
}

fn validate_case(
    case: &Case,
    model: &ModelManifest,
    outputs: &BTreeMap<String, Boundary>,
    trailing: bool,
) -> Result<()> {
    ensure!(
        outputs["prompt_prepare"].value["token_ids"] == json!(case.input_token_ids),
        "tokenized input IDs differ from pinned IDs"
    );
    for (stage, output) in outputs {
        if stage.starts_with("prefill_prepare_aux_l") || stage.starts_with("decode_aux_") {
            let rows = output.value["rows"]
                .as_array()
                .context("missing PLE rows")?;
            let count = if stage.starts_with("prefill_") {
                case.input_token_ids.len()
            } else {
                1
            };
            ensure!(
                rows.len() == count && has_nonzero_payload(&output.value["rows"]),
                "{stage}: PLE is inactive or has the wrong row count"
            );
        }
    }
    let final_value = &outputs["output_finalize"].value;
    let result: InferenceResult = serde_json::from_value(final_value.clone())?;
    ensure!(
        result.generated_token_count == case.tokens
            && result.generated_token_ids.len() == case.tokens as usize,
        "incorrect final token count"
    );
    ensure!(
        result.generated_token_ids_sha256
            == format!(
                "{:x}",
                Sha256::digest(serde_json::to_vec(&result.generated_token_ids)?)
            ),
        "incorrect generated token hash"
    );
    for step in 0..case.tokens + u32::from(trailing) {
        let stage = logits_stage(step);
        let value = &outputs[&stage].value;
        ensure!(
            value["decode_position"] == json!(case.input_token_ids.len() + step as usize),
            "{stage}: incorrect position"
        );
        let logits = value["logits"].as_array().context("missing logits")?;
        ensure!(
            logits.len() == model.shape.vocab_size as usize,
            "{stage}: incomplete logit vector"
        );
        let mut best: Option<(usize, i64)> = None;
        for (id, entry) in logits.iter().enumerate() {
            ensure!(
                entry["token_id"] == json!(id),
                "{stage}: incorrect logit token ID {id}"
            );
            let value = entry["value"]
                .as_i64()
                .context("invalid fixed-point logit")?;
            i32::try_from(value)?;
            ensure!(
                value.abs() <= i64::from(model.shape.final_logit_softcap),
                "{stage}: logit {id} exceeds the configured softcap"
            );
            if best.is_none_or(|(_, maximum)| value > maximum) {
                best = Some((id, value));
            }
        }
        ensure!(
            logits
                .iter()
                .map(|entry| entry["value"].clone().to_string())
                .collect::<BTreeSet<_>>()
                .len()
                > 1,
            "{stage}: constant logits"
        );
        if step < case.tokens {
            let edge = &outputs[&format!("decode_select_t{step}")].value;
            ensure!(
                edge["generated_token_ids"] == json!(&result.generated_token_ids[..=step as usize]),
                "step {step}: incorrect selected-token history"
            );
            ensure!(
                edge["token_id"] == json!(best.unwrap().0)
                    && edge["value"] == json!(best.unwrap().1),
                "step {step}: selection does not match full-vector argmax"
            );
        }
    }
    let passes = if trailing {
        case.tokens
    } else {
        case.tokens.saturating_sub(1)
    };
    for token in 0..passes {
        let position = case.input_token_ids.len() as u32 + token;
        for layer in 0..model.shape.num_hidden_layers {
            let stage = format!("decode_range_t{token}_l{layer}");
            let value = &outputs[&stage].value;
            ensure!(
                value["start_position"] == json!(position),
                "{stage}: activation position did not advance"
            );
            ensure!(
                value["rows"]
                    .as_array()
                    .context("missing activation rows")?
                    .len()
                    == 1,
                "{stage}: decode must contain one row"
            );
            let first = if model.shape.layer_types[layer as usize] == "sliding_attention" {
                (position + 1).saturating_sub(model.shape.sliding_window)
            } else {
                0
            };
            let positions = value["kv"]
                .as_array()
                .context("missing KV cache")?
                .iter()
                .map(|key| key["position"].as_u64().context("missing KV position"))
                .collect::<Result<Vec<_>>>()?;
            ensure!(
                positions == (first..=position).map(u64::from).collect::<Vec<_>>(),
                "{stage}: incorrect KV growth or eviction, positions={positions:?}"
            );
        }
    }
    Ok(())
}

fn has_nonzero_payload(value: &Value) -> bool {
    match value {
        Value::Object(fields) => {
            fields
                .get("bytes")
                .and_then(Value::as_str)
                .is_some_and(|bytes| bytes.chars().any(|ch| ch != '0'))
                || fields.values().any(has_nonzero_payload)
        }
        Value::Array(values) => values.iter().any(has_nonzero_payload),
        _ => false,
    }
}

fn validate_donor_bindings(
    dir: &Path,
    outputs: &BTreeMap<String, Boundary>,
    tokens: u32,
) -> Result<()> {
    for pass in 0..=tokens {
        let stage = |layer| {
            if pass == 0 {
                format!("prefill_range_l{layer}")
            } else {
                format!("decode_range_t{}_l{layer}", pass - 1)
            }
        };
        for layer in [2, 3] {
            let input: Value = read_json(&dir.join(stage(layer)).join("input_manifest.json"))?;
            ensure!(
                input["donor_a_kv"]["commitment"] == json!(outputs[&stage(0)].commitment),
                "{}: incorrect sliding donor binding",
                stage(layer)
            );
            ensure!(
                input["donor_b_kv"]["commitment"] == json!(outputs[&stage(1)].commitment),
                "{}: incorrect full-attention donor binding",
                stage(layer)
            );
        }
    }
    Ok(())
}

fn validate_model_bundle(
    model_bundle: &Path,
    model_dir: &Path,
    model: &ModelManifest,
) -> Result<()> {
    let shape = &model.shape;
    ensure!(
        shape.num_hidden_layers == 4
            && shape.hidden_size == 128
            && shape.num_attention_heads == 4
            && shape.num_key_value_heads == 2
            && shape.head_dim == 32
            && shape.hidden_size_per_layer_input == 8
            && shape.sliding_window == 16
            && shape.num_kv_shared_layers == 2
            && shape.vocab_size == 512
            && shape.norm_eps > 0
            && shape.final_logit_softcap > 0,
        "model bundle shape or active numerics changed"
    );
    let weights = direct_infer::detwgt::MmapDetwgt::open(&model_bundle.join("model.detwgt"))?;
    for layer in 0..4 {
        for tensor in [
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        ] {
            let values =
                weights.values(&format!("model.language_model.layers.{layer}.{tensor}"))?;
            ensure!(
                values.len() == 128 * 512
                    && values.len() * 4 > 196_608
                    && (values.len() * 4) % 196_608 != 0,
                "model bundle does not cross a page with a partial tail"
            );
            ensure!(
                values.iter().all(|v| *v != 0)
                    && values.iter().any(|v| *v < 0)
                    && values.iter().any(|v| *v > 0)
                    && values.iter().any(|v| v % 65536 != 0),
                "model bundle weights are not dense signed fractional values"
            );
        }
        let path = model_dir.join(format!("raster/prefill-range/layer{layer}"));
        let artifact = raster_runtime::read_raster_artifact(
            &path.with_extension("rastered"),
            &path.with_extension("rindex"),
            &raster_runtime::ReadLimits::unbounded(),
        )?;
        ensure!(
            artifact.roots_agree(),
            "imported layer index commitment mismatch"
        );
        let value = compare::value_json(&artifact.value)?;
        for name in ["w_gate", "w_up", "w_down"] {
            let matrix = &value[name];
            let pages = matrix["pages"]
                .as_array()
                .context("missing imported matrix pages")?;
            ensure!(
                matrix["page_size"] == json!(196_608)
                    && matrix["byte_len"] == json!(262_144)
                    && pages.len() == 2
                    && pages[0]["bytes"].as_str().map(str::len) == Some(196_608 * 2)
                    && pages[1]["bytes"].as_str().map(str::len) == Some(65_536 * 2),
                "layer{layer}.{name}: imported matrix must contain a full page and a partial tail"
            );
        }
        let params = &value["params"];
        ensure!(
            params["ffn_size"] == json!(512) && params["ple_width"] == json!(8),
            "incorrect imported FFN or PLE shape"
        );
        ensure!(
            params["kv_donor_layer"] == json!(if layer < 2 { -1 } else { layer - 2 }),
            "incorrect imported KV donor"
        );
        ensure!(
            params["rotary_dim"] == json!(if layer % 2 == 0 { 32 } else { 16 }),
            "incorrect partial RoPE configuration"
        );
    }
    Ok(())
}

pub fn worker(spec: &Path, output: &Path, direct: bool) -> Result<()> {
    fs::create_dir_all(output)?;
    if direct {
        let config = DirectInferenceConfig {
            base_dir: std::env::current_dir()?,
            run_spec_path: spec.to_path_buf(),
        };
        let diagnostic = DirectInferenceExecutor.run_with_diagnostics(config.clone())?;
        let names = diagnostic
            .boundaries
            .iter()
            .map(|record| record.stage.clone())
            .collect::<Vec<_>>();
        for record in diagnostic.boundaries {
            let dir = output.join(&record.stage);
            fs::create_dir(&dir)?;
            fs::write(dir.join("output.bin"), record.output_bytes)?;
            fs::write(dir.join("output.rindex"), record.output_index)?;
            write_json(
                &dir.join("output_manifest.json"),
                &json!({"output": {"commitment": record.output_commitment}}),
            )?;
        }
        write_json(&output.join("boundary-order.json"), &names)?;
        write_json(&output.join("result.json"), &diagnostic.report.result)?;
        let ordinary = DirectInferenceExecutor.run(config)?;
        ensure!(
            ordinary == diagnostic.report.result,
            "ordinary and diagnostic direct execution differ"
        );
    } else {
        let result = build_claim(ClaimBuildOptions {
            base_dir: output.to_path_buf(),
            run_spec_path: spec.to_path_buf(),
            current_exe: std::env::current_exe()?,
            staged_backend: StagedExecutionBackend::InProcess,
            claimed_token_ids: None,
        })?;
        write_json(
            &output.join("result.json"),
            &json!({"chain_dir": result.chain_dir, "result": result.final_result.context("native-staged did not finalize")?}),
        )?;
    }
    Ok(())
}

fn command(program: impl AsRef<std::ffi::OsStr>, root: &Path) -> Command {
    let mut command = Command::new(program);
    command.current_dir(root);
    // These binaries only execute host code. Keep their build cache separate
    // from ordinary inference/proving builds. RISC Zero otherwise regenerates
    // guest metadata on every per-stage Cargo invocation, even on a warm run.
    command.env("CARGO_TARGET_DIR", root.join("target/parity-host"));
    command.env("RISC0_SKIP_BUILD", "1");
    command.env("CARGO_PROFILE_RELEASE_INCREMENTAL", "true");
    // `cargo run` exports its package directory. It must not become an input
    // to the nested dependency build scripts (notably cc/ring).
    command.env_remove("CARGO_MANIFEST_DIR");
    command.env_remove("CARGO_MANIFEST_PATH");
    // Do not inherit debugging instrumentation or cache-disable settings into
    // the canonical gate. Each worker installs its own stage input environment.
    for name in [
        "DIRECT_INFER_COMPARE_TRACE",
        "DIRECT_INFER_TRACE_COMMITMENTS",
        "STAGED_INFER_MATERIALIZATION_CACHE_ENTRIES",
        "DIRECT_NATIVE_MATERIALIZATION_CACHE_ENTRIES",
        "STAGED_INFER_PREFILL_WEIGHT_CACHE_ENTRIES",
        "DIRECT_NATIVE_PREFILL_WEIGHT_CACHE_ENTRIES",
        "STAGED_INFER_AUX_PARALLELISM",
        "DIRECT_NATIVE_AUX_PARALLELISM",
        "RASTER_OUTPUT_DIR",
        "RASTER_TRACE_PATH",
    ] {
        command.env_remove(name);
    }
    command
}

fn logged(mut command: Command, log: &Path) -> Result<f64> {
    let file = File::create(log)?;
    command
        .stdout(Stdio::from(file.try_clone()?))
        .stderr(Stdio::from(file));
    let started = Instant::now();
    let status = command
        .status()
        .with_context(|| format!("failed to start command; log: {}", log.display()))?;
    ensure!(
        status.success(),
        "command failed ({status}); log: {}",
        log.display()
    );
    Ok(started.elapsed().as_secs_f64())
}
