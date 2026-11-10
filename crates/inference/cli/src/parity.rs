//! The local three-path regression gate: the chain through Raster and natively,
//! checked against direct generation position by position. Every worker owns its
//! runtime state.
mod challenge;
pub(crate) mod compare;

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
    check_inventory, compare, read_boundaries, teacher_forced_names, Boundary, RangedShape,
};

#[derive(Debug, Deserialize)]
struct Case {
    name: String,
    prompt: String,
    raw_prompt: bool,
    tokens: u32,
    input_token_ids: Vec<u32>,
}

pub fn run(skip_challenges: bool) -> Result<()> {
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
    report["challenge_leg"] = json!(if skip_challenges { "skipped" } else { "run" });
    let result = run_suite(&root, &run_dir, &mut report, skip_challenges);
    report["elapsed_seconds"] = json!(started.elapsed().as_secs_f64());
    report["passed"] = json!(result.is_ok());
    if let Err(error) = &result {
        report["error"] = json!(format!("{error:#}"));
    }
    write_json(&run_dir.join("report.json"), &report)?;
    result
        .with_context(|| format!("parity failed; artifacts and report: {}", run_dir.display()))?;
    println!(
        "PASS: 3 cases, 3 independent paths, exact parity ({:.2}s total).",
        started.elapsed().as_secs_f64()
    );
    Ok(())
}

fn run_suite(root: &Path, run_dir: &Path, report: &mut Value, skip_challenges: bool) -> Result<()> {
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
        // several ranges.
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
            skip_challenges,
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
    skip_challenges: bool,
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
    // Preparation discovers the claimed ids with direct decode.
    let prepared = prepare_claim_run(&ClaimBuildOptions {
        base_dir: dir.to_path_buf(),
        run_spec_path: spec.clone(),
        current_exe: exe.to_path_buf(),
        staged_backend: StagedExecutionBackend::InProcess,
        claimed_token_ids: None,
    })?;
    let claimed = prepared.claimed_token_ids.clone();
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
    let direct_dir = dir.join("native-direct");

    // Raster and native-staged, byte for byte, in the expected order.
    let order = inference_artifacts::read_stage_order(&native_dir)?
        .context("native chain recorded no stage inventory")?;
    let shape = RangedShape::from_names(&order);
    ensure!(
        shape.layers == model.shape.num_hidden_layers,
        "the chain has {} layers per range, the model has {}",
        shape.layers,
        model.shape.num_hidden_layers
    );
    let names = teacher_forced_names(shape);
    let raster_outputs =
        read_boundaries(&raster_dir, &names, false).context("Raster-staged artifacts")?;
    let native = read_boundaries(&native_dir, &names, false).context("native-staged artifacts")?;
    for name in &names {
        compare(name, &raster_outputs[name], &native[name])
            .context("Raster-staged ↔ native-staged")?;
    }
    validate_donor_bindings(&raster_dir, &raster_outputs, shape)?;
    validate_donor_bindings(&native_dir, &native, shape)?;

    // Direct generation: its own coverage, then the chain position by position.
    ensure!(
        claimed.len() == case.tokens as usize,
        "direct decode produced {} of {} tokens; the pinned corpus fills its budget",
        claimed.len(),
        case.tokens
    );
    let direct_names =
        direct_infer::diagnostics::boundary_names(model.shape.num_hidden_layers, case.tokens);
    let direct =
        read_boundaries(&direct_dir, &direct_names, true).context("native-direct artifacts")?;
    validate_case(case, model, &direct).context("native-direct coverage")?;
    for token in &claimed {
        selected.insert(*token);
    }
    for step in 0..case.tokens {
        let stage = logits_stage(step);
        logit_vectors.insert(serde_json::to_string(&direct[&stage].value["logits"])?);
    }
    let mut comparison = compare_teacher_forced(&native_dir, &direct_dir)?;
    ensure!(
        comparison["prompt_token_ids"] == json!(case.input_token_ids),
        "the prompt ranges' ids differ from the pinned ids"
    );
    ensure!(
        comparison["generated_token_ids"] == json!(claimed),
        "the chain's output differs from the claimed ids"
    );
    let checkpoints = inference_artifacts::read_checkpoint_hashes(
        &native_dir.join(inference_artifacts::CHECKPOINT_HASHES_TXT),
    )?;
    comparison["checkpoints"] = json!(names.len());
    comparison["final_checkpoint_sha256"] = json!(checkpoints.last());

    // The last prompt context is the chain's record of what it was given.
    let last = shape
        .prompt_ranges
        .checked_sub(1)
        .context("the chain has no prompt ranges")?;
    let context = &native[&format!("prompt_check_p{last}")].value["context"];
    ensure!(
        context["fault"] == json!(0) && context["complete"] == json!(true),
        "an honest prompt recorded a preparation fault: {context}"
    );
    ensure!(
        context["input_text_sha256"]
            == json!(format!("{:x}", Sha256::digest(case.prompt.as_bytes())))
            && context["model_id"] == json!(model.model_id)
            && context["prompt_tokens"] == json!(case.input_token_ids.len()),
        "the prompt context does not identify the run: {context}"
    );
    // The final record is what a contract reads: the context it reports is the
    // one the prompt stages established, and its commitment hashes its own text.
    let record = &raster_outputs["output_finalize"].value;
    for (field, source) in [
        ("input_text_sha256", "input_text_sha256"),
        ("model_id", "model_id"),
        ("raw_prompt", "raw_prompt"),
        ("max_new_tokens", "max_new_tokens"),
        ("range_size", "range_size"),
        ("prompt_tokens", "prompt_tokens"),
        ("prompt_complete", "complete"),
        ("prompt_fault", "fault"),
        ("prompt_fault_position", "fault_position"),
    ] {
        ensure!(
            record[field] == context[source],
            "the final record's {field} is {}; the prompt context says {}",
            record[field],
            context[source]
        );
    }
    ensure!(
        record["max_new_tokens"] == json!(case.tokens),
        "the final record does not report the run's token budget"
    );
    let text = record["generated_text"]
        .as_str()
        .context("the final record has no text")?;
    ensure!(
        record["output_text_sha256"] == json!(format!("{:x}", Sha256::digest(text.as_bytes()))),
        "the output commitment is not the hash of the emitted text"
    );
    ensure!(
        record["check_fault"] == json!(0)
            && record["output_fault"] == json!(0)
            && record["model_fault"] == json!(0)
            && record["claim_mismatches"] == json!(0),
        "an honest claim published a fault: {record}"
    );
    ensure!(
        native_info["result"] == *record,
        "the result artifact is not the chain's final record"
    );
    let direct_result: Value = read_json(&direct_dir.join("result.json"))?;
    ensure!(
        direct_result == *record,
        "direct generation's result is not the chain's final record"
    );

    // Only the templated case has a safe first cut to build these from.
    if !case.raw_prompt {
        comparison["prompt_faults"] =
            run_prompt_fault_leg(root, dir, exe, raster, model_path, model, &prepared)?;
        comparison["chain_faults"] = run_chain_fault_leg(
            root, dir, exe, raster, model_path, model, case, &prepared, &claimed,
        )?;
    }
    if !case.raw_prompt && skip_challenges {
        comparison["challenges"] = json!("skipped");
    } else if !case.raw_prompt {
        comparison["challenges"] = challenge::run_challenge_leg(&challenge::Leg {
            root,
            dir: dir.join("challenges"),
            exe,
            raster,
            model_path,
            model,
            case,
            prepared: &prepared,
            claimed: &claimed,
        })
        .context("challenge leg")?;
    }
    println!(
        "  PASS: {} checkpoints, {} direct boundaries, {} complete logit vectors, {} rows in {} prompt + {} claimed ranges ({:.2}s)",
        names.len(),
        direct_names.len(),
        case.tokens,
        comparison["rows_compared"],
        comparison["prompt_ranges"],
        comparison["claimed_ranges"],
        started.elapsed().as_secs_f64()
    );
    report["checkpoints"] = json!(names.len());
    report["direct_boundaries"] = json!(direct_names.len());
    report["logit_vectors_compared"] = json!(case.tokens);
    report["result"] = record.clone();
    report["comparison"] = comparison;
    report["elapsed_seconds"] = json!(started.elapsed().as_secs_f64());
    Ok(())
}

/// A false claim must reach the final record. Runs whole chains through Raster
/// and native-staged over wrong preparations and wrong claims, and checks that
/// every stage finishes, both agree byte for byte, and the record says which.
#[allow(clippy::too_many_arguments)]
fn run_chain_fault_leg(
    root: &Path,
    case_dir: &Path,
    exe: &Path,
    raster: &Path,
    model_path: &Path,
    model: &ModelManifest,
    case: &Case,
    prepared: &crate::claim::PreparedClaimRun,
    claimed: &[u32],
) -> Result<Value> {
    let prompt = &prepared.prepared_run.prompt;
    let range_size = model.range_size;
    let model_dir = model_path
        .parent()
        .context("model manifest has no parent directory")?;
    let tokenizer: Value = read_json(&model_dir.join(&model.bundle.tokenizer_path))?;
    let ranges = run_prep::marked_ranges(
        &prompt.initial_pieces,
        range_size as usize,
        &run_prep::prompt_tokenizer(&tokenizer)?,
    )?;
    let canonical: Vec<Vec<u32>> = ranges.iter().map(|range| range.marks.clone()).collect();
    // Every piece's mark, and the first place after the start where a token begins.
    let all: Vec<u32> = canonical.concat();
    let second_token = all
        .iter()
        .skip(1)
        .position(|mark| *mark != 0)
        .map(|index| index + 1)
        .context("the pinned case's prompt is a single token")?;
    let (template_path, template) = model.verified_raster_template(model_path)?;
    let template_dir = template_path
        .parent()
        .context("manifest template has no parent directory")?;

    let greedy = claimed
        .get(..2)
        .context("the pinned case claims too few tokens")?;
    let not_greedy = [(greedy[0] + 1) % model.shape.vocab_size as u32, greedy[1]];
    // Just past the tables, and far enough past that its page does not exist.
    let unknown = [model.shape.vocab_size as u32, u32::MAX];
    let positions = case.input_token_ids.len() as u32;
    // Name, marks, claimed ids, budget, what the record must say, then what to
    // add to the schedule's three counts.
    // Invalid prompt histories suppress downstream rows and retain the fault.
    let cases = [
        (
            "empty-middle-range",
            vec![
                all[..second_token].to_vec(),
                Vec::new(),
                all[second_token..].to_vec(),
            ],
            greedy,
            2,
            true,
            run_prep::CHECK_FAULT_NO_CLAIMED_TOKENS,
            positions,
            [0, 0],
        ),
        (
            "empty-last-range",
            vec![all.clone(), Vec::new()],
            greedy,
            2,
            true,
            run_prep::CHECK_FAULT_NO_CLAIMED_TOKENS,
            positions,
            [0, 0],
        ),
        (
            "not-greedy",
            canonical.clone(),
            &not_greedy[..],
            2,
            false,
            0,
            0,
            [0, 0],
        ),
        (
            "unknown-ids",
            canonical.clone(),
            &unknown[..],
            2,
            false,
            0,
            0,
            [0, 0],
        ),
        (
            "short-of-budget",
            canonical.clone(),
            greedy,
            8,
            false,
            0,
            0,
            [0, 0],
        ),
        // A run that does not follow its schedule: the schedule names counts the
        // manifest's stages do not run. The last is refused when it is bound, the
        // others by the stages that reach the difference.
        (
            "schedule-extra-prompt-range",
            canonical.clone(),
            greedy,
            2,
            false,
            0,
            0,
            [1, 0],
        ),
        (
            "schedule-extra-claimed-range",
            canonical,
            greedy,
            2,
            false,
            0,
            0,
            [0, 1_i64],
        ),
    ];

    let mut report = Vec::new();
    for (name, marks, ids, budget, prompt_fault, check_fault, check_position, skew) in cases {
        let dir = case_dir.join("chain-faults").join(name);
        let files = dir.join("prompt");
        let manifest = dir.join("Raster.toml");
        let mark_files = run_prep::write_marks(&marks, &files)?;
        let claimed_files = run_prep::write_claimed_ranges(ids, range_size as usize, &files)?;
        let rendered = crate::claim::render_ranged_manifest(
            &template,
            template_dir,
            &run_prep::write_prompt_text(&prompt.resolved_prompt, &files)?,
            &run_prep::write_run_config(
                &run_prep::RunConfig {
                    model_id: model.model_id.clone(),
                    raw_prompt: case.raw_prompt,
                    max_new_tokens: budget,
                    range_size,
                },
                &files,
            )?,
            &run_prep::write_run_schedule(
                (mark_files.len() as i64 + skew[0]) as usize,
                (claimed_files.len() as i64 + skew[1]) as usize,
                &files,
            )?,
            &mark_files,
            &claimed_files,
        )?;
        fs::write(&manifest, rendered)?;

        let raster_dir = dir.join("raster");
        fs::create_dir(&raster_dir)?;
        let mut run = command(raster, root);
        run.args(["raster", "chain", "run"])
            .arg(&manifest)
            .args(["--no-auth", "--run"])
            .arg(&raster_dir);
        logged(run, &dir.join("raster.log"))
            .with_context(|| format!("chain fault `{name}`: a Raster stage did not finish"))?;
        let native_run = staged_infer::CheckpointedInferenceExecutor
            .run(staged_infer::CheckpointedInferenceConfig {
                base_dir: dir.clone(),
                manifest_path: manifest.clone(),
                current_exe: exe.to_path_buf(),
                staged_backend: StagedExecutionBackend::InProcess,
                parity_policy: staged_infer::ParityPolicy::Skip,
            })
            .with_context(|| format!("chain fault `{name}`: a native stage did not finish"))?;

        let order = inference_artifacts::read_stage_order(&raster_dir)?
            .with_context(|| format!("chain fault `{name}`: no stage inventory"))?;
        let raster_outputs = read_boundaries(&raster_dir, &order, false)?;
        let native = read_boundaries(&native_run.chain_dir, &order, false)?;
        for stage in &order {
            compare(stage, &raster_outputs[stage], &native[stage])
                .with_context(|| format!("chain fault `{name}`"))?;
        }

        let record = &raster_outputs["output_finalize"].value;
        ensure!(
            (record["prompt_fault"] != json!(0)) == prompt_fault,
            "chain fault `{name}`: expected a prompt fault: {prompt_fault}; got {record}"
        );
        if skew == [0, 0] {
            ensure!(
                record["check_fault"] == json!(check_fault)
                    && record["check_fault_position"] == json!(check_position),
                "chain fault `{name}`: expected check fault {check_fault} at {check_position}; got {record}"
            );
        } else {
            ensure!(
                record["model_fault"] != json!(0),
                "chain fault `{name}`: a run that left its schedule kept a clean record: {record}"
            );
        }
        if prompt_fault {
            ensure!(
                record["model_fault"] != json!(0),
                "chain fault `{name}` lost its execution fault"
            );
        }
        match name {
            "not-greedy" => ensure!(
                record["claim_mismatches"].as_u64().unwrap_or(0) > 0
                    && record["first_claim_mismatch_position"] == json!(positions),
                "chain fault `{name}`: the verdict does not name the changed token: {record}"
            ),
            "unknown-ids" => ensure!(
                record["claim_mismatches"] == json!(2)
                    && record["first_claim_mismatch_position"] == json!(positions),
                "chain fault `{name}`: the verdict does not reject both ids: {record}"
            ),
            "short-of-budget" => ensure!(
                record["claim_mismatches"] == json!(0)
                    && record["output_fault"] == json!(run_prep::OUTPUT_FAULT_SHORT_WITHOUT_EOS),
                "chain fault `{name}`: the output fault is not published: {record}"
            ),
            _ => {}
        }
        let result: InferenceResult = serde_json::from_value(record.clone())?;
        ensure!(
            native_run.final_result.as_ref() == Some(&result),
            "chain fault `{name}`: the native result is not the final record"
        );
        report.push(json!({
            "case": name,
            "stages": order.len(),
            "model_fault": record["model_fault"],
            "prompt_fault": record["prompt_fault"],
            "check_fault": record["check_fault"],
            "claim_mismatches": record["claim_mismatches"],
            "output_fault": record["output_fault"],
        }));
    }
    Ok(json!(report))
}

/// A wrong preparation must not stop a prompt stage: an aborting stage has no
/// output to compare or replay. Runs the prompt stages over wrong marks through
/// Raster and native-staged, and checks that both finish, agree byte for byte,
/// and record the fault.
fn run_prompt_fault_leg(
    root: &Path,
    case_dir: &Path,
    exe: &Path,
    raster: &Path,
    model_path: &Path,
    model: &ModelManifest,
    prepared: &crate::claim::PreparedClaimRun,
) -> Result<Value> {
    use run_prep::{
        RunFile, FAULT_CUT_BEYOND_TEXT, FAULT_EMPTY_RANGE, FAULT_NONE, FAULT_NOT_THE_MERGE,
    };
    let pieces = &prepared.prepared_run.prompt.initial_pieces;
    let tokenizer: Value = read_json(
        &model_path
            .parent()
            .context("model manifest has no parent directory")?
            .join(&model.bundle.tokenizer_path),
    )?;
    // The merge's own marks, range by range.
    let canonical: Vec<Vec<u32>> = run_prep::marked_ranges(
        pieces,
        model.range_size as usize,
        &run_prep::prompt_tokenizer(&tokenizer)?,
    )?
    .into_iter()
    .map(|range| range.marks)
    .collect();
    let first = canonical[0].clone();
    let with_first = |first: Vec<u32>| {
        let mut ranges = canonical.clone();
        ranges[0] = first;
        ranges
    };
    let second_token = first
        .iter()
        .skip(1)
        .position(|mark| *mark != 0)
        .map(|index| index + 1)
        .context("the pinned case's first range is a single token")?;
    let mut past_the_end = canonical.clone();
    past_the_end
        .last_mut()
        .context("the prompt has no range")?
        .extend([1; 5]);
    let mut empty_range = canonical.clone();
    empty_range.insert(1, Vec::new());
    let mut cases = vec![
        ("past-the-end", past_the_end, FAULT_CUT_BEYOND_TEXT, true),
        ("empty-range", empty_range, FAULT_EMPTY_RANGE, true),
        (
            "left-out",
            vec![first[..second_token].to_vec()],
            FAULT_NONE,
            false,
        ),
    ];
    // A token claimed as two: its pieces are vocabulary entries, and the merge
    // joins them, so the claim is not the merge's result.
    let inside = canonical.iter().enumerate().find_map(|(range, marks)| {
        marks
            .iter()
            .position(|mark| *mark == 0)
            .map(|index| (range, index))
    });
    if let Some((range, index)) = inside {
        let mut split = canonical.clone();
        split[range][index] = 1;
        cases.push(("split-token", split, FAULT_NOT_THE_MERGE, true));
    }
    // Two neighbouring tokens of one segment, claimed as one token and as the
    // ends of two ranges.
    let same_segment = (1..first.len())
        .find(|&index| first[index] != 0 && pieces[index - 1].segment == pieces[index].segment);
    if let Some(boundary) = same_segment {
        let mut joined = first.clone();
        joined[boundary] = 0;
        cases.push((
            "joined-tokens",
            with_first(joined),
            FAULT_NOT_THE_MERGE,
            true,
        ));
        // A range may end between any two tokens, even inside a word.
        let mut ranges = vec![first[..boundary].to_vec(), first[boundary..].to_vec()];
        ranges.extend(canonical[1..].iter().cloned());
        cases.push(("cut-inside-a-segment", ranges, FAULT_NONE, true));
    }

    let template = fs::read_to_string(&prepared.manifest_path)?;
    let mut report = Vec::new();
    for (name, marks, fault, complete) in cases {
        let dir = case_dir.join("prompt-faults").join(name);
        let files: Vec<RunFile> = run_prep::write_marks(&marks, &dir.join("prompt"))?;
        let manifest = dir.join("Raster.toml");
        fs::write(&manifest, prompt_only_manifest(&template, &files)?)?;

        let raster_dir = dir.join("raster");
        fs::create_dir(&raster_dir)?;
        let mut run = command(raster, root);
        run.args(["raster", "chain", "run"])
            .arg(&manifest)
            .args(["--no-auth", "--run"])
            .arg(&raster_dir);
        logged(run, &dir.join("raster.log"))
            .with_context(|| format!("prompt fault `{name}`: a Raster stage did not finish"))?;
        let native_dir = staged_infer::CheckpointedInferenceExecutor
            .run(staged_infer::CheckpointedInferenceConfig {
                base_dir: dir.clone(),
                manifest_path: manifest.clone(),
                current_exe: exe.to_path_buf(),
                staged_backend: StagedExecutionBackend::InProcess,
                parity_policy: staged_infer::ParityPolicy::Skip,
            })
            .with_context(|| format!("prompt fault `{name}`: a native stage did not finish"))?
            .chain_dir;

        let order = inference_artifacts::read_stage_order(&raster_dir)?
            .with_context(|| format!("prompt fault `{name}`: no stage inventory"))?;
        let raster_outputs = read_boundaries(&raster_dir, &order, false)?;
        let native = read_boundaries(&native_dir, &order, false)?;
        for stage in &order {
            compare(stage, &raster_outputs[stage], &native[stage])
                .with_context(|| format!("prompt fault `{name}`"))?;
        }
        let last = format!("prompt_check_p{}", marks.len() - 1);
        let context = &raster_outputs[&last].value["context"];
        ensure!(
            context["fault"] == json!(fault) && context["complete"] == json!(complete),
            "prompt fault `{name}`: expected fault {fault}, complete {complete}; got {context}"
        );
        report.push(json!({"case": name, "stages": order.len(), "context": context}));
    }
    report.push(run_long_text_case(root, case_dir, exe, raster, &template)?);
    Ok(json!(report))
}

/// The most a prompt-stage tile writes per chunk of text: one single-byte atom
/// per byte. Raster limits that and native does not, so only running it through
/// Raster shows that the chunk size fits.
fn run_long_text_case(
    root: &Path,
    case_dir: &Path,
    exe: &Path,
    raster: &Path,
    template: &str,
) -> Result<Value> {
    let dir = case_dir.join("prompt-faults/long-text");
    let text = "ab ".repeat(1024);
    let file = run_prep::write_prompt_text(&text, &dir.join("prompt"))?;
    let binding = crate::claim::RunExternal::new(&file.path, &file.index_path, &file.commitment)
        .binding("text");
    let mut rendered = String::new();
    for block in manifest_blocks(template) {
        let keep = block[0].trim() == "[chain]"
            || (block[0].trim() == "[[chain.stage]]" && block_name(&block) == "prompt_bind");
        if !keep {
            continue;
        }
        for line in block {
            rendered.push_str(if line.starts_with("inputs.text =") {
                &binding
            } else {
                line
            });
            rendered.push('\n');
        }
    }
    let manifest = dir.join("Raster.toml");
    fs::write(&manifest, rendered)?;

    let raster_dir = dir.join("raster");
    fs::create_dir(&raster_dir)?;
    let mut run = command(raster, root);
    run.args(["raster", "chain", "run"])
        .arg(&manifest)
        .args(["--no-auth", "--run"])
        .arg(&raster_dir);
    logged(run, &dir.join("raster.log")).context("a long text did not finish through Raster")?;
    let native_dir = staged_infer::CheckpointedInferenceExecutor
        .run(staged_infer::CheckpointedInferenceConfig {
            base_dir: dir.clone(),
            manifest_path: manifest,
            current_exe: exe.to_path_buf(),
            staged_backend: StagedExecutionBackend::InProcess,
            parity_policy: staged_infer::ParityPolicy::Skip,
        })
        .context("a long text did not finish natively")?
        .chain_dir;
    let names = vec![String::from("prompt_bind")];
    let bound = read_boundaries(&raster_dir, &names, false)?;
    let native = read_boundaries(&native_dir, &names, false)?;
    compare("prompt_bind", &bound["prompt_bind"], &native["prompt_bind"]).context("a long text")?;
    let context = &bound["prompt_bind"].value["context"];
    ensure!(
        context["fault"] == json!(0)
            && context["input_text_sha256"]
                == json!(format!("{:x}", Sha256::digest(text.as_bytes()))),
        "a long text was not bound: {context}"
    );
    Ok(json!({"case": "long-text", "stages": 1, "context": context}))
}

/// A manifest's lines, split at each table header.
fn manifest_blocks(manifest: &str) -> Vec<Vec<&str>> {
    let mut blocks: Vec<Vec<&str>> = Vec::new();
    for line in manifest.lines() {
        if line.trim_start().starts_with('[') || blocks.is_empty() {
            blocks.push(Vec::new());
        }
        blocks.last_mut().expect("a block was pushed").push(line);
    }
    blocks
}

/// The name a manifest table gives its stage or repeat.
fn block_name(block: &[&str]) -> String {
    block
        .iter()
        .find_map(|line| line.trim().strip_prefix("name = \""))
        .map(|name| name.trim_end_matches('"').to_string())
        .unwrap_or_default()
}

/// A run manifest's prompt stages alone, over these marks.
fn prompt_only_manifest(template: &str, cuts: &[run_prep::RunFile]) -> Result<String> {
    let mut out = String::new();
    let mut in_prompt = false;
    for block in &manifest_blocks(template) {
        let name = block_name(block);
        match block[0].trim() {
            "[chain]" => out.push_str(&(block.join("\n") + "\n")),
            "[chain.input.prompt_marks]" => {
                let dir = cuts[0].path.parent().context("marks have no directory")?;
                out.push_str(&format!(
                    "\n[chain.input.prompt_marks]\nindex = \"j\"\npath = {:?}\nindex_path = {:?}\ncommitments = {:?}\n",
                    dir.join("marks_{j}.rastered"),
                    dir.join("marks_{j}.rindex"),
                    cuts.iter().map(|cut| &cut.commitment).collect::<Vec<_>>(),
                ));
            }
            "[[chain.stage]]" if name == "prompt_bind" => {
                out.push_str(&(block.join("\n") + "\n"));
            }
            "[[chain.repeat]]" => {
                in_prompt = name == "prompt";
                if in_prompt {
                    for line in block {
                        if line.trim().starts_with("count =") {
                            out.push_str(&format!("count = {}\n", cuts.len()));
                        } else {
                            out.push_str(line);
                            out.push('\n');
                        }
                    }
                }
            }
            "[[chain.repeat.stage]]" if in_prompt && name.starts_with("prompt_") => {
                out.push_str(&(block.join("\n") + "\n"));
            }
            _ => {}
        }
    }
    Ok(out)
}

/// Compares a chain with direct generation for the same prompt and output, bit
/// for bit and position by position.
///
/// The prompt ranges' token ids must concatenate to direct's tokenization.
/// Stitching each layer's ranges together, row `i` of every embedding, PLE and
/// layer stage must equal direct's prefill row `i` (`i < prompt_len`) or decode
/// step `i - prompt_len`'s single row, and every K/V entry a range publishes,
/// inherited or its own, direct's entry at the same position. Direct never runs a
/// pass for its last token, so the chain's row for it has no reference; nothing
/// is scored from that row. `output_finalize` must be the same record, and the
/// last range's check the same edge as direct's last selection.
///
/// Stages are read one layer at a time, so real-model chains with hundreds of
/// rows fit in memory.
pub(crate) fn compare_teacher_forced(tf_dir: &Path, direct_dir: &Path) -> Result<Value> {
    let load = |dir: &Path, stage: &str| -> Result<Boundary> {
        let boundary = compare::read_boundary(&dir.join(stage))
            .with_context(|| format!("{stage} in {}", dir.display()))?;
        compare::validate_errors(stage, &boundary.value)?;
        Ok(boundary)
    };
    let tf_names = inference_artifacts::read_stage_order(tf_dir)?
        .with_context(|| format!("{}: missing recorded stage inventory", tf_dir.display()))?;
    let shape = RangedShape::from_names(&tf_names);
    ensure!(
        tf_names == teacher_forced_names(shape),
        "{}: not a chain run",
        tf_dir.display()
    );
    let direct_names: Vec<String> = read_json(&direct_dir.join("boundary-order.json"))?;

    let prompt_ids = load(direct_dir, "prompt_prepare")?.value["token_ids"].clone();
    let generated = load(direct_dir, "output_finalize")?.value["generated_token_ids"].clone();
    let prompt_len = prompt_ids.as_array().context("missing prompt ids")?.len();
    let tokens = generated.as_array().context("missing generated ids")?.len();
    ensure!(tokens > 0, "direct generation produced no tokens");
    check_inventory(
        &direct_infer::diagnostics::boundary_names(shape.layers, tokens as u32),
        &direct_names,
    )
    .with_context(|| format!("{}: not a direct run", direct_dir.display()))?;

    let mut ranged_prompt = Vec::new();
    for j in 0..shape.prompt_ranges {
        let ids = load(tf_dir, &format!("prompt_check_p{j}"))?.value["token_ids"].clone();
        ranged_prompt.extend(ids.as_array().cloned().unwrap_or_default());
    }
    ensure!(
        json!(ranged_prompt) == prompt_ids,
        "prompt ranges tokenize differently from the whole prompt"
    );
    compare(
        "output_finalize",
        &load(tf_dir, "output_finalize")?,
        &load(direct_dir, "output_finalize")?,
    )
    .context("chain ↔ direct final record")?;
    let last_check = format!("check_c{}", shape.claimed_ranges - 1);
    let last_select = format!("decode_select_t{}", tokens - 1);
    compare(
        &last_check,
        &load(tf_dir, &last_check)?,
        &load(direct_dir, &last_select)?,
    )
    .with_context(|| format!("chain {last_check} ↔ direct {last_select}"))?;

    // The same family of stages, ranged and direct.
    let ranges = (0..shape.prompt_ranges)
        .map(|j| format!("p{j}"))
        .chain((0..shape.claimed_ranges).map(|k| format!("c{k}")))
        .collect::<Vec<_>>();
    let passes = tokens - 1;
    let mut families = vec![(
        ranges
            .iter()
            .map(|r| format!("embed_{r}"))
            .collect::<Vec<_>>(),
        String::from("input_embedding"),
        (0..passes)
            .map(|t| format!("decode_embed_t{t}"))
            .collect::<Vec<_>>(),
    )];
    for layer in 0..shape.layers {
        families.push((
            ranges.iter().map(|r| format!("aux_{r}_l{layer}")).collect(),
            format!("prefill_prepare_aux_l{layer}"),
            (0..passes)
                .map(|t| format!("decode_aux_t{t}_l{layer}"))
                .collect(),
        ));
        families.push((
            ranges
                .iter()
                .map(|r| format!("range_{r}_l{layer}"))
                .collect(),
            format!("prefill_range_l{layer}"),
            (0..passes)
                .map(|t| format!("decode_range_t{t}_l{layer}"))
                .collect(),
        ));
    }
    let mut rows_compared = 0;
    let mut keys_compared = 0;
    for (ranged, prefill, decode) in &families {
        // Position `prompt_len + t` comes from decode step `t`, which embeds
        // the token selected at step `t`.
        let prefill_boundary = load(direct_dir, prefill)?;
        let decode_boundaries = decode
            .iter()
            .map(|stage| load(direct_dir, stage))
            .collect::<Result<Vec<_>>>()?;
        let source = |position: usize| {
            if position < prompt_len {
                Some((&prefill_boundary, position))
            } else {
                decode_boundaries
                    .get(position - prompt_len)
                    .map(|boundary| (boundary, 0))
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
                if let Some((source, index)) = source(position) {
                    ensure!(
                        source.value["rows"][index] == *row,
                        "{stage}: position {position} differs from direct's row"
                    );
                    rows_compared += 1;
                }
                position += 1;
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
                let Some((source, _)) = source(key_position) else {
                    continue;
                };
                let expected = source.value["kv"]
                    .as_array()
                    .with_context(|| format!("{stage}: direct's source has no KV"))?
                    .iter()
                    .find(|entry| entry["position"] == key["position"])
                    .with_context(|| {
                        format!("{stage}: no direct K/V entry for position {key_position}")
                    })?;
                ensure!(
                    expected == key,
                    "{stage}: K/V at position {key_position} differs from direct's entry"
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

/// Direct generation's own coverage: active PLE, full logit vectors whose argmax
/// is the selection, and K/V growth and eviction over its decode passes.
fn validate_case(
    case: &Case,
    model: &ModelManifest,
    outputs: &BTreeMap<String, Boundary>,
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
    for step in 0..case.tokens {
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
        let edge = &outputs[&format!("decode_select_t{step}")].value;
        ensure!(
            edge["generated_token_ids"] == json!(&result.generated_token_ids[..=step as usize]),
            "step {step}: incorrect selected-token history"
        );
        ensure!(
            edge["token_id"] == json!(best.unwrap().0) && edge["value"] == json!(best.unwrap().1),
            "step {step}: selection does not match full-vector argmax"
        );
    }
    for token in 0..case.tokens.saturating_sub(1) {
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

/// Layers 2 and 3 of every range borrow layers 0 and 1 of the same range.
fn validate_donor_bindings(
    dir: &Path,
    outputs: &BTreeMap<String, Boundary>,
    shape: RangedShape,
) -> Result<()> {
    let ranges = (0..shape.prompt_ranges)
        .map(|j| format!("p{j}"))
        .chain((0..shape.claimed_ranges).map(|k| format!("c{k}")));
    for range in ranges {
        let stage = |layer| format!("range_{range}_l{layer}");
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
            && shape.hidden_size_per_layer_input == 32
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
            let matrix =
                weights.q8_matrix(&format!("model.language_model.layers.{layer}.{tensor}"))?;
            let bytes = matrix.rows() * det_num::q8::row_bytes(matrix.cols()).unwrap_or(0);
            ensure!(
                matrix.rows() * matrix.cols() == 128 * 2048
                    && bytes > 195_840
                    && bytes % 195_840 != 0,
                "model bundle does not cross a Q8_0 page with a partial tail"
            );
            let values = (0..matrix.rows())
                .map(|row| matrix.row_dequantized(row))
                .collect::<Result<Vec<_>>>()?
                .concat();
            ensure!(
                values.iter().any(|v| *v < 0)
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
                matrix["page_size"] == json!(195_840)
                    && matrix["byte_len"] == json!(278_528)
                    && pages.len() == 2
                    && pages[0]["bytes"].as_str().map(str::len) == Some(195_840 * 2)
                    && pages[1]["bytes"].as_str().map(str::len) == Some(82_688 * 2),
                "layer{layer}.{name}: imported matrix must contain a full page and a partial tail"
            );
        }
        let params = &value["params"];
        ensure!(
            params["ffn_size"] == json!(2048) && params["ple_width"] == json!(32),
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
