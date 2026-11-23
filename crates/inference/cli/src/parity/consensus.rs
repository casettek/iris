//! The gate's consensus leg: one inference on two models, and whether they
//! agree.
//!
//! The second model is the other family's bundle, so the chain holds both
//! families' stage programs under their labels. The run's prompt is built from
//! the source leg's signed documents. Each model's answer is learned first by
//! direct inference and named as a form, both with one code, so the claim's
//! result stage finds two different texts that stand for the same thing.
//!
//! The claim is run directly, natively and through Raster. Every stage must
//! give the same bytes both ways, the claim must state what direct inference
//! does, a contract's checks must pass, and a challenger must find a false
//! checkpoint in the second inference and replay it.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{ensure, Context, Result};
use inference_artifacts::contract::CLAIM_SUBMISSION_JSON;
use inference_artifacts::{ModelFamily, ModelManifest, Pipeline, PreparedRun};
use run_prep::source::{FormFile, LoadedSource};
use serde_json::{json, Value};
use staged_infer::chain_runner::StagedExecutionBackend;

use super::compare::{compare, read_boundaries};
use super::{command, logged, source};
use crate::challenge::locate_divergence;
use crate::claim::{
    build_posted_claim, corrupt_checkpoint_hashes, ClaimBuildOptions,
    CorruptCheckpointHashesOptions, CorruptCheckpointSelection,
};

/// The code both models' answers are given.
const AGREED_CODE: u32 = 7;
/// The stage of the second inference whose checkpoint the leg falsifies: long
/// enough to replay on either family's bundle.
const FALSIFIED_STAGE: &str = "m1_prompt_check_p0";

/// Imports the other family's bundle beside the run's own model.
fn import_other_family(
    root: &Path,
    dir: &Path,
    exe: &Path,
    model: &ModelManifest,
) -> Result<PathBuf> {
    let own = model.family()?;
    let other = ModelFamily::ALL
        .into_iter()
        .find(|family| *family != own)
        .context("this build has one model family, and a consensus leg needs two")?
        .parity_bundle();
    let artifact_root = dir.join("model-artifacts");
    let mut import = command(exe, root);
    import
        .args(["model", "import", "--model"])
        .arg(root.join("model-bundles").join(other))
        .arg("--artifact-root")
        .arg(&artifact_root)
        .args(["--model-id", other, "--no-stage-fixtures"])
        .args(["--range-size", &model.range_size.to_string()]);
    logged(import, &dir.join("import.log")).context("importing the second model")?;
    Ok(artifact_root.join(other).join("manifest.json"))
}

/// Writes the run: the source's files, and a spec naming both models.
fn write_run(dir: &Path, source: &LoadedSource, models: &[&Path; 2]) -> Result<PathBuf> {
    let table = source::write_source(dir, source)?;
    let spec = dir.join("inference.toml");
    fs::write(
        &spec,
        format!(
            "pipeline = \"consensus\"\ntokens = 4\n\n[[model]]\nmanifest = {}\n[[model]]\nmanifest = {}\n\n{table}",
            serde_json::to_string(models[0])?,
            serde_json::to_string(models[1])?,
        ),
    )?;
    Ok(spec)
}

pub(super) fn run_consensus_leg(
    root: &Path,
    run_dir: &Path,
    exe: &Path,
    raster: &Path,
    model_path: &Path,
    model: &ModelManifest,
) -> Result<Value> {
    let dir = run_dir.join("consensus");
    fs::create_dir_all(&dir)?;
    let other_path = import_other_family(root, &dir, exe, model)?;
    let models = [model_path, other_path.as_path()];

    // Each model's answer, from direct inference alone. The forms are no
    // part of the prompt, so naming the answers afterwards leaves them as
    // they are.
    let mut source = source::honest_source()?;
    let spec = write_run(&dir.join("answers"), &source, &models)?;
    let answers = crate::infer::run_infer(spec).context("the two models' answers")?;
    let texts: Vec<String> = answers
        .reports
        .iter()
        .map(|report| report.result.generated_text.trim().to_string())
        .collect();
    ensure!(
        texts.len() == 2 && texts[0] != texts[1],
        "the two models gave one answer, so the leg cannot show two texts agreeing: {texts:?}"
    );
    for text in &texts {
        source.recipe.forms.push(FormFile {
            text: text.clone(),
            code: AGREED_CODE,
        });
    }

    // The claim, three ways.
    let claim_dir = dir.join("claim");
    let spec = write_run(&claim_dir, &source, &models)?;
    let direct = crate::infer::run_infer(spec.clone()).context("direct inference")?;
    let direct_consensus = direct
        .consensus
        .as_ref()
        .context("direct inference stated no consensus")?;
    ensure!(
        direct_consensus.agree && direct_consensus.code == AGREED_CODE,
        "direct inference does not find the two answers to agree: {direct_consensus:?}"
    );
    let built = build_posted_claim(ClaimBuildOptions {
        base_dir: claim_dir.clone(),
        run_spec_path: spec.clone(),
        current_exe: exe.to_path_buf(),
        staged_backend: StagedExecutionBackend::InProcess,
        claimed_token_ids: None,
        direct_output_ids: None,
        claimer: None,
    })
    .context("the consensus claim")?;
    ensure!(
        built.pipeline == Pipeline::Consensus
            && built.final_result.as_ref() == Some(&direct.reports[0].result)
            && built.consensus == direct.consensus,
        "direct inference and the claim state different results: {:?} and {:?}",
        direct.consensus,
        built.consensus
    );
    let bundle = inference_artifacts::read_claim_bundle(&built.claim_bundle_path)?;
    let prepared: PreparedRun = inference_artifacts::read_json(
        bundle
            .prepared_run_path
            .as_ref()
            .context("the claim bundle names no prepared run")?,
    )?;
    let manifest = prepared
        .run_manifest_path
        .as_ref()
        .context("the prepared run names no chain manifest")?;

    let raster_dir = claim_dir.join("raster");
    fs::create_dir(&raster_dir)?;
    let mut run = command(raster, root);
    run.args(["raster", "chain", "run"])
        .arg(manifest)
        .args(["--no-auth", "--run"])
        .arg(&raster_dir);
    logged(run, &claim_dir.join("raster.log")).context("Raster-staged execution")?;
    let order = inference_artifacts::read_stage_order(&built.chain_dir)?
        .context("native chain recorded no stage inventory")?;
    let of = |label: &str| {
        order
            .iter()
            .filter(|stage| stage.starts_with(label))
            .count()
    };
    ensure!(
        order.first().map(String::as_str) == Some("source")
            && order.last().map(String::as_str) == Some("result")
            && of("m0_") > 0
            && of("m1_") > 0
            && of("m0_") + of("m1_") + 2 == order.len(),
        "a consensus chain is the source, two inferences and the result; ran {order:?}"
    );
    let raster_outputs = read_boundaries(&raster_dir, &order, false).context("Raster-staged")?;
    let native = read_boundaries(&built.chain_dir, &order, false).context("native-staged")?;
    for stage in &order {
        compare(stage, &raster_outputs[stage], &native[stage])
            .context("Raster-staged ↔ native-staged")?;
    }
    // The result stage states what direct inference does: each answer's
    // code, and that the two agree.
    let stated = &raster_outputs["result"].value;
    ensure!(
        stated["agree"] == json!(true)
            && stated["same_input"] == json!(true)
            && stated["code"] == json!(AGREED_CODE)
            && stated["first"]["code"] == json!(AGREED_CODE)
            && stated["second"]["code"] == json!(AGREED_CODE)
            && stated["first"]["statement"]["bytes"]
                == json!(direct.reports[0].result.statement_sha256)
            && stated["second"]["statement"]["bytes"]
                == json!(direct.reports[1].result.statement_sha256)
            && stated["first"]["statement"] != stated["second"]["statement"],
        "the result stage does not state that the two answers agree: {stated}"
    );

    // What a contract would check, and every native checkpoint against the
    // boundary the Raster run committed.
    let submission_path = built
        .claim_submission_path
        .clone()
        .context("the consensus claim wrote no submission")?;
    // The contract's pins are a file: the consensus pipeline's programs and
    // both models, as `iris claim pins` writes them.
    let pins_path = claim_dir.join(inference_artifacts::contract::CONTRACT_PINS_JSON);
    let pins = crate::checks::run_pins(&claim_dir, &spec).context("the contract's pins")?;
    ensure!(
        pins.pipeline == Pipeline::Consensus
            && pins.models.len() == 2
            && pins.models[0].model_id == prepared.inferences[0].model_id
            && pins.models[1].model_id == prepared.inferences[1].model_id
            && pins.models[0].model_id != pins.models[1].model_id,
        "the pins are not of this claim's two models"
    );
    inference_artifacts::write_json(&pins_path, &pins)?;
    let check = |submission: &Path, raster_run: Option<PathBuf>| {
        crate::checks::check_claim(crate::checks::ClaimCheckOptions {
            base_dir: claim_dir.clone(),
            run_spec_path: spec.clone(),
            submission_path: submission.to_path_buf(),
            raster_run,
            pins_path: Some(pins_path.clone()),
            registry_path: None,
            da_dir: None,
            submitted_at: None,
        })
        .context("contract checks")
    };
    let contract = check(&submission_path, Some(raster_dir.clone()))?;
    for line in &contract.lines {
        ensure!(
            line.rejection.is_none(),
            "a contract would reject an honest consensus claim at `{}`: {}",
            line.check,
            line.rejection.as_deref().unwrap_or_default()
        );
    }
    ensure!(
        contract.lines.len() == 10,
        "the contract checks of a consensus claim did not reach the Raster run"
    );
    // A claim that states another agreement than its outputs have is refused
    // when it is submitted, with no stage replayed.
    let mut overstated: Value = inference_artifacts::read_json(&submission_path)?;
    overstated["consensus"]["agree"] = json!(false);
    overstated["consensus"]["code"] = json!(0);
    let overstated_path = built
        .chain_dir
        .join(format!("denied.{CLAIM_SUBMISSION_JSON}"));
    inference_artifacts::write_json(&overstated_path, &overstated)?;
    let refused = check(&overstated_path, None)?;
    let refusal = refused
        .lines
        .iter()
        .find(|line| line.rejection.is_some())
        .context("a contract accepts a claim that misstates its agreement")?;
    ensure!(
        refusal.check.contains("agreement"),
        "a claim that misstates its agreement is refused at `{}`",
        refusal.check
    );

    // A challenger: nothing to dispute in the honest claim, a false
    // checkpoint in the second inference found and replayed, and a false
    // result found.
    let challenge = |name: &str, hashes: &Path| -> Result<String> {
        let challenge_dir = dir.join("challenges").join(name);
        fs::create_dir_all(&challenge_dir)?;
        let log = challenge_dir.join("challenge.log");
        let mut run = command(exe, root);
        run.current_dir(&challenge_dir)
            .args(["challenge", "build", "--run"])
            .arg(&spec)
            .arg("--claim-context")
            .arg(&built.claim_bundle_path)
            .arg("--checkpoints")
            .arg(hashes);
        logged(run, &log).with_context(|| format!("challenge `{name}`"))?;
        Ok(fs::read_to_string(log)?)
    };
    let has_line = |log: &str, wanted: &str| log.lines().any(|line| line.trim() == wanted);
    ensure!(
        has_line(
            &challenge("honest", &built.checkpoint_hashes_path)?,
            "no divergence found"
        ),
        "an honest consensus claim was disputed"
    );
    let falsified = dir.join("challenges").join("falsified.checkpoints.txt");
    fs::create_dir_all(dir.join("challenges"))?;
    corrupt_checkpoint_hashes(CorruptCheckpointHashesOptions {
        hashes_path: built.checkpoint_hashes_path.clone(),
        trace_path: Some(built.checkpoint_trace_path.clone()),
        output_path: Some(falsified.clone()),
        selection: CorruptCheckpointSelection::Stage(FALSIFIED_STAGE.to_string()),
    })?;
    ensure!(
        has_line(
            &challenge("falsified", &falsified)?,
            &format!("divergence found: {FALSIFIED_STAGE}")
        ),
        "a false checkpoint in the second inference was not found at `{FALSIFIED_STAGE}`"
    );
    let honest_trace = inference_artifacts::read_checkpoint_trace(&built.checkpoint_trace_path)?;
    let mut lying = honest_trace.clone();
    lying
        .checkpoints
        .last_mut()
        .context("the claim has no checkpoints")?
        .output_sha256 = "5a".repeat(32);
    let found = locate_divergence(
        &built.checkpoint_trace_path,
        &lying,
        &built.checkpoint_trace_path,
        &honest_trace,
    )
    .context("a false consensus result did not differ from honest execution")?;
    ensure!(
        found.stage == "result",
        "a false consensus result first differs at `{}`",
        found.stage
    );

    Ok(json!({
        "models": [prepared.inferences[0].model_id, prepared.inferences[1].model_id],
        "checkpoints": order.len(),
        "first_inference_stages": of("m0_"),
        "second_inference_stages": of("m1_"),
        "agree": stated["agree"],
        "code": stated["code"],
        "contract_checks": contract.lines.len(),
        "replayed": FALSIFIED_STAGE,
    }))
}
