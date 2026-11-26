//! The gate's relay scenario: two models, the second on a prompt that holds
//! the first one's output.
//!
//! The second model is the other family's bundle, so the pipeline holds both
//! families' stage programs under their labels, with the relay stage between
//! them. The first prompt is built from the source scenario's signed
//! documents, so the pipeline has both source programs.
//!
//! The claim is run directly, natively and through Raster. Every stage must
//! give the same bytes both ways, the claim must state what direct inference
//! does, the relay's record must be of the first output and be the one the
//! second statement names, a contract's checks must pass, and a challenger
//! must find a false checkpoint in either inference and replay it.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{ensure, Context, Result};
use blockchain_rules::checks::CLAIM_SUBMISSION_JSON;
use inference_manifest::ModelManifest;
use procedure_definition::{PreparedRun, ProcedureKind, SpecificationRelay};
use procedure_pipeline::pipeline_runner::StagedExecutionBackend;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use super::compare::{compare, read_boundaries};
use super::{case_beside, command, logged, source};
use blockchain_host::challenge::locate_divergence;
use blockchain_host::claim::{
    build_posted_claim, corrupt_checkpoint_hashes, ClaimBuildOptions,
    CorruptCheckpointHashesOptions, CorruptCheckpointSelection,
};

/// The text around the first output in the second prompt.
const BEFORE: &str = "An answer was given: ";
const AFTER: &str = "\nSay it again.";
/// The stages whose checkpoints the scenario falsifies and a challenger
/// replays, one of each inference: long enough to replay on either family's
/// bundle.
const FALSIFIED_STAGES: [&str; 2] = ["m0_prompt_check_p0", "m1_prompt_check_p0"];

pub(super) fn run_relay_scenario(
    root: &Path,
    run_dir: &Path,
    exe: &Path,
    raster: &Path,
    model_path: &Path,
    model: &ModelManifest,
) -> Result<Value> {
    let dir = run_dir.join("relay");
    fs::create_dir_all(&dir)?;
    let other_path = super::consensus::import_other_family(root, &dir, exe, model)?;
    let models = [model_path, other_path.as_path()];
    let source = source::honest_source()?;
    let relay = SpecificationRelay {
        before: BEFORE.to_string(),
        after: AFTER.to_string(),
    };

    // The claim, three ways.
    let claim_dir = dir.join("run");
    let specification = source::write_run_of(
        &claim_dir,
        &source,
        ProcedureKind::named("relay")?,
        &models,
        Some(relay),
    )?;
    let direct =
        procedure_native::infer::run_infer(specification.clone(), case_beside(&specification))
            .context("direct inference")?;
    let [first, second] = direct.reports.as_slice() else {
        anyhow::bail!(
            "a relay makes two inferences, and the direct run made {}",
            direct.reports.len()
        );
    };
    let (first, second) = (&first.result, &second.result);
    ensure!(
        first.valid && second.valid,
        "an inference of the relay stated no valid claim record"
    );
    // The second prompt is the recipe around the first output, exactly.
    let relayed = format!("{BEFORE}{}{AFTER}", first.generated_text);
    ensure!(
        second.input_text_sha256 == hex::encode(Sha256::digest(relayed.as_bytes())),
        "the second inference did not run the prompt built around the first output"
    );
    // The first inference is held to the run's source, the second to the
    // relay's record, and the result is of the second output alone.
    ensure!(
        first.source_sha256 == procedure_prepare::source_id(Some(&source.run().record))
            && second.source_sha256 != first.source_sha256,
        "the two inferences do not name their own sources"
    );
    ensure!(
        first.result_forms_sha256.is_empty() && !second.result_forms_sha256.is_empty(),
        "the result stage did not read the second inference alone"
    );

    let built = build_posted_claim(ClaimBuildOptions {
        base_dir: claim_dir.clone(),
        case_path: Some(case_beside(&specification)),
        specification_path: specification.clone(),
        current_exe: exe.to_path_buf(),
        staged_backend: StagedExecutionBackend::InProcess,
        claimed_token_ids: None,
        direct_output_ids: None,
        claimer: None,
    })
    .context("the relay claim")?;
    ensure!(
        built.procedure_kind.name() == "relay"
            && built
                .results
                .iter()
                .eq(direct.reports.iter().map(|report| &report.result))
            && built.consensus.is_none(),
        "direct inference and the claim state different results"
    );
    // And with no blockchain: the same checkpoints as the claim's run.
    super::run_with_no_blockchain(
        root,
        exe,
        &claim_dir.join("procedure-run"),
        &specification,
        &built.pipeline_dir,
    )?;
    let bundle = blockchain_rules::read_claim_bundle(&built.claim_bundle_path)?;
    let prepared: PreparedRun = procedure_definition::read_json(
        bundle
            .prepared_run_path
            .as_ref()
            .context("the claim bundle names no prepared run")?,
    )?;
    let manifest = prepared
        .run_manifest_path
        .as_ref()
        .context("the prepared run names no pipeline manifest")?;

    // Through Raster, and every boundary against the native run's.
    let raster_dir = dir.join("raster");
    fs::create_dir(&raster_dir)?;
    let mut raster_run = command(raster, root);
    raster_run
        .args(["raster", "pipeline", "run"])
        .arg(manifest)
        .args(["--no-auth", "--run"])
        .arg(&raster_dir);
    logged(raster_run, &dir.join("raster.log")).context("Raster-staged execution")?;
    let order = procedure_definition::read_stage_order(&built.pipeline_dir)?
        .context("native pipeline recorded no stage inventory")?;
    let of = |label: &str| {
        order
            .iter()
            .filter(|stage| stage.starts_with(label))
            .count()
    };
    let at = |name: &str| order.iter().position(|stage| stage == name);
    let relay_at = at("relay").context("the pipeline ran no relay stage")?;
    ensure!(
        at("source") == Some(0)
            && at("result") == Some(order.len() - 1)
            && of("m0_") > 0
            && of("m1_") > 0
            && of("m0_") + of("m1_") + 3 == order.len()
            && order[1..relay_at]
                .iter()
                .all(|stage| stage.starts_with("m0_"))
            && order[relay_at + 1..order.len() - 1]
                .iter()
                .all(|stage| stage.starts_with("m1_")),
        "a relay pipeline is the source, the first inference, the relay, the second inference \
         and the result; ran {order:?}"
    );
    let raster_outputs = read_boundaries(&raster_dir, &order, false).context("Raster-staged")?;
    let native = read_boundaries(&built.pipeline_dir, &order, false).context("native-staged")?;
    for stage in &order {
        compare(stage, &raster_outputs[stage], &native[stage])
            .context("Raster-staged ↔ native-staged")?;
    }

    // The relay's record: of the prompt the second inference ran, built on
    // the first inference's claim record, and the record the second
    // statement names.
    let stated = &raster_outputs["relay"].value;
    let record: source_phase::SourceRecord =
        serde_json::from_value(stated.clone()).context("the relay stage's output")?;
    ensure!(
        record.present
            && record.fault == 0
            && record.documents == 1
            && record.text_sha256 == second.input_text_sha256
            && record.text_total == relayed.len() as u64,
        "the relay's record is not of the prompt the second inference ran: {stated}"
    );
    let expected = procedure_prepare::relayed_run(
        &procedure_definition::read_specification(&specification)?,
        first,
    )?
    .record;
    ensure!(
        record == expected,
        "the relay stage states another record than direct inference: {stated}"
    );
    ensure!(
        second.source_sha256 == procedure_prepare::source_id(Some(&record)),
        "the second statement does not name the relay's record"
    );
    let result = &raster_outputs["result"].value;
    ensure!(
        result["statement"]["bytes"] == json!(second.statement_sha256)
            && result["valid"] == json!(true),
        "the result stage is not of the second inference: {result}"
    );

    // What a contract would check, and every native checkpoint against the
    // boundary the Raster run committed. The contract's pins are a file: the
    // relay procedure kind's programs and both models, as `iris claim pins`
    // writes them.
    let submission_path = built
        .claim_submission_path
        .clone()
        .context("the relay claim wrote no submission")?;
    let pins_path = claim_dir.join(blockchain_rules::checks::CONTRACT_PINS_JSON);
    let pins = blockchain_host::claim_check::run_pins(&claim_dir, &specification)
        .context("the contract's pins")?;
    ensure!(
        pins.procedure_kind.name() == "relay"
            && pins.models.len() == 2
            && pins.models[0].model_id == prepared.inferences[0].model_id
            && pins.models[1].model_id == prepared.inferences[1].model_id,
        "the pins are not of this claim's two models"
    );
    procedure_definition::write_json(&pins_path, &pins)?;
    let check = |submission: &Path, raster_run: Option<PathBuf>| {
        blockchain_host::claim_check::check_claim(blockchain_host::claim_check::ClaimCheckOptions {
            base_dir: claim_dir.clone(),
            specification_path: specification.clone(),
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
            "a contract would reject an honest relay claim at `{}`: {}",
            line.check,
            line.rejection.as_deref().unwrap_or_default()
        );
    }
    ensure!(
        contract.lines.len() == 10,
        "the contract checks of a relay claim did not reach the Raster run"
    );
    // A claim whose first output is not the one its relay was built on is
    // refused when it is submitted, with no stage replayed: the first record
    // commits to the swapped text, and the relay's record names another.
    let mut swapped: Value = procedure_definition::read_json(&submission_path)?;
    let other = "another answer";
    swapped["inferences"][0]["output_text"] = json!(other);
    swapped["inferences"][0]["record"]["output_text_sha256"] =
        json!(hex::encode(Sha256::digest(other.as_bytes())));
    let swapped_path = built
        .pipeline_dir
        .join(format!("swapped.{CLAIM_SUBMISSION_JSON}"));
    procedure_definition::write_json(&swapped_path, &swapped)?;
    let refused = check(&swapped_path, None)?;
    let refusal = refused
        .lines
        .iter()
        .find(|line| line.rejection.is_some())
        .context("a contract accepts a relay claim with another first output")?;
    ensure!(
        refusal.check.contains("source record")
            && refusal
                .rejection
                .as_deref()
                .is_some_and(|reason| reason.contains("relay record")),
        "a relay claim with another first output is refused at `{}`: {:?}",
        refusal.check,
        refusal.rejection
    );

    // A challenger, rebuilding the relay from what the claim fixed: nothing
    // to dispute in the honest claim, and a false checkpoint in each
    // inference found and replayed.
    let challenge = |name: &str, hashes: &Path| -> Result<String> {
        let challenge_dir = dir.join("challenges").join(name);
        fs::create_dir_all(&challenge_dir)?;
        let log = challenge_dir.join("challenge.log");
        let mut run = command(exe, root);
        run.current_dir(&challenge_dir)
            .args(["challenge", "build", "--specification"])
            .arg(&specification)
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
        "an honest relay claim was disputed"
    );
    fs::create_dir_all(dir.join("challenges"))?;
    for stage in FALSIFIED_STAGES {
        let falsified = dir
            .join("challenges")
            .join(format!("{stage}.checkpoints.txt"));
        corrupt_checkpoint_hashes(CorruptCheckpointHashesOptions {
            hashes_path: built.checkpoint_hashes_path.clone(),
            list_path: Some(built.checkpoint_list_path.clone()),
            output_path: Some(falsified.clone()),
            selection: CorruptCheckpointSelection::Stage(stage.to_string()),
        })?;
        ensure!(
            has_line(
                &challenge(stage, &falsified)?,
                &format!("divergence found: {stage}")
            ),
            "a false checkpoint was not found at `{stage}`"
        );
    }
    // The relay stage and the result are shorter than the replay window on
    // these models: a false one is where recomputation first differs.
    let honest_trace = procedure_definition::read_checkpoint_list(&built.checkpoint_list_path)?;
    let located = ["relay", "result"];
    for stage in located {
        let mut lying = honest_trace.clone();
        lying
            .checkpoints
            .iter_mut()
            .find(|checkpoint| checkpoint.stage == stage)
            .with_context(|| format!("the relay pipeline ran no `{stage}`"))?
            .output_sha256 = "5a".repeat(32);
        let found = locate_divergence(
            &built.checkpoint_list_path,
            &lying,
            &built.checkpoint_list_path,
            &honest_trace,
        )
        .with_context(|| format!("a false `{stage}` did not differ from honest execution"))?;
        ensure!(
            found.stage == stage,
            "a false `{stage}` first differs at `{}`",
            found.stage
        );
    }

    Ok(json!({
        "models": [prepared.inferences[0].model_id, prepared.inferences[1].model_id],
        "checkpoints": order.len(),
        "first_inference_stages": of("m0_"),
        "second_inference_stages": of("m1_"),
        "relayed": first.generated_text,
        "answered": second.generated_text,
        "contract_checks": contract.lines.len(),
        "replayed": FALSIFIED_STAGES,
        "located": located,
    }))
}
