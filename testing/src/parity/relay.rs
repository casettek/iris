//! The gate's relay scenario: two models, the second on a prompt that holds
//! the first one's output.
//!
//! The second model is the other family's bundle, so the chain holds both
//! families' stage programs under their labels, with the relay stage between
//! them. The first prompt is built from the source scenario's signed
//! documents, so the chain has both source programs.
//!
//! The run is made directly, natively and through Raster. Every stage must
//! give the same bytes both ways, the pipeline must state what direct
//! inference does, the relay's record must be of the first output and be the
//! one the second statement names, and a false checkpoint in either inference
//! or in the relay stage must be located.
//!
//! No claim is built: a claim of a relay is not written yet.

use std::fs;
use std::path::Path;

use anyhow::{ensure, Context, Result};
use model_manifest::ModelManifest;
use procedure_definition::divergence::locate_divergence;
use procedure_definition::{ProcedureKind, SpecificationRelay};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use staged_infer::chain_runner::StagedExecutionBackend;

use super::compare::{compare, read_boundaries};
use super::{case_beside, command, logged, source};

/// The text around the first output in the second prompt.
const BEFORE: &str = "An answer was given: ";
const AFTER: &str = "\nSay it again.";

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

    // Directly, then natively on what the direct run generated. The run
    // fails unless the two state the same results.
    let base = dir.join("run");
    let specification = source::write_run_of(
        &base,
        &source,
        ProcedureKind::named("relay")?,
        &models,
        Some(relay),
    )?;
    let run = procedure_run::run(&procedure_run::RunOptions {
        base_dir: base.clone(),
        specification_path: specification.clone(),
        case_path: case_beside(&specification),
        current_exe: exe.to_path_buf(),
        staged_backend: StagedExecutionBackend::InProcess,
    })
    .context("the relay run, directly and natively")?;
    let [first, second] = run.direct.reports.as_slice() else {
        anyhow::bail!(
            "a relay makes two inferences, and the direct run made {}",
            run.direct.reports.len()
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
        first.source_sha256 == run_prep::source_id(Some(&source.run().record))
            && second.source_sha256 != first.source_sha256,
        "the two inferences do not name their own sources"
    );
    ensure!(
        first.result_forms_sha256.is_empty() && !second.result_forms_sha256.is_empty(),
        "the result stage did not read the second inference alone"
    );

    // Through Raster, and every boundary against the native run's.
    let raster_dir = dir.join("raster");
    fs::create_dir(&raster_dir)?;
    let mut raster_run = command(raster, root);
    raster_run
        .args(["raster", "chain", "run"])
        .arg(&run.staged.manifest_path)
        .args(["--no-auth", "--run"])
        .arg(&raster_dir);
    logged(raster_run, &dir.join("raster.log")).context("Raster-staged execution")?;
    let order = procedure_definition::read_stage_order(&run.staged.chain_dir)?
        .context("native chain recorded no stage inventory")?;
    let of = |label: &str| {
        order
            .iter()
            .filter(|stage| stage.starts_with(label))
            .count()
    };
    let at = |name: &str| order.iter().position(|stage| stage == name);
    let relay_at = at("relay").context("the chain ran no relay stage")?;
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
        "a relay chain is the source, the first inference, the relay, the second inference \
         and the result; ran {order:?}"
    );
    let raster_outputs = read_boundaries(&raster_dir, &order, false).context("Raster-staged")?;
    let native = read_boundaries(&run.staged.chain_dir, &order, false).context("native-staged")?;
    for stage in &order {
        compare(stage, &raster_outputs[stage], &native[stage])
            .context("Raster-staged ↔ native-staged")?;
    }

    // The relay's record: of the prompt the second inference ran, built on
    // the first inference's claim record, and the record the second
    // statement names.
    let stated = &raster_outputs["relay"].value;
    let record: source_core::SourceRecord =
        serde_json::from_value(stated.clone()).context("the relay stage's output")?;
    ensure!(
        record.present
            && record.fault == 0
            && record.documents == 1
            && record.text_sha256 == second.input_text_sha256
            && record.text_total == relayed.len() as u64,
        "the relay's record is not of the prompt the second inference ran: {stated}"
    );
    let expected = run_prep::relayed_run(
        &procedure_definition::read_specification(&specification)?,
        first,
    )?
    .record;
    ensure!(
        record == expected,
        "the relay stage states another record than direct inference: {stated}"
    );
    ensure!(
        second.source_sha256 == run_prep::source_id(Some(&record)),
        "the second statement does not name the relay's record"
    );
    let result = &raster_outputs["result"].value;
    ensure!(
        result["statement"]["bytes"] == json!(second.statement_sha256)
            && result["valid"] == json!(true),
        "the result stage is not of the second inference: {result}"
    );

    // A false checkpoint in either inference, in the relay stage or in the
    // result is located at its stage.
    let list_path = &run.staged.checkpoint_list_path;
    let honest = procedure_definition::read_checkpoint_list(list_path)?;
    let falsified = [
        "m0_prompt_check_p0",
        "relay",
        "m1_prompt_check_p0",
        "result",
    ];
    for stage in falsified {
        let mut lying = honest.clone();
        lying
            .checkpoints
            .iter_mut()
            .find(|checkpoint| checkpoint.stage == stage)
            .with_context(|| format!("the relay chain ran no `{stage}`"))?
            .output_sha256 = "5a".repeat(32);
        let found = locate_divergence(list_path, &lying, list_path, &honest)
            .with_context(|| format!("a false `{stage}` did not differ from honest execution"))?;
        ensure!(
            found.stage == stage,
            "a false `{stage}` first differs at `{}`",
            found.stage
        );
    }

    Ok(json!({
        "checkpoints": order.len(),
        "first_inference_stages": of("m0_"),
        "second_inference_stages": of("m1_"),
        "relayed": first.generated_text,
        "answered": second.generated_text,
        "located": falsified,
    }))
}
