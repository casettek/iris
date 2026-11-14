//! The challenge leg: a claim that differs from honest execution is found by a
//! challenger's own recomputation, and the stage it differs at replays through
//! authenticated Raster. Runs the `challenge build` command a challenger runs.
//!
//! A stage too short for the replay window cannot be challenged yet (see
//! `docs/issues/short-stage-fraud-window.md`), so the leg only checks that
//! recomputation finds it.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{ensure, Context, Result};
use inference_artifacts::contract::{dispute_digest, settlement, stage_challenge, Settlement};
use inference_artifacts::{
    build_checkpoint_trace, checkpoint_hashes, file_sha256, read_checkpoint_trace, read_json,
    write_checkpoint_hashes, CheckpointTrace, ModelManifest,
};
use raster_prover::commitment_file::CommitmentFile;
use run_prep::RunConfig;
use serde_json::{json, Value};
use staged_infer::chain_runner::StagedExecutionBackend;
use staged_infer::{CheckpointedInferenceConfig, CheckpointedInferenceExecutor, ParityPolicy};

use super::{block_name, command, logged, manifest_blocks, Case};
use crate::challenge::locate_divergence;
use crate::claim::{
    build_claim, corrupt_checkpoint_hashes, render_ranged_manifest, ClaimBuildOptions,
    ClaimBuildResult, CorruptCheckpointHashesOptions, CorruptCheckpointSelection, PreparedClaimRun,
};

/// The window a challenge replays at.
const REPLAY_WINDOW: usize = 128;

/// Checkpoints to falsify in an otherwise honest claim, one per family of stage
/// long enough to challenge.
const FALSIFIED_STAGES: [&str; 4] = [
    "prompt_check_p0",
    "prompt_check_p1",
    "range_p1_l2",
    "range_c1_l3",
];

/// Whether this stage's trace exceeds the replay window on the pinned templated
/// case.
fn long_enough_to_challenge(stage: &str) -> bool {
    // A bind stage covers one range, and a range of the pinned model is six
    // tokens: about 110 steps, where a 512-token range of a real model is
    // about 28,000.
    stage.starts_with("prompt_check_p")
        || stage.starts_with("range_p")
        || stage.starts_with("range_c")
}

pub(super) struct Leg<'a> {
    pub root: &'a Path,
    pub dir: PathBuf,
    pub exe: &'a Path,
    pub raster: &'a Path,
    pub model_path: &'a Path,
    pub model: &'a ModelManifest,
    pub case: &'a Case,
    pub prepared: &'a PreparedClaimRun,
    pub claimed: &'a [u32],
}

/// What one run binds besides the model.
#[derive(Clone)]
struct RunInputs {
    text: String,
    config: RunConfig,
    marks: Vec<Vec<u32>>,
    ids: Vec<u32>,
    /// Added to the schedule's two counts. Zero for a schedule matching the run.
    schedule_skew: [i64; 2],
    /// Claim another hash for the text than the text's.
    other_text_hash: bool,
}

pub(super) fn run_challenge_leg(leg: &Leg) -> Result<Value> {
    fs::create_dir_all(&leg.dir)?;
    let spec = run_spec(leg, leg.model_path, "inference.toml")?;
    let inputs = honest_inputs(leg)?;
    let (other_path, other) = import_other_model(leg)?;
    let honest_manifest = render(
        leg.model_path,
        leg.model,
        &inputs,
        &leg.dir.join("honest-run"),
    )?;
    let other_manifest = render(&other_path, &other, &inputs, &leg.dir.join("other-run"))?;

    let honest = claim(leg, "honest", &spec, Some(leg.claimed.to_vec()))?;
    let context = honest.claim_bundle_path.as_path();
    let honest_trace = read_checkpoint_trace(&honest.checkpoint_trace_path)?;
    let mut replays = Vec::new();

    // An honest claim has nothing to dispute and nothing to reject.
    let log = challenge(
        leg,
        "honest",
        &spec,
        context,
        &honest.checkpoint_hashes_path,
    )?;
    ensure!(
        has_line(&log, "no divergence found"),
        "an honest claim was disputed"
    );

    for stage in FALSIFIED_STAGES {
        let name = format!("falsified-{stage}");
        let dir = leg.dir.join(&name);
        fs::create_dir_all(&dir)?;
        let hashes = dir.join("checkpoints.txt");
        corrupt_checkpoint_hashes(CorruptCheckpointHashesOptions {
            hashes_path: honest.checkpoint_hashes_path.clone(),
            trace_path: Some(honest.checkpoint_trace_path.clone()),
            output_path: Some(hashes.clone()),
            selection: CorruptCheckpointSelection::Stage(stage.to_string()),
        })?;
        let outcome = expect_divergence(leg, &name, &spec, context, &hashes, stage)?;
        // The same replay, against a claimer that ran this stage on the honest
        // inputs and stated another output for it. A falsified hash has no
        // checkpoint behind it, so this is the claim a contract would see: a
        // replay settles it, and Raster must admit the challenge.
        let mut lying = honest_trace.clone();
        let stated = lying
            .checkpoints
            .iter_mut()
            .find(|checkpoint| checkpoint.stage == stage)
            .with_context(|| format!("the honest claim has no `{stage}`"))?;
        stated.output_sha256 = "5a".repeat(32);
        let outcome = dispute_tie(leg, &name, &lying, &honest_trace, stage, outcome)?;
        ensure!(
            outcome["settled_by"] == "replay" && outcome["raster_admits"] == true,
            "`{name}`: a claim that states another output was not settled by a replay Raster admits"
        );
        replays.push(outcome);
    }

    // Correct execution over ids that are not the greedy output: every
    // checkpoint agrees, and the claim's own record refutes it.
    let not_greedy = not_greedy_ids(leg)?;
    let refuted = claim(leg, "not-greedy", &spec, Some(not_greedy.clone()))?;
    let log = challenge(
        leg,
        "not-greedy",
        &spec,
        &refuted.claim_bundle_path,
        &refuted.checkpoint_hashes_path,
    )?;
    ensure!(
        has_line(&log, "no divergence found; claimed output rejected"),
        "a non-greedy claim was not rejected on its final record"
    );

    // An honest claimer that ran a different model. Nothing of that model is
    // available to the challenger, and the run differs at the first stage that
    // names it.
    let other_spec = run_spec(leg, &other_path, "other-model.toml")?;
    let elsewhere = claim(leg, "other-model", &other_spec, None)?;
    replays.push(expect_claim_found(
        leg,
        "other-model",
        &spec,
        &elsewhere,
        &honest_trace,
        "prompt_bind_p0",
    )?);
    let log = fs::read_to_string(leg.dir.join("other-model/challenge/challenge.log"))?;
    ensure!(
        log.contains("warning: the claim names model"),
        "the challenger did not report that the claim names another model"
    );

    // A claim of the honest run whose checkpoints came from something else:
    // its inputs, one model input at a time, or its stages' connections.
    // The first range claims its last piece the other way: a token starting
    // there, or not.
    let other_marks = {
        let mut marks = inputs.marks.clone();
        let last = marks[0]
            .last_mut()
            .context("the first prompt range has no pieces")?;
        *last ^= 1;
        marks
    };
    let mut other_budget = inputs.config.clone();
    other_budget.max_new_tokens += 1;
    let doctored = [
        (
            "other-text",
            "prompt_bind_p0",
            RunInputs {
                text: format!("{} ", inputs.text),
                ..inputs.clone()
            },
        ),
        (
            "other-config",
            "prompt_bind_p0",
            RunInputs {
                config: other_budget,
                ..inputs.clone()
            },
        ),
        (
            "other-schedule",
            "prompt_bind_p0",
            RunInputs {
                schedule_skew: [1, 0],
                ..inputs.clone()
            },
        ),
        (
            "other-text-claim",
            "prompt_bind_p0",
            RunInputs {
                other_text_hash: true,
                ..inputs.clone()
            },
        ),
        (
            "other-marks",
            "prompt_check_p0",
            RunInputs {
                marks: other_marks,
                ..inputs.clone()
            },
        ),
        (
            "other-ids",
            "embed_c0",
            RunInputs {
                ids: not_greedy,
                ..inputs.clone()
            },
        ),
    ];
    for (name, stage, run) in doctored {
        let manifest = render(
            leg.model_path,
            leg.model,
            &run,
            &leg.dir.join(name).join("run"),
        )?;
        replays.push(expect_found(
            leg,
            name,
            &spec,
            context,
            &honest_trace,
            &manifest,
            stage,
        )?);
    }

    // What range 0 is given where a later range takes the range before it.
    let empty = |stage, key| binding_value(&honest_manifest, stage, key);
    let take = |stage, key| take_binding(&honest_manifest, &other_manifest, stage, key);
    let table = |header| take_table(&honest_manifest, &other_manifest, header);
    let wire = |stage, key, line: &str| replace_binding(&honest_manifest, stage, key, line);
    // Sparse verification authenticates reads, not unused tokenizer content.
    // Change every bucket so each of these consuming stages sees a real mutation.
    let changed_tokenizer = tokenizer_substitution_binding(
        &honest_manifest,
        &leg.dir.join("tokenizer-substitution-inputs"),
    )?;
    // A family whose layers take no per-layer input has no auxiliary stage to
    // give another model's table to, and none to bind to the wrong range.
    let auxiliary = leg.model.family()?.has_auxiliary_stages();
    let mut substituted = vec![
        (
            "tokenizer-at-bind",
            "prompt_bind_p0",
            wire("prompt_bind_p{j}", "tokenizer", &changed_tokenizer)?,
        ),
        (
            "tokenizer-at-check",
            "prompt_check_p0",
            wire("prompt_check_p{j}", "tokenizer", &changed_tokenizer)?,
        ),
        (
            "embedding-in-prompt",
            "embed_p0",
            take("embed_p{j}", "embedding")?,
        ),
        (
            "embedding-in-claim",
            "embed_c0",
            take("embed_c{k}", "embedding")?,
        ),
        ("other-head", "check_c0", take("check_c{k}", "head")?),
        // The first check is the first stage to read the decoder table: every
        // check states the output record of the transcript so far.
        ("other-decoder", "check_c0", take("check_c{k}", "decoder")?),
    ];
    if auxiliary {
        substituted.push(("other-ple", "aux_p0_l0", table("[chain.input.aux_layer]")?));
    }
    substituted.push((
        "other-weights",
        "range_p0_l0",
        table("[chain.input.transformer_layer]")?,
    ));
    // Transformer layers have no such case: their caches differ by attention
    // type and donor, so a stage given another layer's weights stops, and
    // the claimer has no output to post.
    if auxiliary {
        substituted.push((
            "swapped-ple",
            "aux_p0_l0",
            swapped_ple(leg, &honest_manifest)?,
        ));
    }
    substituted.extend([
        (
            "repeated-context",
            "prompt_check_p1",
            wire(
                "prompt_check_p{j}",
                "previous",
                &empty("prompt_check_p0", "previous")?,
            )?,
        ),
        (
            "dropped-verdict",
            "check_c1",
            wire("check_c{k}", "prior", &empty("check_c0", "prior")?)?,
        ),
        // No wiring presents an earlier range's record as the claim's: the
        // final record is the last check's own output, and the three-path
        // leg checks that an earlier check's record carries a fault.
        (
            "first-prompt-context",
            "check_c0",
            wire("check_c{k}", "prompt", r#"{ from = "prompt_check_p0" }"#)?,
        ),
    ]);
    for (name, stage, manifest) in &substituted {
        replays.push(expect_found(
            leg,
            name,
            &spec,
            context,
            &honest_trace,
            manifest,
            stage,
        )?);
    }

    for (name, _, manifest) in substituted
        .iter()
        .filter(|(name, _, _)| *name != "other-decoder")
    {
        replays.push(expect_semantic_rejection(
            leg,
            &format!("{name}-self-replay"),
            manifest,
        )?);
    }

    // Self-replay the same dishonest model inputs. Agreement is expected;
    // rejection must come from the committed semantic fault.
    let (changed_decoder, changed_proof) = decoder_substitution_manifests(
        &honest_manifest,
        &leg.dir.join("decoder-substitution-inputs"),
    )?;
    replays.push(expect_semantic_rejection(
        leg,
        "decoder-self-replay",
        &changed_decoder,
    )?);
    replays.push(expect_semantic_rejection(
        leg,
        "decoder-and-proof-self-replay",
        &changed_proof,
    )?);

    // Execute the dishonest wiring itself in both engines. Agreement must end
    // in a committed invalid verdict, even when the altered pipeline runs honestly.
    let mut rewired = vec![
        (
            "restarted-positions",
            wire("embed_p{j}", "previous", &empty("embed_p0", "previous")?)?,
        ),
        (
            "dropped-cache",
            wire(
                "range_p{j}_l{l}",
                "prior_kv",
                &empty("range_p0_l{l}", "prior_kv")?,
            )?,
        ),
    ];
    if auxiliary {
        rewired.push((
            "wrong-auxiliary-range",
            wire("range_c{k}_l{l}", "ple", r#"{ from = "aux_p0_l{l}" }"#)?,
        ));
    }
    rewired.extend([
        (
            "stale-cache-range",
            wire(
                "range_c{k}_l{l}",
                "prior_kv",
                r#"{ from = "range_p0_l{l}" }"#,
            )?,
        ),
        (
            "stale-predicting-row",
            wire("check_c{k}", "previous", r#"{ from = "range_p0_l3" }"#)?,
        ),
        (
            "claim-embedding-reset",
            wire("embed_c{k}", "previous", r#"{ from = "embed_p0" }"#)?,
        ),
    ]);
    for (name, manifest) in &rewired {
        replays.push(expect_semantic_rejection(
            leg,
            &format!("{name}-self-replay"),
            manifest,
        )?);
    }

    Ok(json!(replays))
}

fn has_line(log: &str, wanted: &str) -> bool {
    log.lines().any(|line| line.trim() == wanted)
}

fn run_spec(leg: &Leg, model_path: &Path, file: &str) -> Result<PathBuf> {
    let path = leg.dir.join(file);
    fs::write(
        &path,
        format!(
            "model_manifest = {}\nprompt = {}\nraw_prompt = {}\ntokens = {}\n",
            serde_json::to_string(model_path)?,
            serde_json::to_string(&leg.case.prompt)?,
            leg.case.raw_prompt,
            leg.case.tokens
        ),
    )?;
    Ok(path)
}

fn claim(leg: &Leg, name: &str, spec: &Path, ids: Option<Vec<u32>>) -> Result<ClaimBuildResult> {
    let base_dir = leg.dir.join(name).join("claim");
    fs::create_dir_all(&base_dir)?;
    build_claim(ClaimBuildOptions {
        base_dir,
        run_spec_path: spec.to_path_buf(),
        current_exe: leg.exe.to_path_buf(),
        staged_backend: StagedExecutionBackend::InProcess,
        claimed_token_ids: ids,
        direct_output_ids: None,
    })
    .with_context(|| format!("claim `{name}`"))
}

/// Runs `challenge build` the way a challenger does and returns its output.
fn challenge(leg: &Leg, name: &str, spec: &Path, context: &Path, hashes: &Path) -> Result<String> {
    let (built, log) = attempt_challenge(leg, name, spec, context, hashes)?;
    built.with_context(|| format!("challenge `{name}`"))?;
    Ok(log)
}

/// Runs `challenge build` and returns whether it built a challenge and what
/// it printed either way.
fn attempt_challenge(
    leg: &Leg,
    name: &str,
    spec: &Path,
    context: &Path,
    hashes: &Path,
) -> Result<(Result<f64>, String)> {
    let dir = leg.dir.join(name).join("challenge");
    fs::create_dir_all(&dir)?;
    let log = dir.join("challenge.log");
    let mut run = command(leg.exe, leg.root);
    run.current_dir(&dir)
        .args(["challenge", "build", "--run"])
        .arg(spec)
        .arg("--claim-context")
        .arg(context)
        .arg("--checkpoints")
        .arg(hashes);
    let built = logged(run, &log);
    Ok((built, fs::read_to_string(log)?))
}

/// A claim built somewhere else. Recomputation must first differ from its
/// checkpoints at this stage, and a stage long enough to challenge must also
/// replay. For a shorter one the challenger's command is still run, for what
/// it reports before the replay it cannot finish.
fn expect_claim_found(
    leg: &Leg,
    name: &str,
    spec: &Path,
    claim: &ClaimBuildResult,
    honest: &CheckpointTrace,
    stage: &str,
) -> Result<Value> {
    let claimed = read_checkpoint_trace(&claim.checkpoint_trace_path)?;
    let found = locate_divergence(
        &claim.checkpoint_trace_path,
        &claimed,
        &claim.checkpoint_trace_path,
        honest,
    )
    .with_context(|| format!("`{name}` did not differ from honest execution"))?;
    ensure!(
        found.stage == stage,
        "`{name}` first differs at `{}`, not `{stage}`",
        found.stage
    );
    let (context, hashes) = (&claim.claim_bundle_path, &claim.checkpoint_hashes_path);
    if long_enough_to_challenge(stage) {
        let outcome = expect_divergence(leg, name, spec, context, hashes, stage)?;
        return dispute_tie(leg, name, &claimed, honest, stage, outcome);
    }
    // The replay of a stage this short is refused, so whether the command
    // built a challenge is not this case's result.
    let (_built, _log) = attempt_challenge(leg, name, spec, context, hashes)?;
    Ok(json!({"case": name, "stage": stage, "replayed": false}))
}

/// The challenge must name this stage. `challenge build` only succeeds once the
/// authenticated replay matches the challenger's native checkpoint and differs
/// from the claim's.
fn expect_divergence(
    leg: &Leg,
    name: &str,
    spec: &Path,
    context: &Path,
    hashes: &Path,
    stage: &str,
) -> Result<Value> {
    let started = Instant::now();
    let log = challenge(leg, name, spec, context, hashes)?;
    ensure!(
        has_line(&log, &format!("divergence found: {stage}")),
        "challenge `{name}` did not find `{stage}`: {}",
        log.lines()
            .find(|line| line.starts_with("divergence found") || line.starts_with("no divergence"))
            .unwrap_or("it reported no outcome")
    );
    let replay = leg
        .dir
        .join(name)
        .join("challenge/target/staged-infer/runs");
    let run = fs::read_dir(&replay)?
        .next()
        .with_context(|| format!("challenge `{name}` recorded no run"))??
        .path();
    let commitment = CommitmentFile::read_memory(
        &run.join("challenge")
            .join(stage)
            .join("raster-replay")
            .join(stage)
            .join("commit.bin"),
    )?;
    ensure!(
        commitment.window_size() == REPLAY_WINDOW,
        "challenge `{name}` replayed at window {}",
        commitment.window_size()
    );
    Ok(json!({
        "case": name,
        "stage": stage,
        "replayed": true,
        "trace_steps": commitment.len(),
        "seconds": started.elapsed().as_secs_f64(),
    }))
}

/// A claimer runs this manifest natively. Recomputation must first differ from
/// its checkpoints at this stage, and a stage long enough to challenge must
/// also replay.
#[allow(clippy::too_many_arguments)]
fn expect_found(
    leg: &Leg,
    name: &str,
    spec: &Path,
    context: &Path,
    honest: &CheckpointTrace,
    manifest: &str,
    stage: &str,
) -> Result<Value> {
    let (hashes, claimed) = claimed_run(leg, name, manifest)?;
    let found = locate_divergence(&hashes, &claimed, &hashes, honest)
        .with_context(|| format!("`{name}` did not differ from honest execution"))?;
    ensure!(
        found.stage == stage,
        "`{name}` first differs at `{}`, not `{stage}`",
        found.stage
    );
    if long_enough_to_challenge(stage) {
        let outcome = expect_divergence(leg, name, spec, context, &hashes, stage)?;
        return dispute_tie(leg, name, &claimed, honest, stage, outcome);
    }
    Ok(json!({"case": name, "stage": stage, "replayed": false}))
}

/// Where the challenger's authenticated replay of `stage` was written.
fn replay_stage_dir(leg: &Leg, name: &str, stage: &str) -> Result<PathBuf> {
    let runs = leg
        .dir
        .join(name)
        .join("challenge/target/staged-infer/runs");
    let run = fs::read_dir(&runs)?
        .next()
        .with_context(|| format!("challenge `{name}` recorded no run"))??
        .path();
    Ok(run
        .join("challenge")
        .join(stage)
        .join("raster-replay")
        .join(stage))
}

/// Ties the challenger's replay to the claimer's own checkpoint for the stage,
/// as a contract would. Records which rule settles the difference between the
/// claimed checkpoint and the honest one. When that rule is a replay, the
/// Raster stage challenge built from the claimer's checkpoint and the
/// challenger's trace must be one Raster itself admits, about that checkpoint.
fn dispute_tie(
    leg: &Leg,
    name: &str,
    claimed: &CheckpointTrace,
    honest: &CheckpointTrace,
    stage: &str,
    mut outcome: Value,
) -> Result<Value> {
    let of = |trace: &CheckpointTrace| trace.checkpoints.iter().find(|c| c.stage == stage).cloned();
    let (Some(claimed), Some(honest)) = (of(claimed), of(honest)) else {
        outcome["settled_by"] = json!("missing checkpoint");
        return Ok(outcome);
    };
    let rule = settlement(&claimed, &honest);
    outcome["settled_by"] = json!(match rule {
        Settlement::StageOrder => "stage order",
        Settlement::ProgramCheck => "program check",
        Settlement::Inputs => "inputs",
        Settlement::Replay => "replay",
        Settlement::Nothing => "nothing",
    });
    if rule != Settlement::Replay {
        return Ok(outcome);
    }
    let replay = replay_stage_dir(leg, name, stage)?;
    let commitment = CommitmentFile::read_memory(&replay.join("commit.bin"))?;
    let challenge = stage_challenge(
        &claimed,
        commitment.header(),
        &file_sha256(&replay.join("output.bin"))?,
    )
    .with_context(|| {
        format!("`{name}`: no Raster challenge can be built from the claimed checkpoint")
    })?;
    let admitted = raster_prover::challenge::verify_challenge(&challenge)
        .map_err(|refusal| anyhow::anyhow!("`{name}`: Raster refuses the challenge: {refusal}"))?;
    ensure!(
        Some(admitted.chain_digest) == dispute_digest(&claimed)
            && hex::encode(&admitted.stage.program_commitment) == claimed.program_commitment
            && hex::encode(&admitted.stage.input_manifest_commitment) == claimed.input_commitment,
        "`{name}`: the admitted challenge is not about the claimed checkpoint"
    );
    // The same trace against the honest checkpoint disputes nothing, and
    // Raster refuses it: a challenge has to state another output.
    let undisputed = stage_challenge(
        &honest,
        commitment.header(),
        &file_sha256(&replay.join("output.bin"))?,
    )
    .context("no Raster challenge can be built from the honest checkpoint")?;
    ensure!(
        raster_prover::challenge::verify_challenge(&undisputed).is_err(),
        "`{name}`: Raster admitted a challenge that disputes nothing"
    );
    outcome["raster_admits"] = json!(true);
    Ok(outcome)
}

/// What the honest claim binds: the same text, configuration, cuts and ids a
/// claim's preparation wrote.
fn honest_inputs(leg: &Leg) -> Result<RunInputs> {
    let prompt = &leg.prepared.prepared_run.prompt;
    let range_size = leg.model.range_size;
    let model_dir = leg
        .model_path
        .parent()
        .context("model manifest has no parent directory")?;
    let tokenizer: Value = read_json(&model_dir.join(&leg.model.bundle.tokenizer_path))?;
    let ranges = run_prep::marked_ranges(
        &prompt.initial_pieces,
        range_size as usize,
        &run_prep::prompt_tokenizer(&tokenizer)?,
    )?;
    Ok(RunInputs {
        text: prompt.resolved_prompt.clone(),
        config: RunConfig {
            model_id: leg.model.model_id.clone(),
            raw_prompt: leg.case.raw_prompt,
            max_new_tokens: leg.case.tokens,
            range_size,
        },
        marks: ranges.into_iter().map(|range| range.marks).collect(),
        ids: leg.claimed.to_vec(),
        schedule_skew: [0; 2],
        other_text_hash: false,
    })
}

fn not_greedy_ids(leg: &Leg) -> Result<Vec<u32>> {
    let mut ids = leg.claimed.to_vec();
    let first = ids
        .first_mut()
        .context("the pinned case claims no tokens")?;
    *first = (*first + 1) % leg.model.shape.vocab_size() as u32;
    Ok(ids)
}

/// This model's chain over these inputs.
fn render(model_path: &Path, model: &ModelManifest, run: &RunInputs, dir: &Path) -> Result<String> {
    let (template_path, template) = model.verified_raster_template(model_path)?;
    let range_size = run.config.range_size as usize;
    let mark_files = run_prep::write_marks(&run.marks, dir)?;
    let claimed_files = run_prep::write_claimed_ranges(&run.ids, range_size, dir)?;
    let tokenizer: Value = read_json(
        &model_path
            .parent()
            .context("model manifest has no parent directory")?
            .join(&model.bundle.tokenizer_path),
    )?;
    let (slices, mut text_claim) = run_prep::text_for_ranges(
        &run.text,
        run.config.raw_prompt,
        range_size,
        &run_prep::prompt_tokenizer(&tokenizer)?,
        run.marks.len(),
    )?;
    if run.other_text_hash {
        text_claim.input_text_sha256 = "00".repeat(32);
    }
    render_ranged_manifest(
        &template,
        template_path
            .parent()
            .context("manifest template has no parent directory")?,
        &run_prep::write_text_slices(&slices, dir)?,
        &run_prep::write_run_config(&run.config, dir)?,
        &run_prep::write_run_schedule(
            (mark_files.len() as i64 + run.schedule_skew[0]) as usize,
            (claimed_files.len() as i64 + run.schedule_skew[1]) as usize,
            dir,
        )?,
        &run_prep::write_text_claim(&text_claim, dir)?,
        &mark_files,
        &claimed_files,
    )
}

/// The checkpoints a claimer gets from running this manifest natively.
fn claimed_run(leg: &Leg, name: &str, manifest: &str) -> Result<(PathBuf, CheckpointTrace)> {
    let dir = leg.dir.join(name);
    fs::create_dir_all(&dir)?;
    let manifest_path = dir.join("Raster.toml");
    fs::write(&manifest_path, manifest)?;
    let run = CheckpointedInferenceExecutor
        .run(CheckpointedInferenceConfig {
            base_dir: dir.clone(),
            manifest_path: manifest_path.clone(),
            current_exe: leg.exe.to_path_buf(),
            staged_backend: StagedExecutionBackend::InProcess,
            parity_policy: ParityPolicy::Skip,
        })
        .with_context(|| format!("substituted claim `{name}`"))?;
    let trace = build_checkpoint_trace(&run.chain_dir, &manifest_path)?;
    let hashes = dir.join("checkpoints.txt");
    write_checkpoint_hashes(&hashes, &checkpoint_hashes(&trace)?)?;
    Ok((hashes, trace))
}

/// Imports the pinned bundle with its weights and tokenizer changed, so every
/// model input commits to something else.
fn import_other_model(leg: &Leg) -> Result<(PathBuf, ModelManifest)> {
    let source = leg
        .root
        .join("model-bundles")
        .join(leg.model.family()?.parity_bundle());
    let bundle = leg.dir.join("other-bundle");
    fs::create_dir_all(&bundle)?;
    fs::copy(source.join("config.json"), bundle.join("config.json"))?;
    let mut weights = fs::read(source.join("model.detwgt"))?;
    change_every_tensor(&mut weights)?;
    fs::write(bundle.join("model.detwgt"), weights)?;
    let mut tokenizer: Value = read_json(&source.join("tokenizer.json"))?;
    let vocab = &mut tokenizer["model"]["vocab"];
    let (a, b) = (vocab["a"].take(), vocab["b"].take());
    ensure!(
        a.is_u64() && b.is_u64(),
        "the pinned vocabulary lacks `a` and `b`"
    );
    (vocab["a"], vocab["b"]) = (b, a);
    fs::write(
        bundle.join("tokenizer.json"),
        serde_json::to_vec(&tokenizer)?,
    )?;

    let artifact_root = leg.dir.join("other-artifacts");
    let mut import = command(leg.exe, leg.root);
    import
        .args(["model", "import", "--model"])
        .arg(&bundle)
        .arg("--artifact-root")
        .arg(&artifact_root)
        .args(["--model-id", "other-model", "--no-stage-fixtures"])
        .args(["--range-size", &leg.model.range_size.to_string()]);
    logged(import, &leg.dir.join("other-import.log"))?;
    let path = artifact_root.join("other-model/manifest.json");
    let model = ModelManifest::load_verified(&path)?;
    Ok((path, model))
}

/// Flips the low bit of one weight in every tensor of a `.detwgt`. Byte 8 of a
/// payload is the low byte of a 16- or 32-bit value, and a value byte of a Q8
/// block. A tensor too small to have one changes at its first value.
fn change_every_tensor(bytes: &mut [u8]) -> Result<()> {
    const PAYLOAD_ALIGNMENT: usize = 64;
    fn u32_at(bytes: &[u8], at: usize) -> Result<u32> {
        let field = bytes.get(at..at + 4).context("weights ended mid-record")?;
        Ok(u32::from_le_bytes(field.try_into()?))
    }
    fn u64_at(bytes: &[u8], at: usize) -> Result<u64> {
        let field = bytes.get(at..at + 8).context("weights ended mid-record")?;
        Ok(u64::from_le_bytes(field.try_into()?))
    }
    let tensors = u64_at(bytes, 16)?;
    let mut at = 24;
    for _ in 0..tensors {
        let name = u32_at(bytes, at)? as usize;
        at += 4 + name;
        let rank = u32_at(bytes, at)? as usize;
        // Dimensions, element count, element width.
        at += 4 + 8 * rank + 8 + 4;
        let payload = u64_at(bytes, at)? as usize;
        // Payload length, largest row mass.
        at += 16;
        at = at.next_multiple_of(PAYLOAD_ALIGNMENT);
        let value = if payload > 8 { 8 } else { 0 };
        *bytes
            .get_mut(at + value)
            .context("weights ended mid-tensor")? ^= 1;
        at += payload;
    }
    ensure!(at == bytes.len(), "weights have trailing bytes");
    Ok(())
}

/// Whether a manifest table is this stage: the repeated entry of that name,
/// or the plain stage its range 0 is written out as.
fn is_stage(block: &[&str], stage: &str) -> bool {
    let name = block_name(block);
    name == stage || name == stage.replace("{j}", "0").replace("{k}", "0")
}

/// How this table binds `key`, as the manifest writes it.
fn binding_value(manifest: &str, table: &str, key: &str) -> Result<String> {
    let prefix = format!("inputs.{key} =");
    manifest_blocks(manifest)
        .iter()
        .filter(|block| block_name(block) == table)
        .flatten()
        .find_map(|line| line.trim_start().strip_prefix(&prefix))
        .map(|value| value.trim().to_string())
        .with_context(|| format!("no table `{table}` binds `{key}`"))
}

/// The manifest with `key`'s binding replaced in every block of this stage.
fn replace_binding(manifest: &str, stage: &str, key: &str, value: &str) -> Result<String> {
    let line = format!("inputs.{key} = {value}");
    replace_binding_line(manifest, stage, key, &line)
}

fn replace_binding_line(manifest: &str, stage: &str, key: &str, line: &str) -> Result<String> {
    let prefix = format!("inputs.{key} =");
    let mut replaced = 0;
    let mut out = Vec::new();
    for block in manifest_blocks(manifest) {
        let matches = is_stage(&block, stage);
        for &original in &block {
            if matches && original.trim_start().starts_with(&prefix) {
                replaced += 1;
                out.push(line);
            } else {
                out.push(original);
            }
        }
    }
    ensure!(replaced > 0, "no stage `{stage}` binds `{key}`");
    Ok(out.join("\n") + "\n")
}

/// The manifest with this stage's `key` bound as another run binds it.
fn take_binding(manifest: &str, other: &str, stage: &str, key: &str) -> Result<String> {
    let prefix = format!("inputs.{key} =");
    let blocks = manifest_blocks(other);
    let line = blocks
        .iter()
        .filter(|block| is_stage(block, stage))
        .flatten()
        .find(|line| line.trim_start().starts_with(&prefix))
        .with_context(|| format!("no stage `{stage}` binds `{key}`"))?;
    replace_binding_line(manifest, stage, key, line)
}

/// The manifest with this `[chain.input]` table taken from another run.
fn take_table(manifest: &str, other: &str, header: &str) -> Result<String> {
    let blocks = manifest_blocks(other);
    let table = blocks
        .iter()
        .find(|block| block[0].trim() == header)
        .with_context(|| format!("no table {header}"))?;
    let mut out = Vec::new();
    for block in manifest_blocks(manifest) {
        out.extend(if block[0].trim() == header {
            table.clone()
        } else {
            block
        });
    }
    Ok(out.join("\n") + "\n")
}

/// The manifest with layers 0 and 1 of the PLE table exchanged, weights and
/// commitments together, so each file still proves what the table says.
fn swapped_ple(leg: &Leg, manifest: &str) -> Result<String> {
    let model_dir = leg
        .model_path
        .parent()
        .context("model manifest has no parent directory")?;
    let source = model_dir.join("raster/prefill-prepare-aux");
    let target = leg.dir.join("swapped-ple/weights");
    fs::create_dir_all(&target)?;
    for layer in 0..leg.model.shape.num_hidden_layers() {
        let read_as = match layer {
            0 => 1,
            1 => 0,
            other => other,
        };
        for extension in ["rastered", "rindex"] {
            fs::copy(
                source.join(format!("layer{read_as}.{extension}")),
                target.join(format!("layer{layer}.{extension}")),
            )?;
        }
    }
    let (source, target) = (
        source.to_str().context("path is not UTF-8")?,
        target.to_str().context("path is not UTF-8")?,
    );
    let mut out = Vec::new();
    for block in manifest_blocks(manifest) {
        if block[0].trim() != "[chain.input.aux_layer]" {
            out.extend(block.iter().map(|line| line.to_string()));
            continue;
        }
        let commitment = |layer: &str| {
            block
                .iter()
                .find(|line| line.trim_end().ends_with(layer))
                .map(|line| line.to_string())
                .with_context(|| format!("the PLE table has no commitment {layer}"))
        };
        let (first, second) = (commitment("# l0")?, commitment("# l1")?);
        for line in &block {
            out.push(if line.trim_end().ends_with("# l0") {
                second.clone()
            } else if line.trim_end().ends_with("# l1") {
                first.clone()
            } else {
                line.replace(source, target)
            });
        }
    }
    Ok(out.join("\n") + "\n")
}

/// Change all complete buckets without changing any matching rule. This tests
/// membership even when the numerical/tokenization result happens to be equal.
fn tokenizer_substitution_binding(manifest: &str, dir: &Path) -> Result<String> {
    use sequence_check::prompt::{BpeMerge, PromptTokenizer, TokenEntry};
    let parsed: toml::Value = toml::from_str(manifest)?;
    let stage = parsed["chain"]["repeat"]
        .as_array()
        .context("missing repeats")?
        .iter()
        .filter_map(|repeat| repeat["stage"].as_array())
        .flatten()
        .find(|s| s["name"].as_str() == Some("prompt_bind_p{j}"))
        .context("missing prompt bind")?;
    let external = &stage["inputs"]["tokenizer"]["external"];
    let data = Path::new(
        external["path"]
            .as_str()
            .context("missing tokenizer path")?,
    );
    let index = Path::new(
        external["index_path"]
            .as_str()
            .context("missing tokenizer index")?,
    );
    ensure!(
        data.is_absolute() && index.is_absolute(),
        "mutation fixture paths must be absolute"
    );
    let value =
        raster_runtime::read_raster_value(data, index, &raster_runtime::ReadLimits::unbounded())?;
    let mut tokenizer: PromptTokenizer =
        serde_json::from_value(super::compare::value_json(&value)?)?;
    for bucket in tokenizer.vocab_buckets.iter_mut() {
        let mut entries = bucket.entries.as_slice().to_vec();
        entries.push(TokenEntry {
            token: "\0iris-model-substitution".into(),
            id: u32::MAX,
        });
        bucket.entries = entries.into();
    }
    for bucket in tokenizer.merge_buckets.iter_mut() {
        let mut rules = bucket.rules.as_slice().to_vec();
        rules.push(BpeMerge {
            left: "\0iris-model-substitution".into(),
            right: "\0iris-model-substitution".into(),
            merged: "\0iris-model-substitution".into(),
            rank: u32::MAX,
        });
        bucket.rules = rules.into();
    }
    fs::create_dir_all(dir)?;
    let data = dir.join("tokenizer.rastered");
    let index = dir.join("tokenizer.rindex");
    let root = raster::write_raster_files(&tokenizer, &data, &index)?;
    Ok(format!(
        "{{ external = {{ path = {}, index_path = {}, commitment = {} }} }}",
        serde_json::to_string(&data)?,
        serde_json::to_string(&index)?,
        serde_json::to_string(&root)?
    ))
}

/// Change every entry, including hidden text, so every consumed token tests
/// membership. Changes confined to unread entries need not alter execution.
fn decoder_substitution_manifests(manifest: &str, dir: &Path) -> Result<(String, String)> {
    let parsed: toml::Value = toml::from_str(manifest)?;
    let stage = parsed["chain"]["stage"]
        .as_array()
        .context("missing stages")?
        .iter()
        .find(|stage| stage["name"].as_str() == Some("check_c0"))
        .context("missing the first check stage")?;
    let external = &stage["inputs"]["decoder"]["external"];
    let data = Path::new(external["path"].as_str().context("missing decoder path")?);
    let index = Path::new(
        external["index_path"]
            .as_str()
            .context("missing decoder index")?,
    );
    ensure!(
        data.is_absolute() && index.is_absolute(),
        "mutation fixture paths must be absolute"
    );
    let artifact =
        raster_runtime::read_raster_value(data, index, &raster_runtime::ReadLimits::unbounded())?;
    let table: sequence_check::input::DecoderTable =
        serde_json::from_value(super::compare::value_json(&artifact)?)?;
    let tokens = table
        .tokens
        .iter()
        .cloned()
        .map(|mut token| {
            token.token.push_str("model-identity-substitution");
            token
        })
        .collect::<Vec<_>>();
    let table = sequence_check::input::DecoderTable {
        tokens: tokens.into(),
    };
    let witness = sequence_check::decoder_witness(&table).map_err(|f| anyhow::anyhow!("{f:?}"))?;
    fs::create_dir_all(dir)?;
    let binding = |name: &str, root: String| -> Result<String> {
        Ok(format!(
            "{{ external = {{ path = {}, index_path = {}, commitment = {} }} }}",
            serde_json::to_string(&dir.join(format!("{name}.rastered")))?,
            serde_json::to_string(&dir.join(format!("{name}.rindex")))?,
            serde_json::to_string(&root)?
        ))
    };
    let decoder_root = raster::write_raster_files(
        &table,
        &dir.join("decoder.rastered"),
        &dir.join("decoder.rindex"),
    )?;
    let witness_root = raster::write_raster_files(
        &witness,
        &dir.join("witness.rastered"),
        &dir.join("witness.rindex"),
    )?;
    let changed = replace_binding(
        manifest,
        "check_c{k}",
        "decoder",
        &binding("decoder", decoder_root)?,
    )?;
    let changed_proof = replace_binding(
        &changed,
        "check_c{k}",
        "witness",
        &binding("witness", witness_root)?,
    )?;
    Ok((changed, changed_proof))
}

fn expect_semantic_rejection(leg: &Leg, name: &str, manifest: &str) -> Result<Value> {
    replay_semantic_rejection(
        leg.root,
        leg.dir.join(name),
        leg.exe,
        leg.raster,
        &leg.model.model_id,
        name,
        manifest,
    )
}

fn replay_semantic_rejection(
    root: &Path,
    dir: PathBuf,
    exe: &Path,
    raster: &Path,
    expected_model_id: &str,
    name: &str,
    manifest: &str,
) -> Result<Value> {
    fs::create_dir_all(&dir)?;
    let manifest_path = dir.join("Raster.toml");
    fs::write(&manifest_path, manifest)?;
    let native = CheckpointedInferenceExecutor.run(CheckpointedInferenceConfig {
        base_dir: dir.clone(),
        manifest_path: manifest_path.clone(),
        current_exe: exe.to_path_buf(),
        staged_backend: StagedExecutionBackend::InProcess,
        parity_policy: ParityPolicy::Skip,
    })?;
    let raster_dir = dir.join("raster");
    fs::create_dir(&raster_dir)?;
    let mut run = command(raster, root);
    run.args(["raster", "chain", "run"])
        .arg(&manifest_path)
        .args(["--no-auth", "--run"])
        .arg(&raster_dir);
    logged(run, &dir.join("raster.log"))?;
    let order =
        inference_artifacts::read_stage_order(&raster_dir)?.context("missing stage inventory")?;
    let raster_outputs = super::read_boundaries(&raster_dir, &order, false)?;
    let native_outputs = super::read_boundaries(&native.chain_dir, &order, false)?;
    for stage in &order {
        super::compare(stage, &raster_outputs[stage], &native_outputs[stage])?;
    }
    let result = native.final_result.context("missing final record")?;
    ensure!(
        result.model_id == expected_model_id,
        "substitution replaced the statement model ID"
    );
    ensure!(
        result.model_fault != 0,
        "substitution produced a clean model/execution binding verdict"
    );
    let rejection = crate::challenge::claimed_output_rejection(Some(&result))?
        .context("matching dishonest execution was accepted")?;
    ensure!(!rejection.is_empty(), "missing semantic rejection");
    Ok(
        json!({"name": name, "matching_checkpoints": order.len(), "model_fault": result.model_fault, "rejected": true}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires retained three-path parity fixtures; set IRIS_PARITY_RUN"]
    fn decoder_self_replay_rejects_matching_dishonest_execution() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../..")
            .canonicalize()
            .unwrap();
        let base = std::path::PathBuf::from(
            std::env::var_os("IRIS_PARITY_RUN").expect("set IRIS_PARITY_RUN"),
        );
        let model: inference_artifacts::ModelManifest = inference_artifacts::read_json(
            &base.join("model-artifacts/parity-gemma/manifest.json"),
        )
        .unwrap();
        let challenges = base.join("over-window/challenges");
        let changed =
            std::fs::read_to_string(challenges.join("other-decoder/Raster.toml")).unwrap();
        let (changed, changed_proof) = super::decoder_substitution_manifests(
            &changed,
            &base.join(format!("decoder-mutation-inputs-{}", std::process::id())),
        )
        .unwrap();
        let cases = [
            ("decoder-self-replay", changed),
            ("decoder-and-proof-self-replay", changed_proof),
        ];
        for (name, manifest) in cases {
            let dir = base.join(format!("{name}-{}", std::process::id()));
            let result = super::replay_semantic_rejection(
                &root,
                dir,
                &root.join("target/release/iris"),
                &root.join("target/parity-tools/release/cargo-raster"),
                &model.model_id,
                name,
                &manifest,
            )
            .unwrap();
            assert_eq!(result["rejected"], true);
        }
    }

    #[test]
    #[ignore = "replays retained pipeline attacks through native and Raster; set IRIS_PARITY_RUN"]
    fn pipeline_wiring_self_replays_reject_matching_execution() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../..")
            .canonicalize()
            .unwrap();
        let base = PathBuf::from(std::env::var_os("IRIS_PARITY_RUN").expect("set IRIS_PARITY_RUN"));
        let model: ModelManifest =
            read_json(&base.join("model-artifacts/parity-gemma/manifest.json")).unwrap();
        for case in [
            "repeated-context",
            "dropped-verdict",
            "first-prompt-context",
            "restarted-positions",
            "dropped-cache",
            "wrong-auxiliary-range",
            "stale-cache-range",
            "stale-predicting-row",
            "claim-embedding-reset",
        ] {
            let name = format!("{case}-self-replay");
            let manifest = fs::read_to_string(
                base.join("over-window/challenges")
                    .join(&name)
                    .join("Raster.toml"),
            )
            .unwrap();
            let result = replay_semantic_rejection(
                &root,
                base.join(format!("binding-{case}-{}", std::process::id())),
                &root.join("target/release/iris"),
                &root.join("target/parity-tools/release/cargo-raster"),
                &model.model_id,
                &name,
                &manifest,
            )
            .unwrap();
            assert_eq!(result["rejected"], true);
        }
    }

    #[test]
    #[ignore = "requires retained parity fixtures; set IRIS_PARITY_RUN"]
    fn tokenizer_substitutions_touch_consumed_buckets() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../..")
            .canonicalize()
            .unwrap();
        let base = PathBuf::from(std::env::var_os("IRIS_PARITY_RUN").expect("set IRIS_PARITY_RUN"))
            .canonicalize()
            .unwrap();
        let model: ModelManifest =
            read_json(&base.join("model-artifacts/parity-gemma/manifest.json")).unwrap();
        let runs = base.join("over-window/target/iris/runs");
        let prepared = fs::read_dir(runs).unwrap().next().unwrap().unwrap().path();
        let manifest = fs::read_to_string(prepared.join("Raster.toml")).unwrap();
        let work = base.join(format!("sparse-tokenizer-native-{}", std::process::id()));
        let binding = tokenizer_substitution_binding(&manifest, &work.join("inputs")).unwrap();
        for (name, stage) in [("bind", "prompt_bind_p{j}"), ("check", "prompt_check_p{j}")] {
            let dir = work.join(name);
            fs::create_dir_all(&dir).unwrap();
            let path = dir.join("Raster.toml");
            fs::write(
                &path,
                replace_binding(&manifest, stage, "tokenizer", &binding).unwrap(),
            )
            .unwrap();
            let run = CheckpointedInferenceExecutor
                .run(CheckpointedInferenceConfig {
                    base_dir: dir,
                    manifest_path: path,
                    current_exe: root.join("target/release/iris"),
                    staged_backend: StagedExecutionBackend::InProcess,
                    parity_policy: ParityPolicy::Skip,
                })
                .unwrap();
            let result = run.final_result.unwrap();
            assert_eq!(result.model_id, model.model_id);
            assert_ne!(
                result.model_fault, 0,
                "{name} must consume a mutated bucket"
            );
        }
    }

    const MANIFEST: &str = "[chain]\nname = \"iris\"\n\n[chain.input.layers]\nindex = \"l\"\npath = \"a/layer{l}.rastered\"\ncommitments = [\n  \"one\", # l0\n  \"two\", # l1\n]\n\n[[chain.stage]]\nname = \"first\"\ninputs.value = { from = \"source\" }\n\n[[chain.repeat]]\nname = \"more\"\n\n  [[chain.repeat.stage]]\nname = \"later_{j}\"\ninputs.value = { from = \"first\" }\ninputs.other = { from = \"first\" }\n";

    #[test]
    fn a_binding_changes_in_the_named_stage_only() {
        let changed =
            replace_binding(MANIFEST, "later_{j}", "value", r#"{ from = "source" }"#).unwrap();
        assert_eq!(
            changed,
            MANIFEST.replace(
                "name = \"later_{j}\"\ninputs.value = { from = \"first\" }",
                "name = \"later_{j}\"\ninputs.value = { from = \"source\" }"
            )
        );
        assert!(replace_binding(MANIFEST, "absent", "value", "{}").is_err());
        assert!(replace_binding(MANIFEST, "first", "absent", "{}").is_err());
    }

    #[test]
    fn a_table_and_a_binding_come_from_the_other_manifest() {
        let other = MANIFEST
            .replace("a/layer", "b/layer")
            .replace("{ from = \"source\" }", "{ from = \"elsewhere\" }");
        let table = take_table(MANIFEST, &other, "[chain.input.layers]").unwrap();
        assert_eq!(table, MANIFEST.replace("a/layer", "b/layer"));
        let binding = take_binding(MANIFEST, &other, "first", "value").unwrap();
        assert_eq!(
            binding,
            MANIFEST.replace("{ from = \"source\" }", "{ from = \"elsewhere\" }")
        );
    }

    #[test]
    fn every_tensor_changes_by_one_bit() {
        // Two tensors: header, then name, rank, dims, count, width, payload
        // length and row mass, padded to a 64-byte offset.
        let mut bytes = b"DNWGTV0\0".to_vec();
        bytes.extend(3_u32.to_le_bytes());
        bytes.extend(1_u32.to_le_bytes());
        bytes.extend(2_u64.to_le_bytes());
        for name in ["a", "bc"] {
            bytes.extend((name.len() as u32).to_le_bytes());
            bytes.extend(name.as_bytes());
            bytes.extend(1_u32.to_le_bytes());
            bytes.extend(8_u64.to_le_bytes());
            bytes.extend(8_u64.to_le_bytes());
            bytes.extend(16_u32.to_le_bytes());
            bytes.extend(16_u64.to_le_bytes());
            bytes.extend(0_u64.to_le_bytes());
            bytes.resize(bytes.len().next_multiple_of(64), 0);
            bytes.extend([0_u8; 16]);
        }
        let original = bytes.clone();
        change_every_tensor(&mut bytes).unwrap();
        let changed: Vec<usize> = (0..bytes.len())
            .filter(|&i| bytes[i] != original[i])
            .collect();
        assert_eq!(changed.len(), 2);
        for at in changed {
            assert_eq!(bytes[at] ^ original[at], 1);
            assert_eq!(at % 64, 8);
        }
        bytes.push(0);
        assert!(change_every_tensor(&mut bytes).is_err());
    }
}
