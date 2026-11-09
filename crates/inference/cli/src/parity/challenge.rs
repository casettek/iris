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
use inference_artifacts::{
    build_checkpoint_trace, checkpoint_hashes, read_checkpoint_trace, read_json,
    write_checkpoint_hashes, CheckpointTrace, ModelManifest,
};
use raster_prover::commitment_file::CommitmentFile;
use run_prep::{RangeCut, RunConfig};
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
const FALSIFIED_STAGES: [&str; 6] = [
    "prompt_bind",
    "prompt_seed_p0",
    "prompt_merge_p1_b0",
    "prompt_prepare_p1",
    "range_p1_l2",
    "range_c1_l3",
];

/// Whether this stage's trace exceeds the replay window on the pinned templated
/// case.
fn long_enough_to_challenge(stage: &str) -> bool {
    stage.starts_with("prompt_") || stage.starts_with("range_p") || stage.starts_with("range_c")
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
    cuts: Vec<RangeCut>,
    ids: Vec<u32>,
    merge_repeats: u32,
    /// Added to the schedule's three counts. Zero for a schedule matching the run.
    schedule_skew: [i64; 3],
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
        replays.push(expect_divergence(
            leg, &name, &spec, context, &hashes, stage,
        )?);
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
    replays.push(expect_divergence(
        leg,
        "other-model",
        &spec,
        &elsewhere.claim_bundle_path,
        &elsewhere.checkpoint_hashes_path,
        "prompt_bind",
    )?);
    let log = fs::read_to_string(leg.dir.join("other-model/challenge/challenge.log"))?;
    ensure!(
        log.contains("warning: the claim names model"),
        "the challenger did not report that the claim names another model"
    );

    // A claim of the honest run whose checkpoints came from something else:
    // its inputs, one model input at a time, or its stages' connections.
    let moved_cut = {
        let mut cuts = inputs.cuts.clone();
        cuts[0].end -= 1;
        cuts
    };
    let mut other_budget = inputs.config.clone();
    other_budget.max_new_tokens += 1;
    let doctored = [
        (
            "other-text",
            "prompt_bind",
            RunInputs {
                text: format!("{} ", inputs.text),
                ..inputs.clone()
            },
        ),
        (
            "other-config",
            "prompt_bind",
            RunInputs {
                config: other_budget,
                ..inputs.clone()
            },
        ),
        (
            "other-schedule",
            "prompt_bind",
            RunInputs {
                schedule_skew: [0, 1, 0],
                ..inputs.clone()
            },
        ),
        (
            "other-cuts",
            "prompt_seed_p0",
            RunInputs {
                cuts: moved_cut,
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

    let take = |stage, key| take_binding(&honest_manifest, &other_manifest, stage, key);
    let table = |header| take_table(&honest_manifest, &other_manifest, header);
    let wire = |stage, key, line: &str| replace_binding(&honest_manifest, stage, key, line);
    // Sparse verification authenticates reads, not unused tokenizer content.
    // Change every bucket so each of these consuming stages sees a real mutation.
    let changed_tokenizer = tokenizer_substitution_binding(
        &honest_manifest,
        &leg.dir.join("tokenizer-substitution-inputs"),
    )?;
    let substituted = [
        (
            "tokenizer-at-bind",
            "prompt_bind",
            wire("prompt_bind", "tokenizer", &changed_tokenizer)?,
        ),
        (
            "tokenizer-at-seed",
            "prompt_seed_p0",
            wire("prompt_seed_p{j}", "tokenizer", &changed_tokenizer)?,
        ),
        (
            "tokenizer-at-merge",
            "prompt_merge_p0_b0",
            wire("prompt_merge_p{j}_b{b}", "tokenizer", &changed_tokenizer)?,
        ),
        (
            "tokenizer-at-prepare",
            "prompt_prepare_p0",
            wire("prompt_prepare_p{j}", "tokenizer", &changed_tokenizer)?,
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
        (
            "other-decoder",
            "output_finalize",
            take("output_finalize", "decoder")?,
        ),
        ("other-ple", "aux_p0_l0", table("[chain.input.aux_layer]")?),
        (
            "other-weights",
            "range_p0_l0",
            table("[chain.input.transformer_layer]")?,
        ),
        // Transformer layers have no such case: their caches differ by attention
        // type and donor, so a stage given another layer's weights stops, and
        // the claimer has no output to post.
        (
            "swapped-ple",
            "aux_p0_l0",
            swapped_ple(leg, &honest_manifest)?,
        ),
        (
            "repeated-context",
            "prompt_seed_p1",
            wire(
                "prompt_seed_p{j}",
                "previous",
                r#"{ from = "prompt_bind" }"#,
            )?,
        ),
        (
            "dropped-verdict",
            "check_c1",
            wire("check_c{k}", "prior", r#"{ from = "decode_init" }"#)?,
        ),
        (
            "unchecked-range",
            "output_finalize",
            wire("output_finalize", "edge", r#"{ from = "check_c0" }"#)?,
        ),
        (
            "first-prompt-context",
            "output_finalize",
            wire(
                "output_finalize",
                "prompt",
                r#"{ from = "prompt_prepare_p0" }"#,
            )?,
        ),
    ];
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
    let rewired = [
        (
            "restarted-positions",
            wire("embed_p{j}", "previous", r#"{ from = "range_origin" }"#)?,
        ),
        (
            "dropped-cache",
            wire(
                "range_p{j}_l{l}",
                "prior_kv",
                r#"{ from = "range_origin" }"#,
            )?,
        ),
        (
            "wrong-auxiliary-range",
            wire("range_c{k}_l{l}", "ple", r#"{ from = "aux_p0_l{l}" }"#)?,
        ),
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
    ];
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
    })
    .with_context(|| format!("claim `{name}`"))
}

/// Runs `challenge build` the way a challenger does and returns its output.
fn challenge(leg: &Leg, name: &str, spec: &Path, context: &Path, hashes: &Path) -> Result<String> {
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
    logged(run, &log).with_context(|| format!("challenge `{name}`"))?;
    Ok(fs::read_to_string(log)?)
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
        return expect_divergence(leg, name, spec, context, &hashes, stage);
    }
    Ok(json!({"case": name, "stage": stage, "replayed": false}))
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
    let ranges = run_prep::prompt_ranges(
        &prompt.initial_pieces,
        range_size as usize,
        &run_prep::vocabulary_pairs(&tokenizer)?,
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
        cuts: run_prep::range_cuts(&ranges),
        ids: leg.claimed.to_vec(),
        merge_repeats: run_prep::range_tokenizer_repeat_count(&ranges)?,
        schedule_skew: [0; 3],
    })
}

fn not_greedy_ids(leg: &Leg) -> Result<Vec<u32>> {
    let mut ids = leg.claimed.to_vec();
    let first = ids
        .first_mut()
        .context("the pinned case claims no tokens")?;
    *first = (*first + 1) % leg.model.shape.vocab_size as u32;
    Ok(ids)
}

/// This model's chain over these inputs.
fn render(model_path: &Path, model: &ModelManifest, run: &RunInputs, dir: &Path) -> Result<String> {
    let (template_path, template) = model.verified_raster_template(model_path)?;
    let range_size = run.config.range_size as usize;
    let cut_files = run_prep::write_cuts(&run.cuts, dir)?;
    let claimed_files = run_prep::write_claimed_ranges(&run.ids, range_size, dir)?;
    render_ranged_manifest(
        &template,
        template_path
            .parent()
            .context("manifest template has no parent directory")?,
        &run_prep::write_prompt_text(&run.text, dir)?,
        &run_prep::write_run_config(&run.config, dir)?,
        &run_prep::write_run_schedule(
            (cut_files.len() as i64 + run.schedule_skew[0]) as usize,
            (i64::from(run.merge_repeats) + run.schedule_skew[1]) as u32,
            (claimed_files.len() as i64 + run.schedule_skew[2]) as usize,
            dir,
        )?,
        &cut_files,
        &claimed_files,
        run.merge_repeats,
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
    let source = leg.root.join("model-bundles/parity-gemma");
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
        let matches = block_name(&block) == stage;
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
        .filter(|block| block_name(block) == stage)
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
    for layer in 0..leg.model.shape.num_hidden_layers {
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
    use output_finalize::prompt::{BoundaryPair, BpeMerge, PromptTokenizer, TokenEntry};
    let parsed: toml::Value = toml::from_str(manifest)?;
    let stage = parsed["chain"]["stage"]
        .as_array()
        .context("missing stages")?
        .iter()
        .find(|s| s["name"].as_str() == Some("prompt_bind"))
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
    for bucket in tokenizer.boundary_buckets.iter_mut() {
        let mut pairs = bucket.pairs.as_slice().to_vec();
        pairs.push(BoundaryPair {
            left: u32::MAX,
            right: u32::MAX,
        });
        bucket.pairs = pairs.into();
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
        .find(|stage| stage["name"].as_str() == Some("output_finalize"))
        .context("missing finalization stage")?;
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
    let table: output_finalize::input::DecoderTable =
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
    let table = output_finalize::input::DecoderTable {
        tokens: tokens.into(),
    };
    let witness = output_finalize::decoder_witness(&table).map_err(|f| anyhow::anyhow!("{f:?}"))?;
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
        "output_finalize",
        "decoder",
        &binding("decoder", decoder_root)?,
    )?;
    let changed_proof = replace_binding(
        &changed,
        "output_finalize",
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
    // Only truncating an otherwise honest checked transcript may preserve
    // model/history integrity; it must fail the explicit completion rule.
    if name == "unchecked-range-self-replay" {
        ensure!(
            result.output_fault == run_prep::OUTPUT_FAULT_SHORT_WITHOUT_EOS,
            "truncated checked history was not rejected as incomplete"
        );
    } else {
        ensure!(
            result.model_fault != 0,
            "substitution produced a clean model/execution binding verdict"
        );
    }
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
            "unchecked-range",
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
        for (name, stage) in [
            ("bind", "prompt_bind"),
            ("seed", "prompt_seed_p{j}"),
            ("merge", "prompt_merge_p{j}_b{b}"),
            ("prepare", "prompt_prepare_p{j}"),
        ] {
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
