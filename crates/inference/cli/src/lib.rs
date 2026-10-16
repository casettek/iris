use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command as ProcessCommand, ExitCode, ExitStatus};

use anyhow::{bail, Context, Result};
use clap::{Args, Parser, Subcommand};

mod challenge;
mod claim;
mod infer;
mod parity;
#[cfg(feature = "replay-profile")]
mod replay_profile;
#[cfg(test)]
mod test_support;

pub use challenge::{
    build_challenge, ChallengeBuildOptions, ChallengeBuildOutcome, ChallengeBuildResult,
    ChallengeInput,
};
pub use claim::{
    build_claim, corrupt_checkpoint_hashes, ClaimBuildOptions, ClaimBuildResult,
    CorruptCheckpointHashesOptions, CorruptCheckpointHashesResult, CorruptCheckpointSelection,
};
pub use infer::run_infer;
#[cfg(feature = "replay-profile")]
pub use replay_profile::{run_replay_profile, ReplayProfileOptions};
pub use inference_artifacts::{
    build_checkpoint_trace, write_claim_artifacts, ChallengeBundle, ChallengeTrace, Checkpoint,
    CheckpointTrace, ClaimBundle, ClaimEndpoint, Divergence, DivergenceReason, ReplayPackage,
    CHALLENGE_BUNDLE_JSON, CHALLENGE_TRACE_JSON, CHECKPOINT_HASHES_TXT, CHECKPOINT_TRACE_JSON,
    CLAIM_BUNDLE_JSON, DIVERGENCE_JSON, REPLAY_PACKAGE_JSON,
};

#[derive(Debug, Parser)]
#[command(name = "iris")]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    /// Run one staged-infer stage and publish its canonical output artifact.
    #[arg(long = "run-stage", alias = "run-prefill-range-stage", hide = true)]
    run_stage: Option<PathBuf>,

    /// Compare one already-completed no-auth stage without rerunning the chain.
    #[arg(long = "compare-stage", alias = "compare-prefill-range-stage")]
    compare_stage: Option<PathBuf>,

    #[arg(long, hide = true)]
    input: Option<PathBuf>,

    #[arg(long = "input-manifest", hide = true)]
    input_manifest: Option<PathBuf>,
}

#[derive(Debug, Subcommand)]
enum Commands {
    /// Model preparation workflows.
    Model {
        #[command(subcommand)]
        command: ModelCommand,
    },
    /// Proposer claim workflows.
    Claim {
        #[command(subcommand)]
        command: ClaimCommand,
    },
    /// Fast unconstrained deterministic inference.
    Infer(InferArgs),
    /// Measure the heaviest tile replays in one exact stage instance.
    #[cfg(feature = "replay-profile")]
    ReplayProfile(ReplayProfileArgs),
    /// Run the pinned three-path end-to-end parity suite.
    TestParity,
    #[command(hide = true)]
    ParityWorker {
        #[arg(long)]
        run: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        direct: bool,
    },
    /// Challenge workflows reserved for the verifier path.
    Challenge {
        #[command(subcommand)]
        command: ChallengeCommand,
    },
    /// Fault proof workflows reserved for the proof path.
    Fault {
        #[command(subcommand)]
        command: FaultCommand,
    },
}

#[derive(Debug, Subcommand)]
enum ModelCommand {
    /// Import a model bundle into committed Raster inference externals.
    Import(ModelImportArgs),
}

#[derive(Debug, Args)]
struct ModelImportArgs {
    #[arg(long = "model")]
    model_dir: PathBuf,
    /// Also save a copy of the model-specific Raster template at this path.
    #[arg(long)]
    manifest: Option<PathBuf>,
    #[arg(long = "artifact-root", default_value = inference_artifacts::MODEL_ARTIFACTS_DIR)]
    artifact_root: PathBuf,
    #[arg(long = "model-id")]
    model_id: Option<String>,
    #[arg(long = "only-tokenizer")]
    only_tokenizer: bool,
    #[arg(long = "only-layers")]
    only_layers: bool,
    #[arg(long = "only-embedding")]
    only_embedding: bool,
    #[arg(long = "only-ple")]
    only_ple: bool,
    #[arg(long = "only-direct")]
    only_direct: bool,
    /// Do not overwrite the stage-local manual input fixtures.
    #[arg(long)]
    no_stage_fixtures: bool,
}

#[derive(Debug, Args)]
struct InferArgs {
    /// Run spec to execute.
    #[arg(long, default_value = inference_artifacts::INFERENCE_RUN_SPEC_TOML)]
    run: PathBuf,
}

#[cfg(feature = "replay-profile")]
#[derive(Debug, Args)]
struct ReplayProfileArgs {
    #[arg(long, default_value = inference_artifacts::INFERENCE_RUN_SPEC_TOML)]
    run: PathBuf,
    /// Exact expanded stage name, e.g. prefill_range_l13 or prompt_merge_b1.
    #[arg(long)]
    stage: String,
    /// Number of distinct tiles selected by their slowest native invocation.
    #[arg(long, default_value = "3")]
    top_n: std::num::NonZeroUsize,
}

#[derive(Debug, Subcommand)]
enum ClaimCommand {
    /// Build a staged-infer checkpoint claim.
    Build(ClaimBuildArgs),
    /// Copy and corrupt one claimed checkpoint hash for manual challenge tests.
    Corrupt(ClaimCorruptHashesArgs),
}

#[derive(Debug, Args)]
struct ClaimBuildArgs {
    /// Run spec to execute.
    #[arg(long, default_value = inference_artifacts::INFERENCE_RUN_SPEC_TOML)]
    run: PathBuf,

    /// Use the previous per-stage child-process staged-infer runner.
    #[arg(long = "staged-subprocess", alias = "direct-subprocess", hide = true)]
    staged_subprocess: bool,
}

#[derive(Debug, Args)]
struct ClaimCorruptHashesArgs {
    /// Checkpoints hash list to copy and corrupt.
    #[arg(long = "checkpoints")]
    checkpoints: PathBuf,

    /// Optional output path for the corrupted hash list.
    #[arg(long)]
    output: Option<PathBuf>,

    /// Optional checkpoint trace used only to resolve --stage to an index.
    #[arg(long)]
    trace: Option<PathBuf>,

    /// Corrupt this zero-based checkpoint index.
    #[arg(long)]
    index: Option<usize>,

    /// Corrupt the checkpoint with this stage name.
    #[arg(long)]
    stage: Option<String>,

    /// Pick a checkpoint to corrupt. This is the default when no selector is supplied.
    #[arg(long)]
    random: bool,

    /// Seed for reproducible random selection.
    #[arg(long)]
    seed: Option<String>,
}

#[derive(Debug, Subcommand)]
enum ChallengeCommand {
    /// Locate the first divergent checkpoint in a claim.
    Locate,
    /// Build a challenge bundle for a divergent claim.
    Build(ChallengeBuildArgs),
}

#[derive(Debug, Subcommand)]
enum FaultCommand {
    /// Prove a bad Raster challenge.
    Prove,
}

pub fn run_from_env() -> Result<ExitCode> {
    execute(Cli::parse())
}

pub fn execute_from<I, T>(args: I) -> Result<ExitCode>
where
    I: IntoIterator<Item = T>,
    T: Into<std::ffi::OsString> + Clone,
{
    execute(Cli::try_parse_from(args)?)
}

pub fn parse_for_test<I, T>(args: I) -> Result<()>
where
    I: IntoIterator<Item = T>,
    T: Into<std::ffi::OsString> + Clone,
{
    Cli::try_parse_from(args)?;
    Ok(())
}

fn execute(cli: Cli) -> Result<ExitCode> {
    if let Some(stage_dir) = cli.run_stage.as_ref() {
        require_input_args(&cli)?;
        staged_infer::parity::run_hidden_stage_direct(stage_dir)?;
        return Ok(ExitCode::SUCCESS);
    }

    if let Some(stage_dir) = cli.compare_stage {
        if cli.input.is_none() || cli.input_manifest.is_none() {
            let status = run_hidden_compare(&stage_dir)?;
            return finish_parity_run(status, &stage_dir);
        }
        let report = staged_infer::parity::run_hidden_stage_compare(&stage_dir)?;
        return Ok(if report.matched {
            ExitCode::SUCCESS
        } else {
            ExitCode::from(1)
        });
    }

    match cli.command {
        #[cfg(feature = "replay-profile")]
        Some(Commands::ReplayProfile(args)) => {
            warn_if_debug_build("replay-profile");
            run_replay_profile(ReplayProfileOptions {
                base_dir: std::env::current_dir()?,
                run_spec_path: args.run,
                current_exe: std::env::current_exe()?,
                stage: args.stage,
                top_n: args.top_n.get(),
            })?;
            Ok(ExitCode::SUCCESS)
        }
        Some(Commands::TestParity) => {
            parity::run()?;
            Ok(ExitCode::SUCCESS)
        }
        Some(Commands::ParityWorker { run, output, direct }) => {
            parity::worker(&run, &output, direct)?;
            Ok(ExitCode::SUCCESS)
        }
        Some(Commands::Model {
            command: ModelCommand::Import(args),
        }) => {
            model_import::import_model(args.into_import_config())
                .map_err(|error| anyhow::anyhow!("{error}"))?;
            Ok(ExitCode::SUCCESS)
        }
        Some(Commands::Claim {
            command: ClaimCommand::Build(args),
        }) => {
            warn_if_debug_build("claim build");
            let run_spec_path = args.run.clone();
            let result = build_claim(args.into_options()?)?;
            if let Some(final_result) = result.final_result.as_ref() {
                print_infer_result(final_result, Some(&run_spec_path));
            }
            println!("checkpoint trace: {}", result.checkpoint_trace_path.display());
            println!("checkpoints: {}", result.checkpoint_hashes_path.display());
            println!("claim bundle: {}", result.claim_bundle_path.display());
            Ok(ExitCode::SUCCESS)
        }
        Some(Commands::Claim {
            command: ClaimCommand::Corrupt(args),
        }) => {
            let result = corrupt_checkpoint_hashes(args.into_options()?)?;
            println!("corrupted checkpoints: {}", result.corrupted_hashes_path.display());
            println!("corruption manifest: {}", result.corruption_manifest_path.display());
            println!("corrupted checkpoint index: {}", result.checkpoint_index);
            println!("corrupted checkpoint number: {}", result.checkpoint_number);
            if let Some(stage) = result.stage.as_ref() {
                println!("corrupted stage: {stage}");
            }
            println!("original hash: {}", result.original_hash);
            println!("corrupted hash: {}", result.corrupted_hash);
            Ok(ExitCode::SUCCESS)
        }
        Some(Commands::Infer(args)) => {
            warn_if_debug_build("infer");
            let run_spec_path = args.run.clone();
            let report = run_infer(args.run)?;
            print_infer_result(&report.result, Some(&run_spec_path));
            print_infer_timing_summary(&report.timings);
            Ok(ExitCode::SUCCESS)
        }
        Some(Commands::Challenge { command }) => match command {
            ChallengeCommand::Locate => {
                bail!("iris challenge locate is not implemented yet")
            }
            ChallengeCommand::Build(args) => {
                warn_if_debug_build("challenge build");
                let result = build_challenge(args.into_options()?)?;
                print_challenge_result(&result);
                Ok(ExitCode::SUCCESS)
            }
        },
        Some(Commands::Fault { command }) => match command {
            FaultCommand::Prove => bail!("iris fault prove is not implemented yet"),
        },
        None => bail!(
            "iris requires a workflow command; try `model import`, `claim build`, or `infer`"
        ),
    }
}

#[cfg(debug_assertions)]
fn warn_if_debug_build(workflow: &str) {
    eprintln!(
        "warning: iris {workflow} is running from a debug build; \
         use `cargo run --release -p iris-cli -- {workflow}` \
         for the optimized staged-infer path"
    );
}

#[cfg(not(debug_assertions))]
fn warn_if_debug_build(_workflow: &str) {}

#[derive(Debug, Args)]
struct ChallengeBuildArgs {
    /// Run spec selecting the model to verify against the claim.
    #[arg(long, default_value = inference_artifacts::INFERENCE_RUN_SPEC_TOML)]
    run: PathBuf,

    /// Claim bundle to challenge. Preferred over --trace because it carries frozen run metadata.
    #[arg(long)]
    claim: Option<PathBuf>,

    /// Existing checkpoint trace to challenge.
    #[arg(long)]
    trace: Option<PathBuf>,

    /// Public claimed checkpoints hash list to challenge.
    #[arg(long = "checkpoints")]
    checkpoint_hashes: Option<PathBuf>,

    /// Claim bundle used only as frozen run context for --checkpoints.
    #[arg(long = "claim-context")]
    claim_context: Option<PathBuf>,

    /// Use the previous per-stage child-process staged-infer runner.
    #[arg(long = "staged-subprocess", alias = "direct-subprocess", hide = true)]
    staged_subprocess: bool,
}

fn require_input_args(cli: &Cli) -> Result<()> {
    if cli.input.is_none() || cli.input_manifest.is_none() {
        bail!("hidden stage mode requires --input and --input-manifest");
    }
    Ok(())
}

fn print_challenge_result(result: &ChallengeBuildResult) {
    match &result.outcome {
        ChallengeBuildOutcome::NoDivergence {
            claimed_reference_path,
            recomputed_trace_path,
            recomputed_chain_dir,
        } => {
            println!("no divergence found");
            println!("claimed reference: {}", claimed_reference_path.display());
            println!("recomputed trace: {}", recomputed_trace_path.display());
            println!("recomputed chain: {}", recomputed_chain_dir.display());
        }
        ChallengeBuildOutcome::DivergenceBuilt {
            divergence,
            replay_package,
            challenge_bundle_path,
            ..
        } => {
            println!("divergence found: {}", divergence.stage);
            println!("checkpoint index: {}", divergence.checkpoint_index);
            println!("checkpoint number: {}", divergence.checkpoint_index + 1);
            println!("reason: {:?}", divergence.reason);
            if let Some(claimed_hash) = divergence.claimed_hash.as_ref() {
                println!("claimed checkpoint hash: {claimed_hash}");
            }
            if let Some(recomputed_hash) = divergence.recomputed_hash.as_ref() {
                println!("recomputed checkpoint hash: {recomputed_hash}");
            }
            if let Some(claimed) = divergence.claimed.as_ref() {
                println!("claimed commitment: {}", claimed.output_commitment);
                println!("claimed sha256: {}", claimed.output_sha256);
            }
            if let Some(recomputed) = divergence.recomputed.as_ref() {
                println!("recomputed commitment: {}", recomputed.output_commitment);
                println!("recomputed sha256: {}", recomputed.output_sha256);
            }
            println!(
                "raster fingerprint: {}",
                replay_package.commit_path.display()
            );
            println!("challenge bundle: {}", challenge_bundle_path.display());
        }
    }
}

fn print_infer_result(result: &staged_infer::InferenceResult, run_spec_path: Option<&Path>) {
    println!("generated token count: {}", result.generated_token_count);
    println!("generated token ids: {:?}", result.generated_token_ids);
    println!(
        "generated token ids sha256: {}",
        result.generated_token_ids_sha256
    );
    if let Some(run_spec_path) = run_spec_path {
        match render_token_trace_for_run(result, run_spec_path) {
            Ok(trace) => print!("{trace}"),
            Err(error) => eprintln!("generated token trace unavailable: {error:#}"),
        }
    }
    println!("stop reason: {}", result.stop_reason);
    println!("generated text:");
    println!("{}", result.generated_text);
}

fn render_token_trace_for_run(
    result: &staged_infer::InferenceResult,
    run_spec_path: &Path,
) -> Result<String> {
    let base_dir = std::env::current_dir().context("failed to read current directory")?;
    let run_spec_path = absolute(&base_dir, run_spec_path);
    let run_spec = inference_artifacts::read_run_spec(&run_spec_path)?;
    let run_spec_dir = run_spec_path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("run spec has no parent directory"))?;
    let model_manifest_path = absolute(run_spec_dir, &run_spec.model_manifest);
    let model_manifest: inference_artifacts::ModelManifest =
        inference_artifacts::read_json(&model_manifest_path)?;
    let model_manifest_dir = model_manifest_path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("model manifest has no parent directory"))?;
    let tokenizer_path = absolute(model_manifest_dir, &model_manifest.bundle.tokenizer_path);
    let tokenizer: serde_json::Value = inference_artifacts::read_json(&tokenizer_path)?;
    let decoder = TokenDecoder::from_tokenizer(&tokenizer, &model_manifest.eos_token_ids)?;
    Ok(render_token_trace(result, &decoder))
}

fn render_token_trace(result: &staged_infer::InferenceResult, decoder: &TokenDecoder) -> String {
    let mut out = String::from("generated tokens:\n");
    let mut stopped = false;
    for (idx, token_id) in result.generated_token_ids.iter().copied().enumerate() {
        let display = decoder.display(token_id, stopped);
        if display.kind == TokenKind::Terminal {
            stopped = true;
        }
        out.push_str(&format!(
            "  {idx}: id={token_id} token={} kind={} rendered={}\n",
            display.token,
            display.kind.as_str(),
            display.rendered
        ));
    }
    out
}

#[derive(Debug)]
struct TokenDecoder {
    tokens: BTreeMap<u32, String>,
    special_ids: BTreeSet<u32>,
    terminal_ids: BTreeSet<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TokenKind {
    Text,
    Special,
    Terminal,
    Byte,
    Unknown,
    AfterStop,
}

struct TokenDisplay {
    token: String,
    kind: TokenKind,
    rendered: String,
}

impl TokenKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Special => "special",
            Self::Terminal => "terminal",
            Self::Byte => "byte",
            Self::Unknown => "unknown",
            Self::AfterStop => "after_stop",
        }
    }
}

impl TokenDecoder {
    fn from_tokenizer(tokenizer: &serde_json::Value, eos_ids: &[u32]) -> Result<Self> {
        let mut tokens = tokenizer
            .get("model")
            .and_then(|model| model.get("vocab"))
            .and_then(serde_json::Value::as_object)
            .ok_or_else(|| anyhow::anyhow!("tokenizer.json has no vocab"))?
            .iter()
            .map(|(token, id)| {
                Ok((
                    id.as_u64()
                        .ok_or_else(|| anyhow::anyhow!("token '{token}' has non-integer id"))?
                        as u32,
                    token.clone(),
                ))
            })
            .collect::<Result<BTreeMap<_, _>>>()?;
        let mut special_ids = BTreeSet::new();
        if let Some(added_tokens) = tokenizer
            .get("added_tokens")
            .and_then(serde_json::Value::as_array)
        {
            for entry in added_tokens {
                let Some(id) = entry.get("id").and_then(serde_json::Value::as_u64) else {
                    continue;
                };
                if let Some(content) = entry.get("content").and_then(serde_json::Value::as_str) {
                    tokens
                        .entry(id as u32)
                        .or_insert_with(|| content.to_string());
                }
                if entry
                    .get("special")
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or(false)
                {
                    special_ids.insert(id as u32);
                }
            }
        }
        Ok(Self {
            tokens,
            special_ids,
            terminal_ids: eos_ids.iter().copied().collect(),
        })
    }

    fn display(&self, token_id: u32, stopped: bool) -> TokenDisplay {
        let Some(token) = self.tokens.get(&token_id) else {
            return TokenDisplay {
                token: String::from("<unknown>"),
                kind: TokenKind::Unknown,
                rendered: String::from("<unknown>"),
            };
        };
        if stopped {
            return TokenDisplay {
                token: quote_token(token),
                kind: TokenKind::AfterStop,
                rendered: String::from("<skipped>"),
            };
        }
        if self.terminal_ids.contains(&token_id) {
            return TokenDisplay {
                token: quote_token(token),
                kind: TokenKind::Terminal,
                rendered: String::from("<stops text>"),
            };
        }
        if self.special_ids.contains(&token_id) {
            return TokenDisplay {
                token: quote_token(token),
                kind: TokenKind::Special,
                rendered: String::from("<skipped>"),
            };
        }
        if let Some(byte) = byte_fallback(token) {
            return TokenDisplay {
                token: quote_token(token),
                kind: TokenKind::Byte,
                rendered: format!("<byte 0x{byte:02X}>"),
            };
        }
        TokenDisplay {
            token: quote_token(token),
            kind: TokenKind::Text,
            rendered: quote_token(&token.replace('\u{2581}', " ")),
        }
    }
}

fn byte_fallback(token: &str) -> Option<u8> {
    let bytes = token.as_bytes();
    if bytes.len() != 6 || &bytes[..3] != b"<0x" || bytes[5] != b'>' {
        return None;
    }
    Some(hex_nibble(bytes[3])? * 16 + hex_nibble(bytes[4])?)
}

fn hex_nibble(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

fn quote_token(token: &str) -> String {
    format!("{token:?}")
}

fn absolute(base: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        base.join(path)
    }
}

fn print_infer_timing_summary(timings: &staged_infer::InferenceTimings) {
    eprint!("{}", render_infer_timing_summary(timings));
}

fn render_infer_timing_summary(timings: &staged_infer::InferenceTimings) -> String {
    if is_direct_infer_timing(timings) {
        return render_direct_infer_timing_summary(timings);
    }

    let mut out = String::new();
    out.push_str("infer timing:\n");
    out.push_str(&format!(
        "  total: {}\n",
        format_duration(timings.total_duration)
    ));
    if !timings.aux_waves.is_empty() {
        for wave in &timings.aux_waves {
            out.push_str(&format!(
                "  aux wave {}: {} stages, parallelism {}, wall {}, stage-sum {}",
                wave.name,
                wave.stage_count,
                wave.parallelism,
                format_duration(wave.wall_duration),
                format_duration(wave.stage_duration_sum)
            ));
            out.push('\n');
        }
    }

    let synthesis = timings
        .stages
        .iter()
        .map(|stage| stage.input_synthesis_duration)
        .sum();
    let input_load = timings
        .stages
        .iter()
        .map(|stage| stage.input_load_duration)
        .sum();
    let kernel = timings
        .stages
        .iter()
        .map(|stage| stage.kernel_duration)
        .sum();
    let encode = timings
        .stages
        .iter()
        .map(|stage| stage.encode_duration)
        .sum();
    out.push_str(&format!("  stages: {}\n", timings.stages.len()));
    out.push_str(&format!(
        "  input synthesis: {}\n",
        format_duration(synthesis)
    ));
    out.push_str(&format!("  input load: {}\n", format_duration(input_load)));
    out.push_str(&format!("  kernels: {}\n", format_duration(kernel)));
    out.push_str(&format!("  encode: {}\n", format_duration(encode)));
    out
}

fn is_direct_infer_timing(timings: &staged_infer::InferenceTimings) -> bool {
    !timings.stages.is_empty()
        && timings
            .stages
            .iter()
            .all(|stage| stage.routine == "direct-infer")
}

fn render_direct_infer_timing_summary(timings: &staged_infer::InferenceTimings) -> String {
    let named = |name: &str| {
        timings
            .stages
            .iter()
            .find(|stage| stage.stage == name)
            .map(|stage| stage.total_duration)
            .unwrap_or_default()
    };
    let accounted = named("load_direct_model")
        + named("prompt_prepare")
        + named("input_embedding")
        + named("prefill")
        + named("decode")
        + named("output_finalize");
    let kernel_records = timings
        .stages
        .iter()
        .filter(|stage| !matches!(stage.stage.as_str(), "prefill" | "decode"))
        .count();

    let mut out = String::new();
    out.push_str("direct-infer timing:\n");
    out.push_str(&format!(
        "  total: {}\n",
        format_duration(timings.total_duration)
    ));
    out.push_str(&format!("  phase records: {kernel_records}\n"));
    out.push_str(&format!(
        "  model load: {}\n",
        format_duration(named("load_direct_model"))
    ));
    out.push_str(&format!(
        "  prompt prepare: {}\n",
        format_duration(named("prompt_prepare"))
    ));
    out.push_str(&format!(
        "  input embedding: {}\n",
        format_duration(named("input_embedding"))
    ));
    out.push_str(&format!(
        "  prefill: {}\n",
        format_duration(named("prefill"))
    ));
    out.push_str(&format!("  decode: {}\n", format_duration(named("decode"))));
    out.push_str(&format!(
        "  output finalize: {}\n",
        format_duration(named("output_finalize"))
    ));
    out.push_str(&format!(
        "  accounted phase time: {}\n",
        format_duration(accounted)
    ));
    out
}

fn format_duration(duration: std::time::Duration) -> String {
    if duration.as_secs() >= 1 {
        format!("{:.2}s", duration.as_secs_f64())
    } else if duration.as_millis() >= 1 {
        format!("{}ms", duration.as_millis())
    } else if duration.as_micros() >= 1 {
        format!("{}us", duration.as_micros())
    } else {
        format!("{}ns", duration.as_nanos())
    }
}

fn finish_parity_run(status: ExitStatus, stage_dir: &Path) -> Result<ExitCode> {
    match staged_infer::parity::read_parity_report(stage_dir) {
        Ok(report) => {
            let chain_dir = stage_dir
                .parent()
                .ok_or_else(|| anyhow::anyhow!("selected stage has no chain run directory"))?;
            print!("{}", render_timing_summary_or_warning(chain_dir, &report));
        }
        Err(error) if status.success() => return Err(error),
        Err(_) => {}
    }
    Ok(exit_code_from_status(status))
}

fn render_timing_summary_or_warning(
    chain_dir: &Path,
    report: &staged_infer::parity::ParityReport,
) -> String {
    staged_infer::parity::read_chain_execution_times(chain_dir)
        .and_then(|timings| staged_infer::parity::render_timing_summary(&timings, report))
        .unwrap_or_else(|error| format!("\ntiming summary unavailable: {error:#}\n"))
}

fn run_hidden_compare(stage_dir: &Path) -> Result<ExitStatus> {
    let input = stage_dir.join("input.json");
    let input_manifest = stage_dir.join("input_manifest.json");
    let prior_report = staged_infer::parity::parity_dir(stage_dir).join("report.json");
    if let Err(error) = fs::remove_file(&prior_report) {
        if error.kind() != std::io::ErrorKind::NotFound {
            return Err(error)
                .with_context(|| format!("failed to remove {}", prior_report.display()));
        }
    }
    let exe = std::env::current_exe().context("failed to locate current executable")?;
    let status = comparison_command(&exe, stage_dir, &input, &input_manifest)
        .status()
        .context("failed to start staged-infer comparison child")?;
    Ok(status)
}

fn comparison_command(
    exe: &Path,
    stage_dir: &Path,
    input: &Path,
    input_manifest: &Path,
) -> ProcessCommand {
    let mut command = ProcessCommand::new(exe);
    command
        .arg("--compare-stage")
        .arg(stage_dir)
        .arg("--input")
        .arg(input)
        .arg("--input-manifest")
        .arg(input_manifest)
        .env(raster_runtime::auth::AUTH_ENV, "0")
        .env_remove(raster_runtime::TRACE_PATH_ENV)
        .env_remove(raster_runtime::TRACE_FORMAT_ENV)
        .env_remove(raster_runtime::OUTPUT_DIR_ENV)
        .env_remove(raster_runtime::PROFILE_PATH_ENV)
        .env_remove(raster_runtime::PROFILE_STREAM_PATH_ENV)
        .env_remove(raster_runtime::PROFILE_RUN_ID_ENV);
    command
}

fn exit_code_from_status(status: ExitStatus) -> ExitCode {
    match status.code() {
        Some(code) => ExitCode::from(code.clamp(0, u8::MAX as i32) as u8),
        None => ExitCode::from(1),
    }
}

impl ModelImportArgs {
    fn into_import_config(self) -> model_import::ImportConfig {
        model_import::ImportConfig {
            model_dir: self.model_dir,
            manifest: self.manifest,
            artifact_root: self.artifact_root,
            model_id: self.model_id,
            only_tokenizer: self.only_tokenizer,
            only_layers: self.only_layers,
            only_embedding: self.only_embedding,
            only_ple: self.only_ple,
            only_direct: self.only_direct,
            no_stage_fixtures: self.no_stage_fixtures,
        }
    }
}

impl ClaimBuildArgs {
    fn into_options(self) -> Result<ClaimBuildOptions> {
        let current_exe = std::env::current_exe().context("failed to locate current executable")?;
        let staged_backend = if self.staged_subprocess {
            staged_infer::chain_runner::StagedExecutionBackend::Subprocess
        } else {
            staged_infer::chain_runner::StagedExecutionBackend::InProcess
        };
        ClaimBuildOptions::from_current_dir(current_exe, staged_backend, self.run)
    }
}

impl ClaimCorruptHashesArgs {
    fn into_options(self) -> Result<CorruptCheckpointHashesOptions> {
        let selector_count = usize::from(self.index.is_some())
            + usize::from(self.stage.is_some())
            + usize::from(self.random);
        if selector_count > 1 {
            anyhow::bail!("claim corrupt accepts only one of --index, --stage, or --random");
        }
        let selection = if let Some(index) = self.index {
            CorruptCheckpointSelection::Index(index)
        } else if let Some(stage) = self.stage {
            CorruptCheckpointSelection::Stage(stage)
        } else {
            CorruptCheckpointSelection::Random { seed: self.seed }
        };
        Ok(CorruptCheckpointHashesOptions {
            hashes_path: self.checkpoints,
            trace_path: self.trace,
            output_path: self.output,
            selection,
        })
    }
}

impl ChallengeBuildArgs {
    fn into_options(self) -> Result<ChallengeBuildOptions> {
        let input = match (
            self.claim,
            self.trace,
            self.checkpoint_hashes,
            self.claim_context,
        ) {
            (Some(claim), None, None, None) => ChallengeInput::Claim(claim),
            (None, Some(trace), None, None) => ChallengeInput::Trace(trace),
            (None, None, Some(checkpoint_hashes_path), Some(claim_context_path)) => {
                ChallengeInput::CheckpointHashes {
                    claim_context_path,
                    checkpoint_hashes_path,
                }
            }
            (None, None, Some(_), None) => {
                anyhow::bail!("challenge build --checkpoints requires --claim-context")
            }
            (None, None, None, Some(_)) => {
                anyhow::bail!("challenge build --claim-context requires --checkpoints")
            }
            (None, None, None, None) => anyhow::bail!(
                "challenge build requires --claim <claim_bundle.json> or --checkpoints <checkpoints.txt> --claim-context <claim_bundle.json>"
            ),
            _ => anyhow::bail!(
                "challenge build accepts only one input mode: --claim, --trace, or --checkpoints with --claim-context"
            ),
        };
        let current_exe = std::env::current_exe().context("failed to locate current executable")?;
        let staged_backend = if self.staged_subprocess {
            staged_infer::chain_runner::StagedExecutionBackend::Subprocess
        } else {
            staged_infer::chain_runner::StagedExecutionBackend::InProcess
        };
        ChallengeBuildOptions::from_current_dir(current_exe, staged_backend, input, self.run)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_phase_one_workflows() {
        parse_for_test([
            "iris",
            "model",
            "import",
            "--model",
            "model-bundles/model",
        ])
        .unwrap();
        parse_for_test([
            "iris",
            "claim",
            "build",
            "--run",
            "inference.toml",
        ])
        .unwrap();
        parse_for_test([
            "iris",
            "claim",
            "corrupt",
            "--checkpoints",
            "checkpoints.txt",
            "--index",
            "1",
        ])
        .unwrap();
        parse_for_test([
            "iris",
            "challenge",
            "build",
            "--trace",
            "checkpoint_trace.json",
        ])
        .unwrap();
        parse_for_test([
            "iris",
            "challenge",
            "build",
            "--checkpoints",
            "checkpoints.txt",
            "--claim-context",
            "claim_bundle.json",
        ])
        .unwrap();
        parse_for_test(["iris", "infer", "--run", "inference.toml"]).unwrap();
        parse_for_test([
            "iris",
            "model",
            "import",
            "--model",
            "model-bundles/model",
            "--only-direct",
        ])
        .unwrap();
    }

    #[test]
    fn challenge_run_spec_defaults_and_overrides_reach_options() {
        for run in [None, Some("specs/model-b.toml")] {
            let mut argv = vec![
                "iris",
                "challenge",
                "build",
                "--claim",
                "claim_bundle.json",
            ];
            if let Some(path) = run {
                argv.extend(["--run", path]);
            }
            let cli = Cli::try_parse_from(argv).unwrap();
            let Some(Commands::Challenge {
                command: ChallengeCommand::Build(args),
            }) = cli.command
            else {
                panic!("expected challenge build");
            };
            let options = args.into_options().unwrap();
            assert_eq!(
                options.run_spec_path,
                PathBuf::from(run.unwrap_or("inference.toml"))
            );
        }
    }

    #[test]
    fn infer_requires_an_imported_workspace() {
        let error = execute_from(["iris", "infer"]).unwrap_err();

        assert!(format!("{error:#}").contains("failed to load run spec"));
    }

    #[test]
    fn direct_infer_timing_summary_uses_phase_language() {
        let timings = staged_infer::InferenceTimings {
            total_duration: std::time::Duration::from_millis(10),
            stages: vec![
                direct_timing("load_direct_model", 1),
                direct_timing("prompt_prepare", 1),
                direct_timing("input_embedding", 1),
                direct_timing("prefill_layer", 3),
                direct_timing("prefill", 3),
                direct_timing("decode_layer", 2),
                direct_timing("decode", 2),
                direct_timing("output_finalize", 1),
            ],
            aux_waves: Vec::new(),
        };

        let summary = render_infer_timing_summary(&timings);

        assert!(summary.contains("direct-infer timing:"));
        assert!(summary.contains("model load: 1ms"));
        assert!(summary.contains("phase records: 6"));
        assert!(summary.contains("accounted phase time: 9ms"));
        assert!(!summary.contains("validation"));
        assert!(!summary.contains("stages:"));
        assert!(!summary.contains("input synthesis:"));
        assert!(!summary.contains("encode:"));
    }

    #[test]
    fn token_trace_labels_special_terminal_byte_unknown_and_text_tokens() {
        let decoder = TokenDecoder {
            tokens: std::collections::BTreeMap::from([
                (1, String::from("<eos>")),
                (2, String::from("<bos>")),
                (3, String::from("late")),
                (10, String::from("<0xC3>")),
                (150917, String::from("Ciao")),
            ]),
            special_ids: std::collections::BTreeSet::from([1, 2]),
            terminal_ids: std::collections::BTreeSet::from([1]),
        };
        let result = staged_infer::InferenceResult {
            generated_token_count: 6,
            generated_token_ids: vec![2, 150917, 10, 999, 1, 3],
            generated_token_ids_sha256: String::from("sha"),
            generated_text: String::from("Ciao"),
            stop_reason: String::from("eos"),
        };

        let trace = render_token_trace(&result, &decoder);

        assert!(trace.contains(r#"0: id=2 token="<bos>" kind=special rendered=<skipped>"#));
        assert!(trace.contains(r#"1: id=150917 token="Ciao" kind=text rendered="Ciao""#));
        assert!(trace.contains(r#"2: id=10 token="<0xC3>" kind=byte rendered=<byte 0xC3>"#));
        assert!(trace.contains("3: id=999 token=<unknown> kind=unknown rendered=<unknown>"));
        assert!(trace.contains(r#"4: id=1 token="<eos>" kind=terminal rendered=<stops text>"#));
        assert!(trace.contains(r#"5: id=3 token="late" kind=after_stop rendered=<skipped>"#));
    }

    #[test]
    fn model_import_args_build_typed_config() {
        let config = ModelImportArgs {
            model_dir: PathBuf::from("model"),
            manifest: Some(PathBuf::from("Raster.toml")),
            artifact_root: PathBuf::from(inference_artifacts::MODEL_ARTIFACTS_DIR),
            model_id: Some(String::from("gemma")),
            only_tokenizer: true,
            only_layers: false,
            only_embedding: true,
            only_ple: false,
            only_direct: false,
            no_stage_fixtures: false,
        }
        .into_import_config();

        assert_eq!(
            config,
            model_import::ImportConfig {
                model_dir: PathBuf::from("model"),
                manifest: Some(PathBuf::from("Raster.toml")),
                artifact_root: PathBuf::from(inference_artifacts::MODEL_ARTIFACTS_DIR),
                model_id: Some(String::from("gemma")),
                only_tokenizer: true,
                only_layers: false,
                only_embedding: true,
                only_ple: false,
                only_direct: false,
                no_stage_fixtures: false,
            }
        );
    }

    fn direct_timing(stage: &str, millis: u64) -> staged_infer::InferStageTiming {
        let duration = std::time::Duration::from_millis(millis);
        staged_infer::InferStageTiming {
            stage: stage.to_string(),
            routine: String::from("direct-infer"),
            input_synthesis_duration: std::time::Duration::ZERO,
            input_load_duration: std::time::Duration::ZERO,
            kernel_duration: duration,
            encode_duration: std::time::Duration::ZERO,
            total_duration: duration,
        }
    }
}
