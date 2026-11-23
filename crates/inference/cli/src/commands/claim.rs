//! A claim: `iris claim`.

use super::infer::{print_consensus, print_infer_result};
use crate::*;

#[derive(Debug, Subcommand)]
pub(crate) enum ClaimCommand {
    /// Build a staged-infer checkpoint claim.
    Build(ClaimBuildArgs),
    /// Copy and corrupt one claimed checkpoint hash for manual challenge
    /// tests. With `--submission`, write the claim a lying claimer would
    /// submit: another output stated for one checkpoint, with its own list.
    Corrupt(ClaimCorruptHashesArgs),
    /// Read a submitted claim's data back from the data-availability layer
    /// into a new directory, with a run spec over what it posted: what a
    /// challenger starts from.
    Fetch(ClaimFetchArgs),
    /// Ask of a fetched claim what no replay shows: whether every checkpoint
    /// names an approved program, and whether every input a stage takes from
    /// an earlier one is that stage's output. Nothing is run.
    Audit(ClaimAuditArgs),
    /// Post the data `claim build` prepared, and write where it is into the
    /// claim's submission.
    Post(ClaimPostArgs),
    /// Make the checks a contract would make on a claim's submission.
    Check(ClaimCheckArgs),
    /// Write what a contract of a run's pipeline over its models would pin.
    Pins(ClaimPinsArgs),
}

#[derive(Debug, Args)]
pub(crate) struct ClaimPostArgs {
    /// The `claim_submission.json` that `claim build` wrote. The blobs to
    /// post are beside it (`posting-blob-<n>.bin`).
    #[arg(long)]
    submission: PathBuf,

    /// The stand-in data-availability layer's directory. Default:
    /// `IRIS_DA_DIR`, or `~/.cache/iris/stub-celestia`.
    #[arg(long = "da-dir")]
    da_dir: Option<PathBuf>,
}

#[derive(Debug, Args)]
pub(crate) struct ClaimAuditArgs {
    /// The claim's `claim_submission.json`, in the directory `iris claim
    /// fetch` wrote: with its checkpoint list, checkpoints and input lists.
    #[arg(long)]
    submission: PathBuf,

    /// A run spec of the configuration the claim names.
    #[arg(long, default_value = inference_artifacts::INFERENCE_RUN_SPEC_TOML)]
    run: PathBuf,

    /// The pins the contract holds. As for `iris claim check`.
    #[arg(long)]
    pins: Option<PathBuf>,

    /// Write every call a contract can be asked of the claim with no replay,
    /// as a JSON list: a `voidClaim` for each checkpoint's program and for
    /// each input its program takes, each with what the contract is
    /// expected to answer. The calls that point at a fault are the ones to
    /// send.
    #[arg(long)]
    calls: Option<PathBuf>,
}

/// What a lying claim says falsely of its checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub(crate) enum ClaimFaultArg {
    /// Another output than the stage's.
    Output,
    /// A program outside the approved set.
    Program,
    /// An input that is not what its producer made.
    Input,
    /// An input the claim may name, in a place the chain does not give it:
    /// the value an earlier stage of the same program is given.
    Arrangement,
    /// The check's record stated valid where the check does not find it so,
    /// with the result stage run on that record. Of a claim whose claimed
    /// output is not the model's.
    Verdict,
}

#[derive(Debug, Args)]
pub(crate) struct ClaimFetchArgs {
    /// The claim's `claim_submission.json` as it was submitted, with where
    /// its data is.
    #[arg(long)]
    submission: PathBuf,

    /// Your own run spec of the configuration the claim names: its
    /// pipeline, models and parameters, and for a claim with a source the
    /// recipe and the signer's key. Its prompt and its documents are
    /// replaced by the ones the claim posted.
    #[arg(long, default_value = inference_artifacts::INFERENCE_RUN_SPEC_TOML)]
    run: PathBuf,

    /// The pins the contract holds. As for `iris claim check`.
    #[arg(long)]
    pins: Option<PathBuf>,

    /// The stand-in data-availability layer's directory. As for `iris claim
    /// check`.
    #[arg(long = "da-dir")]
    da_dir: Option<PathBuf>,

    /// A new directory to write the claim's files and `inference.toml` to.
    #[arg(long)]
    out: PathBuf,
}

#[derive(Debug, Args)]
pub(crate) struct ClaimPinsArgs {
    /// Run spec naming the pipeline and its models.
    #[arg(long, default_value = inference_artifacts::INFERENCE_RUN_SPEC_TOML)]
    run: PathBuf,

    /// Where to write the pins. They are made from the models' imports and
    /// the stage projects' locks as they are now, so run this from the iris
    /// root.
    #[arg(long)]
    out: PathBuf,
}

#[derive(Debug, Args)]
pub(crate) struct ClaimCheckArgs {
    /// Run spec naming the model and the parameters the contract requires.
    #[arg(long, default_value = inference_artifacts::INFERENCE_RUN_SPEC_TOML)]
    run: PathBuf,

    /// The `claim_submission.json` that `claim build` wrote.
    #[arg(long)]
    submission: PathBuf,

    /// A Raster chain run directory of the same claim. With it, every
    /// checkpoint of the claim is held to the boundary that run committed:
    /// the same program, input manifest and output at every stage.
    #[arg(long = "raster-run")]
    raster_run: Option<PathBuf>,

    /// The pins the contract holds, as `iris claim pins` wrote them. Without
    /// it a `single` contract's are the model import's, and those of a
    /// pipeline over several models are made from the imports now.
    #[arg(long)]
    pins: Option<PathBuf>,

    /// The registry of configurations, as `iris registry add` writes it. The
    /// claim is judged by the configuration it names there. Without it, by
    /// the one the run spec and the pins describe.
    #[arg(long)]
    registry: Option<PathBuf>,

    /// The stand-in data-availability layer's directory. Default:
    /// `IRIS_DA_DIR`, or `~/.cache/iris/stub-celestia`.
    #[arg(long = "da-dir")]
    da_dir: Option<PathBuf>,

    /// When the claim is taken to be submitted, in Unix seconds. Default: now.
    #[arg(long = "submitted-at")]
    submitted_at: Option<u64>,
}

#[derive(Debug, Args)]
pub(crate) struct ClaimBuildArgs {
    /// Run spec to execute.
    #[arg(long, default_value = inference_artifacts::INFERENCE_RUN_SPEC_TOML)]
    run: PathBuf,

    /// Debug: claim these comma-separated output ids
    /// instead of discovering them, to build claims that should be refuted.
    #[arg(long = "claimed-ids", value_delimiter = ',', hide = true)]
    claimed_ids: Option<Vec<u32>>,

    /// The comma-separated output ids `iris infer` printed for this same run
    /// spec. The claim is built from them without running direct inference
    /// again; the chain still re-derives every token and rejects any other ids.
    #[arg(
        long = "direct-output-ids",
        value_delimiter = ',',
        conflicts_with = "claimed_ids"
    )]
    direct_output_ids: Option<Vec<u32>>,

    /// Use the previous per-stage child-process staged-infer runner.
    #[arg(long = "staged-subprocess", alias = "direct-subprocess", hide = true)]
    staged_subprocess: bool,

    /// The address the claim is attributed to: `0x` and 40 lowercase hex
    /// digits. Every blob the claim posts names it. Default: `IRIS_CLAIMER`,
    /// or the zero address.
    #[arg(long)]
    claimer: Option<String>,
}

#[derive(Debug, Args)]
pub(crate) struct ClaimCorruptHashesArgs {
    /// Checkpoints hash list to copy and corrupt.
    #[arg(long = "checkpoints", required_unless_present = "submission")]
    checkpoints: Option<PathBuf>,

    /// An honest claim's `claim_submission.json`, in its directory. The
    /// lying claim is written to `--output`, a new directory, ready for
    /// `iris claim post`.
    #[arg(long, conflicts_with_all = ["checkpoints", "trace"], requires = "output")]
    submission: Option<PathBuf>,

    /// With `--submission`: the address the lying claim is attributed to.
    /// Default: the honest claim's.
    #[arg(long, requires = "submission")]
    claimer: Option<String>,

    /// With `--submission`: what the lying claim says falsely of the
    /// checkpoint. An output is a lie a replay shows. A program or an input
    /// is one a challenger points at (`iris claim audit`).
    #[arg(long, value_enum, default_value = "output", requires = "submission")]
    fault: ClaimFaultArg,

    /// With `--fault input`: the input to lie about. Default: the first the
    /// checkpoint names a producer for.
    #[arg(long, requires = "submission")]
    input: Option<String>,

    /// Optional output path for the corrupted hash list. With
    /// `--submission`, the lying claim's directory.
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

impl ClaimBuildArgs {
    pub(crate) fn into_options(self) -> Result<ClaimBuildOptions> {
        let current_exe = std::env::current_exe().context("failed to locate current executable")?;
        let staged_backend = if self.staged_subprocess {
            staged_infer::chain_runner::StagedExecutionBackend::Subprocess
        } else {
            staged_infer::chain_runner::StagedExecutionBackend::InProcess
        };
        let mut options =
            ClaimBuildOptions::from_current_dir(current_exe, staged_backend, self.run)?;
        options.claimed_token_ids = self.claimed_ids;
        options.direct_output_ids = self.direct_output_ids;
        options.claimer = self.claimer.or_else(|| std::env::var("IRIS_CLAIMER").ok());
        Ok(options)
    }
}

impl ClaimCorruptHashesArgs {
    pub(crate) fn selection(&self) -> Result<CorruptCheckpointSelection> {
        let selector_count = usize::from(self.index.is_some())
            + usize::from(self.stage.is_some())
            + usize::from(self.random);
        if selector_count > 1 {
            anyhow::bail!("claim corrupt accepts only one of --index, --stage, or --random");
        }
        Ok(if let Some(index) = self.index {
            CorruptCheckpointSelection::Index(index)
        } else if let Some(stage) = self.stage.clone() {
            CorruptCheckpointSelection::Stage(stage)
        } else {
            CorruptCheckpointSelection::Random {
                seed: self.seed.clone(),
            }
        })
    }

    pub(crate) fn into_options(self) -> Result<CorruptCheckpointHashesOptions> {
        let selection = self.selection()?;
        Ok(CorruptCheckpointHashesOptions {
            hashes_path: self
                .checkpoints
                .context("claim corrupt requires --checkpoints or --submission")?,
            trace_path: self.trace,
            output_path: self.output,
            selection,
        })
    }
}

pub(crate) fn run(command: ClaimCommand) -> Result<ExitCode> {
    match command {
        ClaimCommand::Build(args) => {
            warn_if_debug_build("claim build");
            let run_spec_path = args.run.clone();
            let result = build_claim(args.into_options()?)?;
            println!("pipeline: {}", result.pipeline);
            match (result.final_result.as_ref(), result.consensus.as_ref()) {
                (Some(first), Some(consensus)) => {
                    print_consensus(first, consensus);
                    for (inference, result) in [first, &consensus.second].into_iter().enumerate() {
                        println!("inference: {inference}");
                        print_infer_result(result, Some(&run_spec_path), inference);
                    }
                }
                (Some(final_result), None) => {
                    print_infer_result(final_result, Some(&run_spec_path), 0)
                }
                (None, _) => {}
            }
            println!(
                "checkpoint trace: {}",
                result.checkpoint_trace_path.display()
            );
            println!("checkpoints: {}", result.checkpoint_hashes_path.display());
            println!("claim bundle: {}", result.claim_bundle_path.display());
            if let Some(path) = result.claim_submission_path.as_ref() {
                println!("claim submission: {}", path.display());
                if path
                    .with_file_name(availability::POSTING_OUTPUTS_JSON)
                    .exists()
                {
                    println!("posting: prepared, not posted (`iris claim post`)");
                }
            }
            Ok(ExitCode::SUCCESS)
        }
        ClaimCommand::Post(args) => {
            let base_dir =
                std::env::current_dir().context("failed to resolve current directory")?;
            let submission = absolute(&base_dir, &args.submission);
            let da_dir = args.da_dir.map(|dir| absolute(&base_dir, &dir));
            let posted = contract::post_claim_submission(&submission, da_dir.as_deref(), None)?;
            for blob in &posted.blobs {
                println!(
                    "blob: height {} commitment {} ({} bytes)",
                    blob.pointer.height, blob.pointer.commitment, blob.length
                );
            }
            println!("claim submission: {}", submission.display());
            Ok(match posted.rejection {
                None => {
                    println!("ok        data posted, proven and retrievable through the challenge period");
                    ExitCode::SUCCESS
                }
                Some(reason) => {
                    println!("REJECTED  data posted: {reason}");
                    ExitCode::FAILURE
                }
            })
        }
        ClaimCommand::Check(args) => {
            let report = check_claim(ClaimCheckOptions {
                base_dir: std::env::current_dir().context("failed to resolve current directory")?,
                run_spec_path: args.run,
                submission_path: args.submission,
                raster_run: args.raster_run,
                pins_path: args.pins,
                registry_path: args.registry,
                da_dir: args.da_dir,
                submitted_at: args.submitted_at,
            })?;
            for line in &report.lines {
                match &line.rejection {
                    None => println!("ok        {}", line.check),
                    Some(reason) => println!("REJECTED  {}: {reason}", line.check),
                }
            }
            Ok(if report.accepted() {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            })
        }
        ClaimCommand::Pins(args) => {
            let base_dir =
                std::env::current_dir().context("failed to resolve current directory")?;
            let pins = contract::run_pins(&base_dir, &args.run)?;
            inference_artifacts::write_json(&args.out, &pins)?;
            println!("pipeline: {}", pins.pipeline);
            for model in &pins.models {
                println!("model id: {}", model.model_id);
            }
            println!("stage programs: {}", pins.programs.stage_programs.len());
            println!("contract pins: {}", args.out.display());
            Ok(ExitCode::SUCCESS)
        }
        ClaimCommand::Fetch(args) => {
            let base_dir =
                std::env::current_dir().context("failed to resolve current directory")?;
            let fetched = posted::fetch_claim(posted::FetchClaimOptions {
                submission_path: absolute(&base_dir, &args.submission),
                run_spec_path: absolute(&base_dir, &args.run),
                pins_path: args.pins.map(|path| absolute(&base_dir, &path)),
                da_dir: args.da_dir.map(|path| absolute(&base_dir, &path)),
                out_dir: absolute(&base_dir, &args.out),
            })?;
            println!("blobs read: {} ({} bytes)", fetched.blobs, fetched.bytes);
            println!("checkpoints: {}", fetched.checkpoints);
            println!("input manifests: {}", fetched.input_manifests);
            println!("claim submission: {}", fetched.submission_path.display());
            println!("run spec: {}", fetched.run_spec_path.display());
            Ok(ExitCode::SUCCESS)
        }
        ClaimCommand::Audit(args) => {
            let base_dir =
                std::env::current_dir().context("failed to resolve current directory")?;
            let audit = audit::audit_claim(audit::AuditClaimOptions {
                submission_path: absolute(&base_dir, &args.submission),
                run_spec_path: absolute(&base_dir, &args.run),
                pins_path: args.pins.map(|path| absolute(&base_dir, &path)),
            })?;
            println!("checkpoints: {}", audit.checkpoints);
            println!("programs approved: {}", audit.programs_approved);
            println!("inputs held to their origin: {}", audit.inputs_linked);
            println!("faults: {}", audit.faults.len());
            for fault in &audit.faults {
                let input = fault
                    .input
                    .as_ref()
                    .map(|input| format!(", input `{input}`"))
                    .unwrap_or_default();
                println!(
                    "fault: checkpoint {} (`{}`){input}: {} (call {})",
                    fault.checkpoint, fault.stage, fault.fault, fault.call
                );
            }
            if let Some(calls) = args.calls {
                let calls = absolute(&base_dir, &calls);
                inference_artifacts::write_json(&calls, &audit.calls)?;
                println!("calls: {}", calls.display());
            }
            Ok(ExitCode::SUCCESS)
        }
        ClaimCommand::Corrupt(args) if args.submission.is_some() => {
            let base_dir =
                std::env::current_dir().context("failed to resolve current directory")?;
            let selection = args.selection()?;
            anyhow::ensure!(
                args.input.is_none()
                    || matches!(
                        args.fault,
                        ClaimFaultArg::Input | ClaimFaultArg::Arrangement
                    ),
                "--input names the input of a `--fault input` or `--fault arrangement` lie"
            );
            let lying = lies::corrupt_claim(lies::CorruptClaimOptions {
                submission_path: absolute(&base_dir, &args.submission.expect("checked above")),
                selection,
                fault: match args.fault {
                    ClaimFaultArg::Output => lies::CorruptFault::Output,
                    ClaimFaultArg::Program => lies::CorruptFault::Program,
                    ClaimFaultArg::Input => lies::CorruptFault::Input(args.input),
                    ClaimFaultArg::Arrangement => lies::CorruptFault::Arrangement(args.input),
                    ClaimFaultArg::Verdict => lies::CorruptFault::Verdict,
                },
                claimer: args.claimer,
                out_dir: absolute(&base_dir, &args.output.expect("clap requires it")),
            })?;
            println!("corrupted checkpoints: {}", lying.hashes_path.display());
            println!("corruption manifest: {}", lying.manifest_path.display());
            println!("corrupted checkpoint index: {}", lying.checkpoint_index);
            println!(
                "corrupted checkpoint number: {}",
                lying.checkpoint_index + 1
            );
            println!("corrupted stage: {}", lying.stage);
            println!("original hash: {}", lying.original_hash);
            println!("corrupted hash: {}", lying.corrupted_hash);
            println!("claim submission: {}", lying.submission_path.display());
            Ok(ExitCode::SUCCESS)
        }
        ClaimCommand::Corrupt(args) => {
            let result = corrupt_checkpoint_hashes(args.into_options()?)?;
            println!(
                "corrupted checkpoints: {}",
                result.corrupted_hashes_path.display()
            );
            println!(
                "corruption manifest: {}",
                result.corruption_manifest_path.display()
            );
            println!("corrupted checkpoint index: {}", result.checkpoint_index);
            println!("corrupted checkpoint number: {}", result.checkpoint_number);
            if let Some(stage) = result.stage.as_ref() {
                println!("corrupted stage: {stage}");
            }
            println!("original hash: {}", result.original_hash);
            println!("corrupted hash: {}", result.corrupted_hash);
            Ok(ExitCode::SUCCESS)
        }
    }
}
