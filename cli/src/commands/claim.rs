//! A claim: `iwebs claim`.

use super::infer::{print_consensus, print_infer_result};
use crate::*;

#[derive(Debug, Subcommand)]
pub(crate) enum ClaimCommand {
    /// Build a checkpoint claim from a pipeline run.
    Build(ClaimBuildArgs),
    /// Read a submitted claim's data back from the data-availability layer
    /// into a new directory, with the case it posted: what a challenger
    /// starts from.
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
    /// Write what a contract of a run's procedure kind over its models would pin.
    Pins(ClaimPinsArgs),
}

#[derive(Debug, Args)]
pub(crate) struct ClaimPostArgs {
    /// The `claim_submission.json` that `claim build` wrote. The blobs to
    /// post are beside it (`posting-blob-<n>.bin`).
    #[arg(long)]
    submission: PathBuf,

    /// The stand-in data-availability layer's directory. Default:
    /// `IWEBS_DA_DIR`, or `~/.cache/iwebs/stub-celestia`.
    #[arg(long = "da-dir")]
    da_dir: Option<PathBuf>,
}

#[derive(Debug, Args)]
pub(crate) struct ClaimAuditArgs {
    /// The claim's `claim_submission.json`, in the directory `iwebs claim
    /// fetch` wrote: with its checkpoint list, checkpoints and input lists.
    #[arg(long)]
    submission: PathBuf,

    /// A specification of the procedure the claim names.
    #[arg(long, default_value = procedure_definition::SPECIFICATION_TOML)]
    specification: PathBuf,

    /// The pins the contract holds. As for `iwebs claim check`.
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

#[derive(Debug, Args)]
pub(crate) struct ClaimFetchArgs {
    /// The claim's `claim_submission.json` as it was submitted, with where
    /// its data is.
    #[arg(long)]
    submission: PathBuf,

    /// Your own specification of the procedure the claim names: its
    /// procedure kind, models and parameters, and for a claim with a source the
    /// recipe and the signer's key. The claim's procedure is held to it
    /// before anything is read.
    #[arg(long, default_value = procedure_definition::SPECIFICATION_TOML)]
    specification: PathBuf,

    /// The pins the contract holds. As for `iwebs claim check`.
    #[arg(long)]
    pins: Option<PathBuf>,

    /// The stand-in data-availability layer's directory. As for `iwebs claim
    /// check`.
    #[arg(long = "da-dir")]
    da_dir: Option<PathBuf>,

    /// A new directory to write the claim's files and its `case.toml` to.
    #[arg(long)]
    out: PathBuf,
}

#[derive(Debug, Args)]
pub(crate) struct ClaimPinsArgs {
    /// Specification naming the procedure kind and its models.
    #[arg(long, default_value = procedure_definition::SPECIFICATION_TOML)]
    specification: PathBuf,

    /// Where to write the pins. They are made from the models' imports and
    /// the stage projects' locks as they are now, so run this from the IWEBS
    /// root.
    #[arg(long)]
    out: PathBuf,
}

#[derive(Debug, Args)]
pub(crate) struct ClaimCheckArgs {
    /// Specification naming the models and the parameters the contract requires.
    #[arg(long, default_value = procedure_definition::SPECIFICATION_TOML)]
    specification: PathBuf,

    /// The `claim_submission.json` that `claim build` wrote.
    #[arg(long)]
    submission: PathBuf,

    /// A Raster pipeline run directory of the same claim. With it, every
    /// checkpoint of the claim is held to the boundary that run committed:
    /// the same program, input manifest and output at every stage.
    #[arg(long = "raster-run")]
    raster_run: Option<PathBuf>,

    /// The pins the contract holds, as `iwebs claim pins` wrote them. Without
    /// it a `single` contract's are the model import's, and those of a
    /// procedure kind over several models are made from the imports now.
    #[arg(long)]
    pins: Option<PathBuf>,

    /// The registry of procedures, as `iwebs registry add` writes it. The
    /// claim is judged by the procedure it names there. Without it, by
    /// the one the specification and the pins describe.
    #[arg(long)]
    registry: Option<PathBuf>,

    /// The stand-in data-availability layer's directory. Default:
    /// `IWEBS_DA_DIR`, or `~/.cache/iwebs/stub-celestia`.
    #[arg(long = "da-dir")]
    da_dir: Option<PathBuf>,

    /// When the claim is taken to be submitted, in Unix seconds. Default: now.
    #[arg(long = "submitted-at")]
    submitted_at: Option<u64>,
}

#[derive(Debug, Args)]
pub(crate) struct ClaimBuildArgs {
    /// The specification of the procedure the claim is of.
    #[arg(long, default_value = procedure_definition::SPECIFICATION_TOML)]
    pub(crate) specification: PathBuf,

    /// The case the claim is about: the prompt, or the documents of the
    /// specification's source.
    #[arg(long, default_value = procedure_definition::CASE_TOML)]
    pub(crate) case: PathBuf,

    /// The output the claim states, as comma-separated token ids, for a
    /// procedure kind of one model. It is the case's `[[output]]` table given
    /// here instead. The pipeline holds every token to the model's own
    /// choice, and a claim it rejects is not written.
    #[arg(
        long = "output-ids",
        alias = "native-output-ids",
        value_delimiter = ','
    )]
    native_output_ids: Option<Vec<u32>>,

    /// Run each native stage in a child process of its own, as the previous runner did.
    #[arg(long = "native-stage-subprocess", hide = true)]
    native_stage_subprocess: bool,

    /// The address the claim is attributed to: `0x` and 40 lowercase hex
    /// digits. Every blob the claim posts names it. Default: `IWEBS_CLAIMER`,
    /// or the zero address.
    #[arg(long)]
    claimer: Option<String>,
}

impl ClaimBuildArgs {
    pub(crate) fn into_options(self) -> Result<ClaimBuildOptions> {
        let current_exe = std::env::current_exe().context("failed to locate current executable")?;
        let native_stage_backend = if self.native_stage_subprocess {
            procedure_pipeline::pipeline_runner::NativeStageBackend::Subprocess
        } else {
            procedure_pipeline::pipeline_runner::NativeStageBackend::InProcess
        };
        let mut options = ClaimBuildOptions::from_current_dir(
            current_exe,
            native_stage_backend,
            self.specification,
            self.case,
        )?;
        options.native_output_ids = self.native_output_ids;
        options.claimer = self.claimer.or_else(|| std::env::var("IWEBS_CLAIMER").ok());
        Ok(options)
    }
}

/// Build a claim and print what it states and where it is.
pub(crate) fn build(options: ClaimBuildOptions, specification_path: PathBuf) -> Result<ExitCode> {
    warn_if_debug_build("claim build");
    // A claim is of a verification procedure, which is given the output it
    // verifies. Nothing here finds one.
    if options.claimed_token_ids.is_none() && options.native_output_ids.is_none() {
        if let Some(case_path) = &options.case_path {
            let case = procedure_definition::read_case(&absolute(&options.base_dir, case_path))?;
            anyhow::ensure!(
                !case.outputs.is_empty(),
                "{} states no output, and a claim states one: give each model's output in the \
                 case as an `[[output]]` table with its `ids`, or one model's with \
                 `--output-ids`. A generation procedure finds an output (`iwebs run`)",
                case_path.display()
            );
        }
    }
    let result = build_claim(options)?;
    println!("procedure kind: {}", result.procedure_kind);
    match (result.results.as_slice(), result.consensus.as_ref()) {
        ([first, second], Some(consensus)) => {
            print_consensus(first, second, consensus);
            for (inference, result) in [first, second].into_iter().enumerate() {
                println!("inference: {inference}");
                print_infer_result(result, Some(&specification_path), inference);
            }
        }
        ([only], _) => print_infer_result(only, Some(&specification_path), 0),
        (results, _) => {
            if !results.is_empty() {
                println!("result of inference: {}", result.procedure_kind.answer());
            }
            for (inference, result) in results.iter().enumerate() {
                println!("inference: {inference}");
                print_infer_result(result, Some(&specification_path), inference);
            }
        }
    }
    println!("checkpoint list: {}", result.checkpoint_list_path.display());
    println!("checkpoints: {}", result.checkpoint_hashes_path.display());
    println!("claim bundle: {}", result.claim_bundle_path.display());
    if let Some(path) = result.claim_submission_path.as_ref() {
        println!("claim submission: {}", path.display());
        if path.with_file_name(posting::POSTING_OUTPUTS_JSON).exists() {
            println!("posting: prepared, not posted (`iwebs claim post`)");
        }
    }
    Ok(ExitCode::SUCCESS)
}

pub(crate) fn run(command: ClaimCommand) -> Result<ExitCode> {
    match command {
        ClaimCommand::Build(args) => {
            let specification_path = args.specification.clone();
            build(args.into_options()?, specification_path)
        }
        ClaimCommand::Post(args) => {
            let base_dir =
                std::env::current_dir().context("failed to resolve current directory")?;
            let submission = absolute(&base_dir, &args.submission);
            let da_dir = args.da_dir.map(|dir| absolute(&base_dir, &dir));
            let posted = claim_check::post_claim_submission(&submission, da_dir.as_deref(), None)?;
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
                specification_path: args.specification,
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
            let pins = claim_check::run_pins(&base_dir, &args.specification)?;
            procedure_definition::write_json(&args.out, &pins)?;
            println!("procedure kind: {}", pins.procedure_kind);
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
                specification_path: absolute(&base_dir, &args.specification),
                pins_path: args.pins.map(|path| absolute(&base_dir, &path)),
                da_dir: args.da_dir.map(|path| absolute(&base_dir, &path)),
                out_dir: absolute(&base_dir, &args.out),
            })?;
            println!("blobs read: {} ({} bytes)", fetched.blobs, fetched.bytes);
            println!("checkpoints: {}", fetched.checkpoints);
            println!("input manifests: {}", fetched.input_manifests);
            println!("claim submission: {}", fetched.submission_path.display());
            println!("case: {}", fetched.case_path.display());
            Ok(ExitCode::SUCCESS)
        }
        ClaimCommand::Audit(args) => {
            let base_dir =
                std::env::current_dir().context("failed to resolve current directory")?;
            let audit = audit::audit_claim(audit::AuditClaimOptions {
                submission_path: absolute(&base_dir, &args.submission),
                specification_path: absolute(&base_dir, &args.specification),
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
                procedure_definition::write_json(&calls, &audit.calls)?;
                println!("calls: {}", calls.display());
            }
            Ok(ExitCode::SUCCESS)
        }
    }
}
