//! A challenge: `iris challenge`.

use crate::*;

#[derive(Debug, Args)]
pub(crate) struct ChallengePostArgs {
    /// The `challenge_bundle.json` that `challenge build` wrote.
    #[arg(long)]
    challenge: PathBuf,

    /// The claim's `claim_submission.json`. The claim's checkpoints are read
    /// from the directory it is in.
    #[arg(long)]
    submission: PathBuf,

    /// Specification naming the procedure kind and models of the contract.
    #[arg(long, default_value = procedure_definition::SPECIFICATION_TOML)]
    specification: PathBuf,

    /// The pins the contract holds. As for `iris claim check`.
    #[arg(long)]
    pins: Option<PathBuf>,

    /// The address the challenge is attributed to, as `claim build
    /// --claimer`. Default: `IRIS_CHALLENGER`, or the zero address.
    #[arg(long)]
    challenger: Option<String>,

    /// The stand-in data-availability layer's directory. As for `iris claim
    /// check`.
    #[arg(long = "da-dir")]
    da_dir: Option<PathBuf>,

    /// When the challenge is taken to be submitted, in Unix seconds. Default:
    /// now.
    #[arg(long = "submitted-at")]
    submitted_at: Option<u64>,

    /// Post this trace commitment in place of the bundle's own: the copy
    /// `iris test corrupt-challenge` wrote, for a challenge a defender can refute.
    #[arg(long)]
    commitment: Option<PathBuf>,

    /// Where to write the challenge submission. Default:
    /// `challenge_submission.json` beside the bundle.
    #[arg(long)]
    out: Option<PathBuf>,
}

#[derive(Debug, Args)]
pub(crate) struct ChallengeFetchArgs {
    /// The `challenge_submission.json` as it was submitted, with where its
    /// trace body is.
    #[arg(long)]
    challenge: PathBuf,

    /// The stand-in data-availability layer's directory. As for `iris claim
    /// check`.
    #[arg(long = "da-dir")]
    da_dir: Option<PathBuf>,

    /// The file to write the trace commitment to (a `commit.bin`).
    #[arg(long)]
    out: PathBuf,
}

#[derive(Debug, Args)]
pub(crate) struct ChallengeCheckArgs {
    /// The `challenge_submission.json` that `challenge post` wrote.
    #[arg(long)]
    challenge: PathBuf,

    /// The disputed claim's `claim_submission.json`, in its directory.
    #[arg(long)]
    submission: PathBuf,

    /// Specification naming the procedure kind and models of the contract.
    #[arg(long, default_value = procedure_definition::SPECIFICATION_TOML)]
    specification: PathBuf,

    /// The pins the contract holds. As for `iris claim check`.
    #[arg(long)]
    pins: Option<PathBuf>,

    /// The stand-in data-availability layer's directory. As for `iris claim
    /// check`.
    #[arg(long = "da-dir")]
    da_dir: Option<PathBuf>,

    /// When the challenge is taken to be submitted, in Unix seconds. Default:
    /// now.
    #[arg(long = "submitted-at")]
    submitted_at: Option<u64>,
}

#[derive(Debug, Subcommand)]
pub(crate) enum ChallengeCommand {
    /// Build a challenge bundle for a divergent claim.
    Build(ChallengeBuildArgs),
    /// Post a challenge's trace body to the data-availability layer, write
    /// the challenge submission, and make the checks a contract would make
    /// on it.
    Post(ChallengePostArgs),
    /// Read the trace a submitted challenge names back from the
    /// data-availability layer: what a defender audits its own run against.
    Fetch(ChallengeFetchArgs),
    /// Make the checks a contract would make on a challenge's submission as
    /// it is. Nothing is posted.
    Check(ChallengeCheckArgs),
}

#[derive(Debug, Args)]
pub(crate) struct ChallengeBuildArgs {
    /// Your own specification of the procedure the claim names. Its models
    /// are the ones the claim is verified against.
    #[arg(long, default_value = procedure_definition::SPECIFICATION_TOML)]
    specification: PathBuf,

    /// For a test that has the claimer's own directory: the claim's
    /// checkpoint list. A challenger of a submitted claim gives
    /// `--submission`.
    #[arg(long = "checkpoints", requires = "claim_context")]
    checkpoint_hashes: Option<PathBuf>,

    /// With `--checkpoints`: the claimer's `claim_bundle.json`, for the run
    /// the claim was built from.
    #[arg(long = "claim-context", requires = "checkpoint_hashes")]
    claim_context: Option<PathBuf>,

    /// A submitted claim as `iris claim fetch` wrote it: its
    /// `claim_submission.json`, with its checkpoint list and the `case.toml`
    /// that command wrote beside it. Nothing of the claimer's own run is
    /// read.
    #[arg(long, conflicts_with_all = ["checkpoint_hashes", "claim_context"])]
    submission: Option<PathBuf>,

    /// With `--submission`: the comma-separated output ids your own `iris
    /// infer` printed for your specification and the fetched case, taken in
    /// place of running direct inference again.
    #[arg(
        long = "direct-output-ids",
        value_delimiter = ',',
        requires = "submission"
    )]
    direct_output_ids: Option<Vec<u32>>,

    /// Use the previous per-stage child-process staged-infer runner.
    #[arg(long = "staged-subprocess", alias = "direct-subprocess", hide = true)]
    staged_subprocess: bool,
}

pub(crate) fn print_challenge_result(result: &ChallengeBuildResult) {
    match &result.outcome {
        ChallengeBuildOutcome::NoDivergence {
            claimed_reference_path,
            recomputed_list_path,
            recomputed_pipeline_dir,
        } => {
            println!("no divergence found");
            println!("claimed reference: {}", claimed_reference_path.display());
            println!("recomputed list: {}", recomputed_list_path.display());
            println!("recomputed pipeline: {}", recomputed_pipeline_dir.display());
        }
        ChallengeBuildOutcome::ClaimedOutputRejected {
            reason,
            claimed_reference_path,
            recomputed_list_path,
            recomputed_pipeline_dir,
        } => {
            println!("no divergence found; claimed output rejected");
            println!("reason: {reason}");
            println!("claimed reference: {}", claimed_reference_path.display());
            println!("recomputed list: {}", recomputed_list_path.display());
            println!("recomputed pipeline: {}", recomputed_pipeline_dir.display());
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

impl ChallengeBuildArgs {
    pub(crate) fn into_options(self) -> Result<ChallengeBuildOptions> {
        let posted = self
            .submission
            .map(|submission_path| -> Result<ChallengeInput> {
                Ok(ChallengeInput::Posted {
                    submission_path: absolute(
                        &std::env::current_dir().context("failed to resolve current directory")?,
                        &submission_path,
                    ),
                })
            });
        let input = match (posted, self.checkpoint_hashes, self.claim_context) {
            (Some(posted), None, None) => posted?,
            (None, Some(checkpoint_hashes_path), Some(claim_context_path)) => {
                ChallengeInput::CheckpointHashes {
                    claim_context_path,
                    checkpoint_hashes_path,
                }
            }
            _ => anyhow::bail!(
                "challenge build requires --submission <claim_submission.json>, or for a claim \
                 in its claimer's directory --checkpoints <checkpoints.txt> --claim-context \
                 <claim_bundle.json>"
            ),
        };
        let current_exe = std::env::current_exe().context("failed to locate current executable")?;
        let staged_backend = if self.staged_subprocess {
            procedure_pipeline::pipeline_runner::StagedExecutionBackend::Subprocess
        } else {
            procedure_pipeline::pipeline_runner::StagedExecutionBackend::InProcess
        };
        let mut options = ChallengeBuildOptions::from_current_dir(
            current_exe,
            staged_backend,
            input,
            self.specification,
        )?;
        options.direct_output_ids = self.direct_output_ids;
        Ok(options)
    }
}

pub(crate) fn run(command: ChallengeCommand) -> Result<ExitCode> {
    match command {
        ChallengeCommand::Build(args) => {
            warn_if_debug_build("challenge build");
            let result = build_challenge(args.into_options()?)?;
            print_challenge_result(&result);
            Ok(ExitCode::SUCCESS)
        }
        ChallengeCommand::Post(args) => {
            let base_dir =
                std::env::current_dir().context("failed to resolve current directory")?;
            let posted = claim_check::post_challenge(claim_check::ChallengePostOptions {
                specification_path: absolute(&base_dir, &args.specification),
                pins_path: args.pins.map(|path| absolute(&base_dir, &path)),
                challenge_bundle_path: absolute(&base_dir, &args.challenge),
                submission_path: absolute(&base_dir, &args.submission),
                challenger: args
                    .challenger
                    .or_else(|| std::env::var("IRIS_CHALLENGER").ok()),
                da_dir: args.da_dir.map(|path| absolute(&base_dir, &path)),
                submitted_at: args.submitted_at,
                commitment_path: args.commitment.map(|path| absolute(&base_dir, &path)),
                out_path: args.out.map(|path| absolute(&base_dir, &path)),
            })?;
            for line in &posted.report.lines {
                match &line.rejection {
                    None => println!("ok        {}", line.check),
                    Some(reason) => println!("REJECTED  {}: {reason}", line.check),
                }
            }
            println!("challenge submission: {}", posted.submission_path.display());
            Ok(if posted.report.accepted() {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            })
        }
        ChallengeCommand::Fetch(args) => {
            let base_dir =
                std::env::current_dir().context("failed to resolve current directory")?;
            let fetched = posted::fetch_challenge(posted::FetchChallengeOptions {
                challenge_path: absolute(&base_dir, &args.challenge),
                da_dir: args.da_dir.map(|path| absolute(&base_dir, &path)),
                out_path: absolute(&base_dir, &args.out),
            })?;
            println!("trace header: {}", fetched.header_digest);
            println!(
                "trace commitment: {} ({} bytes)",
                fetched.commitment_path.display(),
                fetched.bytes
            );
            Ok(ExitCode::SUCCESS)
        }
        ChallengeCommand::Check(args) => {
            let base_dir =
                std::env::current_dir().context("failed to resolve current directory")?;
            let report =
                claim_check::check_challenge_submission(claim_check::ChallengeCheckOptions {
                    specification_path: absolute(&base_dir, &args.specification),
                    pins_path: args.pins.map(|path| absolute(&base_dir, &path)),
                    challenge_path: absolute(&base_dir, &args.challenge),
                    submission_path: absolute(&base_dir, &args.submission),
                    da_dir: args.da_dir.map(|path| absolute(&base_dir, &path)),
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
    }
}
