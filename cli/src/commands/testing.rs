//! The tools a test stages a dispute with: `iris test`.
//!
//! Nothing here is anything an honest party runs. A claimer runs `iris claim`,
//! a challenger `iris claim fetch` and `iris challenge`, a defender `iris
//! fault`. These make the wrong claims and challenges those commands must
//! catch, measure what catching them costs, and drive the local blockchains.

use super::claim::ClaimBuildArgs;
use super::devstack::DevstackCommand;
use crate::*;

#[derive(Debug, Subcommand)]
pub(crate) enum TestCommand {
    /// Run the pinned three-path end-to-end parity suite.
    Parity {
        /// Leave out the challenge scenario, whose authenticated replays take about an
        /// hour. The report records that it was skipped.
        #[arg(long)]
        skip_challenges: bool,
        /// The bundle under `model-bundles` to run: one family's small model.
        #[arg(long, default_value = "parity-gemma")]
        bundle: String,
    },
    /// Build a claim of the output ids given, whatever the model's own
    /// output is: a claim that should be refuted. Otherwise as `iris claim
    /// build`.
    Claim {
        /// The comma-separated output ids to claim.
        #[arg(long, value_delimiter = ',', required = true)]
        ids: Vec<u32>,
        #[command(flatten)]
        build: ClaimBuildArgs,
    },
    /// Write the claim a lying claimer would submit: an honest claim with one
    /// checkpoint false, under its own checkpoint list, ready for `iris
    /// claim post`.
    CorruptClaim(CorruptClaimArgs),
    /// Copy a checkpoint hash list with one hash changed, for a challenge
    /// built from the claimer's own directory (`challenge build
    /// --checkpoints`). A changed hash has no checkpoint behind it, so this
    /// is no claim a contract could hold: for that, `corrupt-claim`.
    CorruptCheckpoints(CorruptCheckpointsArgs),
    /// Flip an interior fingerprint bit in a copy of a challenge's trace: the
    /// challenge a defender can refute with a fault proof.
    CorruptChallenge(fault::CorruptArgs),
    /// Measure guest cycles and select a heavy fault window from a completed challenge.
    Profile(challenge_profile::ProfileArgs),
    /// Drive the local blockchains (blockchain/devstack/README.md): send
    /// iris's calls to the contracts deployed there, and hold the contracts
    /// to iris.
    /// Transactions go out through an account the development node keeps
    /// unlocked, so this works against a development node and nothing else.
    Devstack {
        #[command(subcommand)]
        command: DevstackCommand,
    },
}

/// Which checkpoint to make false. One at random when none is named.
#[derive(Debug, Args)]
pub(crate) struct CheckpointSelectionArgs {
    /// The checkpoint at this zero-based index.
    #[arg(long)]
    index: Option<usize>,

    /// The checkpoint with this stage name.
    #[arg(long)]
    stage: Option<String>,

    /// A checkpoint picked at random. This is the default.
    #[arg(long)]
    random: bool,

    /// Seed for a reproducible random pick.
    #[arg(long)]
    seed: Option<String>,
}

impl CheckpointSelectionArgs {
    fn selection(&self) -> Result<CorruptCheckpointSelection> {
        let selector_count = usize::from(self.index.is_some())
            + usize::from(self.stage.is_some())
            + usize::from(self.random);
        if selector_count > 1 {
            anyhow::bail!("give only one of --index, --stage, or --random");
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
    /// An input the claim may name, in a place the pipeline does not give it:
    /// the value an earlier stage of the same program is given.
    Arrangement,
    /// The check's record stated valid where the check does not find it so,
    /// with the result stage run on that record. Of a claim whose claimed
    /// output is not the model's (`iris test claim --ids`).
    Verdict,
}

#[derive(Debug, Args)]
pub(crate) struct CorruptClaimArgs {
    /// An honest claim's `claim_submission.json`, in its directory.
    #[arg(long)]
    submission: PathBuf,

    /// The lying claim's directory, which must be new.
    #[arg(long)]
    output: PathBuf,

    /// The address the lying claim is attributed to. Default: the honest
    /// claim's.
    #[arg(long)]
    claimer: Option<String>,

    /// What the lying claim says falsely of the checkpoint. An output, an
    /// arrangement or a verdict is a lie a replay shows. A program or an
    /// input is one a challenger points at (`iris claim audit`).
    #[arg(long, value_enum, default_value = "output")]
    fault: ClaimFaultArg,

    /// With `--fault input` or `--fault arrangement`: the input to lie
    /// about. Default: the first the lie can be told of.
    #[arg(long)]
    input: Option<String>,

    #[command(flatten)]
    checkpoint: CheckpointSelectionArgs,
}

#[derive(Debug, Args)]
pub(crate) struct CorruptCheckpointsArgs {
    /// The checkpoint hash list to copy and change.
    #[arg(long)]
    checkpoints: PathBuf,

    /// Where to write the changed list. Default: beside the original.
    #[arg(long)]
    output: Option<PathBuf>,

    /// A checkpoint list, used only to find `--stage`.
    #[arg(long)]
    list: Option<PathBuf>,

    #[command(flatten)]
    checkpoint: CheckpointSelectionArgs,
}

pub(crate) fn run(command: TestCommand) -> Result<ExitCode> {
    match command {
        TestCommand::Parity {
            skip_challenges,
            bundle,
        } => {
            parity::run(skip_challenges, &bundle)?;
            Ok(ExitCode::SUCCESS)
        }
        TestCommand::Claim { ids, build } => {
            let specification_path = build.specification.clone();
            let mut options = build.into_options()?;
            anyhow::ensure!(
                options.native_output_ids.is_none(),
                "give the ids to claim with --ids alone: --native-output-ids names the model's \
                 own output"
            );
            options.claimed_token_ids = Some(ids);
            super::claim::build(options, specification_path)
        }
        TestCommand::CorruptClaim(args) => {
            let base_dir =
                std::env::current_dir().context("failed to resolve current directory")?;
            let selection = args.checkpoint.selection()?;
            anyhow::ensure!(
                args.input.is_none()
                    || matches!(
                        args.fault,
                        ClaimFaultArg::Input | ClaimFaultArg::Arrangement
                    ),
                "--input names the input of a `--fault input` or `--fault arrangement` lie"
            );
            let lying = lies::corrupt_claim(lies::CorruptClaimOptions {
                submission_path: absolute(&base_dir, &args.submission),
                selection,
                fault: match args.fault {
                    ClaimFaultArg::Output => lies::CorruptFault::Output,
                    ClaimFaultArg::Program => lies::CorruptFault::Program,
                    ClaimFaultArg::Input => lies::CorruptFault::Input(args.input),
                    ClaimFaultArg::Arrangement => lies::CorruptFault::Arrangement(args.input),
                    ClaimFaultArg::Verdict => lies::CorruptFault::Verdict,
                },
                claimer: args.claimer,
                out_dir: absolute(&base_dir, &args.output),
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
        TestCommand::CorruptCheckpoints(args) => {
            let result = corrupt_checkpoint_hashes(CorruptCheckpointHashesOptions {
                selection: args.checkpoint.selection()?,
                hashes_path: args.checkpoints,
                list_path: args.list,
                output_path: args.output,
            })?;
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
        TestCommand::CorruptChallenge(args) => {
            fault::corrupt(args)?;
            Ok(ExitCode::SUCCESS)
        }
        TestCommand::Profile(args) => {
            challenge_profile::run(args)?;
            Ok(ExitCode::SUCCESS)
        }
        TestCommand::Devstack { command } => super::devstack::run(command),
    }
}
