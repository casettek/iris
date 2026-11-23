//! The calldata of the contracts' calls: `iris evm`.

use crate::*;

#[derive(Debug, Subcommand)]
pub(crate) enum EvmCommand {
    /// The call that registers a run spec's configuration.
    Configuration {
        #[arg(long, default_value = inference_artifacts::INFERENCE_RUN_SPEC_TOML)]
        run: PathBuf,
        #[arg(long)]
        pins: Option<PathBuf>,
    },
    /// The call that submits a claim.
    Claim {
        /// The claim's `claim_submission.json`, posted and attested.
        #[arg(long)]
        submission: PathBuf,
    },
    /// The call that submits a challenge.
    Challenge {
        /// The `challenge_submission.json` that `challenge post` wrote,
        /// attested.
        #[arg(long)]
        challenge: PathBuf,
        /// The disputed claim's `claim_submission.json`, in its directory.
        #[arg(long)]
        submission: PathBuf,
    },
    /// The call that answers a challenge with a Raster fault proof of its
    /// trace. Without a challenge: the proof's journal and seal as a contract
    /// takes them, and what a contract reads from the journal.
    FaultProof {
        /// The `fault.receipt` that `fault prove` wrote.
        #[arg(long)]
        receipt: PathBuf,
        /// The answered challenge's `challenge_submission.json`.
        #[arg(long, requires = "submission")]
        challenge: Option<PathBuf>,
        /// The disputed claim's `claim_submission.json`, in its directory.
        #[arg(long, requires = "challenge")]
        submission: Option<PathBuf>,
    },
    /// The ids of Raster's proof programs that a contract pins.
    ProofPrograms,
}

pub(crate) fn run(command: EvmCommand) -> Result<ExitCode> {
    {
        let base_dir = std::env::current_dir().context("failed to resolve current directory")?;
        let call = match command {
            EvmCommand::Configuration { run, pins } => {
                evm::configuration_call(&checks::spec_configuration(
                    &absolute(&base_dir, &run),
                    pins.map(|path| absolute(&base_dir, &path)).as_deref(),
                )?)?
            }
            EvmCommand::Claim { submission } => evm::claim_call(&absolute(&base_dir, &submission))?,
            EvmCommand::Challenge {
                challenge,
                submission,
            } => evm::challenge_call(
                &absolute(&base_dir, &challenge),
                &absolute(&base_dir, &submission),
            )?,
            EvmCommand::FaultProof {
                receipt,
                challenge: Some(challenge),
                submission: Some(submission),
            } => evm::fault_proof_call(
                &absolute(&base_dir, &challenge),
                &absolute(&base_dir, &submission),
                &absolute(&base_dir, &receipt),
            )?,
            EvmCommand::FaultProof { receipt, .. } => {
                evm::fault_proof_parts(&absolute(&base_dir, &receipt))?
            }
            EvmCommand::ProofPrograms => evm::proof_programs(),
        };
        println!("{}", serde_json::to_string_pretty(&call)?);
        Ok(ExitCode::SUCCESS)
    }
}
