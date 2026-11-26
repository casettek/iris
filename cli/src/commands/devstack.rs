//! The local blockchains: `iwebs test devstack`, under the test tools.

use crate::blockchain::devstack;
use crate::*;

/// What a `devstack submit` sends.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub(crate) enum SubmitWhat {
    Claim,
    Challenge,
    FaultProof,
    Fault,
}

/// What a `devstack replay` holds the contracts to IWEBS on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub(crate) enum ReplayWhat {
    Claim,
    Challenge,
    Faults,
}

#[derive(Debug, Subcommand)]
pub(crate) enum DevstackCommand {
    /// Write the deployment file `blockchain/devstack/deploy.sh` leaves:
    /// where the contracts are, what they were deployed with, and their
    /// errors by selector.
    Deployment {
        /// The file to write.
        #[arg(long)]
        out: PathBuf,
        /// A field of the file, as `name=value`. A value that is a whole
        /// number is written as one.
        #[arg(long = "set", value_name = "NAME=VALUE")]
        fields: Vec<String>,
        /// What `forge inspect <contract> errors --json` printed for each
        /// contract, one after another.
        #[arg(long)]
        errors: PathBuf,
    },
    /// Send a claim, a challenge, a fault proof or a fault to the contracts,
    /// after attesting its blobs, relaying their block roots and
    /// registering the claim's procedure. Prints what the contract said
    /// and exits nonzero when it refused.
    Submit {
        what: SubmitWhat,
        /// The specification of the claim's procedure, to register it.
        #[arg(long)]
        specification: Option<PathBuf>,
        /// The claim's `claim_submission.json`.
        #[arg(long)]
        submission: Option<PathBuf>,
        /// The `challenge_submission.json` that `challenge post` wrote.
        #[arg(long)]
        challenge: Option<PathBuf>,
        /// The `fault.receipt` that `fault prove` wrote.
        #[arg(long)]
        receipt: Option<PathBuf>,
        /// The calls `iwebs claim audit --calls` wrote.
        #[arg(long)]
        calls: Option<PathBuf>,
        /// Which of those calls to send.
        #[arg(long)]
        call: Option<usize>,
        #[arg(long)]
        no_attest: bool,
        #[arg(long)]
        no_relay: bool,
        #[arg(long)]
        no_register: bool,
    },
    /// Hold the contracts to IWEBS's own verdict, case by case: on a posted
    /// claim and changed copies of it, on a posted challenge and changed
    /// copies of it, or on every on-the-spot question of a fetched claim.
    /// The claim or challenge itself is then submitted. Exits nonzero when
    /// the two differ on any case.
    Replay {
        what: ReplayWhat,
        /// The specification of the procedure.
        #[arg(long)]
        specification: PathBuf,
        /// The claim's `claim_submission.json`.
        #[arg(long)]
        submission: PathBuf,
        /// The `challenge_submission.json` that `challenge post` wrote.
        #[arg(long)]
        challenge: Option<PathBuf>,
    },
}

pub(crate) fn run(command: DevstackCommand) -> Result<ExitCode> {
    {
        let base_dir = std::env::current_dir().context("failed to resolve current directory")?;
        let at = |path: &PathBuf| absolute(&base_dir, path);
        match command {
            DevstackCommand::Deployment {
                out,
                fields,
                errors,
            } => {
                let mut written = serde_json::Map::new();
                for field in &fields {
                    let (name, value) = field
                        .split_once('=')
                        .with_context(|| format!("`{field}` is not name=value"))?;
                    let value = match value.parse::<u64>() {
                        Ok(number) => serde_json::json!(number),
                        Err(_) => serde_json::json!(value),
                    };
                    written.insert(name.to_string(), value);
                }
                let errors_text = fs::read_to_string(at(&errors))
                    .with_context(|| format!("failed to read {}", errors.display()))?;
                let count = devstack::write_deployment(
                    &at(&out),
                    serde_json::Value::Object(written.clone()),
                    &errors_text,
                )?;
                println!("{}", serde_json::to_string(&written)?);
                println!("{count} errors");
                Ok(ExitCode::SUCCESS)
            }
            DevstackCommand::Submit {
                what,
                specification,
                submission,
                challenge,
                receipt,
                calls,
                call,
                no_attest,
                no_relay,
                no_register,
            } => {
                let node = devstack::Node::open()?;
                let layer = posting::open_layer(None)?;
                let before = devstack::submit::Before {
                    attest: !no_attest,
                    relay: !no_relay,
                    register: !no_register,
                };
                let need = |path: &Option<PathBuf>, flag: &str| {
                    path.as_ref()
                        .map(&at)
                        .with_context(|| format!("this submission needs {flag}"))
                };
                let outcome = match what {
                    SubmitWhat::Claim => devstack::submit::claim(
                        &node,
                        layer.as_ref(),
                        specification.as_ref().map(&at).as_deref(),
                        &need(&submission, "--submission")?,
                        before,
                    )?,
                    SubmitWhat::Challenge => devstack::submit::challenge(
                        &node,
                        layer.as_ref(),
                        &need(&submission, "--submission")?,
                        &need(&challenge, "--challenge")?,
                        before,
                    )?,
                    SubmitWhat::FaultProof => devstack::submit::fault_proof(
                        &node,
                        &need(&submission, "--submission")?,
                        &need(&challenge, "--challenge")?,
                        &need(&receipt, "--receipt")?,
                    )?,
                    SubmitWhat::Fault => devstack::submit::fault(
                        &node,
                        &need(&calls, "--calls")?,
                        call.context("a fault needs --call")?,
                    )?,
                };
                println!("{}", serde_json::to_string_pretty(&outcome)?);
                Ok(if outcome["accepted"] == true {
                    ExitCode::SUCCESS
                } else {
                    ExitCode::FAILURE
                })
            }
            DevstackCommand::Replay {
                what,
                specification,
                submission,
                challenge,
            } => {
                let node = devstack::Node::open()?;
                let layer = posting::open_layer(None)?;
                let (specification, submission) = (at(&specification), at(&submission));
                match what {
                    ReplayWhat::Claim => {
                        devstack::replay::claim(&node, layer.as_ref(), &specification, &submission)?
                    }
                    ReplayWhat::Challenge => devstack::replay::challenge(
                        &node,
                        layer.as_ref(),
                        &specification,
                        &submission,
                        &at(&challenge.context("a challenge needs --challenge")?),
                    )?,
                    ReplayWhat::Faults => devstack::replay::faults(
                        &node,
                        layer.as_ref(),
                        &specification,
                        &submission,
                    )?,
                }
                Ok(ExitCode::SUCCESS)
            }
        }
    }
}
