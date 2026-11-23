use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command as ProcessCommand, ExitCode, ExitStatus};

use anyhow::{bail, Context, Result};
use clap::{Args, Parser, Subcommand, ValueEnum};

mod arrangement;
mod audit;
mod celestia;
mod challenge;
mod challenge_profile;
mod checks;
mod claim;
mod commands;
mod convert;
mod devstack;
mod evm;
mod fault;
mod infer;
mod lies;
mod measure;
mod parity;
mod posted;
mod posting;
#[cfg(feature = "replay-profile")]
mod replay_profile;
mod run_inputs;
mod source;
mod stage_bindings;
#[cfg(test)]
mod test_support;

pub use challenge::{
    build_challenge, ChallengeBuildOptions, ChallengeBuildOutcome, ChallengeBuildResult,
    ChallengeInput,
};
pub use checks::{check_claim, CheckLine, ClaimCheckOptions, ClaimCheckReport};
pub use claim::{
    build_claim, corrupt_checkpoint_hashes, ClaimBuildOptions, ClaimBuildResult,
    CorruptCheckpointHashesOptions, CorruptCheckpointHashesResult, CorruptCheckpointSelection,
};
pub use infer::run_infer;
pub use inference_artifacts::{
    build_checkpoint_trace, write_claim_artifacts, ChallengeBundle, ChallengeTrace, Checkpoint,
    CheckpointTrace, ClaimBundle, ClaimEndpoint, Divergence, DivergenceReason, ReplayPackage,
    CHALLENGE_BUNDLE_JSON, CHALLENGE_TRACE_JSON, CHECKPOINT_HASHES_TXT, CHECKPOINT_TRACE_JSON,
    CLAIM_BUNDLE_JSON, DIVERGENCE_JSON, REPLAY_PACKAGE_JSON,
};
#[cfg(feature = "replay-profile")]
pub use replay_profile::{run_replay_profile, ReplayProfileOptions};

use commands::challenge::ChallengeCommand;
use commands::claim::ClaimCommand;
use commands::da::DaCommand;
use commands::evm::EvmCommand;
use commands::fault::FaultCommand;
use commands::infer::InferArgs;
use commands::model::ModelCommand;
use commands::registry::RegistryCommand;
use commands::stage::*;
use commands::testing::TestCommand;

#[derive(Debug, Parser)]
#[command(name = "iris")]
struct Cli {
    /// Persistent auxiliary worker; value is its retained-weight byte budget.
    #[arg(long, hide = true)]
    aux_worker: Option<usize>,

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
    /// Set up a model: convert a GGUF model, import a bundle.
    Model {
        #[command(subcommand)]
        command: ModelCommand,
    },
    /// A claim: build, post and check one, or read one back and audit it.
    Claim {
        #[command(subcommand)]
        command: ClaimCommand,
    },
    /// Fast deterministic inference: the output a claim then states.
    Infer(InferArgs),
    /// The host signer: a trusted stand-in for a notary.
    Attest {
        #[command(subcommand)]
        command: source::AttestCommand,
    },
    /// A run's source, without running the model.
    Source {
        #[command(subcommand)]
        command: source::SourceCommand,
    },
    /// Measure the heaviest tile replays in one exact stage instance.
    #[cfg(feature = "replay-profile")]
    ReplayProfile(ReplayProfileArgs),
    /// Compare a chain run with direct generation of the same prompt and
    /// output, position by position.
    #[command(hide = true)]
    CompareTeacherForced {
        #[arg(long = "teacher-forced")]
        teacher_forced: PathBuf,
        #[arg(long)]
        direct: PathBuf,
    },
    #[command(hide = true)]
    ParityWorker {
        #[arg(long)]
        run: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        direct: bool,
    },
    /// A challenge of a claim: build, post and check one, or read one back.
    Challenge {
        #[command(subcommand)]
        command: ChallengeCommand,
    },
    /// The registry of configurations claims are made of.
    Registry {
        #[command(subcommand)]
        command: RegistryCommand,
    },
    /// The data-availability layer: what a relayer and an attester need.
    Da {
        #[command(subcommand)]
        command: DaCommand,
    },
    /// The calldata of the calls iris's contracts take. Nothing is sent.
    Evm {
        #[command(subcommand)]
        command: EvmCommand,
    },
    /// The tools a test stages a dispute with, and the test suites
    /// themselves: lying claims and challenges, what catching them costs,
    /// the parity gate, and the local chains. Nothing an honest party runs.
    Test {
        #[command(subcommand)]
        command: TestCommand,
    },
    /// A fault proof: what answers a challenge whose trace is wrong.
    Fault {
        #[command(subcommand)]
        command: FaultCommand,
    },
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

#[cfg(debug_assertions)]
fn warn_if_debug_build(workflow: &str) {
    eprintln!(
        "warning: iris {workflow} is running from a debug build; \
         use `cargo run --release -p iris-cli -- {workflow}` \
         for the optimized staged-infer path"
    );
}

fn execute(cli: Cli) -> Result<ExitCode> {
    if let Some(bytes) = cli.aux_worker {
        staged_infer::worker::run_worker(bytes)?;
        return Ok(ExitCode::SUCCESS);
    }
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
        Some(Commands::CompareTeacherForced {
            teacher_forced,
            direct,
        }) => {
            let comparison = parity::compare_teacher_forced(&teacher_forced, &direct)?;
            println!("{}", serde_json::to_string_pretty(&comparison)?);
            Ok(ExitCode::SUCCESS)
        }
        Some(Commands::ParityWorker {
            run,
            output,
            direct,
        }) => {
            parity::worker(&run, &output, direct)?;
            Ok(ExitCode::SUCCESS)
        }
        Some(Commands::Model { command }) => commands::model::run(command),
        Some(Commands::Claim { command }) => commands::claim::run(command),
        Some(Commands::Infer(args)) => commands::infer::run(args),
        Some(Commands::Attest { command }) => {
            match command {
                source::AttestCommand::Keygen { out } => source::keygen(&out)?,
                source::AttestCommand::Sign(args) => source::sign(args)?,
            }
            Ok(ExitCode::SUCCESS)
        }
        Some(Commands::Source {
            command: source::SourceCommand::Inputs { run, out },
        }) => {
            source::stage_inputs(&run, &out)?;
            Ok(ExitCode::SUCCESS)
        }
        Some(Commands::Source {
            command: source::SourceCommand::Render { run, text_out },
        }) => {
            let record = source::render(&run, text_out.as_deref())?;
            Ok(if record.fault == 0 {
                ExitCode::SUCCESS
            } else {
                ExitCode::from(1)
            })
        }
        Some(Commands::Challenge { command }) => commands::challenge::run(command),
        Some(Commands::Da { command }) => commands::da::run(command),
        Some(Commands::Test { command }) => commands::testing::run(command),
        Some(Commands::Evm { command }) => commands::evm::run(command),
        Some(Commands::Registry { command }) => commands::registry::run(command),
        Some(Commands::Fault { command }) => commands::fault::run(command),
        None => {
            bail!("iris requires a workflow command; try `model import`, `claim build`, or `infer`")
        }
    }
}

#[cfg(not(debug_assertions))]
fn warn_if_debug_build(_workflow: &str) {}

fn absolute(base: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        base.join(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_phase_one_workflows() {
        parse_for_test(["iris", "model", "import", "--model", "model-bundles/model"]).unwrap();
        parse_for_test(["iris", "claim", "build", "--run", "inference.toml"]).unwrap();
        parse_for_test([
            "iris",
            "test",
            "corrupt-checkpoints",
            "--checkpoints",
            "checkpoints.txt",
            "--index",
            "1",
        ])
        .unwrap();
        parse_for_test([
            "iris",
            "test",
            "corrupt-claim",
            "--submission",
            "claim_submission.json",
            "--output",
            "lying",
            "--stage",
            "range_p0_l1",
            "--fault",
            "arrangement",
        ])
        .unwrap();
        parse_for_test(["iris", "test", "claim", "--ids", "1,2,3"]).unwrap();
        parse_for_test(["iris", "test", "parity", "--skip-challenges"]).unwrap();
        // The test tools are under `iris test` and nowhere else.
        for moved in [
            vec![
                "iris",
                "claim",
                "corrupt",
                "--checkpoints",
                "checkpoints.txt",
            ],
            vec!["iris", "challenge", "corrupt", "--challenge", "bundle.json"],
            vec!["iris", "test-parity"],
            vec!["iris", "claim", "build", "--claimed-ids", "1,2,3"],
        ] {
            assert!(parse_for_test(moved).is_err());
        }
        parse_for_test([
            "iris",
            "challenge",
            "build",
            "--submission",
            "claim_submission.json",
        ])
        .unwrap();
        // The inputs that read a claimer's own trace or bundle are gone.
        for removed in ["--trace", "--claim"] {
            assert!(parse_for_test(["iris", "challenge", "build", removed, "file.json"]).is_err());
        }
        assert!(parse_for_test(["iris", "challenge", "locate"]).is_err());
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
                "--submission",
                "claim_submission.json",
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

    /// The one table of formats and versions says what the code says. The
    /// weight file's two versions are `detwgt`'s, which this crate does not
    /// link: its row is checked by its presence alone.
    #[test]
    fn the_formats_table_is_current() {
        use inference_artifacts::availability::BLOB_MAGIC;
        use inference_artifacts::contract::{
            CHALLENGE_SUBMISSION_VERSION, CLAIM_SUBMISSION_VERSION, CONTRACT_PINS_VERSION,
        };
        use inference_artifacts::registry::{CONFIGURATION_VERSION, REGISTRY_VERSION};
        use inference_artifacts::{MODEL_MANIFEST_VERSION, PREPARED_RUN_VERSION};
        use model_identity::pipeline_core::EXECUTION_VERSION;
        use model_identity::DESCRIPTOR_VERSION;

        let table = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../docs/internals/formats.md"
        ));
        let blob = std::str::from_utf8(BLOB_MAGIC).unwrap().trim_end();
        let blob_version: u32 = blob.rsplit("/v").next().unwrap().parse().unwrap();
        assert!(table.contains(&format!("`{blob}`")), "{blob} is not named");
        for (constant, version) in [
            ("DESCRIPTOR_VERSION", DESCRIPTOR_VERSION),
            ("MODEL_MANIFEST_VERSION", MODEL_MANIFEST_VERSION),
            ("PREPARED_RUN_VERSION", PREPARED_RUN_VERSION),
            ("EXECUTION_VERSION", EXECUTION_VERSION),
            ("CONTRACT_PINS_VERSION", CONTRACT_PINS_VERSION),
            ("CONFIGURATION_VERSION", CONFIGURATION_VERSION),
            ("REGISTRY_VERSION", REGISTRY_VERSION),
            ("CLAIM_SUBMISSION_VERSION", CLAIM_SUBMISSION_VERSION),
            ("CHALLENGE_SUBMISSION_VERSION", CHALLENGE_SUBMISSION_VERSION),
            ("BLOB_MAGIC", blob_version),
        ] {
            let row = table
                .lines()
                .find(|line| line.starts_with('|') && line.contains(&format!("`{constant}`")))
                .unwrap_or_else(|| panic!("the formats table has no row for {constant}"));
            let cells: Vec<&str> = row.split('|').map(str::trim).collect();
            assert_eq!(
                cells.get(3).copied(),
                Some(version.to_string().as_str()),
                "the formats table gives {constant} another version than the code: {row}"
            );
        }
        for constant in ["DETWGT_VERSION", "DET_NUM_SPEC_VERSION"] {
            assert!(
                table.contains(&format!("`{constant}`")),
                "no row for {constant}"
            );
        }
    }
}
