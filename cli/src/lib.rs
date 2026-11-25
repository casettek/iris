use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command as ProcessCommand, ExitCode, ExitStatus};

use anyhow::{bail, Context, Result};
use clap::{Args, Parser, Subcommand, ValueEnum};

mod commands;
#[cfg(feature = "replay-profile")]
mod replay_profile;
mod source;

use blockchain_host::{
    audit, blockchain, challenge, challenge_profile, claim, claim_check, fault, measure, posted,
    posting,
};
use model_import::convert;
use procedure_direct::{absolute, infer};
use testing::{lies, parity};

pub use blockchain_rules::{
    write_claim_artifacts, ChallengeBundle, ChallengeTrace, ClaimBundle, ClaimEndpoint,
    ReplayPackage, CHALLENGE_BUNDLE_JSON, CHALLENGE_TRACE_JSON, CLAIM_BUNDLE_JSON,
    REPLAY_PACKAGE_JSON,
};
pub use challenge::{
    build_challenge, ChallengeBuildOptions, ChallengeBuildOutcome, ChallengeBuildResult,
    ChallengeInput,
};
pub use claim::{
    build_claim, corrupt_checkpoint_hashes, ClaimBuildOptions, ClaimBuildResult,
    CorruptCheckpointHashesOptions, CorruptCheckpointHashesResult, CorruptCheckpointSelection,
};
pub use claim_check::{check_claim, CheckLine, ClaimCheckOptions, ClaimCheckReport};
pub use infer::run_infer;
pub use procedure_definition::{
    build_checkpoint_list, Checkpoint, CheckpointList, Divergence, DivergenceReason,
    CHECKPOINT_HASHES_TXT, CHECKPOINT_LIST_JSON, DIVERGENCE_JSON,
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
        specification: PathBuf,
        #[arg(long)]
        case: PathBuf,
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
    /// The registry of procedures claims are made of.
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
    /// the parity gate, and the local blockchains. Nothing an honest party runs.
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
                specification_path: args.specification,
                case_path: args.case,
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
            specification,
            case,
            output,
            direct,
        }) => {
            parity::worker(&specification, &case, &output, direct)?;
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
            command:
                source::SourceCommand::Inputs {
                    specification,
                    case,
                    out,
                },
        }) => {
            source::stage_inputs(&specification, &case, &out)?;
            Ok(ExitCode::SUCCESS)
        }
        Some(Commands::Source {
            command:
                source::SourceCommand::Render {
                    specification,
                    case,
                    text_out,
                },
        }) => {
            let record = source::render(&specification, &case, text_out.as_deref())?;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_phase_one_workflows() {
        parse_for_test(["iris", "model", "import", "--model", "model-bundles/model"]).unwrap();
        parse_for_test([
            "iris",
            "claim",
            "build",
            "--specification",
            "specification.toml",
            "--case",
            "case.toml",
        ])
        .unwrap();
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
        parse_for_test([
            "iris",
            "infer",
            "--specification",
            "specification.toml",
            "--case",
            "case.toml",
        ])
        .unwrap();
        // The flag the two replaced is no flag any more.
        assert!(parse_for_test(["iris", "infer", "--run", "inference.toml"]).is_err());
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
    fn challenge_specification_defaults_and_overrides_reach_options() {
        for specification in [None, Some("specs/model-b.toml")] {
            let mut argv = vec![
                "iris",
                "challenge",
                "build",
                "--submission",
                "claim_submission.json",
            ];
            if let Some(path) = specification {
                argv.extend(["--specification", path]);
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
                options.specification_path,
                PathBuf::from(specification.unwrap_or("specification.toml"))
            );
        }
    }

    #[test]
    fn infer_requires_an_imported_workspace() {
        let error = execute_from(["iris", "infer"]).unwrap_err();

        assert!(format!("{error:#}").contains("failed to load specification"));
    }

    /// The one table of formats and versions says what the code says. The
    /// weight file's two versions are `detwgt`'s, which this crate does not
    /// link: its row is checked by its presence alone.
    #[test]
    fn the_formats_table_is_current() {
        use blockchain_rules::availability::BLOB_MAGIC;
        use blockchain_rules::checks::{
            CHALLENGE_SUBMISSION_VERSION, CLAIM_SUBMISSION_VERSION, CONTRACT_PINS_VERSION,
        };
        use blockchain_rules::registry::{PROCEDURE_VERSION, REGISTRY_VERSION};
        use model_identity::statement_core::EXECUTION_VERSION;
        use model_identity::DESCRIPTOR_VERSION;
        use model_manifest::MODEL_MANIFEST_VERSION;
        use procedure_definition::PREPARED_RUN_VERSION;

        let table = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../docs/protocol/formats.md"
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
            ("PROCEDURE_VERSION", PROCEDURE_VERSION),
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

    /// The core does not depend on the blockchain code: of the crates under
    /// `segments/`, `procedure/`, `cli/`, `testing/` and `raster-stages/`,
    /// only this one and the tests name a path that leads into `blockchain/`.
    #[test]
    fn nothing_but_the_cli_depends_on_blockchain() {
        fn manifests(dir: &Path, found: &mut Vec<PathBuf>) {
            for entry in fs::read_dir(dir).unwrap() {
                let path = entry.unwrap().path();
                // A linked directory is searched where it really is.
                if path.is_symlink() {
                    continue;
                }
                if path.is_dir() {
                    if !path.ends_with("target") {
                        manifests(&path, found);
                    }
                } else if path.ends_with("Cargo.toml") {
                    found.push(path);
                }
            }
        }
        // Every `path = "..."` of a manifest, whichever table it is in.
        fn paths(value: &toml::Value, found: &mut Vec<String>) {
            match value {
                toml::Value::Table(table) => {
                    for (key, value) in table {
                        match value {
                            toml::Value::String(path) if key == "path" => found.push(path.clone()),
                            other => paths(other, found),
                        }
                    }
                }
                toml::Value::Array(values) => values.iter().for_each(|value| paths(value, found)),
                _ => {}
            }
        }
        // The path with its `..` steps taken, whether or not it exists.
        fn resolved(base: &Path, path: &str) -> PathBuf {
            let mut out = PathBuf::new();
            for part in base.join(path).components() {
                match part {
                    std::path::Component::ParentDir => {
                        out.pop();
                    }
                    std::path::Component::CurDir => {}
                    other => out.push(other),
                }
            }
            out
        }

        let here = Path::new(env!("CARGO_MANIFEST_DIR"));
        let root = resolved(here, "..");
        let blockchain = root.join("blockchain");
        let reaches_blockchain = |manifest: &Path| {
            let text = fs::read_to_string(manifest).unwrap();
            let mut named = Vec::new();
            paths(&toml::from_str(&text).unwrap(), &mut named);
            named
                .iter()
                .any(|path| resolved(manifest.parent().unwrap(), path).starts_with(&blockchain))
        };
        let mut found = Vec::new();
        for dir in ["segments", "procedure", "cli", "testing", "raster-stages"] {
            manifests(&root.join(dir), &mut found);
        }
        let own = here.join("Cargo.toml");
        let tests = root.join("testing/Cargo.toml");
        assert!(found.contains(&own), "the search did not find this crate");
        assert!(
            reaches_blockchain(&own),
            "the check does not see this crate's own dependency on blockchain/"
        );
        let others: Vec<&PathBuf> = found
            .iter()
            .filter(|manifest| {
                **manifest != own && **manifest != tests && reaches_blockchain(manifest)
            })
            .collect();
        assert!(
            others.is_empty(),
            "only the CLI and the tests may depend on blockchain/: {others:?}"
        );
    }
}
