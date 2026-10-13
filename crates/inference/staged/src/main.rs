use std::path::{Path, PathBuf};
use std::process::{Command as ProcessCommand, ExitCode, ExitStatus};

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};

#[derive(Debug, Parser)]
#[command(name = "staged-infer")]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    /// Compare one already-completed no-auth stage without rerunning the chain.
    #[arg(long = "compare-stage", alias = "compare-prefill-range-stage")]
    compare_stage: Option<PathBuf>,

    /// Run one stage directly and publish its canonical output artifact.
    #[arg(long = "run-stage", alias = "run-prefill-range-stage", hide = true)]
    run_stage: Option<PathBuf>,

    #[arg(long, hide = true)]
    input: Option<PathBuf>,

    #[arg(long = "input-manifest", hide = true)]
    input_manifest: Option<PathBuf>,
}

#[derive(Debug, Subcommand)]
enum Commands {
    /// Run staged-infer chain commands.
    Chain {
        #[command(subcommand)]
        command: ChainCommand,
    },
}

#[derive(Debug, Subcommand)]
enum ChainCommand {
    /// Run the staged-infer chain, optionally leaving one stage on Raster as the reference.
    Run {
        /// Stage to run on Raster while staged-infer owns supported stages around it.
        /// Omit this to run every stage staged-infer.
        #[arg(long = "raster-stage")]
        raster_stage: Option<String>,

        /// Use the previous per-stage child-process staged-infer runner.
        #[arg(long = "staged-subprocess", alias = "direct-subprocess", hide = true)]
        staged_subprocess: bool,
    },
}

fn main() -> ExitCode {
    match try_main() {
        Ok(code) => code,
        Err(error) => {
            eprintln!("staged-infer error: {error:#}");
            ExitCode::from(1)
        }
    }
}

fn try_main() -> Result<ExitCode> {
    execute(Cli::parse())
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
        Some(Commands::Chain {
            command:
                ChainCommand::Run {
                    raster_stage,
                    staged_subprocess,
                },
        }) => {
            let exe = std::env::current_exe().context("failed to locate current executable")?;
            let staged_backend = if staged_subprocess {
                staged_infer::chain_runner::StagedExecutionBackend::Subprocess
            } else {
                staged_infer::chain_runner::StagedExecutionBackend::InProcess
            };
            let manifest_path = std::env::current_dir()
                .context("failed to read current directory")?
                .join("Raster.toml");
            let run = staged_infer::chain_runner::run(
                &manifest_path,
                raster_stage.as_deref(),
                &exe,
                staged_backend,
            )?;
            if let Some(selected_stage_dir) = run.selected_stage_dir.as_ref() {
                let report = staged_infer::parity::read_parity_report(selected_stage_dir)?;
                print!(
                    "{}",
                    render_timing_summary_or_warning(&run.chain_dir, &report)
                );
            }
            Ok(ExitCode::SUCCESS)
        }
        None => bail!(
            "staged-infer requires `chain run`; \
             use the Raster CLI directly for Raster-only execution"
        ),
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

fn require_input_args(cli: &Cli) -> Result<()> {
    if cli.input.is_none() || cli.input_manifest.is_none() {
        bail!("hidden stage mode requires --input and --input-manifest");
    }
    Ok(())
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
    if let Err(error) = std::fs::remove_file(&prior_report) {
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

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::ffi::{OsStr, OsString};

    use super::*;

    #[test]
    fn bare_invocation_is_rejected() {
        let error = execute(Cli {
            command: None,
            compare_stage: None,
            run_stage: None,
            input: None,
            input_manifest: None,
        })
        .unwrap_err();

        assert!(error.to_string().contains("requires `chain run`"));
    }

    #[test]
    fn hidden_direct_stage_requires_input_files() {
        let error = execute(Cli {
            command: None,
            compare_stage: None,
            run_stage: Some(PathBuf::from("prefill_range_l3")),
            input: None,
            input_manifest: None,
        })
        .unwrap_err();

        assert!(error.to_string().contains("--input and --input-manifest"));
    }

    #[test]
    fn comparison_command_is_unauthenticated_and_isolated() {
        let command = comparison_command(
            Path::new("/tmp/staged-infer"),
            Path::new("/tmp/staged-runs/run/prefill_range_l3"),
            Path::new("/tmp/input.json"),
            Path::new("/tmp/input_manifest.json"),
        );
        let env: HashMap<OsString, Option<OsString>> = command
            .get_envs()
            .map(|(name, value)| (name.to_owned(), value.map(OsStr::to_owned)))
            .collect();

        assert_eq!(
            env.get(OsStr::new(raster_runtime::auth::AUTH_ENV)),
            Some(&Some(OsString::from("0")))
        );
        for name in [
            raster_runtime::TRACE_PATH_ENV,
            raster_runtime::TRACE_FORMAT_ENV,
            raster_runtime::OUTPUT_DIR_ENV,
            raster_runtime::PROFILE_PATH_ENV,
            raster_runtime::PROFILE_STREAM_PATH_ENV,
            raster_runtime::PROFILE_RUN_ID_ENV,
        ] {
            assert_eq!(env.get(OsStr::new(name)), Some(&None), "{name}");
        }
    }

    #[test]
    fn missing_execution_times_do_not_override_parity() {
        let report = staged_infer::parity::ParityReport {
            version: 3,
            stage: String::from("prefill_range_l3"),
            routine: String::from("prefill_range"),
            instance: Some(3),
            authority: staged_infer::parity::ParityAuthority::NonAuthoritative,
            raster_source_mode: staged_infer::parity::RasterSourceMode::Unauthenticated,
            matched: true,
            input_load_duration_ns: 1,
            kernel_duration_ns: 1,
            encode_write_duration_ns: 1,
            direct_stage_duration_ns: 1,
        };

        let summary = render_timing_summary_or_warning(Path::new("/missing/chain/run"), &report);

        assert!(summary.contains("timing summary unavailable"));
    }
}
