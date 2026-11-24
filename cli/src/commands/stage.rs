//! The hidden modes that run or compare one stage, and the profile command's arguments.

use crate::*;

#[cfg(feature = "replay-profile")]
#[derive(Debug, Args)]
pub(crate) struct ReplayProfileArgs {
    #[arg(long, default_value = inference_artifacts::SPECIFICATION_TOML)]
    pub(crate) specification: PathBuf,
    #[arg(long, default_value = inference_artifacts::CASE_TOML)]
    pub(crate) case: PathBuf,
    /// Exact expanded stage name, e.g. range_p0_l13 or prompt_check_p0.
    #[arg(long)]
    pub(crate) stage: String,
    /// Number of distinct tiles selected by their slowest native invocation.
    #[arg(long, default_value = "3")]
    pub(crate) top_n: std::num::NonZeroUsize,
}

pub(crate) fn require_input_args(cli: &Cli) -> Result<()> {
    if cli.input.is_none() || cli.input_manifest.is_none() {
        bail!("hidden stage mode requires --input and --input-manifest");
    }
    Ok(())
}

pub(crate) fn finish_parity_run(status: ExitStatus, stage_dir: &Path) -> Result<ExitCode> {
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

pub(crate) fn render_timing_summary_or_warning(
    chain_dir: &Path,
    report: &staged_infer::parity::ParityReport,
) -> String {
    staged_infer::parity::read_chain_execution_times(chain_dir)
        .and_then(|timings| staged_infer::parity::render_timing_summary(&timings, report))
        .unwrap_or_else(|error| format!("\ntiming summary unavailable: {error:#}\n"))
}

pub(crate) fn run_hidden_compare(stage_dir: &Path) -> Result<ExitStatus> {
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

pub(crate) fn comparison_command(
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
        .env_remove(raster_runtime::tracing::COMMIT_PATH_ENV)
        .env_remove(raster_runtime::tracing::COMMIT_PROJECT_ENV)
        .env_remove(raster_runtime::tracing::COMMIT_WINDOW_ENV)
        .env_remove(raster_runtime::TRACE_FORMAT_ENV)
        .env_remove(raster_runtime::OUTPUT_DIR_ENV)
        .env_remove(raster_runtime::PROFILE_PATH_ENV)
        .env_remove(raster_runtime::PROFILE_STREAM_PATH_ENV)
        .env_remove(raster_runtime::PROFILE_RUN_ID_ENV);
    command
}

pub(crate) fn exit_code_from_status(status: ExitStatus) -> ExitCode {
    match status.code() {
        Some(code) => ExitCode::from(code.clamp(0, u8::MAX as i32) as u8),
        None => ExitCode::from(1),
    }
}
