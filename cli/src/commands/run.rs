//! A procedure run with no blockchain, `iwebs run`.

use super::infer::print_native_run;
use crate::*;

#[derive(Debug, Args)]
pub(crate) struct RunArgs {
    /// The specification of the procedure to run.
    #[arg(long, default_value = procedure_definition::SPECIFICATION_TOML)]
    specification: PathBuf,

    /// The case to run it on: the prompt, or the documents of the
    /// specification's source.
    #[arg(long, default_value = procedure_definition::CASE_TOML)]
    case: PathBuf,

    /// The `checkpoint_list.json` of another run of the same specification
    /// and case. This run's checkpoints are compared with it in order, and
    /// the first stage where the two differ is named.
    #[arg(long)]
    compare: Option<PathBuf>,

    /// Run each native stage in a child process of its own, as the previous runner did.
    #[arg(long = "native-stage-subprocess", hide = true)]
    native_stage_subprocess: bool,
}

pub(crate) fn run(args: RunArgs) -> Result<ExitCode> {
    warn_if_debug_build("run");
    let base_dir = std::env::current_dir().context("failed to read current directory")?;
    let options = procedure_run::RunOptions {
        specification_path: args.specification.clone(),
        case_path: args.case.clone(),
        current_exe: std::env::current_exe().context("failed to locate current executable")?,
        native_stage_backend: if args.native_stage_subprocess {
            procedure_pipeline::pipeline_runner::NativeStageBackend::Subprocess
        } else {
            procedure_pipeline::pipeline_runner::NativeStageBackend::InProcess
        },
        base_dir: base_dir.clone(),
    };
    // A list that cannot be read is found now, not after the run.
    let compare = args.compare.map(|other| absolute(&base_dir, &other));
    if let Some(other) = &compare {
        procedure_definition::read_checkpoint_list(other)
            .with_context(|| format!("the checkpoint list to compare with, {}", other.display()))?;
    }
    // The result first: it is known before any stage runs.
    let native_run = procedure_run::run_native(&options)?;
    print_native_run(&native_run, &args.specification, &args.case)?;
    let pipeline_run = procedure_run::run_pipeline(&options, &native_run)?;
    println!(
        "pipeline run: {} stages, and the result the native run stated",
        pipeline_run.checkpoints
    );
    println!(
        "checkpoint list: {}",
        pipeline_run.checkpoint_list_path.display()
    );
    println!(
        "checkpoints: {}",
        pipeline_run.checkpoint_hashes_path.display()
    );
    let Some(other) = compare else {
        return Ok(ExitCode::SUCCESS);
    };
    Ok(
        match procedure_run::first_difference(&pipeline_run, &other)? {
            None => {
                println!("compared with {}: the same at every stage", other.display());
                ExitCode::SUCCESS
            }
            Some(divergence) => {
                println!(
                    "compared with {}: differs first at stage `{}` (checkpoint {}): {}",
                    other.display(),
                    divergence.stage,
                    divergence.checkpoint_index,
                    serde_json::to_value(divergence.reason)?
                        .as_str()
                        .unwrap_or("unknown")
                );
                ExitCode::FAILURE
            }
        },
    )
}
