//! A procedure run with no blockchain, `iris run`.

use super::infer::print_direct_run;
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

    /// Use the previous per-stage child-process staged-infer runner.
    #[arg(long = "staged-subprocess", hide = true)]
    staged_subprocess: bool,
}

pub(crate) fn run(args: RunArgs) -> Result<ExitCode> {
    warn_if_debug_build("run");
    let base_dir = std::env::current_dir().context("failed to read current directory")?;
    let options = procedure_run::RunOptions {
        specification_path: args.specification.clone(),
        case_path: args.case.clone(),
        current_exe: std::env::current_exe().context("failed to locate current executable")?,
        staged_backend: if args.staged_subprocess {
            staged_infer::pipeline_runner::StagedExecutionBackend::Subprocess
        } else {
            staged_infer::pipeline_runner::StagedExecutionBackend::InProcess
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
    let direct = procedure_run::run_direct(&options)?;
    print_direct_run(&direct, &args.specification, &args.case)?;
    let staged = procedure_run::run_staged(&options, &direct)?;
    println!(
        "staged run: {} stages, and the result the direct run stated",
        staged.checkpoints
    );
    println!("checkpoint list: {}", staged.checkpoint_list_path.display());
    println!("checkpoints: {}", staged.checkpoint_hashes_path.display());
    let Some(other) = compare else {
        return Ok(ExitCode::SUCCESS);
    };
    Ok(match procedure_run::first_difference(&staged, &other)? {
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
    })
}
