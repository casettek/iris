//! The record of a run, checked by making the run again: `iwebs record`.

use crate::*;

#[derive(Debug, Subcommand)]
pub(crate) enum RecordCommand {
    /// Make a recorded run again and compare. The run is made on the
    /// record's own inputs, under your specification of its procedure, and
    /// nothing is generated: the stages hold the output the record states to
    /// the model's own choice. The result is compared with the record's,
    /// and every checkpoint, in order, when the record carries them. Exits
    /// with status 1 when they differ.
    Check(RecordCheckArgs),
}

#[derive(Debug, Args)]
pub(crate) struct RecordCheckArgs {
    /// The record's directory, as `iwebs run --record` wrote it.
    #[arg(long)]
    record: PathBuf,

    /// Your own specification of the record's procedure, naming your imports
    /// of its models. The command fails, before anything runs, when it
    /// describes another procedure than the record names.
    #[arg(long, default_value = procedure_definition::SPECIFICATION_TOML)]
    specification: PathBuf,

    /// Run each native stage in a child process of its own, as the previous runner did.
    #[arg(long = "native-stage-subprocess", hide = true)]
    native_stage_subprocess: bool,
}

pub(crate) fn run(command: RecordCommand) -> Result<ExitCode> {
    match command {
        RecordCommand::Check(args) => check(args),
    }
}

fn check(args: RecordCheckArgs) -> Result<ExitCode> {
    warn_if_debug_build("record check");
    let options = procedure_run::record::CheckOptions {
        base_dir: std::env::current_dir().context("failed to read current directory")?,
        specification_path: args.specification,
        record_dir: args.record,
        current_exe: std::env::current_exe().context("failed to locate current executable")?,
        native_stage_backend: if args.native_stage_subprocess {
            procedure_pipeline::pipeline_runner::NativeStageBackend::Subprocess
        } else {
            procedure_pipeline::pipeline_runner::NativeStageBackend::InProcess
        },
    };
    if procedure_run::record::record_generates(&options)? {
        return check_generation(&options);
    }
    let check = procedure_run::record::check_record(&options)?;
    println!("procedure: {}", check.procedure);
    println!(
        "pipeline run: {} stages, on the record's inputs",
        check.run.checkpoints
    );
    println!(
        "checkpoint list: {}",
        check.run.checkpoint_list_path.display()
    );
    if let Some(divergence) = &check.divergence {
        println!(
            "record: differs first at stage `{}` (checkpoint {}): {}",
            divergence.stage,
            divergence.checkpoint_index,
            serde_json::to_value(divergence.reason)?
                .as_str()
                .unwrap_or("unknown")
        );
        return Ok(ExitCode::FAILURE);
    }
    if let Some(difference) = &check.result_difference {
        println!("record: {difference}");
        return Ok(ExitCode::FAILURE);
    }
    for (inference, result) in check.run.results.iter().enumerate() {
        println!(
            "inference {inference}: valid: {}, result code: {}",
            result.valid, result.result_code
        );
    }
    if check.checkpoints_compared {
        println!("record: the same at every stage, and the result it states");
    } else {
        println!("record: the result it states; it carries no checkpoints to compare");
    }
    Ok(ExitCode::SUCCESS)
}

/// The record of a generation: the generation is made again natively, and
/// its results are compared with the record's.
fn check_generation(options: &procedure_run::record::CheckOptions) -> Result<ExitCode> {
    let check = procedure_run::record::check_generation_record(options)?;
    println!("procedure: {}", check.procedure);
    println!("generation: made again natively, on the record's inputs");
    if let Some(difference) = &check.result_difference {
        println!("record: {difference}");
        return Ok(ExitCode::FAILURE);
    }
    for (inference, result) in check.results.iter().enumerate() {
        println!(
            "inference {inference}: valid: {}, result code: {}",
            result.valid, result.result_code
        );
    }
    println!("record: the result it states");
    Ok(ExitCode::SUCCESS)
}
