//! The registry of procedures: `iris registry`.

use crate::*;

#[derive(Debug, Subcommand)]
pub(crate) enum RegistryCommand {
    /// Register the procedure a specification describes: its procedure kind's
    /// programs, its models, its parameters and what it requires of the
    /// source and the result. Prints its id.
    Add(RegistryAddArgs),
}

#[derive(Debug, Args)]
pub(crate) struct RegistryAddArgs {
    /// Specification naming the procedure kind, its models and the parameters.
    #[arg(long, default_value = inference_artifacts::SPECIFICATION_TOML)]
    specification: PathBuf,

    /// The pins of the procedure kind's programs and models, as `iris claim pins`
    /// wrote them. Without it, as `iris claim check` finds them.
    #[arg(long)]
    pins: Option<PathBuf>,

    /// The registry file to add to. Made when it does not exist.
    #[arg(long)]
    registry: PathBuf,
}

pub(crate) fn run(command: RegistryCommand) -> Result<ExitCode> {
    match command {
        RegistryCommand::Add(args) => {
            let base_dir =
                std::env::current_dir().context("failed to resolve current directory")?;
            let procedure = claim_check::spec_procedure(
                &absolute(&base_dir, &args.specification),
                args.pins.map(|path| absolute(&base_dir, &path)).as_deref(),
            )?;
            let registry_path = absolute(&base_dir, &args.registry);
            let mut registry = inference_artifacts::registry::Registry::read(&registry_path)?;
            let id = registry.register(procedure);
            registry.write(&registry_path)?;
            println!("procedure: {id}");
            println!("registry: {}", registry_path.display());
            Ok(ExitCode::SUCCESS)
        }
    }
}
