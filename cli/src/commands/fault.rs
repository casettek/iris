//! A fault proof: `iwebs fault`.

use crate::*;

#[derive(Debug, Subcommand)]
pub(crate) enum FaultCommand {
    /// Check that this binary embeds real proving guests.
    Check {
        /// Verify program identities used by automatic backend challenges.
        #[arg(long)]
        raster_manifest: Option<PathBuf>,
    },
    /// Prove a bad Raster challenge.
    Prove(fault::ProveArgs),
}

pub(crate) fn run(command: FaultCommand) -> Result<ExitCode> {
    match command {
        FaultCommand::Check { raster_manifest } => {
            fault::check_runtime()?;
            if let Some(manifest) = raster_manifest {
                fault::check_challenge_programs(&manifest)?;
            }
            println!("Fault proving runtime available");
            println!(
                "Proving build: {}",
                if cfg!(feature = "cuda") {
                    "CUDA"
                } else {
                    "CPU"
                }
            );
            Ok(ExitCode::SUCCESS)
        }
        FaultCommand::Prove(args) => {
            fault::prove(args)?;
            Ok(ExitCode::SUCCESS)
        }
    }
}
