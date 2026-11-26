//! The data-availability layer: `iwebs da`.

use crate::blockchain::evm;
use crate::*;

#[derive(Debug, Subcommand)]
pub(crate) enum DaCommand {
    /// Print a block's height, data root and time, as a relayer posts them
    /// to a contract's oracle.
    BlockRoot {
        #[arg(long)]
        height: u64,
        /// The stand-in layer's directory. Default: `IWEBS_CELESTIA_RPC`'s
        /// node, or the default stand-in.
        #[arg(long = "da-dir")]
        da_dir: Option<PathBuf>,
    },
    /// As the trusted host, check each blob a submission posted and sign
    /// where it is, its block's root and what it holds, with the key in
    /// `IWEBS_ATTESTER_KEY`. The signatures are written into the submission.
    Attest {
        /// A claim's or a challenge's submission.
        #[arg(long)]
        submission: PathBuf,
        #[arg(long = "da-dir")]
        da_dir: Option<PathBuf>,
    },
    /// What a submission put on the layer, measured there: every blob it
    /// names is read back and its size split among what it holds.
    Contents {
        /// A claim's or a challenge's submission, posted.
        #[arg(long)]
        submission: PathBuf,
        #[arg(long = "da-dir")]
        da_dir: Option<PathBuf>,
    },
}

pub(crate) fn run(command: DaCommand) -> Result<ExitCode> {
    {
        let base_dir = std::env::current_dir().context("failed to resolve current directory")?;
        match command {
            DaCommand::BlockRoot { height, da_dir } => {
                let da_dir = da_dir.map(|dir| absolute(&base_dir, &dir));
                let root = posting::open_layer(da_dir.as_deref())?.block_root(height)?;
                println!("{}", serde_json::to_string_pretty(&root)?);
            }
            DaCommand::Attest { submission, da_dir } => {
                let da_dir = da_dir.map(|dir| absolute(&base_dir, &dir));
                let layer = posting::open_layer(da_dir.as_deref())?;
                let path = absolute(&base_dir, &submission);
                let attested = evm::attest_submission(&path, layer.as_ref())?;
                println!("attested blobs: {attested}");
                println!("submission: {}", path.display());
            }
            DaCommand::Contents { submission, da_dir } => {
                let da_dir = da_dir.map(|dir| absolute(&base_dir, &dir));
                let measured =
                    measure::measure_posting(&absolute(&base_dir, &submission), da_dir.as_deref())?;
                println!("{}", serde_json::to_string_pretty(&measured)?);
            }
        }
        Ok(ExitCode::SUCCESS)
    }
}
