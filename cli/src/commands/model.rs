//! Model preparation: `iris model`.

use crate::*;

#[derive(Debug, Subcommand)]
pub(crate) enum ModelCommand {
    /// Import a model bundle into committed Raster inference externals.
    Import(ModelImportArgs),
    /// Convert a GGUF model (Q8_0 weights) into a model bundle.
    Convert(ModelConvertArgs),
}

#[derive(Debug, Args)]
pub(crate) struct ModelConvertArgs {
    /// The GGUF file.
    #[arg(long)]
    gguf: PathBuf,
    /// Directory holding the model's config.json and tokenizer.json, copied
    /// into the bundle.
    #[arg(long)]
    metadata: PathBuf,
    /// Bundle directory to write.
    #[arg(long)]
    out: PathBuf,
    /// Write block tensors as dequantized Q16.16 values instead of Q8_0.
    #[arg(long, hide = true)]
    dequantize: bool,
}

#[derive(Debug, Args)]
pub(crate) struct ModelImportArgs {
    #[arg(long = "model")]
    model_dir: PathBuf,
    /// Also save a copy of the model-specific Raster template at this path.
    #[arg(long)]
    manifest: Option<PathBuf>,
    #[arg(long = "artifact-root", default_value = inference_manifest::MODEL_ARTIFACTS_DIR)]
    artifact_root: PathBuf,
    #[arg(long = "model-id")]
    model_id: Option<String>,
    #[arg(long = "only-tokenizer")]
    only_tokenizer: bool,
    #[arg(long = "only-layers")]
    only_layers: bool,
    #[arg(long = "only-embedding")]
    only_embedding: bool,
    #[arg(long = "only-ple")]
    only_ple: bool,
    #[arg(long = "only-native")]
    only_native: bool,
    /// Accepted and ignored: import no longer writes fixtures into stage directories.
    #[arg(long)]
    no_stage_fixtures: bool,
    /// Largest range, in prompt tokens or claimed ids.
    #[arg(long = "range-size", default_value_t = inference_import::DEFAULT_RANGE_SIZE)]
    range_size: u32,
}

impl ModelImportArgs {
    pub(crate) fn into_import_config(self) -> inference_import::ImportConfig {
        inference_import::ImportConfig {
            model_dir: self.model_dir,
            manifest: self.manifest,
            artifact_root: self.artifact_root,
            model_id: self.model_id,
            only_tokenizer: self.only_tokenizer,
            only_layers: self.only_layers,
            only_embedding: self.only_embedding,
            only_ple: self.only_ple,
            only_native: self.only_native,
            no_stage_fixtures: self.no_stage_fixtures,
            range_size: self.range_size,
        }
    }
}

pub(crate) fn run(command: ModelCommand) -> Result<ExitCode> {
    match command {
        ModelCommand::Convert(args) => {
            convert::convert_bundle(&args.gguf, &args.metadata, &args.out, args.dequantize)?;
            Ok(ExitCode::SUCCESS)
        }
        ModelCommand::Import(args) => {
            let imported = inference_import::import_model(args.into_import_config())
                .map_err(|error| anyhow::anyhow!("{error}"))?;
            if let (Some(manifest), Some(template)) =
                (&imported.model_manifest_path, &imported.template)
            {
                write_contract_pins(manifest, template);
            }
            Ok(ExitCode::SUCCESS)
        }
    }
}

/// Writes what a contract pins for this import beside its manifest, as two
/// entries. The procedure kind: the program id each stage project has locked, and
/// for each program the inputs it takes from earlier stages, read from the
/// pipeline template of the `single` procedure kind over this import. The model: its
/// id with the commitments it hashes, and the range size.
///
/// An import made where the stage projects are not locked gets no pins and
/// says so. It can still run; it cannot have a claim checked against it.
fn write_contract_pins(model_manifest_path: &std::path::Path, template: &str) {
    let pins = inference_manifest::ModelManifest::load_verified(model_manifest_path).and_then(
        |manifest| blockchain_rules::checks::ContractPins::from_import(&manifest, template),
    );
    let path = model_manifest_path.with_file_name(blockchain_rules::checks::CONTRACT_PINS_JSON);
    match pins.and_then(|pins| inference_manifest::write_json(&path, &pins)) {
        Ok(()) => println!("wrote {}", path.display()),
        Err(error) => println!("no contract pins written: {error:#}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_import_args_build_typed_config() {
        let config = ModelImportArgs {
            model_dir: PathBuf::from("model"),
            manifest: Some(PathBuf::from("Raster.toml")),
            artifact_root: PathBuf::from(inference_manifest::MODEL_ARTIFACTS_DIR),
            model_id: Some(String::from("gemma")),
            only_tokenizer: true,
            only_layers: false,
            only_embedding: true,
            only_ple: false,
            only_native: false,
            no_stage_fixtures: false,
            range_size: inference_import::DEFAULT_RANGE_SIZE,
        }
        .into_import_config();

        assert_eq!(
            config,
            inference_import::ImportConfig {
                model_dir: PathBuf::from("model"),
                manifest: Some(PathBuf::from("Raster.toml")),
                artifact_root: PathBuf::from(inference_manifest::MODEL_ARTIFACTS_DIR),
                model_id: Some(String::from("gemma")),
                only_tokenizer: true,
                only_layers: false,
                only_embedding: true,
                only_ple: false,
                only_native: false,
                no_stage_fixtures: false,
                range_size: inference_import::DEFAULT_RANGE_SIZE,
            }
        );
    }
}
