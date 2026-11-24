use std::collections::BTreeSet;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

use inference_artifacts::{
    model_id_of, write_json, DirectInferBundle, DirectInferProvenance, FamilyShape,
    ModelCommitment, ModelFamily, ModelManifest, MODEL_MANIFEST_JSON, MODEL_MANIFEST_VERSION,
};
use sha2::{Digest, Sha256};

use super::ImportConfig;

pub fn write_model_manifest(
    args: &ImportConfig,
    eos_ids: &BTreeSet<u32>,
    family: ModelFamily,
    shape: FamilyShape,
    raster_manifest: Option<(&Path, &str)>,
    model_commitments: &[ModelCommitment],
) -> Result<PathBuf, Box<dyn Error>> {
    let artifact_dir = args.artifact_dir();
    fs::create_dir_all(&artifact_dir)?;
    let template_provenance = |(path, text): (&Path, &str)| -> Result<_, Box<dyn Error>> {
        Ok((
            manifest_relative_path(&artifact_dir, path)?,
            format!("{:x}", Sha256::digest(text.as_bytes())),
        ))
    };
    let manifest_path = artifact_dir.join(MODEL_MANIFEST_JSON);
    let model_detwgt = args.model_dir.join("model.detwgt");
    let config = args.model_dir.join("config.json");
    let tokenizer_path = args.model_dir.join("tokenizer.json");

    let direct = ModelManifest {
        version: MODEL_MANIFEST_VERSION,
        family: family.name().to_string(),
        model_id: model_id_of(model_commitments),
        model_commitments: model_commitments.to_vec(),
        range_size: args.range_size,
        bundle: DirectInferBundle {
            model_detwgt_path: manifest_relative_path(&artifact_dir, &model_detwgt)?,
            model_detwgt_sha256: sha256_file(&model_detwgt)?,
            config_path: manifest_relative_path(&artifact_dir, &config)?,
            config_sha256: sha256_file(&config)?,
            tokenizer_path: manifest_relative_path(&artifact_dir, &tokenizer_path)?,
            tokenizer_sha256: sha256_file(&tokenizer_path)?,
        },
        shape,
        eos_token_ids: eos_ids.iter().copied().collect(),
        provenance: raster_manifest.map(template_provenance).transpose()?.map(
            |(raster_manifest_path, raster_manifest_sha256)| DirectInferProvenance {
                raster_manifest_path,
                raster_manifest_sha256,
            },
        ),
    };

    if direct.provenance.is_some() {
        let descriptor = direct.descriptor().map_err(|error| error.to_string())?;
        let bytes = descriptor
            .encode()
            .map_err(|fault| format!("invalid model descriptor: {fault:?}"))?;
        fs::write(
            artifact_dir.join(inference_artifacts::model_identity::MODEL_DESCRIPTOR_BIN),
            bytes,
        )?;
    }

    write_json(&manifest_path, &direct)
        .map_err(|error| -> Box<dyn Error> { error.to_string().into() })?;
    println!("wrote {}", manifest_path.display());
    Ok(manifest_path)
}

pub fn warn_partial_direct_manifest_not_refreshed(args: &ImportConfig) {
    eprintln!(
        "model manifest not refreshed; run a full `iris model import ...` \
         to regenerate {}",
        args.artifact_dir().join(MODEL_MANIFEST_JSON).display()
    );
}

fn manifest_relative_path(artifact_dir: &Path, source: &Path) -> Result<PathBuf, Box<dyn Error>> {
    let dir = fs::canonicalize(artifact_dir)?;
    let source = fs::canonicalize(source)?;
    Ok(source.strip_prefix(dir).unwrap_or(&source).to_path_buf())
}

fn sha256_file(path: &Path) -> Result<String, Box<dyn Error>> {
    inference_artifacts::file_sha256(path).map_err(|error| error.to_string().into())
}
