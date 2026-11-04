use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::ChainKind;

pub const MODEL_ARTIFACTS_DIR: &str = "runtime/model-artifacts";
pub const MODEL_MANIFEST_JSON: &str = "manifest.json";
pub const TEACHER_FORCED_TEMPLATE_TOML: &str = "Raster.teacher-forced.toml";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelManifest {
    pub version: u32,
    pub bundle: DirectInferBundle,
    pub shape: DirectInferShape,
    pub eos_token_ids: Vec<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provenance: Option<DirectInferProvenance>,
}

impl ModelManifest {
    /// Validate the bundle once at the workflow boundary, before any inference.
    /// Callers retain this manifest; tensor and stage loaders do no extra hashing.
    pub fn load_verified(path: &Path) -> Result<Self> {
        let manifest: Self = crate::read_json(path)?;
        if manifest.version != 2 {
            bail!("unsupported model manifest v{}", manifest.version);
        }
        let dir = path
            .parent()
            .context("model manifest has no parent directory")?;
        for (file, expected, label) in [
            (
                &manifest.bundle.config_path,
                &manifest.bundle.config_sha256,
                "model config",
            ),
            (
                &manifest.bundle.tokenizer_path,
                &manifest.bundle.tokenizer_sha256,
                "model tokenizer",
            ),
            (
                &manifest.bundle.model_detwgt_path,
                &manifest.bundle.model_detwgt_sha256,
                "model weights",
            ),
        ] {
            verify_file_sha256(&dir.join(file), expected, label)?;
        }
        Ok(manifest)
    }

    /// Claims and challenges must use the template tied to this model import.
    pub fn verified_raster_template(&self, manifest_path: &Path) -> Result<(PathBuf, String)> {
        self.verified_chain_template(manifest_path, ChainKind::Autoregressive)
    }

    /// The teacher-forced range size the model's template was built for.
    pub fn teacher_forced_range_size(&self) -> Result<u32> {
        self.provenance
            .as_ref()
            .and_then(|provenance| provenance.teacher_forced_range_size)
            .context(
                "model manifest has no teacher-forced range size; rerun a full `iris model import --model <bundle-dir>`",
            )
    }

    /// The import-bound template for one chain shape.
    pub fn verified_chain_template(
        &self,
        manifest_path: &Path,
        chain: ChainKind,
    ) -> Result<(PathBuf, String)> {
        let provenance = self.provenance.as_ref().context(
            "model manifest has no Raster template provenance; run a full `iris model import --model <bundle-dir>` before building claims or challenges",
        )?;
        let (template_path, expected) = match chain {
            ChainKind::Autoregressive => (
                &provenance.raster_manifest_path,
                &provenance.raster_manifest_sha256,
            ),
            ChainKind::TeacherForced => provenance
                .teacher_forced_manifest_path
                .as_ref()
                .zip(provenance.teacher_forced_manifest_sha256.as_ref())
                .context(
                    "model manifest has no teacher-forced Raster template; rerun a full `iris model import --model <bundle-dir>`",
                )?,
        };
        let dir = manifest_path
            .parent()
            .context("model manifest has no parent directory")?;
        let path = dir.join(template_path);
        let template = std::fs::read_to_string(&path)
            .with_context(|| format!("failed to read Raster template {}", path.display()))?;
        let actual = format!("{:x}", Sha256::digest(template.as_bytes()));
        if &actual != expected {
            bail!(
                "Raster template hash mismatch for {}; rerun a full model import",
                path.display()
            );
        }
        Ok((path, template))
    }
}

/// Stream large weight bundles so startup validation uses bounded memory.
pub fn file_sha256(path: &Path) -> Result<String> {
    let mut file =
        File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let len = file
            .read(&mut buffer)
            .with_context(|| format!("failed to read {}", path.display()))?;
        if len == 0 {
            break;
        }
        hasher.update(&buffer[..len]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

pub fn verify_file_sha256(path: &Path, expected: &str, label: &str) -> Result<()> {
    let actual = file_sha256(path)?;
    if actual != expected {
        bail!(
            "{label} hash mismatch for {}: expected {expected}, got {actual}",
            path.display()
        );
    }
    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DirectInferBundle {
    pub model_detwgt_path: PathBuf,
    pub model_detwgt_sha256: String,
    pub config_path: PathBuf,
    pub config_sha256: String,
    pub tokenizer_path: PathBuf,
    pub tokenizer_sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DirectInferShape {
    pub hidden_size: u32,
    pub num_hidden_layers: u32,
    pub num_attention_heads: u32,
    pub num_key_value_heads: u32,
    pub head_dim: u32,
    pub global_head_dim: u32,
    pub vocab_size: u32,
    pub hidden_size_per_layer_input: u32,
    pub sliding_window: u32,
    pub layer_types: Vec<String>,
    pub num_kv_shared_layers: u32,
    pub norm_eps: i64,
    pub rope_base_sliding: i64,
    pub rope_base_full: i64,
    pub full_partial_rotary_factor_q16: i32,
    pub embedding_scale: i32,
    pub ple_embedding_scale: i32,
    pub ple_projection_scalar: i32,
    pub ple_input_scale: i32,
    pub final_logit_softcap: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DirectInferProvenance {
    pub raster_manifest_path: PathBuf,
    pub raster_manifest_sha256: String,
    /// Absent for imports made before the teacher-forced chain existed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub teacher_forced_manifest_path: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub teacher_forced_manifest_sha256: Option<String>,
    /// Largest teacher-forced range, in prompt pieces or claimed ids.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub teacher_forced_range_size: Option<u32>,
}
