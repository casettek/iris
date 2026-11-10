use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const MODEL_ARTIFACTS_DIR: &str = "runtime/model-artifacts";
pub const MODEL_MANIFEST_JSON: &str = "manifest.json";

pub const MODEL_MANIFEST_VERSION: u32 = 9;

/// One committed model input, named by the role it plays in the chain.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelCommitment {
    pub role: String,
    pub commitment: String,
}

/// The model's identifier: a hash of its commitments by role, in order.
///
/// This is the legacy role-list adapter, also used for direct-only imports.
/// Claim-capable imports validate the inventory through ModelDescriptor; stages
/// bind that descriptor to the expected ID and verify actual component data.
/// Paths are excluded, so relocating an import preserves identity.
pub fn model_id_of(commitments: &[ModelCommitment]) -> String {
    let mut hasher = Sha256::new();
    for entry in commitments {
        hasher.update(entry.role.as_bytes());
        hasher.update([0]);
        hasher.update(entry.commitment.as_bytes());
        hasher.update([b'\n']);
    }
    format!("{:x}", hasher.finalize())
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelManifest {
    pub version: u32,
    pub model_id: String,
    pub model_commitments: Vec<ModelCommitment>,
    /// Largest range the chain cuts a prompt or an output into, in tokens.
    pub range_size: u32,
    pub bundle: DirectInferBundle,
    pub shape: DirectInferShape,
    pub eos_token_ids: Vec<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provenance: Option<DirectInferProvenance>,
}

impl ModelManifest {
    /// Validate the bundle once at the workflow boundary, before any inference.
    /// This validates host bundle provenance. Stage-level component authentication
    /// remains mandatory and is independent of this host check.
    pub fn load_verified(path: &Path) -> Result<Self> {
        let manifest: Self = crate::read_json(path)?;
        if manifest.version != MODEL_MANIFEST_VERSION {
            bail!(
                "unsupported model manifest v{}, expected v{MODEL_MANIFEST_VERSION}; rerun a full `iris model import --model <bundle-dir>`",
                manifest.version
            );
        }
        if manifest.model_id != model_id_of(&manifest.model_commitments) {
            bail!("model id does not match the manifest's model commitments");
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
        let provenance = self.provenance.as_ref().context(
            "model manifest has no Raster template provenance; run a full `iris model import --model <bundle-dir>` before building claims or challenges",
        )?;
        let dir = manifest_path
            .parent()
            .context("model manifest has no parent directory")?;
        let path = dir.join(&provenance.raster_manifest_path);
        let template = std::fs::read_to_string(&path)
            .with_context(|| format!("failed to read Raster template {}", path.display()))?;
        let actual = format!("{:x}", Sha256::digest(template.as_bytes()));
        if actual != provenance.raster_manifest_sha256 {
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
}
