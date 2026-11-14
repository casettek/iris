use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const MODEL_ARTIFACTS_DIR: &str = "runtime/model-artifacts";
pub const MODEL_MANIFEST_JSON: &str = "manifest.json";

pub const MODEL_MANIFEST_VERSION: u32 = 16;

/// Which layer mathematics, tokenizer rules and stage programs a model uses.
///
/// The chain, its schedule and its numerics are shared by every model. A family
/// is what differs: how a layer computes, what the importer reads from a bundle,
/// and which stage programs the import's chain template names. The manifest's
/// `shape` is the family's own.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFamily {
    /// Gemma 3n and Gemma 4 dense models: sliding and full attention with two
    /// RoPE bases, K/V sharing, a gated-GELU feed-forward block, per-layer
    /// embedding inputs and a soft-capped head.
    GemmaDense,
    /// Mistral's Ministral 3 dense models: full attention in every layer with
    /// one RoPE table rotating consecutive pairs, a pre-norm layer with a
    /// gated-SiLU feed-forward block, no per-layer inputs and a plain head.
    MistralDense,
}

impl ModelFamily {
    /// Every family this build can import and run.
    pub const ALL: [Self; 2] = [Self::GemmaDense, Self::MistralDense];

    /// The name a manifest records.
    pub const fn name(self) -> &'static str {
        match self {
            Self::GemmaDense => "gemma-dense",
            Self::MistralDense => "mistral-dense",
        }
    }

    /// Whether each layer has a stage before it that prepares a per-layer
    /// input, and with it a step direct inference records.
    pub const fn has_auxiliary_stages(self) -> bool {
        matches!(self, Self::GemmaDense)
    }

    /// The stage program that runs one layer over one range.
    pub const fn layer_program(self) -> &'static str {
        match self {
            Self::GemmaDense => "prefill-range",
            Self::MistralDense => "layer-mistral",
        }
    }

    /// The bundle under `model-bundles` that the parity gate runs this
    /// family's programs on.
    pub const fn parity_bundle(self) -> &'static str {
        match self {
            Self::GemmaDense => "parity-gemma",
            Self::MistralDense => "parity-mistral",
        }
    }

    pub fn parse(name: &str) -> Result<Self> {
        Self::ALL
            .into_iter()
            .find(|family| family.name() == name)
            .with_context(|| {
                format!(
                    "model family `{name}` is not supported by this build (supported: {})",
                    Self::ALL.map(Self::name).join(", ")
                )
            })
    }
}

fn first_family() -> String {
    ModelFamily::GemmaDense.name().to_string()
}

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
    /// A [`ModelFamily`] name. Manifests written before families were
    /// recorded are the first family's.
    #[serde(default = "first_family")]
    pub family: String,
    pub model_id: String,
    pub model_commitments: Vec<ModelCommitment>,
    /// Largest range the chain cuts a prompt or an output into, in tokens.
    pub range_size: u32,
    pub bundle: DirectInferBundle,
    pub shape: FamilyShape,
    pub eos_token_ids: Vec<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provenance: Option<DirectInferProvenance>,
}

impl ModelManifest {
    /// The family this model is run as, or why this build cannot run it.
    pub fn family(&self) -> Result<ModelFamily> {
        ModelFamily::parse(&self.family)
    }

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
        let family = manifest.family()?;
        if manifest.shape.family() != family {
            bail!(
                "model manifest says family `{}` but its shape is `{}`'s",
                family.name(),
                manifest.shape.family().name()
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

/// `mistral-dense`'s shape.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct MistralShape {
    pub hidden_size: u32,
    pub num_hidden_layers: u32,
    pub num_attention_heads: u32,
    pub num_key_value_heads: u32,
    pub head_dim: u32,
    pub intermediate_size: u32,
    pub vocab_size: u32,
    pub norm_eps: i64,
    /// `2^32 / sqrt(head_dim)`, rounded down: what an attention score is
    /// multiplied by.
    pub attention_scale_q32: u32,
    /// One Q32.32 angle per position for each consecutive pair of a head's
    /// values. The model's YaRN scaling is already in these.
    pub rope_inverse_frequencies: Vec<i64>,
    /// Positions a run may use. From here on the model scales its queries by
    /// position, which is not built.
    pub max_positions: u32,
    /// The head multiplies by the token embedding table.
    pub tied_head: bool,
}

/// A model's shape: the numbers its family's code reads. Each family has its
/// own, and a manifest holds the one its `family` names.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(untagged)]
pub enum FamilyShape {
    GemmaDense(DirectInferShape),
    MistralDense(MistralShape),
}

impl FamilyShape {
    pub fn family(&self) -> ModelFamily {
        match self {
            Self::GemmaDense(_) => ModelFamily::GemmaDense,
            Self::MistralDense(_) => ModelFamily::MistralDense,
        }
    }

    pub fn hidden_size(&self) -> u32 {
        match self {
            Self::GemmaDense(shape) => shape.hidden_size,
            Self::MistralDense(shape) => shape.hidden_size,
        }
    }

    pub fn num_hidden_layers(&self) -> u32 {
        match self {
            Self::GemmaDense(shape) => shape.num_hidden_layers,
            Self::MistralDense(shape) => shape.num_hidden_layers,
        }
    }

    pub fn vocab_size(&self) -> u32 {
        match self {
            Self::GemmaDense(shape) => shape.vocab_size,
            Self::MistralDense(shape) => shape.vocab_size,
        }
    }

    /// `gemma-dense`'s shape, or why this model has none.
    pub fn gemma(&self) -> Result<&DirectInferShape> {
        match self {
            Self::GemmaDense(shape) => Ok(shape),
            other => bail!(
                "this needs a gemma-dense model, and the model is {}",
                other.family().name()
            ),
        }
    }

    /// `mistral-dense`'s shape, or why this model has none.
    pub fn mistral(&self) -> Result<&MistralShape> {
        match self {
            Self::MistralDense(shape) => Ok(shape),
            other => bail!(
                "this needs a mistral-dense model, and the model is {}",
                other.family().name()
            ),
        }
    }
}

impl From<DirectInferShape> for FamilyShape {
    fn from(shape: DirectInferShape) -> Self {
        Self::GemmaDense(shape)
    }
}

impl From<MistralShape> for FamilyShape {
    fn from(shape: MistralShape) -> Self {
        Self::MistralDense(shape)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DirectInferProvenance {
    pub raster_manifest_path: PathBuf,
    pub raster_manifest_sha256: String,
}
