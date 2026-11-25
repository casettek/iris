//! Adapters from legacy JSON model manifests to the bounded stage descriptor.
use crate::direct::{ModelCommitment, ModelManifest};
use anyhow::{ensure, Context, Result};
use model_identity::{ModelDescriptor, DESCRIPTOR_VERSION, MAX_LAYERS};

pub const MODEL_DESCRIPTOR_BIN: &str = "model-descriptor.bin";

pub fn descriptor_from_commitments(
    entries: &[ModelCommitment],
    layers: usize,
) -> Result<ModelDescriptor> {
    ensure!(
        layers > 0 && layers <= MAX_LAYERS,
        "unsupported model layer count"
    );
    // A model whose layers take no per-layer input commits no auxiliary roots.
    let auxiliary = entries.len() == 4 + 2 * layers;
    ensure!(
        auxiliary || entries.len() == 4 + layers,
        "model commitment role count mismatch"
    );
    let first_layer = if auxiliary { 4 + layers } else { 4 };
    let root = |index: usize, role: &str| -> Result<[u8; 32]> {
        let entry = &entries[index];
        ensure!(
            entry.role == role,
            "expected model role {role}, found {}",
            entry.role
        );
        ensure!(
            entry.commitment.len() == 64
                && entry
                    .commitment
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
            "model root must be canonical lowercase SHA-256"
        );
        let mut bytes = [0; 32];
        hex::decode_to_slice(&entry.commitment, &mut bytes).context("invalid model root")?;
        Ok(bytes)
    };
    Ok(ModelDescriptor {
        version: DESCRIPTOR_VERSION,
        tokenizer: root(0, "tokenizer")?,
        decoder: root(1, "decoder")?,
        embedding: root(2, "embedding")?,
        head: root(3, "head")?,
        auxiliary: (0..if auxiliary { layers } else { 0 })
            .map(|i| root(4 + i, &format!("aux_layer_{i}")))
            .collect::<Result<_>>()?,
        transformer: (0..layers)
            .map(|i| root(first_layer + i, &format!("transformer_layer_{i}")))
            .collect::<Result<_>>()?,
    })
}

impl ModelManifest {
    pub fn descriptor(&self) -> Result<ModelDescriptor> {
        let descriptor = descriptor_from_commitments(
            &self.model_commitments,
            self.shape.num_hidden_layers() as usize,
        )?;
        let id = descriptor
            .model_id()
            .map_err(|fault| anyhow::anyhow!("invalid descriptor: {fault:?}"))?;
        ensure!(
            hex::encode(id) == self.model_id,
            "descriptor disagrees with declared model ID"
        );
        Ok(descriptor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::direct::model_id_of;
    fn entries() -> Vec<ModelCommitment> {
        [
            "tokenizer",
            "decoder",
            "embedding",
            "head",
            "aux_layer_0",
            "transformer_layer_0",
        ]
        .into_iter()
        .enumerate()
        .map(|(i, role)| ModelCommitment {
            role: role.into(),
            commitment: hex::encode([i as u8; 32]),
        })
        .collect()
    }
    #[test]
    fn compact_id_matches_existing_manifest_id() {
        let entries = entries();
        let descriptor = descriptor_from_commitments(&entries, 1).unwrap();
        assert_eq!(
            hex::encode(descriptor.model_id().unwrap()),
            model_id_of(&entries)
        );
        assert_eq!(
            ModelDescriptor::decode(&descriptor.encode().unwrap()).unwrap(),
            descriptor
        );
    }
    #[test]
    fn malformed_role_packages_are_not_accepted() {
        let valid = entries();
        let mut swapped = valid.clone();
        swapped.swap(4, 5);
        assert!(descriptor_from_commitments(&swapped, 1).is_err());
        let mut duplicate = valid.clone();
        duplicate[1] = duplicate[0].clone();
        assert!(descriptor_from_commitments(&duplicate, 1).is_err());
        assert!(descriptor_from_commitments(&valid[..5], 1).is_err());
        assert!(descriptor_from_commitments(&valid, usize::MAX).is_err());
        let mut invalid = valid;
        invalid[0].commitment = "FF".repeat(32);
        assert!(descriptor_from_commitments(&invalid, 1).is_err());
    }
}
