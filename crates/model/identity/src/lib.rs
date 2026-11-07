//! Iris model descriptors and verification of consumed values.
//! These checks supplement Raster authentication with the expected model root.
#![no_std]
extern crate alloc;

use alloc::{format, vec::Vec};
use raster_core::input::{Hash32, SelectionProof, SelectorPath};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const DESCRIPTOR_VERSION: u32 = 1;
pub const MAX_LAYERS: usize = 1024;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelDescriptor {
    pub version: u32,
    pub tokenizer: Hash32,
    pub decoder: Hash32,
    pub embedding: Hash32,
    pub head: Hash32,
    pub auxiliary: Vec<Hash32>,
    pub transformer: Vec<Hash32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    Tokenizer,
    Decoder,
    Embedding,
    Head,
    Auxiliary(u32),
    Transformer(u32),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelFault {
    DescriptorVersion,
    LayerCount,
    ModelId,
    LayerIndex,
    SourceRoot,
    SelectionPath,
    ValueEncoding,
    Membership,
    Coverage,
}

impl ModelDescriptor {
    pub fn validate(&self) -> Result<(), ModelFault> {
        if self.version != DESCRIPTOR_VERSION {
            return Err(ModelFault::DescriptorVersion);
        }
        if self.transformer.is_empty()
            || self.transformer.len() > MAX_LAYERS
            || self.auxiliary.len() != self.transformer.len()
        {
            return Err(ModelFault::LayerCount);
        }
        Ok(())
    }

    /// Matches the existing Iris role/NUL/lowercase-hex/newline preimage.
    pub fn model_id(&self) -> Result<Hash32, ModelFault> {
        self.validate()?;
        let mut hash = Sha256::new();
        let mut entry = |role: &str, root: &Hash32| {
            hash.update(role.as_bytes());
            hash.update([0]);
            const HEX: &[u8; 16] = b"0123456789abcdef";
            for byte in root {
                hash.update([HEX[(byte >> 4) as usize], HEX[(byte & 15) as usize]]);
            }
            hash.update([b'\n']);
        };
        entry("tokenizer", &self.tokenizer);
        entry("decoder", &self.decoder);
        entry("embedding", &self.embedding);
        entry("head", &self.head);
        for (index, root) in self.auxiliary.iter().enumerate() {
            entry(&format!("aux_layer_{index}"), root);
        }
        for (index, root) in self.transformer.iter().enumerate() {
            entry(&format!("transformer_layer_{index}"), root);
        }
        Ok(hash.finalize().into())
    }

    pub fn bind(&self, expected: &Hash32) -> Result<BoundModel<'_>, ModelFault> {
        if &self.model_id()? != expected {
            return Err(ModelFault::ModelId);
        }
        Ok(BoundModel { descriptor: self })
    }
}

/// Constructible only after checking the descriptor against the expected ID.
/// This is an in-tile helper, not a transferable proof of stage provenance.
pub struct BoundModel<'a> {
    descriptor: &'a ModelDescriptor,
}

impl BoundModel<'_> {
    pub fn root(&self, role: Role) -> Result<&Hash32, ModelFault> {
        let d = self.descriptor;
        match role {
            Role::Tokenizer => Ok(&d.tokenizer),
            Role::Decoder => Ok(&d.decoder),
            Role::Embedding => Ok(&d.embedding),
            Role::Head => Ok(&d.head),
            Role::Auxiliary(i) => d.auxiliary.get(i as usize).ok_or(ModelFault::LayerIndex),
            Role::Transformer(i) => d.transformer.get(i as usize).ok_or(ModelFault::LayerIndex),
        }
    }

    /// The caller derives role/path from the operation, never from the proof.
    /// Consume this returned value; do not verify one value and use another.
    /// Container geometry and full-sweep coverage must be checked separately.
    pub fn verify_value<T: Serialize>(
        &self,
        role: Role,
        expected_path: &SelectorPath,
        value: T,
        proof: &SelectionProof,
    ) -> Result<T, ModelFault> {
        if &proof.root_hash != self.root(role)? {
            return Err(ModelFault::SourceRoot);
        }
        if &proof.path != expected_path {
            return Err(ModelFault::SelectionPath);
        }
        verify_list_positions(proof)?;
        let root = raster_core::tree::value_root(&value).map_err(|_| ModelFault::ValueEncoding)?;
        if !raster_core::input::verify_selection_reference(&root, proof) {
            return Err(ModelFault::Membership);
        }
        Ok(value)
    }
}

/// Tracks a full ordered sweep. Expected count must come from verified metadata.
pub struct Coverage {
    expected: u64,
    next: u64,
    failed: bool,
}

impl Coverage {
    pub fn new(expected: u64) -> Self {
        Self {
            expected,
            next: 0,
            failed: false,
        }
    }
    pub fn observe(&mut self, index: u64) -> Result<(), ModelFault> {
        if self.failed || index != self.next || self.next >= self.expected {
            self.failed = true;
            return Err(ModelFault::Coverage);
        }
        self.next += 1;
        Ok(())
    }
    pub fn finish(self) -> Result<(), ModelFault> {
        if self.failed || self.next != self.expected {
            return Err(ModelFault::Coverage);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use raster_core::input::SelectorSegment;

    fn descriptor(root: Hash32) -> ModelDescriptor {
        ModelDescriptor {
            version: 1,
            tokenizer: root,
            decoder: [2; 32],
            embedding: [3; 32],
            head: [4; 32],
            auxiliary: vec![[5; 32]],
            transformer: vec![[6; 32]],
        }
    }

    #[test]
    fn actual_value_and_expected_root_are_both_required() {
        let root = raster_core::tree::value_root(&7u32).unwrap();
        let d = descriptor(root);
        let id = d.model_id().unwrap();
        let bound = d.bind(&id).unwrap();
        let path = SelectorPath::new(vec![]);
        let proof = SelectionProof {
            path: path.clone(),
            root_hash: root,
            steps: vec![],
        };
        assert_eq!(
            bound.verify_value(Role::Tokenizer, &path, 7u32, &proof),
            Ok(7)
        );
        assert_eq!(
            bound.verify_value(Role::Tokenizer, &path, 8u32, &proof),
            Err(ModelFault::Membership)
        );
        assert_eq!(
            bound.verify_value(Role::Decoder, &path, 7u32, &proof),
            Err(ModelFault::SourceRoot)
        );
        let other_path = SelectorPath::new(vec![SelectorSegment::Field("other".into())]);
        assert_eq!(
            bound.verify_value(Role::Tokenizer, &other_path, 7u32, &proof),
            Err(ModelFault::SelectionPath)
        );
        assert!(matches!(d.bind(&[0; 32]), Err(ModelFault::ModelId)));
    }

    #[test]
    fn descriptors_reject_bad_shapes_and_bind_every_role() {
        let d = descriptor([1; 32]);
        let id = d.model_id().unwrap();
        let mut changed = d.clone();
        changed.transformer[0][0] ^= 1;
        assert_ne!(changed.model_id().unwrap(), id);
        changed = d.clone();
        changed.auxiliary.clear();
        assert_eq!(changed.model_id(), Err(ModelFault::LayerCount));
        changed = d.clone();
        changed.version += 1;
        assert_eq!(changed.model_id(), Err(ModelFault::DescriptorVersion));
        assert_eq!(
            d.bind(&id).unwrap().root(Role::Transformer(1)),
            Err(ModelFault::LayerIndex)
        );
    }

    #[test]
    fn coverage_rejects_omissions_duplicates_and_ignored_errors() {
        let mut good = Coverage::new(2);
        good.observe(0).unwrap();
        good.observe(1).unwrap();
        good.finish().unwrap();
        let mut short = Coverage::new(2);
        short.observe(0).unwrap();
        assert_eq!(short.finish(), Err(ModelFault::Coverage));
        let mut duplicate = Coverage::new(2);
        duplicate.observe(0).unwrap();
        assert_eq!(duplicate.observe(0), Err(ModelFault::Coverage));
        assert_eq!(duplicate.observe(1), Err(ModelFault::Coverage));
        assert_eq!(duplicate.finish(), Err(ModelFault::Coverage));
    }
}

impl ModelDescriptor {
    /// Fixed-width wire format: version, layer count, then role-ordered roots.
    /// Length is checked before any allocation, including on hostile inputs.
    pub fn decode(bytes: &[u8]) -> Result<Self, ModelFault> {
        if bytes.len() < 8 {
            return Err(ModelFault::ValueEncoding);
        }
        let version = u32::from_le_bytes(bytes[..4].try_into().unwrap());
        if version != DESCRIPTOR_VERSION {
            return Err(ModelFault::DescriptorVersion);
        }
        let layers = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
        if layers == 0 || layers > MAX_LAYERS {
            return Err(ModelFault::LayerCount);
        }
        if bytes.len() != 8 + (4 + 2 * layers) * 32 {
            return Err(ModelFault::ValueEncoding);
        }
        let mut roots = bytes[8..].chunks_exact(32).map(|b| {
            let mut root = [0; 32];
            root.copy_from_slice(b);
            root
        });
        Ok(Self {
            version,
            tokenizer: roots.next().unwrap(),
            decoder: roots.next().unwrap(),
            embedding: roots.next().unwrap(),
            head: roots.next().unwrap(),
            auxiliary: roots.by_ref().take(layers).collect(),
            transformer: roots.collect(),
        })
    }

    pub fn encode(&self) -> Result<Vec<u8>, ModelFault> {
        self.validate()?;
        let mut bytes = Vec::with_capacity(8 + (4 + 2 * self.transformer.len()) * 32);
        bytes.extend_from_slice(&self.version.to_le_bytes());
        bytes.extend_from_slice(&(self.transformer.len() as u32).to_le_bytes());
        for root in [&self.tokenizer, &self.decoder, &self.embedding, &self.head]
            .into_iter()
            .chain(self.auxiliary.iter())
            .chain(self.transformer.iter())
        {
            bytes.extend_from_slice(root);
        }
        Ok(bytes)
    }
}

#[cfg(test)]
mod wire_tests {
    use super::*;
    use alloc::vec;
    #[test]
    fn exact_length_and_layer_bound_are_checked_before_allocation() {
        let d = ModelDescriptor {
            version: 1,
            tokenizer: [1; 32],
            decoder: [2; 32],
            embedding: [3; 32],
            head: [4; 32],
            auxiliary: vec![[5; 32]; 35],
            transformer: vec![[6; 32]; 35],
        };
        let encoded = d.encode().unwrap();
        assert_eq!(encoded.len(), 2376);
        assert_eq!(ModelDescriptor::decode(&encoded).unwrap(), d);
        for end in 0..encoded.len() {
            assert!(ModelDescriptor::decode(&encoded[..end]).is_err());
        }
        let mut extra = encoded.clone();
        extra.push(0);
        assert_eq!(
            ModelDescriptor::decode(&extra),
            Err(ModelFault::ValueEncoding)
        );
        let mut huge = encoded;
        huge[4..8].copy_from_slice(&u32::MAX.to_le_bytes());
        assert_eq!(ModelDescriptor::decode(&huge), Err(ModelFault::LayerCount));
    }
}

/// Raster's root folding is reused, but Iris additionally pins sibling geometry
/// to each list index. A matching path string alone cannot establish position.
fn verify_list_positions(proof: &SelectionProof) -> Result<(), ModelFault> {
    use raster_core::input::{ListProofDirection, SelectionProofStep};
    for step in &proof.steps {
        if let SelectionProofStep::List {
            index,
            len,
            siblings,
        } = step
        {
            if *len == 0 || index >= len {
                return Err(ModelFault::Membership);
            }
            let (mut at, mut width) = (*index, *len);
            let mut levels = siblings.iter();
            while width > 1 {
                let sibling = levels.next().ok_or(ModelFault::Membership)?;
                let expected = if at % 2 == 0 {
                    ListProofDirection::Right
                } else {
                    ListProofDirection::Left
                };
                if sibling.direction != expected {
                    return Err(ModelFault::Membership);
                }
                at /= 2;
                width = width / 2 + width % 2;
            }
            if levels.next().is_some() {
                return Err(ModelFault::Membership);
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod position_tests {
    use super::*;
    use alloc::vec;
    use raster_core::input::{
        ListProofDirection, ListProofSibling, SelectionProofStep, SelectorSegment,
    };
    #[test]
    fn membership_proof_cannot_be_relabelled_to_another_index() {
        let values = vec![7u32, 9u32];
        let root = raster_core::tree::value_root(&values).unwrap();
        let d = ModelDescriptor {
            version: 1,
            tokenizer: root,
            decoder: root,
            embedding: root,
            head: root,
            auxiliary: vec![root],
            transformer: vec![root],
        };
        let bound = d.bind(&d.model_id().unwrap()).unwrap();
        let path = SelectorPath::new(vec![SelectorSegment::Index(0)]);
        let mut proof = SelectionProof {
            root_hash: root,
            path: path.clone(),
            steps: vec![SelectionProofStep::List {
                index: 0,
                len: 2,
                siblings: vec![ListProofSibling {
                    direction: ListProofDirection::Right,
                    hash: raster_core::tree::value_root(&9u32).unwrap(),
                }],
            }],
        };
        assert_eq!(
            bound.verify_value(Role::Tokenizer, &path, 7u32, &proof),
            Ok(7)
        );
        proof.path = SelectorPath::new(vec![SelectorSegment::Index(1)]);
        if let SelectionProofStep::List { index, .. } = &mut proof.steps[0] {
            *index = 1;
        }
        assert_eq!(
            bound.verify_value(Role::Tokenizer, &proof.path, 7u32, &proof),
            Err(ModelFault::Membership)
        );
    }
}

pub mod context;
pub use context::ModelContext;

pub mod stream;

pub mod sparse;

mod pipeline;
pub use pipeline_core;
