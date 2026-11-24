//! Reusable membership paths for selectively consumed model lists.
//! Roots are authenticated as part of the component header before use.
use crate::{stream, ModelFault};
use alloc::vec::Vec;
use raster::{BytesPage, List};
use raster_core::input::{
    Hash32, ListProofDirection, ListProofSibling, SelectionProof, SelectionProofStep, SelectorPath,
    SelectorSegment,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct SparsePath {
    pub siblings: BytesPage,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct SparseHeader {
    pub root: BytesPage,
    pub count: u64,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct SparseWitness {
    pub header: SparseHeader,
    pub paths: List<SparsePath>,
}

/// Build once at import, without changing the committed model payload.
pub fn witness<T: Serialize>(values: &[T]) -> Result<SparseWitness, ModelFault> {
    let roots = values
        .iter()
        .map(|v| raster_core::tree::value_root(v).map_err(|_| ModelFault::ValueEncoding))
        .collect::<Result<Vec<_>, _>>()?;
    let mut frontier = stream::empty();
    for root in &roots {
        frontier = stream::append(&frontier, &stream::page(root));
    }
    let root = stream::finish(&frontier);
    let mut levels = Vec::new();
    let mut current = roots;
    while current.len() > 1 {
        if current.len() % 2 != 0 {
            current.push(*current.last().unwrap());
        }
        let next = current
            .chunks_exact(2)
            .map(|pair| {
                let mut h = Sha256::new();
                h.update(b"list-node");
                h.update(pair[0]);
                h.update(pair[1]);
                Hash32::from(h.finalize())
            })
            .collect();
        levels.push(current);
        current = next;
    }
    let paths = (0..values.len())
        .map(|mut index| {
            let mut bytes = Vec::with_capacity(levels.len() * 32);
            for level in &levels {
                bytes.extend_from_slice(&level[index ^ 1]);
                index /= 2;
            }
            SparsePath {
                siblings: stream::page(&bytes),
            }
        })
        .collect::<Vec<_>>()
        .into();
    Ok(SparseWitness {
        header: SparseHeader {
            root,
            count: values.len() as u64,
        },
        paths,
    })
}

/// The index comes from the numerical lookup, never from the witness.
/// The caller supplies the digest of the complete selected bucket or page.
pub fn verify(
    root: &BytesPage,
    count: u64,
    index: u64,
    path: &SparsePath,
    value: &BytesPage,
) -> bool {
    let Ok(root_hash) = Hash32::try_from(root.as_slice()) else {
        return false;
    };
    let Ok(value_hash) = Hash32::try_from(value.as_slice()) else {
        return false;
    };
    if index >= count || path.siblings.as_slice().len() > 64 * 32 {
        return false;
    }
    let mut position = index;
    let mut width = count;
    let mut chunks = path.siblings.as_slice().chunks_exact(32);
    if !chunks.remainder().is_empty() {
        return false;
    }
    let mut siblings = Vec::new();
    while width > 1 {
        let Some(hash) = chunks.next() else {
            return false;
        };
        siblings.push(ListProofSibling {
            direction: if position % 2 == 0 {
                ListProofDirection::Right
            } else {
                ListProofDirection::Left
            },
            hash: hash.try_into().unwrap(),
        });
        position /= 2;
        width = width / 2 + width % 2;
    }
    if chunks.next().is_some() {
        return false;
    }
    raster_core::input::verify_selection_reference(
        &value_hash,
        &SelectionProof {
            root_hash,
            path: SelectorPath::new(alloc::vec![SelectorSegment::Index(index)]),
            steps: alloc::vec![SelectionProofStep::List {
                index,
                len: count,
                siblings
            }],
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn every_padding_shape_and_index_is_bound() {
        for count in 0..130usize {
            let values = (0..count as u32).collect::<Vec<_>>();
            let w = witness(&values).unwrap();
            assert_eq!(w.header.root, stream::digest(&values));
            for (index, value) in values.iter().enumerate() {
                let path = &w.paths.as_slice()[index];
                assert!(verify(
                    &w.header.root,
                    w.header.count,
                    index as u64,
                    path,
                    &stream::digest(value)
                ));
                assert!(!verify(
                    &w.header.root,
                    w.header.count,
                    index as u64,
                    path,
                    &stream::digest(&(value + 1))
                ));
                assert!(!verify(
                    &w.header.root,
                    w.header.count + 1,
                    index as u64,
                    path,
                    &stream::digest(value)
                ));
                assert!(!verify(
                    &w.header.root,
                    w.header.count,
                    w.header.count,
                    path,
                    &stream::digest(value)
                ));
            }
        }
    }
    #[test]
    fn omitted_extra_and_relabelled_paths_fail() {
        let values = [10u32, 20, 30, 40, 50];
        let w = witness(&values).unwrap();
        for i in 0..values.len() {
            let value = stream::digest(&values[i]);
            let good = &w.paths.as_slice()[i];
            for len in [0, 1, 31, 32, 64, 97, 2080] {
                let bad = SparsePath {
                    siblings: stream::page(&alloc::vec![0;len]),
                };
                assert!(!verify(
                    &w.header.root,
                    w.header.count,
                    i as u64,
                    &bad,
                    &value
                ));
            }
            assert!(!verify(
                &w.header.root,
                w.header.count,
                ((i + 1) % values.len()) as u64,
                good,
                &value
            ));
        }
    }
    #[test]
    fn unread_data_need_not_be_scanned_but_read_data_cannot_change() {
        let mut values = [1u32, 2, 3];
        let w = witness(&values).unwrap();
        values[2] = 99;
        assert!(verify(
            &w.header.root,
            w.header.count,
            0,
            &w.paths.as_slice()[0],
            &stream::digest(&values[0])
        ));
        assert!(!verify(
            &w.header.root,
            w.header.count,
            2,
            &w.paths.as_slice()[2],
            &stream::digest(&values[2])
        ));
        let replacement = witness(&values).unwrap();
        assert!(!verify(
            &w.header.root,
            w.header.count,
            0,
            &replacement.paths.as_slice()[0],
            &stream::digest(&values[0])
        ));
    }
}
