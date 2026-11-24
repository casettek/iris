//! Bounded canonical structural hashing for complete component sweeps.
use crate::{ModelContext, ModelDescriptor, Role};
use alloc::vec::Vec;
use raster::BytesPage;
use raster_core::input::{struct_commitments_root, AppendFrontier, Hash32};
use serde::Serialize;

pub fn page(bytes: &[u8]) -> BytesPage {
    BytesPage::__from_parts(0, 0, bytes.into())
}
pub fn digest<T: Serialize>(value: &T) -> BytesPage {
    page(&raster_core::tree::value_root(value).expect("serializable model value"))
}
pub fn structure(fields: &[(&str, BytesPage)]) -> BytesPage {
    page(&struct_commitments_root(
        fields.iter().map(|(name, hash)| (*name, hash.as_slice())),
    ))
}
pub fn empty() -> BytesPage {
    page(&0u64.to_le_bytes())
}
fn decode(bytes: &BytesPage) -> Option<AppendFrontier> {
    let b = bytes.as_slice();
    let len = u64::from_le_bytes(b.get(..8)?.try_into().ok()?);
    if len == u64::MAX {
        return None;
    }
    let count = if len == 0 {
        0
    } else {
        1 + (len - 1).count_ones() as usize
    };
    if b.len() != 8 + 32 * count {
        return None;
    }
    let mut chunks = b[8..]
        .chunks_exact(32)
        .map(|v| <Hash32>::try_from(v).unwrap());
    Some(AppendFrontier {
        len,
        leaf: chunks.next(),
        ommers: chunks.collect(),
    })
}
pub fn append(state: &BytesPage, root: &BytesPage) -> BytesPage {
    let Some(mut frontier) = decode(state) else {
        return page(&[]);
    };
    let Ok(hash) = <Hash32>::try_from(root.as_slice()) else {
        return page(&[]);
    };
    if frontier.len == u64::MAX {
        return page(&[]);
    }
    frontier.push(hash);
    let mut bytes = Vec::with_capacity(8 + 32 * 65);
    bytes.extend_from_slice(&frontier.len.to_le_bytes());
    if let Some(leaf) = frontier.leaf {
        bytes.extend_from_slice(&leaf);
    }
    for hash in frontier.ommers {
        bytes.extend_from_slice(&hash);
    }
    page(&bytes)
}
pub fn finish(state: &BytesPage) -> BytesPage {
    decode(state)
        .and_then(|f| f.root())
        .map(|v| page(&v))
        .unwrap_or_else(|| page(&[]))
}
pub fn verify(mut context: ModelContext, actual: &BytesPage, role: Role) -> ModelContext {
    let valid = (|| {
        let d = ModelDescriptor::decode(context.descriptor.as_slice()).ok()?;
        let id = crate::context::parse_id(&context.model_id)?;
        let bound = d.bind(&id).ok()?;
        Some(
            actual.as_slice() == bound.root(role).ok()?.as_slice()
                && context.layers as usize == d.transformer.len(),
        )
    })() == Some(true);
    context.require(valid);
    context
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::{format, string::String, vec};
    #[test]
    fn streaming_matches_whole_lists_for_all_padding_shapes() {
        for len in 0..257 {
            let values: Vec<u32> = (0..len).collect();
            let mut fold = empty();
            for value in &values {
                fold = append(&fold, &digest(value));
            }
            assert_eq!(finish(&fold), digest(&values), "length {len}");
            assert!(fold.as_slice().len() <= 8 + 32 * 65);
        }
    }
    #[test]
    fn malformed_frontiers_fail_boundedly() {
        for len in 0..2200 {
            assert!(finish(&page(&vec![255; len])).as_slice().is_empty());
        }
        assert!(append(&empty(), &page(&[])).as_slice().is_empty());
    }
    #[test]
    fn omitted_repeated_reordered_or_changed_values_do_not_authenticate() {
        let values = vec![4u32, 7, 9];
        let root: Hash32 = digest(&values).as_slice().try_into().unwrap();
        let d = ModelDescriptor {
            version: 2,
            tokenizer: root,
            decoder: root,
            embedding: root,
            head: root,
            auxiliary: vec![root],
            transformer: vec![root],
        };
        let id = d
            .model_id()
            .unwrap()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>();
        let context = ModelContext::bind(&d.encode().unwrap(), &id);
        assert_eq!(
            verify(context.clone(), &digest(&values), Role::Embedding).fault,
            0
        );
        for bad in [
            vec![4u32, 7],
            vec![4, 7, 9, 9],
            vec![7, 4, 9],
            vec![4, 7, 10],
            vec![],
        ] {
            let mut fold = empty();
            for value in bad {
                fold = append(&fold, &digest(&value));
            }
            assert_ne!(
                verify(context.clone(), &finish(&fold), Role::Embedding).fault,
                0
            );
        }
    }
}

#[cfg(test)]
mod page_tests {
    use super::*;
    use alloc::{format, string::String, vec};
    use raster::{Bytes, List};
    #[test]
    fn every_byte_container_field_and_page_position_changes_identity() {
        let honest = Bytes::<4>::paged(vec![1, 2, 3, 4, 5, 6, 7, 8, 9]).unwrap();
        let root: Hash32 = digest(&honest).as_slice().try_into().unwrap();
        let d = ModelDescriptor {
            version: 2,
            tokenizer: root,
            decoder: root,
            embedding: root,
            head: root,
            auxiliary: vec![root],
            transformer: vec![root],
        };
        let id = d
            .model_id()
            .unwrap()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>();
        let context = ModelContext::bind(&d.encode().unwrap(), &id);
        let mut folded = empty();
        for p in honest.pages.iter() {
            folded = append(&folded, &digest(p));
        }
        let streamed = structure(&[
            ("byte_len", digest(&honest.byte_len)),
            ("page_size", digest(&honest.page_size)),
            ("pages", finish(&folded)),
        ]);
        assert_eq!(streamed, digest(&honest));
        assert_eq!(verify(context.clone(), &streamed, Role::Embedding).fault, 0);
        for case in 0..7 {
            let mut changed = honest.clone();
            let mut pages = changed.pages.as_slice().to_vec();
            match case {
                0 => changed.byte_len -= 1,
                1 => changed.page_size += 1,
                2 => {
                    pages.pop();
                }
                3 => pages.push(pages[2].clone()),
                4 => pages.swap(0, 1),
                5 => pages[0] = BytesPage::__from_parts(1, 0, vec![1, 2, 3, 4]),
                _ => pages[0] = BytesPage::__from_parts(0, 4, vec![1, 2, 3, 4]),
            }
            changed.pages = List::from(pages);
            assert_ne!(
                verify(context.clone(), &digest(&changed), Role::Embedding).fault,
                0,
                "case {case}"
            );
        }
    }
}
