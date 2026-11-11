//! Native mirror of the authored sparse tokenizer lookups.
use model_identity::{sparse, stream};
use prompt_check::input::*;

pub(crate) fn root(t: &PromptTokenizer, w: &TokenizerWitness) -> raster::BytesPage {
    stream::structure(&[
        ("vocab_bucket_count", stream::digest(&t.vocab_bucket_count)),
        ("merge_bucket_count", stream::digest(&t.merge_bucket_count)),
        ("profile", stream::digest(&t.profile)),
        ("vocab_buckets", w.vocab.root.clone()),
        ("merge_buckets", w.merges.root.clone()),
    ])
}
fn check<T: serde::Serialize>(
    value: Option<&T>,
    header: &SparseHeader,
    paths: &raster::List<SparsePath>,
    index: u32,
) -> bool {
    value
        .zip(paths.as_slice().get(index as usize))
        .map(|(v, p)| {
            sparse::verify(
                &header.root,
                header.count,
                u64::from(index),
                p,
                &stream::digest(v),
            )
        })
        .unwrap_or(false)
}
/// Whether the bucket `text` hashes to is the one the witness commits to.
#[cfg(test)]
fn vocab(t: &PromptTokenizer, w: &TokenizerWitness, text: &str) -> bool {
    if t.vocab_bucket_count == 0 {
        return false;
    }
    let i = vocab_bucket_of(text, t.vocab_bucket_count);
    check(
        t.vocab_buckets.as_slice().get(i as usize),
        &w.vocab,
        &w.vocab_paths,
        i,
    )
}
/// Whether merge bucket `index` is the one the witness commits to.
pub(crate) fn merge_bucket(t: &PromptTokenizer, w: &TokenizerWitness, index: u32) -> bool {
    check(
        t.merge_buckets.as_slice().get(index as usize),
        &w.merges,
        &w.merge_paths,
        index,
    )
}
/// Whether vocabulary bucket `index` is the one the witness commits to.
pub(crate) fn vocab_bucket(t: &PromptTokenizer, w: &TokenizerWitness, index: u32) -> bool {
    check(
        t.vocab_buckets.as_slice().get(index as usize),
        &w.vocab,
        &w.vocab_paths,
        index,
    )
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sparse_tokenizer_authenticates_complete_selected_buckets() {
        let text = "hello";
        let index = vocab_bucket_of(text, 4) as usize;
        let mut t = PromptTokenizer {
            vocab_bucket_count: 4,
            merge_bucket_count: 1,
            vocab_buckets: vec![VocabBucket::default(); 4].into(),
            merge_buckets: vec![MergeBucket::default()].into(),
            ..Default::default()
        };
        let mut buckets = t.vocab_buckets.as_slice().to_vec();
        buckets[index].entries = vec![
            TokenEntry {
                token: text.into(),
                id: 7,
            },
            TokenEntry {
                token: "collision".into(),
                id: 8,
            },
        ]
        .into();
        t.vocab_buckets = buckets.clone().into();
        let w = tokenizer_witness(&t).unwrap();
        assert_eq!(root(&t, &w), stream::digest(&t));
        assert!(vocab(&t, &w, text));
        buckets[(index + 1) % 4].entries = vec![TokenEntry {
            token: "unread".into(),
            id: 9,
        }]
        .into();
        t.vocab_buckets = buckets.clone().into();
        assert!(vocab(&t, &w, text));
        let original = buckets[index].entries.as_slice().to_vec();
        for entries in [
            vec![original[0].clone()],
            vec![original[1].clone(), original[0].clone()],
            vec![
                original[0].clone(),
                original[1].clone(),
                original[1].clone(),
            ],
        ] {
            buckets[index].entries = entries.into();
            t.vocab_buckets = buckets.clone().into();
            assert!(
                !vocab(&t, &w, text),
                "complete bucket must authenticate, even if matching entry is unchanged"
            );
        }
        t.profile.fuse_unk = !t.profile.fuse_unk;
        assert_ne!(root(&t, &w), stream::digest(&PromptTokenizer::default()));
    }
}
