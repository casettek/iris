//! Native mirror of the authored sparse tokenizer lookups.
use inference_kernels::kernels::prompt_prepare::{best_merge, MERGES_PER_BATCH};
use model_identity::{sparse, stream};
use prompt_prepare::input::*;

pub(crate) fn root(t: &PromptTokenizer, w: &TokenizerWitness) -> raster::BytesPage {
    stream::structure(&[
        ("vocab_bucket_count", stream::digest(&t.vocab_bucket_count)),
        ("merge_bucket_count", stream::digest(&t.merge_bucket_count)),
        (
            "boundary_bucket_count",
            stream::digest(&t.boundary_bucket_count),
        ),
        ("profile", stream::digest(&t.profile)),
        ("vocab_buckets", w.vocab.root.clone()),
        ("merge_buckets", w.merges.root.clone()),
        ("boundary_buckets", w.boundaries.root.clone()),
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
/// Merge buckets whose proof this stage has already checked, and the answer.
///
/// A bucket's check depends only on the tokenizer, the witness and the bucket's
/// index, none of which change within a stage, so repeating it gives the same
/// answer. Every bucket a stage reads is still checked against that stage's
/// witness; a merge round rereads almost the same buckets as the round before.
type CheckedBuckets = std::collections::HashMap<u32, bool>;

fn merges(
    t: &PromptTokenizer,
    w: &TokenizerWitness,
    pieces: &[BpePiece],
    checked: &mut CheckedBuckets,
) -> bool {
    if t.merge_bucket_count == 0 {
        return false;
    }
    // Raster performs the lookup even for the first piece and segment boundaries.
    let mut previous = "";
    let mut valid = true;
    for piece in pieces {
        let i = merge_bucket_of(previous, &piece.text, t.merge_bucket_count);
        valid &= *checked.entry(i).or_insert_with(|| {
            check(
                t.merge_buckets.as_slice().get(i as usize),
                &w.merges,
                &w.merge_paths,
                i,
            )
        });
        previous = &piece.text;
    }
    valid
}
pub(crate) fn batch(t: &PromptTokenizer, w: &TokenizerWitness, pieces: &[BpePiece]) -> bool {
    let mut pieces = pieces.to_vec();
    let mut valid = true;
    let mut checked = CheckedBuckets::new();
    for _ in 0..MERGES_PER_BATCH {
        valid &= merges(t, w, &pieces, &mut checked);
        let Ok(best) = best_merge(&pieces, t) else {
            return false;
        };
        if best.matched {
            let i = best.left as usize;
            pieces[i].text = best.merged;
            pieces.remove(i + 1);
        } else {
            break;
        } // Further Raster rounds query exactly the same buckets.
    }
    valid
}
pub(crate) fn prepare(t: &PromptTokenizer, w: &TokenizerWitness, pieces: &[BpePiece]) -> bool {
    merges(t, w, pieces, &mut CheckedBuckets::new()) && pieces.iter().all(|p| vocab(t, w, &p.text))
}
pub(crate) fn bind(
    t: &PromptTokenizer,
    w: &TokenizerWitness,
    text: &PromptText,
    config: &RunConfig,
) -> bool {
    let fold = text
        .chunks
        .iter()
        .fold(begin_text_fold_state(), |f, c| fold_text(f, c));
    let plan = scan_plan_of(&fold, config, &t.profile);
    let mut scan = begin_scan_state(&plan);
    let mut valid = true;
    for chunk in text.chunks.iter() {
        scan = scan_chunk_into(scan, chunk, &plan, &t.profile, &mut |atom| {
            valid &= vocab(t, w, &atom.text);
        });
    }
    valid
}
pub(crate) fn seed(
    t: &PromptTokenizer,
    w: &TokenizerWitness,
    bound: &PromptState,
    previous: &PromptContext,
    cut: &RangeCut,
) -> bool {
    let previous = join_prompt_models(previous.clone(), &bound.context);
    let mut slice = begin_slice_state(&previous, cut.end);
    for piece in bound.pieces.iter() {
        observe_piece(&mut slice, piece);
    }
    let query = boundary_query_of(&slice);
    if t.boundary_bucket_count == 0 {
        return false;
    }
    let i = boundary_bucket_of(query.left, query.right, t.boundary_bucket_count);
    let valid = check(
        t.boundary_buckets.as_slice().get(i as usize),
        &w.boundaries,
        &w.boundary_paths,
        i,
    );
    let pieces = bound
        .pieces
        .iter()
        .enumerate()
        .filter(|(i, _)| (*i as u64) >= slice.start && (*i as u64) < slice.end)
        .map(|(_, p)| p.clone())
        .collect::<Vec<_>>();
    valid && batch(t, w, &pieces)
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
            boundary_bucket_count: 1,
            vocab_buckets: vec![VocabBucket::default(); 4].into(),
            merge_buckets: vec![MergeBucket::default()].into(),
            boundary_buckets: vec![BoundaryBucket::default()].into(),
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

    /// A batch checks each bucket it reads once, and a bucket that does not
    /// match the witness still fails the batch, whichever round first reads it.
    #[test]
    fn a_batch_fails_on_any_altered_merge_bucket_it_reads() {
        const BUCKETS: u32 = 16;
        let rule = |left: &str, right: &str, rank| BpeMerge {
            rank,
            left: left.into(),
            right: right.into(),
            merged: format!("{left}{right}"),
        };
        let mut buckets = vec![MergeBucket::default(); BUCKETS as usize];
        for merge in [rule("a", "b", 0), rule("ab", "c", 1)] {
            let index = merge_bucket_of(&merge.left, &merge.right, BUCKETS) as usize;
            let mut rules = buckets[index].rules.as_slice().to_vec();
            rules.push(merge);
            buckets[index].rules = rules.into();
        }
        let tokenizer = |buckets: &[MergeBucket]| PromptTokenizer {
            vocab_bucket_count: 1,
            merge_bucket_count: BUCKETS,
            boundary_bucket_count: 1,
            vocab_buckets: vec![VocabBucket::default()].into(),
            merge_buckets: buckets.to_vec().into(),
            boundary_buckets: vec![BoundaryBucket::default()].into(),
            ..Default::default()
        };
        let honest = tokenizer(&buckets);
        let w = tokenizer_witness(&honest).unwrap();
        let pieces = ["a", "b", "c"].map(|text| BpePiece {
            text: text.into(),
            segment: 0,
        });
        assert!(batch(&honest, &w, &pieces));

        // Every lookup the three rounds make: the first piece has no left
        // neighbour, and the third round reads the bucket for the lone piece.
        let read = [
            ("", "a"),
            ("a", "b"),
            ("b", "c"),
            ("", "ab"),
            ("ab", "c"),
            ("", "abc"),
        ]
        .map(|(left, right)| merge_bucket_of(left, right, BUCKETS) as usize);
        for index in 0..BUCKETS as usize {
            let mut altered = buckets.clone();
            let mut rules = altered[index].rules.as_slice().to_vec();
            rules.push(rule("x", "y", 9));
            altered[index].rules = rules.into();
            assert_eq!(
                batch(&tokenizer(&altered), &w, &pieces),
                !read.contains(&index),
                "bucket {index}"
            );
        }
    }
}
