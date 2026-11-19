//! Celestia's commitment to a blob, computed here so that iris can name a
//! blob before it posts it and check the one it reads back.
//!
//! A blob is cut into 512-byte shares under its namespace. The shares are
//! grouped into subtrees whose sizes follow a Merkle mountain range, each
//! subtree's root is taken in a namespaced Merkle tree, and the commitment is
//! the Merkle root over those roots. This is the commitment a
//! `MsgPayForBlobs` signs and a node's blob API names a blob by
//! (celestia-app's `inclusion.CreateCommitment`, with a subtree root
//! threshold of 64). The tests hold it to commitments a local celestia-node
//! 0.35.0 computed.
//!
//! Share version 0 only: a blob with no signer in its shares.

use sha2::{Digest, Sha256};

/// A namespace: one version byte and 28 bytes of id.
pub const NAMESPACE_SIZE: usize = 29;
/// A share's size in bytes.
pub const SHARE_SIZE: usize = 512;
/// Shares per subtree root, at most, before subtrees widen.
const SUBTREE_ROOT_THRESHOLD: usize = 64;
/// Bytes of blob data in a blob's first share: after the namespace, the info
/// byte and the 4-byte length of the whole blob.
const FIRST_SHARE_DATA: usize = SHARE_SIZE - NAMESPACE_SIZE - 1 - 4;
/// Bytes of blob data in every later share.
const LATER_SHARE_DATA: usize = SHARE_SIZE - NAMESPACE_SIZE - 1;

/// A node of a namespaced Merkle tree: the least and greatest namespace
/// below it, and its digest.
type NamespacedNode = [u8; 2 * NAMESPACE_SIZE + 32];

/// How many shares a blob of this many bytes takes.
pub fn share_count(data_length: usize) -> usize {
    if data_length <= FIRST_SHARE_DATA {
        1
    } else {
        1 + (data_length - FIRST_SHARE_DATA).div_ceil(LATER_SHARE_DATA)
    }
}

/// The blob's shares, in order. The first states the blob's length; the last
/// is padded with zeros.
pub fn shares(namespace: &[u8; NAMESPACE_SIZE], data: &[u8]) -> Vec<[u8; SHARE_SIZE]> {
    let mut shares = Vec::with_capacity(share_count(data.len()));
    let mut rest = data;
    loop {
        let first = shares.is_empty();
        let mut share = [0u8; SHARE_SIZE];
        share[..NAMESPACE_SIZE].copy_from_slice(namespace);
        // The info byte: the share version, 0, and whether a blob starts here.
        share[NAMESPACE_SIZE] = u8::from(first);
        let mut at = NAMESPACE_SIZE + 1;
        if first {
            share[at..at + 4].copy_from_slice(&(data.len() as u32).to_be_bytes());
            at += 4;
        }
        let take = rest.len().min(SHARE_SIZE - at);
        share[at..at + take].copy_from_slice(&rest[..take]);
        rest = &rest[take..];
        shares.push(share);
        if rest.is_empty() {
            return shares;
        }
    }
}

fn namespaced_leaf(namespace: &[u8; NAMESPACE_SIZE], share: &[u8; SHARE_SIZE]) -> NamespacedNode {
    let mut hasher = Sha256::new();
    hasher.update([0]);
    hasher.update(namespace);
    hasher.update(share);
    let mut node = [0u8; 2 * NAMESPACE_SIZE + 32];
    node[..NAMESPACE_SIZE].copy_from_slice(namespace);
    node[NAMESPACE_SIZE..2 * NAMESPACE_SIZE].copy_from_slice(namespace);
    node[2 * NAMESPACE_SIZE..].copy_from_slice(&hasher.finalize());
    node
}

fn namespaced_parent(left: &NamespacedNode, right: &NamespacedNode) -> NamespacedNode {
    let mut hasher = Sha256::new();
    hasher.update([1]);
    hasher.update(left);
    hasher.update(right);
    let mut node = [0u8; 2 * NAMESPACE_SIZE + 32];
    // A blob's shares are all of one namespace, so the range is the left
    // child's least and the right child's greatest.
    node[..NAMESPACE_SIZE].copy_from_slice(&left[..NAMESPACE_SIZE]);
    node[NAMESPACE_SIZE..2 * NAMESPACE_SIZE]
        .copy_from_slice(&right[NAMESPACE_SIZE..2 * NAMESPACE_SIZE]);
    node[2 * NAMESPACE_SIZE..].copy_from_slice(&hasher.finalize());
    node
}

/// The largest power of two strictly below `count`, for `count` of 2 or more:
/// where a tree of that many leaves splits.
fn split_point(count: usize) -> usize {
    let power = count.next_power_of_two();
    if power == count {
        count / 2
    } else {
        power / 2
    }
}

fn namespaced_root(leaves: &[NamespacedNode]) -> NamespacedNode {
    match leaves {
        [leaf] => *leaf,
        _ => {
            let (left, right) = leaves.split_at(split_point(leaves.len()));
            namespaced_parent(&namespaced_root(left), &namespaced_root(right))
        }
    }
}

/// The Merkle root over byte strings as CometBFT takes it: a leaf is hashed
/// after a zero byte, a pair after a one.
fn merkle_root(leaves: &[&[u8]]) -> [u8; 32] {
    match leaves {
        [] => Sha256::digest([]).into(),
        [leaf] => {
            let mut hasher = Sha256::new();
            hasher.update([0]);
            hasher.update(leaf);
            hasher.finalize().into()
        }
        _ => {
            let (left, right) = leaves.split_at(split_point(leaves.len()));
            let mut hasher = Sha256::new();
            hasher.update([1]);
            hasher.update(merkle_root(left));
            hasher.update(merkle_root(right));
            hasher.finalize().into()
        }
    }
}

/// The width of the subtrees a blob of this many shares is committed by: the
/// shares over the threshold, rounded up to a power of two, and no wider than
/// the smallest square the blob fits in.
fn subtree_width(share_count: usize) -> usize {
    let over_threshold = share_count
        .div_ceil(SUBTREE_ROOT_THRESHOLD)
        .next_power_of_two();
    let mut square = 1;
    while square * square < share_count {
        square += 1;
    }
    over_threshold.min(square.next_power_of_two())
}

/// The commitment to a blob's bytes under a namespace.
pub fn blob_commitment(namespace: &[u8; NAMESPACE_SIZE], data: &[u8]) -> [u8; 32] {
    let leaves: Vec<NamespacedNode> = shares(namespace, data)
        .iter()
        .map(|share| namespaced_leaf(namespace, share))
        .collect();
    // The sizes of a Merkle mountain range over the shares, no tree wider
    // than the subtree width.
    let width = subtree_width(leaves.len());
    let mut roots = Vec::new();
    let mut rest = leaves.as_slice();
    while !rest.is_empty() {
        let size = if rest.len() >= width {
            width
        } else {
            // The largest power of two that fits.
            (rest.len() + 1).next_power_of_two() / 2
        };
        let (tree, after) = rest.split_at(size);
        roots.push(namespaced_root(tree));
        rest = after;
    }
    let roots: Vec<&[u8]> = roots.iter().map(|root| root.as_slice()).collect();
    merkle_root(&roots)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The namespace iris posts claims under: version 0, and `iris-claim` as
    /// the last ten bytes of the id.
    fn namespace() -> [u8; NAMESPACE_SIZE] {
        let mut namespace = [0; NAMESPACE_SIZE];
        namespace[19..].copy_from_slice(b"iris-claim");
        namespace
    }

    fn data(length: usize) -> Vec<u8> {
        (0..length).map(|i| ((i * 7 + 3) % 256) as u8).collect()
    }

    #[test]
    fn a_blob_is_cut_into_shares_that_state_its_length_once() {
        assert_eq!(share_count(0), 1);
        assert_eq!(share_count(478), 1);
        assert_eq!(share_count(479), 2);
        assert_eq!(share_count(478 + 482), 2);
        assert_eq!(share_count(478 + 482 + 1), 3);
        let blob = data(1000);
        let shares = shares(&namespace(), &blob);
        assert_eq!(shares.len(), share_count(1000));
        assert_eq!(shares[0][29], 1);
        assert_eq!(shares[0][30..34], 1000u32.to_be_bytes());
        assert_eq!(shares[0][34..], blob[..478]);
        assert_eq!(shares[1][29], 0);
        assert_eq!(shares[1][30..], blob[478..960]);
        assert_eq!(shares[2][30..70], blob[960..]);
        assert!(shares[2][70..].iter().all(|byte| *byte == 0));
    }

    /// Commitments a local celestia-node 0.35.0 gave for these blobs under
    /// this namespace (`celestia blob submit`, 2026-11-18). The node was given
    /// each blob after the first as the text `0x` and the hex of `data(n)`,
    /// and stored that text, so the blobs are 2n + 2 bytes long: one share,
    /// two shares that end exactly, two with one byte over, 21 shares, 166
    /// shares in subtrees of four, and 1,660 in subtrees of thirty-two.
    #[test]
    fn the_commitment_is_the_one_a_celestia_node_computes() {
        let of = |blob: &[u8]| hex::encode(blob_commitment(&namespace(), blob));
        assert_eq!(
            of(b"hello iris"),
            "3731394c554d6144a1314a802700dfd8d9f0c250ebac67a66b65132a3fc33b5b"
        );
        for (length, commitment) in [
            (
                1,
                "19f558d23c1ce4b312acd9982f78d3dc1aa24a7596c55ff62dfd30fe4ba601c8",
            ),
            (
                478,
                "0af21962f76b900e47fa6e2f115728f73b4368cd38f4126b506b8211395d7f1f",
            ),
            (
                479,
                "653861b4ec2860b1a64c33470f6887d695bbe74d00c6b23d5762bc89a944f30d",
            ),
            (
                5_000,
                "51c45c8a85a63409a75b7c65936073c15263875c42d5e6c9abad1fec7ad43f83",
            ),
            (
                40_000,
                "473e7b55ec974dfcc413fb67bd9a04aaf7bb98ee012f8671133450d3fa538619",
            ),
            (
                400_000,
                "6e3ce19cb99e928d4e191d0814fe8cb566dae63b14bd013d39883d8fd796d4b5",
            ),
        ] {
            let blob = format!("0x{}", hex::encode(data(length)));
            assert_eq!(
                of(blob.as_bytes()),
                commitment,
                "the text of {length} bytes"
            );
        }
    }

    #[test]
    fn subtrees_widen_with_the_blob() {
        assert_eq!(subtree_width(1), 1);
        assert_eq!(subtree_width(64), 1);
        assert_eq!(subtree_width(83), 2);
        assert_eq!(subtree_width(166), 4);
        assert_eq!(subtree_width(830), 16);
        assert_eq!(subtree_width(1_660), 32);
        // A small blob's subtrees are no wider than the square it fits in.
        assert_eq!(subtree_width(2), 1);
    }
}
