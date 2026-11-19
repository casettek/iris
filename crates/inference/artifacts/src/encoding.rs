//! The byte encodings of what a contract hashes itself: a checkpoint, the
//! checkpoint list and a configuration.
//!
//! A contract cannot hash JSON it has to build, nor open one line of a list of
//! hex lines without the whole list. So these follow the scheme the stages'
//! own digests use (`Statement::digest`, `source_id`): a tag ending in a zero
//! byte, then fields in order, numbers little-endian, text and lists prefixed
//! by a 32-bit count, hashes as their 32 bytes. Everything is SHA-256, which
//! an EVM has as a precompile and the proof programs have accelerated.
//!
//! The checkpoint list is a Merkle tree of fixed depth, so that one
//! checkpoint is opened at its position with [`TREE_DEPTH`] hashes.

use sha2::{Digest, Sha256};

/// A hash in the making, over fields in the encoding above.
pub struct Encoder {
    hasher: Sha256,
}

impl Encoder {
    /// `tag` names what is hashed and its version, and ends in a zero byte.
    pub fn new(tag: &[u8]) -> Self {
        debug_assert_eq!(tag.last(), Some(&0));
        let mut hasher = Sha256::new();
        hasher.update(tag);
        Self { hasher }
    }

    pub fn byte(&mut self, value: u8) -> &mut Self {
        self.hasher.update([value]);
        self
    }

    pub fn number(&mut self, value: u32) -> &mut Self {
        self.hasher.update(value.to_le_bytes());
        self
    }

    /// A 64-bit number.
    pub fn long(&mut self, value: u64) -> &mut Self {
        self.hasher.update(value.to_le_bytes());
        self
    }

    /// Bytes of a length the reader knows: a namespace, an address.
    pub fn raw(&mut self, bytes: &[u8]) -> &mut Self {
        self.hasher.update(bytes);
        self
    }

    /// How many items of a list follow.
    pub fn count(&mut self, items: usize) -> &mut Self {
        self.number(items as u32)
    }

    pub fn text(&mut self, text: &str) -> &mut Self {
        self.count(text.len());
        self.hasher.update(text.as_bytes());
        self
    }

    /// A 32-byte hash that the files hold as 64 hex digits: a byte 1 and the
    /// 32 bytes. An empty field is a byte 0. Any other text, which no honest
    /// file holds, is a byte 2 and the text, so that every value still has
    /// one encoding and a contract, which only ever encodes 32 bytes, can
    /// never produce it.
    pub fn hash(&mut self, hex_digits: &str) -> &mut Self {
        match parse_hash(hex_digits) {
            Some(bytes) => {
                self.byte(1);
                self.hasher.update(bytes);
                self
            }
            None if hex_digits.is_empty() => self.byte(0),
            None => self.byte(2).text(hex_digits),
        }
    }

    pub fn finish(self) -> [u8; 32] {
        self.hasher.finalize().into()
    }
}

/// The 32 bytes of 64 lowercase hex digits.
pub fn parse_hash(hex_digits: &str) -> Option<[u8; 32]> {
    if hex_digits.len() != 64
        || !hex_digits
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return None;
    }
    let mut bytes = [0; 32];
    hex::decode_to_slice(hex_digits, &mut bytes).ok()?;
    Some(bytes)
}

/// The depth of the checkpoint tree: room for about a million checkpoints.
pub const TREE_DEPTH: usize = 20;

fn leaf(value: &[u8; 32]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update([0]);
    hasher.update(value);
    hasher.finalize().into()
}

fn node(left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update([1]);
    hasher.update(left);
    hasher.update(right);
    hasher.finalize().into()
}

/// The root of the empty subtree at each height: 32 zero bytes at the
/// leaves, which no leaf hashes to.
fn empty_roots() -> [[u8; 32]; TREE_DEPTH + 1] {
    let mut roots = [[0; 32]; TREE_DEPTH + 1];
    for height in 0..TREE_DEPTH {
        roots[height + 1] = node(&roots[height], &roots[height]);
    }
    roots
}

/// Every level of the tree over these values, leaves first, each padded on
/// the right with the empty subtree of its height.
fn levels(values: &[[u8; 32]]) -> Vec<Vec<[u8; 32]>> {
    let empty = empty_roots();
    let mut levels = vec![values.iter().map(leaf).collect::<Vec<_>>()];
    for height in 0..TREE_DEPTH {
        let below = &levels[height];
        let above = below
            .chunks(2)
            .map(|pair| node(&pair[0], pair.get(1).unwrap_or(&empty[height])))
            .collect();
        levels.push(above);
    }
    levels
}

/// The root of the tree over these values, in order. `None` when there are
/// more than the tree holds.
pub fn tree_root(values: &[[u8; 32]]) -> Option<[u8; 32]> {
    if values.len() > 1 << TREE_DEPTH {
        return None;
    }
    Some(
        levels(values)[TREE_DEPTH]
            .first()
            .copied()
            .unwrap_or(empty_roots()[TREE_DEPTH]),
    )
}

/// The [`TREE_DEPTH`] siblings from the value at `index` up to the root.
pub fn tree_path(values: &[[u8; 32]], index: usize) -> Option<Vec<[u8; 32]>> {
    if index >= values.len() || values.len() > 1 << TREE_DEPTH {
        return None;
    }
    let empty = empty_roots();
    let levels = levels(values);
    Some(
        (0..TREE_DEPTH)
            .map(|height| {
                let sibling = (index >> height) ^ 1;
                levels[height]
                    .get(sibling)
                    .copied()
                    .unwrap_or(empty[height])
            })
            .collect(),
    )
}

/// The root a value at `index` and its path give. This is all a contract
/// computes to open one position.
pub fn root_from_path(value: &[u8; 32], index: u32, path: &[[u8; 32]]) -> Option<[u8; 32]> {
    if path.len() != TREE_DEPTH || u64::from(index) >= 1 << TREE_DEPTH {
        return None;
    }
    let mut hash = leaf(value);
    for (height, sibling) in path.iter().enumerate() {
        hash = if (index >> height) & 1 == 0 {
            node(&hash, sibling)
        } else {
            node(sibling, &hash)
        };
    }
    Some(hash)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_hash_field_has_one_encoding_whatever_it_holds() {
        let of = |text: &str| {
            let mut encoder = Encoder::new(b"test\0");
            encoder.hash(text);
            encoder.finish()
        };
        let hash = "ab".repeat(32);
        // 32 bytes, not 64 characters.
        let mut by_hand = Sha256::new();
        by_hand.update(b"test\0");
        by_hand.update([1]);
        by_hand.update([0xab; 32]);
        assert_eq!(of(&hash), <[u8; 32]>::from(by_hand.finalize()));
        // Empty, upper case and short text are three other values.
        let others = [of(""), of(&hash.to_uppercase()), of("ab")];
        for (at, other) in others.iter().enumerate() {
            assert_ne!(*other, of(&hash));
            assert!(others[..at].iter().all(|earlier| earlier != other));
        }
    }

    #[test]
    fn a_position_opens_with_its_path_and_no_other() {
        let values: Vec<[u8; 32]> = (0u8..5).map(|byte| [byte + 1; 32]).collect();
        let root = tree_root(&values).unwrap();
        for (index, value) in values.iter().enumerate() {
            let path = tree_path(&values, index).unwrap();
            assert_eq!(path.len(), TREE_DEPTH);
            assert_eq!(root_from_path(value, index as u32, &path), Some(root));
            // The same value at another position, or another value here.
            assert_ne!(root_from_path(value, index as u32 + 1, &path), Some(root));
            assert_ne!(root_from_path(&[0xee; 32], index as u32, &path), Some(root));
        }
        // One more value, or one fewer, is another tree; none is the empty one.
        assert_ne!(tree_root(&values[..4]), Some(root));
        assert_eq!(tree_root(&[]), Some(empty_roots()[TREE_DEPTH]));
        assert_eq!(tree_path(&values, 5), None);
        // An empty position does not open as a value of zeros.
        let path = tree_path(&values, 4).unwrap();
        assert_ne!(root_from_path(&[0; 32], 5, &path), Some(root));
    }
}
