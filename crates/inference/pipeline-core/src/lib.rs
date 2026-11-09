//! Statement and token-history primitives. Authenticated producer links remain required.
#![no_std]
extern crate alloc;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Statement {
    pub input_text: [u8; 32],
    pub model: [u8; 32],
    pub raw_prompt: bool,
    pub max_new_tokens: u32,
    pub range_size: u32,
}
impl Statement {
    pub fn digest(&self) -> [u8; 32] {
        let mut h = Sha256::new();
        h.update(b"iris/statement/v1\0");
        h.update(self.input_text);
        h.update(self.model);
        h.update([u8::from(self.raw_prompt)]);
        h.update(self.max_new_tokens.to_le_bytes());
        h.update(self.range_size.to_le_bytes());
        h.finalize().into()
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Prefix {
    pub position: u32,
    pub digest: [u8; 32],
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fault {
    Overflow,
    Empty,
    RangeLimit,
    Statement,
    History,
    Schedule,
}
mod schedule;
pub use schedule::*;
impl Prefix {
    pub fn empty(statement: &Statement) -> Self {
        let mut h = Sha256::new();
        h.update(b"iris/token-prefix/v1\0");
        h.update(statement.digest());
        Self {
            position: 0,
            digest: h.finalize().into(),
        }
    }
    pub fn append(&mut self, token: u32) -> Result<(), Fault> {
        let next = self.position.checked_add(1).ok_or(Fault::Overflow)?;
        let mut h = Sha256::new();
        h.update(b"iris/token/v1\0");
        h.update(self.digest);
        h.update(self.position.to_le_bytes());
        h.update(token.to_le_bytes());
        self.digest = h.finalize().into();
        self.position = next;
        Ok(())
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenRange {
    pub statement: [u8; 32],
    pub before: Prefix,
    pub after: Prefix,
}
impl TokenRange {
    pub fn from_tokens(
        s: &Statement,
        before: Prefix,
        ids: impl IntoIterator<Item = u32>,
    ) -> Result<Self, Fault> {
        let mut after = before;
        for id in ids {
            after.append(id)?;
            if after.position - before.position > s.range_size {
                return Err(Fault::RangeLimit);
            }
        }
        if after == before {
            return Err(Fault::Empty);
        }
        Ok(Self {
            statement: s.digest(),
            before,
            after,
        })
    }
    pub fn follows(&self, previous: &Self) -> Result<(), Fault> {
        if self.statement != previous.statement {
            return Err(Fault::Statement);
        }
        if self.before != previous.after {
            return Err(Fault::History);
        }
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    fn statement() -> Statement {
        Statement {
            input_text: [1; 32],
            model: [2; 32],
            raw_prompt: false,
            max_new_tokens: 8,
            range_size: 8,
        }
    }
    fn range(s: &Statement, p: Prefix, ids: &[u32]) -> TokenRange {
        TokenRange::from_tokens(s, p, ids.iter().copied()).unwrap()
    }
    #[test]
    fn partition_independent() {
        let s = statement();
        let p = Prefix::empty(&s);
        let a = range(&s, p, &[1, 2]);
        let b = range(&s, a.after, &[3, 4]);
        assert_eq!(b.follows(&a), Ok(()));
        assert_eq!(b.after, range(&s, p, &[1, 2, 3, 4]).after);
    }
    #[test]
    fn rejects_same_length_same_tail_different_history() {
        let s = statement();
        let p = Prefix::empty(&s);
        let a = range(&s, p, &[1, 2, 9]);
        let b = range(&s, p, &[1, 3, 9]);
        assert_eq!(range(&s, a.after, &[4]).follows(&b), Err(Fault::History));
    }
    #[test]
    fn statement_fields_are_bound() {
        let s = statement();
        let mut variants = [s.clone(), s.clone(), s.clone(), s.clone(), s.clone()];
        variants[0].input_text[0] ^= 1;
        variants[1].model[0] ^= 1;
        variants[2].raw_prompt = true;
        variants[3].max_new_tokens += 1;
        variants[4].range_size += 1;
        for v in variants {
            assert_ne!(Prefix::empty(&s), Prefix::empty(&v));
        }
    }
    #[test]
    fn overflow_and_empty_ranges_fail() {
        let s = statement();
        let p = Prefix::empty(&s);
        assert_eq!(TokenRange::from_tokens(&s, p, []), Err(Fault::Empty));
        assert_eq!(
            TokenRange::from_tokens(&s, p, [0; 9]),
            Err(Fault::RangeLimit)
        );
        let mut full = Prefix {
            position: u32::MAX,
            digest: [7; 32],
        };
        let old = full;
        assert_eq!(full.append(0), Err(Fault::Overflow));
        assert_eq!(full, old);
    }
}

/// Versioned, bounded execution metadata. It is encoded inside the existing
/// authenticated ModelContext so every boundary conversion carries it intact.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Execution {
    pub version: u32,
    pub statement: Statement,
    pub before: Prefix,
    pub end: Prefix,
    pub prompt: Option<Prefix>,
    pub phase: u8,
    pub claimed: bool,
    pub expected: Option<Prefix>,
    /// The run's stage schedule, all zero until `prompt-bind` verifies one.
    pub schedule: StageSchedule,
    /// The stage that produced this record.
    pub step: Step,
}
pub const EXECUTION_VERSION: u32 = 2;
pub const BOUND: u8 = 0;
pub const TOKENIZING: u8 = 1;
pub const PREPARED: u8 = 2;
pub const EMBEDDED: u8 = 3;
pub const CHECKED: u8 = 4;
impl Execution {
    pub fn bind(statement: Statement) -> Self {
        let before = Prefix::empty(&statement);
        Self {
            version: EXECUTION_VERSION,
            statement,
            before,
            end: before,
            prompt: None,
            phase: BOUND,
            claimed: false,
            expected: None,
            schedule: StageSchedule::default(),
            step: Step::bind(),
        }
    }
    pub fn encode(&self) -> alloc::vec::Vec<u8> {
        postcard::to_allocvec(self).expect("bounded execution context")
    }
    pub fn decode(bytes: &[u8]) -> Option<Self> {
        let (value, rest): (Self, &[u8]) = postcard::take_from_bytes(bytes).ok()?;
        (rest.is_empty()
            && value.version == EXECUTION_VERSION
            && value.before.position <= value.end.position)
            .then_some(value)
    }
    pub fn same_statement(&self, other: &Self) -> bool {
        self.statement == other.statement
    }
    pub fn same_range(&self, other: &Self) -> bool {
        self.same_statement(other)
            && self.before == other.before
            && self.end == other.end
            && self.claimed == other.claimed
            && self.prompt == other.prompt
    }
    pub fn nonempty_bounded(&self) -> bool {
        self.end.position > self.before.position
            && self.end.position - self.before.position <= self.statement.range_size
    }
}
