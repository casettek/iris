//! The relay step: a prompt that holds an earlier inference's output.
//!
//! ```text
//!   recipe.before ───────────────┐
//!   the earlier output's text ───┼─ sha256 ─▶ text_sha256
//!   recipe.after ────────────────┘
//!   the earlier inference's claim record ─▶ documents_sha256
//! ```
//!
//! It states a source record, as the source step does, so the later
//! inference's first bind takes it exactly as it takes that step's: the text
//! the run gives the later inference is held to this record's hash, and that
//! inference's statement names the record.
//!
//! Plain functions over a small state, one per step a stage takes, so native
//! inference, the native stage and the tiles run one copy. Nothing aborts: an
//! earlier inference whose record is not valid, or a part of the recipe over
//! its limit, sets a fault in the record.

use alloc::string::String;
use alloc::vec::Vec;
use raster::{BytesPage, List};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::source::{
    begin_byte_fold, fold_bytes, fold_sha256, hex, text_chunks, ByteFold, SourceRecord, FAULT_NONE,
    FAULT_RECIPE, MAX_PART,
};

/// The earlier inference's claim record is not valid, or its output is not
/// the text that record commits to. A [`SourceRecord::fault`] value, after
/// the source step's own.
pub const FAULT_RELAYED: u32 = 10;

/// What a fault of a relay record is, in words.
pub fn relay_fault_description(fault: u32) -> &'static str {
    match fault {
        FAULT_RELAYED => "the inference whose output is relayed stated no valid claim record",
        other => crate::source::fault_description(other),
    }
}

/// How the later prompt is made: `before`, the earlier output, `after`.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct RelayRecipe {
    /// The text before the output, in chunks of at most [`MAX_PART`] bytes.
    /// Where it is cut changes nothing.
    pub before: List<String>,
    /// The text after the output, cut the same way.
    pub after: List<String>,
}

/// A relay recipe of two texts, cut as the step reads them.
pub fn relay_recipe(before: &str, after: &str) -> RelayRecipe {
    RelayRecipe {
        before: text_chunks(before).into(),
        after: text_chunks(after).into(),
    }
}

/// What the relay step carries from chunk to chunk of the recipe.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct RelayFold {
    /// The prompt text so far.
    pub text: ByteFold,
    /// The part of the recipe being read: `before`, then `after`.
    pub part: ByteFold,
    /// How many parts have been closed.
    pub parts: u32,
    /// A hash chained over every part of the recipe closed so far: 32 bytes.
    pub recipe: BytesPage,
    /// A hash over the claim record relayed: 32 bytes, all zero before it.
    pub relayed: BytesPage,
    pub fault: u32,
    pub fault_position: u64,
}

fn page(bytes: Vec<u8>) -> BytesPage {
    BytesPage::__from_parts(0, 0, bytes)
}

fn chain(previous: &[u8], parts: &[&[u8]]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(previous);
    for part in parts {
        hasher.update((part.len() as u32).to_le_bytes());
        hasher.update(part);
    }
    hasher.finalize().into()
}

fn record_fault(fold: &mut RelayFold, fault: u32, position: u64) {
    if fold.fault == FAULT_NONE && fault != FAULT_NONE {
        fold.fault = fault;
        fold.fault_position = position;
    }
}

/// The fold before the recipe's first chunk.
pub fn begin_relay_fold() -> RelayFold {
    RelayFold {
        text: begin_byte_fold(),
        part: begin_byte_fold(),
        parts: 0,
        recipe: page([0_u8; 32].to_vec()),
        relayed: page([0_u8; 32].to_vec()),
        fault: FAULT_NONE,
        fault_position: 0,
    }
}

/// Adds one chunk of the part being read. A fault in a part is at that
/// part's index: 0 for `before`, 1 for `after`.
pub fn fold_part_chunk(mut fold: RelayFold, chunk: &str) -> RelayFold {
    if chunk.len() > MAX_PART {
        let part = fold.parts as u64;
        record_fault(&mut fold, FAULT_RECIPE, part);
    }
    fold.text = fold_bytes(fold.text, chunk.as_bytes());
    fold.part = fold_bytes(fold.part, chunk.as_bytes());
    fold
}

/// Closes the part being read: its hash joins the recipe's, under its name.
pub fn close_part(mut fold: RelayFold) -> RelayFold {
    let name: &[u8] = if fold.parts == 0 { b"before" } else { b"after" };
    let part = fold_sha256(&fold.part);
    fold.recipe = page(chain(fold.recipe.as_slice(), &[name, &part[..]]).to_vec());
    fold.part = begin_byte_fold();
    fold.parts += 1;
    fold
}

/// Adds the earlier inference's output to the text, and states which claim
/// record it is the output of. `statement`, `output_text_sha256` and `valid`
/// are that record's three values.
pub fn fold_relayed(
    mut fold: RelayFold,
    output: &str,
    statement: &[u8],
    output_text_sha256: &[u8],
    valid: bool,
) -> RelayFold {
    let stated: [u8; 32] = Sha256::digest(output.as_bytes()).into();
    if !valid || stated[..] != *output_text_sha256 {
        record_fault(&mut fold, FAULT_RELAYED, 0);
    }
    fold.text = fold_bytes(fold.text, output.as_bytes());
    fold.relayed = page(chain(&[0_u8; 32], &[statement, output_text_sha256]).to_vec());
    fold
}

/// Closes the last part and states the record.
///
/// The record is a source record of one document, the earlier output, which
/// no signer vouches for: its claim record does, and `documents_sha256` is
/// over that record.
pub fn close_relay(fold: RelayFold) -> SourceRecord {
    let fold = close_part(fold);
    SourceRecord {
        present: true,
        text_sha256: hex(&fold_sha256(&fold.text)),
        text_total: fold.text.total,
        recipe_sha256: hex(fold.recipe.as_slice()),
        documents: 1,
        documents_sha256: hex(fold.relayed.as_slice()),
        signer: String::new(),
        fault: fold.fault,
        fault_position: fold.fault_position,
    }
}

/// The relay step from end to end, and the prompt text it stands for.
pub struct RelayRun {
    pub record: SourceRecord,
    /// The text whose hash the record states.
    pub text: String,
}

/// Runs every step in the order the stage takes them, over the values the
/// stage takes.
pub fn run_relay_steps(
    recipe: &RelayRecipe,
    output: &str,
    statement: &[u8],
    output_text_sha256: &[u8],
    valid: bool,
) -> RelayRun {
    let mut text = String::new();
    let mut fold = begin_relay_fold();
    for chunk in recipe.before.iter() {
        fold = fold_part_chunk(fold, chunk);
        text.push_str(chunk);
    }
    fold = close_part(fold);
    fold = fold_relayed(fold, output, statement, output_text_sha256, valid);
    text.push_str(output);
    for chunk in recipe.after.iter() {
        fold = fold_part_chunk(fold, chunk);
        text.push_str(chunk);
    }
    RelayRun {
        record: close_relay(fold),
        text,
    }
}
