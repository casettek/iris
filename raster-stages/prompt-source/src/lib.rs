//! `prompt-source`: tiles.
//!
//! States what a run's prompt text was built from: checks each document's
//! attestation and bytes, takes the recipe's values out of the documents, and
//! hashes the text the recipe makes of them. The first `prompt-bind` holds
//! the text the run gives to this stage's record.
//!
//! The logic is in `input`, shared with the native run and the native
//! stage. A tile calls it, so it is inside the tile's image id.
//!
//! Nothing a run gives can stop a tile: a document, attestation or recipe that
//! is wrong sets a fault in the record.
//!
//! `#![no_std]` so the same tiles compile into RISC0 replay guests.

#![no_std]

extern crate alloc;

use alloc::string::String;
use raster::prelude::*;

pub mod input;

use input::*;

// Each loop below starts from a value a tile returns, so its seed is a whole
// stored value: Raster ties a loop's first step to a seed only when it is one.

#[tile(kind = iter, description = "No document's bytes hashed yet")]
pub fn bodies_seed() -> BodiesFold {
    begin_bodies_fold()
}

/// Hashes one chunk of the documents. A document's hash joins the chain when
/// its last chunk has been hashed.
#[tile(kind = recur, description = "Hash one chunk of the documents")]
pub fn fold_body(
    input: RecurInput<DocumentChunk>,
    state: RecurState<BodiesFold>,
) -> RecurState<BodiesFold> {
    let mut state = state;
    let chunk = input.into_value();
    *state = fold_body_chunk((*state).clone(), &chunk);
    state
}

#[tile(kind = iter, description = "No attestation checked yet")]
pub fn attested_seed() -> DocumentsFold {
    begin_documents_fold()
}

/// Verifies one attestation's signature and chains what it states.
#[tile(
    kind = recur,
    description = "Verify one attestation against the signer's key",
    estimated_cycles = 6_000_000
)]
pub fn fold_attested(
    input: RecurInput<Attestation>,
    state: RecurState<DocumentsFold>,
    signer: SignerKey,
) -> RecurState<DocumentsFold> {
    let mut state = state;
    let attestation = input.into_value();
    *state = fold_attestation((*state).clone(), &signer, &attestation);
    state
}

/// Holds the documents hashed to the ones the attestations are over.
#[tile(kind = iter, description = "Compare the documents with their attestations")]
pub fn hold_bodies(attested: DocumentsFold, bodies: BodiesFold) -> DocumentsFold {
    check_bodies(attested, &bodies)
}

#[tile(kind = iter, description = "No prompt text hashed yet")]
pub fn text_seed() -> SourceFold {
    begin_source_fold()
}

#[tile(kind = recur, description = "Hash one chunk of the base prompt")]
pub fn fold_base(
    input: RecurInput<String>,
    state: RecurState<SourceFold>,
) -> RecurState<SourceFold> {
    let mut state = state;
    let chunk = input.into_value();
    *state = fold_base_chunk((*state).clone(), &chunk);
    state
}

#[tile(kind = iter, description = "Close the base prompt")]
pub fn end_base(fold: SourceFold) -> SourceFold {
    fold_base_end(fold)
}

#[tile(kind = recur, description = "Chain one document the recipe names")]
pub fn fold_name(
    input: RecurInput<DocumentName>,
    state: RecurState<SourceFold>,
) -> RecurState<SourceFold> {
    let mut state = state;
    let name = input.into_value();
    *state = fold_document_name((*state).clone(), &name);
    state
}

/// Holds the documents given to the ones the recipe names.
#[tile(kind = iter, description = "Compare the documents with the recipe's")]
pub fn hold_documents(fold: SourceFold, attested: DocumentsFold) -> SourceFold {
    check_documents(fold, &attested)
}

#[tile(kind = iter, description = "Open the scan for one field's value")]
pub fn begin_scan(field: Field) -> FieldScan {
    begin_field_scan(&field.path)
}

/// Moves one field's scan over one chunk of the documents, and stops the loop
/// once nothing later can change the outcome.
#[tile(kind = recur, description = "Scan one chunk for a field's value")]
pub fn scan_chunk(
    input: RecurInput<DocumentChunk>,
    state: RecurState<FieldScan>,
    document: u32,
    path: String,
) -> RecurControl<RecurState<FieldScan>> {
    let mut state = state;
    let chunk = input.into_value();
    *state = scan_field_chunk((*state).clone(), &chunk, document, &path);
    if state.done {
        RecurControl::Break(state)
    } else {
        RecurControl::Continue(state)
    }
}

#[tile(kind = iter, description = "A field's value, or why it has none")]
pub fn finish_scan(scan: FieldScan) -> FieldValue {
    finish_field_scan(scan)
}

/// Adds one field's lead and value to the text, and the field to the recipe's
/// hash.
#[tile(kind = iter, description = "Hash one field into the prompt text")]
pub fn fold_one_field(fold: SourceFold, field: Field, value: FieldValue) -> SourceFold {
    fold_field(fold, &field, &value)
}

/// Adds the tail and states the record.
#[tile(kind = iter, description = "State the source record")]
pub fn close(
    fold: SourceFold,
    attested: DocumentsFold,
    signer: SignerKey,
    tail: String,
) -> SourceRecord {
    close_source(fold, &attested, &signer, &tail)
}
