//! Shared Rastered data types for the `range-origin` stage.
//!
//! The activation types are duplicated field-for-field from
//! `input-embedding/src/input.rs`: the chain links stages by the structural
//! commitment of their values, not by Rust type name.

extern crate alloc;

use alloc::string::String;
use raster::{BytesPage, List};
use serde::{Deserialize, Serialize};

/// MUST match `ActivationRow` in every stage that carries activations.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct ActivationRow {
    pub token_id: u32,
    pub values: BytesPage,
}

/// MUST match `KeyRow` in every stage that carries activations.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct KeyRow {
    pub position: u32,
    pub k: BytesPage,
    pub v: BytesPage,
}

/// MUST match `ActivationSequence` in every stage that carries activations.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct ActivationSequence {
    pub rows: List<ActivationRow>,
    pub errors: List<String>,
    pub kv: List<KeyRow>,
    pub start_position: u32,
}
