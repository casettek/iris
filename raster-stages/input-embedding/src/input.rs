//! Shared Rastered data types for the `input-embedding` stage.

extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;
use raster::{Bytes, BytesPage, List};
use serde::{Deserialize, Serialize};

/// Replay-unit page size for the embedding table: whole `Q8_0` rows of
/// `hidden_size / 32 · 34` bytes for every width this chain is imported with
/// (128 → 136, 1536 → 1632). MUST match the importer's `Q8_PAGE_SIZE`.
pub const PAGE_SIZE: u64 = 195_840;

/// The stage's committed first input, bound by the chain to `prompt_prepare`'s
/// authorized output. The stage reads only its `token_ids`.
pub use crate::prompt::{PromptState, PromptTokenization};

/// A teacher-forced claim's output ids, embedded after the prompt so one pass
/// covers `prompt ‖ claimed`. A prompt range binds an empty one.
///
/// MUST match `ClaimedOutput` in `sequence-check`.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct ClaimedOutput {
    pub token_ids: List<u32>,
}

/// The committed embedding table: vocab × hidden, row-major, dense so
/// `token_id` is the row index.
///
/// `values` is a paged byte region — not `Materializable` — so the sequence
/// indexes one page rather than scanning the vocabulary.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct EmbeddingTable {
    pub hidden_size: u32,
    /// Gemma multiplies every token embedding by `sqrt(hidden_size)` before the
    /// first layer. Committed as Q16.16 bits so the tile applies exactly the
    /// scalar the importer derived, with no float anywhere near a replay.
    pub embedding_scale: i32,
    /// `Q8_0` rows: per 32 values, a half-precision scale then 32 signed bytes.
    #[page_size = 195_840]
    pub values: Bytes<195_840>,
}

/// Loop-carried count of the previous range's rows.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct RowCount {
    pub count: u32,
}

/// Loop-carried summary of failed rows: how many, and the first message.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct ErrorSummary {
    pub count: u32,
    pub first: String,
}

/// One prompt position: the token that produced it and its activation row.
///
/// MUST match `ActivationRow` in every stage that passes activations on —
/// the chain links by structural commitment, not by Rust type name.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct ActivationRow {
    pub token_id: u32,
    pub values: BytesPage,
}

/// Key/value side of one token — the layer's KV cache entry.
///
/// MUST match `KeyRow` in every stage that carries activations: Gemma 3n's last
/// 20 layers attend over an earlier layer's cache, so the cache has to survive a
/// stage boundary, and a chain binds by structural commitment.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct KeyRow {
    pub position: u32,
    pub k: BytesPage,
    pub v: BytesPage,
}

/// The stage's authorized output, and the pipeline's activation type.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct ActivationSequence {
    pub rows: List<ActivationRow>,
    pub errors: List<String>,
    /// This layer's K/V cache after this stage: everything it was handed as
    /// `prior_kv`, pruned to the sliding window, plus one row per token this
    /// stage processed.
    ///
    /// Every layer publishes it, and it is read by two different consumers: a
    /// later *sharing* layer attends over its donor's cache in the same step,
    /// and the *same* layer one decode step later takes it as `prior_kv`. A
    /// layer that is neither a donor nor decoded again still pays to publish
    /// it, which is the price of one uniform program across all 35 stages.
    pub kv: List<KeyRow>,
    /// Absolute position of `rows[0]` in the full sequence — 0 for the prompt,
    /// `prompt_len + t` for decode step `t`.
    ///
    /// Positions cannot be recovered by counting: a decode stage sees one row
    /// but sits at position `prompt_len + t`, and RoPE and sliding-window
    /// visibility are both defined over the absolute value. Carrying it
    /// explicitly is what lets one program serve prefill and decode.
    pub start_position: u32,
}

/// Packs Q16.16 values as little-endian i32s into a standalone page.
///
/// A computed vector is a new leaf, not a page of an imported region, so
/// `(index, offset) = (0, 0)`.
pub fn pack_i32s(values: &[i32]) -> BytesPage {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    BytesPage::__from_parts(0, 0, bytes)
}

/// Decodes a page of little-endian i32s. A malformed leaf is a committed tile
/// error, never a panic.
pub fn unpack_i32s(page: &BytesPage) -> core::result::Result<Vec<i32>, String> {
    let bytes = page.as_slice();
    if bytes.len() % 4 != 0 {
        return Err(alloc::format!(
            "page length {} is not i32-aligned",
            bytes.len()
        ));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| i32::from_le_bytes(chunk.try_into().expect("chunks_exact(4)")))
        .collect())
}

/// Reads `n_values` i32s starting at global `byte_off` inside `page`.
pub fn q8_row_at(
    page: &BytesPage,
    byte_off: u64,
    n_values: u32,
) -> core::result::Result<Vec<i32>, String> {
    let local = byte_off
        .checked_sub(page.offset())
        .ok_or_else(|| String::from("row offset is before this page"))? as usize;
    let len = det_num::q8::row_bytes(n_values as usize)
        .ok_or_else(|| alloc::format!("row width {n_values} is not whole Q8_0 blocks"))?;
    let bytes = page.as_slice().get(local..local + len).ok_or_else(|| {
        alloc::format!(
            "row at byte {byte_off} spans past this page (local {local}, need {len}, have {})",
            page.len()
        )
    })?;
    let mut values = alloc::vec![0; n_values as usize];
    det_num::q8::dequantize_row(bytes, &mut values).map_err(String::from)?;
    Ok(values)
}
