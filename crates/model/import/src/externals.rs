//! Mirrors of every stage's committed external type.
//!
//! Each of these must match the stage's own definition **field-for-field**:
//! the chain authorizes an external by the structural commitment of its bytes,
//! so a renamed field or a reordered struct here produces an artifact the stage
//! will not accept. They are duplicated rather than shared for the same reason
//! the stages duplicate their boundary types — a chain member is a standalone
//! program, and its guest cannot depend on a host-side crate.

use raster::{Bytes, BytesPage, List};
use serde::{Deserialize, Serialize};

/// Replay-unit page size shared by every `Bytes<P>` region this importer writes.
pub const PAGE_SIZE: u64 = 196_608;

/// Packs Q16.16 values as little-endian i32s.
pub fn pack_i32_bytes(values: &[i32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 4);
    for value in values {
        out.extend_from_slice(&value.to_le_bytes());
    }
    out
}

/// A computed / imported vector as a standalone page (index 0, offset 0).
pub fn page_of_i32s(values: &[i32]) -> BytesPage {
    BytesPage::__from_parts(0, 0, pack_i32_bytes(values))
}

pub fn paged_i32s(values: &[i32]) -> raster::runtime::Result<Bytes<196_608>> {
    Bytes::<PAGE_SIZE>::paged(pack_i32_bytes(values))
}

// --- prompt-prepare ---------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct TokenEntry {
    pub token: String,
    pub id: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct BpeMerge {
    pub rank: u32,
    pub left: String,
    pub right: String,
    pub merged: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct VocabBucket {
    pub entries: List<TokenEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct MergeBucket {
    pub rules: List<BpeMerge>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct PromptTokenizer {
    pub vocab_bucket_count: u32,
    pub merge_bucket_count: u32,
    pub vocab_buckets: List<VocabBucket>,
    pub merge_buckets: List<MergeBucket>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct BpePieces {
    pub pieces: List<prompt_prepare::input::BpePiece>,
}

// --- output-finalize -------------------------------------------------------

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct DecoderToken {
    pub token: String,
    pub special: bool,
    pub terminal: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct DecoderTable {
    pub tokens: List<DecoderToken>,
}

// --- input-embedding --------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct EmbeddingTable {
    pub hidden_size: u32,
    pub embedding_scale: i32,
    #[page_size = 196_608]
    pub values: Bytes<196_608>,
}

// --- prefill-prepare-aux ----------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct PleLayerParams {
    pub layer_idx: u32,
    pub hidden_size: u32,
    pub ple_width: u32,
    pub embedding_scale: i32,
    pub projection_scalar: i32,
    pub input_scale: i32,
    pub norm_eps: i64,
    pub norm_weights: BytesPage,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct PleLayer {
    pub params: PleLayerParams,
    #[page_size = 196_608]
    pub embeddings: Bytes<196_608>,
    #[page_size = 196_608]
    pub projection: Bytes<196_608>,
}

// --- prefill-range ----------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct LayerParams {
    pub layer_idx: u32,
    pub hidden_size: u32,
    pub ffn_size: u32,
    pub num_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub sliding_window: u32,
    pub attn_scale: i32,
    pub layer_scalar: i32,
    pub norm_eps: i64,
    pub rope_base: i64,
    pub rotary_dim: u32,
    pub rope_freq_base_dim: u32,
    pub kv_donor_layer: i32,
    pub donor_a_layer: i32,
    pub donor_b_layer: i32,
    pub norm_input: BytesPage,
    pub norm_post_attn: BytesPage,
    pub norm_pre_ffw: BytesPage,
    pub norm_post_ffw: BytesPage,
    pub q_norm: BytesPage,
    pub k_norm: BytesPage,
    pub ple_width: u32,
    pub ple_post_norm: BytesPage,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct TransformerLayer {
    pub params: LayerParams,
    #[page_size = 196_608]
    pub w_q: Bytes<196_608>,
    #[page_size = 196_608]
    pub w_k: Bytes<196_608>,
    #[page_size = 196_608]
    pub w_v: Bytes<196_608>,
    #[page_size = 196_608]
    pub w_o: Bytes<196_608>,
    #[page_size = 196_608]
    pub w_gate: Bytes<196_608>,
    #[page_size = 196_608]
    pub w_up: Bytes<196_608>,
    #[page_size = 196_608]
    pub w_down: Bytes<196_608>,
    #[page_size = 196_608]
    pub ple_input_gate: Bytes<196_608>,
    #[page_size = 196_608]
    pub ple_layer_projection: Bytes<196_608>,
}

// --- prefill-finalize -------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct FinalHeadParams {
    pub hidden_size: u32,
    pub norm_eps: i64,
    pub softcap: i32,
    pub norm_weights: BytesPage,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct FinalHead {
    pub params: FinalHeadParams,
    #[page_size = 196_608]
    pub projection: Bytes<196_608>,
}
