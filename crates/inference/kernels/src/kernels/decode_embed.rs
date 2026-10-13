use anyhow::{bail, Result};
use decode_embed::input::{
    pack_i32s, unpack_i32s_at, ActivationRow, ActivationSequence, DecodeEdge, EmbeddingTable,
};
use det_num::ops::mul_sat;
use det_num::Act;
use raster::List;

pub struct DecodeEmbedDirectInputs<'a> {
    pub selected: &'a DecodeEdge,
    pub embedding: &'a EmbeddingTable,
}

pub fn run_decode_embed_direct(inputs: DecodeEmbedDirectInputs<'_>) -> Result<ActivationSequence> {
    if !inputs.selected.has_selected {
        bail!("decode-embed received the empty decode edge with no selected token");
    }

    let token_id = inputs.selected.token_id;
    let hidden_size = inputs.embedding.hidden_size;
    let byte_off = token_id as u64 * hidden_size as u64 * 4;
    let page_size = inputs.embedding.values.page_size();
    let page_idx = if page_size == 0 {
        0
    } else {
        byte_off / page_size
    };
    let page = inputs.embedding.values.pages().get(page_idx as usize);
    let Some(page) = page else {
        bail!("1 prompt token(s) could not be embedded; first: token {token_id} has no embedding row of the declared width");
    };
    let values = unpack_i32s_at(page, byte_off, hidden_size).map_err(|_| {
        anyhow::anyhow!(
            "1 prompt token(s) could not be embedded; first: token {token_id} has no embedding row of the declared width"
        )
    })?;
    let scale = Act::from_bits(inputs.embedding.embedding_scale);
    let scaled = values
        .iter()
        .map(|bits| mul_sat(Act::from_bits(*bits), scale).to_bits())
        .collect::<Vec<_>>();

    Ok(ActivationSequence {
        rows: List::from(vec![ActivationRow {
            token_id,
            values: pack_i32s(&scaled),
        }]),
        errors: List::new(),
        kv: List::new(),
        start_position: inputs.selected.decode_position,
    })
}
