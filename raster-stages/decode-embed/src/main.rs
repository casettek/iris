//! Phase 7 — `decode-embed`: sequences.
//!
//! ```text
//!   selected ──▶ [ offset · page_of · select page · extract ] ──▶ ActivationSequence (1 row)
//!   misses   ──recur──▶ count ──▶ assert none
//! ```
//!
//! Straight-line, not a recur: a decode step embeds exactly one token, so there
//! is no collection to sweep. That is the whole difference from
//! `input-embedding` — same table, same addressing, same scale.

use raster::prelude::*;

use decode_embed::input::*;
use decode_embed::*;

/// Phase 7 entrypoint, for one decode step.
///
/// `selected` is bound to this iteration's `decode-select-token` output;
/// `embedding` is the same committed external `input_embedding` reads.
#[sequence]
fn main(selected: DecodeEdge, embedding: EmbeddingTable) -> Result<ActivationSequence> {
    let has_selected = select!(bool, selected.clone().has_selected);
    call!(require_selected_token, has_selected)?;
    let token_id = select!(u32, selected.clone().token_id);
    let decode_position = select!(u32, selected.decode_position);

    let hidden_size = select!(u32, embedding.clone().hidden_size);
    let page_size = select!(u64, embedding.clone().values.page_size);
    let embedding_scale = select!(i32, embedding.clone().embedding_scale);
    let values = select!(Bytes<196_608>, embedding.values);

    let byte_off = call!(embedding_byte_offset, token_id.clone(), hidden_size.clone());
    let page_idx = call!(page_of, byte_off.clone(), page_size);
    let page = select!(BytesPage, values[page_idx]);

    let draft = call!(
        begin_decode_activations,
        new!(ActivationSequence),
        decode_position
    );
    let embedded = call!(
        append_activation_row,
        draft,
        token_id,
        page,
        byte_off,
        hidden_size,
        embedding_scale
    );
    let embedded = finalize(embedded);
    raster::println!("decode embed → {:?}", embedded);

    let errors = select!(List<String>, embedded.clone().errors);
    let summary = call_recur!(
        tile = summarise_errors,
        input = errors,
        state = ErrorSummary {
            count: 0,
            first: String::new()
        },
        args = ()
    );
    let error_count = select!(u32, summary.clone().count);
    let first_error = select!(String, summary.first);
    call!(assert_all_tokens_embedded, error_count, first_error)?;

    Ok(embedded)
}
