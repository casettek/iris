//! Phase 2 — `input-embedding`: sequences.
//!
//! ```text
//!   prompt ids  ──recur seq──▶ [ offset · page_of · select page · extract ] ──▶ rows
//!   claimed ids ──recur seq──▶ [ same, appending after the prompt's        ] ──▶ ActivationSequence
//!   misses    ──recur─────▶ count ──▶ assert none
//! ```

use raster::prelude::*;

use input_embedding::input::*;
use input_embedding::*;

/// One prompt token: locate its row in the embedding table and append it.
#[sequence(kind = recur)]
fn embed_prompt_token(
    input: RecurSequenceInput<u32>,
    output: RecurSequenceOutput<ActivationSequence>,
    values: Bytes<196_608>,
    hidden_size: u32,
    page_size: u64,
    embedding_scale: i32,
) -> RecurSequenceOutput<ActivationSequence> {
    let token_id = call!(begin_row_lookup, input);
    let byte_off = call!(embedding_byte_offset, token_id.clone(), hidden_size.clone());
    let page_idx = call!(page_of, byte_off.clone(), page_size);
    let page = select!(BytesPage, values[page_idx]);
    call!(
        append_activation_row,
        output,
        token_id,
        page,
        byte_off,
        hidden_size,
        embedding_scale
    )
}

/// Phase 2 entrypoint, for one range of the sequence.
///
/// `prompt` is this range's chain-tokenized prompt ids (empty for a range of
/// claimed output); `claimed` is this range's claimed output ids (empty for a
/// prompt range and in the autoregressive chain); `previous` is the previous
/// range's embedding, or `range-origin` for the first; `embedding` is a
/// committed external.
///
/// The claimed ids are embedded after the prompt's by a second push-only
/// site deriving from the first, so the rows are `prompt ‖ claimed` in order.
#[sequence]
fn main(
    prompt: PromptTokenization,
    claimed: ClaimedOutput,
    previous: ActivationSequence,
    embedding: EmbeddingTable,
) -> Result<ActivationSequence> {
    let previous_start = select!(u32, previous.clone().start_position);
    let previous_rows = select!(List<ActivationRow>, previous.rows);
    let count_seed = call!(begin_row_count);
    let previous_count = call_recur!(
        tile = count_row_chunk,
        input = previous_rows,
        chunk = 256,
        state = count_seed,
        args = ()
    );
    let start_position = call!(range_start, previous_start, previous_count)?;

    let token_ids = select!(List<u32>, prompt.token_ids);
    let claimed_ids = select!(List<u32>, claimed.token_ids);
    let hidden_size = select!(u32, embedding.clone().hidden_size);
    let page_size = select!(u64, embedding.clone().values.page_size);
    let embedding_scale = select!(i32, embedding.clone().embedding_scale);
    let values = select!(Bytes<196_608>, embedding.values);

    let draft = call!(begin_prompt_activations, start_position);
    let prompt_rows = call_recur_seq!(
        sequence = embed_prompt_token,
        input = token_ids,
        output = draft,
        args = (
            values.clone(),
            hidden_size.clone(),
            page_size.clone(),
            embedding_scale.clone()
        )
    );
    let embedded = call_recur_seq!(
        sequence = embed_prompt_token,
        input = claimed_ids,
        output = prompt_rows,
        args = (values, hidden_size, page_size, embedding_scale)
    );
    raster::println!("embedding pass → {:?}", embedded);

    let errors = select!(List<String>, embedded.clone().errors);
    let summary_seed = call!(begin_error_summary);
    let summary = call_recur!(
        tile = summarise_errors,
        input = errors,
        state = summary_seed,
        args = ()
    );
    let error_count = select!(u32, summary.clone().count);
    let first_error = select!(String, summary.first);
    call!(assert_all_tokens_embedded, error_count, first_error)?;

    Ok(embedded)
}
