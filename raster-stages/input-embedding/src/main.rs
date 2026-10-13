//! Phase 2 — `input-embedding`: sequences.
//!
//! ```text
//!   token_ids ──recur seq──▶ [ offset · page_of · select page · extract ] ──▶ ActivationSequence
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

/// Phase 2 entrypoint.
///
/// `prompt` is bound by the chain to `prompt_prepare`'s authorized output;
/// `embedding` is a committed external.
#[sequence]
fn main(prompt: PromptTokenization, embedding: EmbeddingTable) -> Result<ActivationSequence> {
    let token_ids = select!(List<u32>, prompt.token_ids);
    let hidden_size = select!(u32, embedding.clone().hidden_size);
    let page_size = select!(u64, embedding.clone().values.page_size);
    let embedding_scale = select!(i32, embedding.clone().embedding_scale);
    let values = select!(Bytes<196_608>, embedding.values);

    let draft = call!(begin_prompt_activations, new!(ActivationSequence));
    let embedded = call_recur_seq!(
        sequence = embed_prompt_token,
        input = token_ids,
        output = draft,
        args = (values, hidden_size, page_size, embedding_scale)
    );
    raster::println!("embedding pass → {:?}", embedded);

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
