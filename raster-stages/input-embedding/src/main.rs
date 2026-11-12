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
    values: Bytes<195_840>,
    hidden_size: u32,
    page_size: u64,
    table_bytes: u64,
    embedding_scale: i32,
) -> RecurSequenceOutput<ActivationSequence> {
    let token_id = call!(begin_row_lookup, input);
    let byte_off = call!(
        embedding_byte_offset,
        token_id.clone(),
        hidden_size.clone(),
        table_bytes
    );
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
/// prompt range); `previous` is the previous range's embedding, or
/// the empty sequence for the first; `embedding` is a committed external.
///
/// The claimed ids are embedded after the prompt's by a second push-only
/// site deriving from the first, so the rows are `prompt ‖ claimed` in order.
#[sequence]
fn main(
    prompt: PromptState,
    claimed: ClaimedOutput,
    previous: ActivationSequence,
    embedding: EmbeddingTable,
    witness: SparseWitness,
) -> Result<ActivationSequence> {
    let prompt_model = select!(ModelContext, prompt.clone().context.model);
    let previous_model = select!(ModelContext, previous.clone().model);
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
    let start_position = call!(range_start, previous_start, previous_count.clone())?;

    let token_ids = select!(List<u32>, prompt.token_ids);
    let claimed_ids = select!(List<u32>, claimed.token_ids);
    let hidden_size = select!(u32, embedding.clone().hidden_size);
    let page_size = select!(u64, embedding.clone().values.page_size);
    let table_bytes = select!(u64, embedding.clone().values.byte_len);
    let embedding_scale = select!(i32, embedding.clone().embedding_scale);
    let values = select!(Bytes<195_840>, embedding.clone().values);

    let header = select!(SparseHeader, witness.clone().header);
    let paths = select!(List<SparsePath>, witness.paths);
    let root = call!(
        identity_sparse_header,
        hidden_size.clone(),
        embedding_scale.clone(),
        table_bytes.clone(),
        page_size.clone(),
        header.clone()
    );
    let zero = call!(identity_zero_count);
    let prompt_count = call_recur!(
        tile = identity_count_tokens,
        input = token_ids.clone(),
        chunk = 256,
        state = zero,
        args = ()
    );
    let model = call!(
        identity_embedding_context,
        prompt_model,
        previous_model,
        root,
        previous_count,
        prompt_count.clone()
    );
    let model = call_recur_seq!(
        sequence = verify_token_page,
        input = token_ids.clone(),
        state = model,
        args = (
            values.clone(),
            hidden_size.clone(),
            page_size.clone(),
            table_bytes.clone(),
            header.clone(),
            paths.clone()
        )
    );
    let model = call_recur_seq!(
        sequence = verify_token_page,
        input = claimed_ids.clone(),
        state = model,
        args = (
            values.clone(),
            hidden_size.clone(),
            page_size.clone(),
            table_bytes.clone(),
            header,
            paths
        )
    );
    let model = call_recur!(
        tile = bind_embedding_token,
        input = token_ids.clone(),
        state = model,
        args = ()
    );
    let model = call_recur!(
        tile = bind_embedding_token,
        input = claimed_ids.clone(),
        state = model,
        args = ()
    );
    let zero = call!(identity_zero_count);
    let claimed_count = call_recur!(
        tile = identity_count_tokens,
        input = claimed_ids.clone(),
        chunk = 256,
        state = zero,
        args = ()
    );
    let model = call!(
        finish_embedding_binding,
        model,
        start_position.clone(),
        prompt_count,
        claimed_count
    );
    let draft = call!(begin_prompt_activations, start_position, model);
    let prompt_rows = call_recur_seq!(
        sequence = embed_prompt_token,
        input = token_ids,
        output = draft,
        args = (
            values.clone(),
            hidden_size.clone(),
            page_size.clone(),
            table_bytes.clone(),
            embedding_scale.clone()
        )
    );
    let embedded = call_recur_seq!(
        sequence = embed_prompt_token,
        input = claimed_ids,
        output = prompt_rows,
        args = (values, hidden_size, page_size, table_bytes, embedding_scale)
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

/// Preflight uses precisely the same lookup and immutable page reference as execution.
#[sequence(kind=recur)]
fn verify_token_page(
    input: RecurSequenceInput<u32>,
    state: RecurSequenceState<ModelContext>,
    values: Bytes<195_840>,
    hidden_size: u32,
    page_size: u64,
    table_bytes: u64,
    header: SparseHeader,
    paths: List<SparsePath>,
) -> RecurSequenceState<ModelContext> {
    let token_id = call!(begin_row_lookup, input);
    let byte_off = call!(embedding_byte_offset, token_id, hidden_size, table_bytes);
    let index = call!(page_of, byte_off, page_size);
    let page = select!(BytesPage, values[index]);
    let path = select!(SparsePath, paths[index]);
    call!(identity_sparse_page, state, page, index, header, path)
}
