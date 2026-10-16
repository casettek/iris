//! Phase 3 — `prefill-prepare-aux`: sequences.
//!
//! ```text
//!   tokens ──recur seq──▶ [ page_of · project pages · combine ] ──▶ PleLayerInputs
//!   misses ──recur─────▶ count ──▶ assert none
//! ```

use raster::prelude::*;

use prefill_prepare_aux::input::*;
use prefill_prepare_aux::*;

/// One prompt position: gather its PLE embedding, project its activation row,
/// and append the combined layer input.
#[sequence(kind = recur)]
fn prepare_ple_row(
    input: RecurSequenceInput<ActivationRow>,
    output: RecurSequenceOutput<PleLayerInputs>,
    embeddings: Bytes<196_608>,
    projection_pages: List<BytesPage>,
    params: PleLayerParams,
    page_size: u64,
) -> RecurSequenceOutput<PleLayerInputs> {
    let task = call!(begin_ple_task, input);
    let token_id = select!(u32, task.clone().token_id);
    let ple_width = select!(u32, params.clone().ple_width);
    let hidden = select!(u32, params.clone().hidden_size);
    let byte_off = call!(ple_byte_offset, token_id, ple_width.clone());
    let page_idx = call!(page_of, byte_off.clone(), page_size);
    let emb_page = select!(BytesPage, embeddings[page_idx]);
    let acc = call!(zero_accum, ple_width);
    let activation = call!(ple_activation, task.clone());
    let projected = call_recur!(
        tile = mac_weight_page,
        input = projection_pages,
        state = acc,
        args = (activation, hidden)
    );
    let projected = call!(finish_ple_projection, projected, params.clone());
    call!(
        combine_ple_row,
        output,
        task,
        emb_page,
        byte_off,
        projected,
        params
    )
}

/// Phase 3 entrypoint, for one PLE layer.
#[sequence]
fn main(embedded: ActivationSequence, layer: PleLayer) -> Result<PleLayerInputs> {
    let declared_params = select!(PleLayerParams, layer.clone().params);
    let params = call!(validate_layer_params, declared_params)?;
    let page_size = select!(u64, layer.clone().embeddings.page_size);
    let embeddings = select!(Bytes<196_608>, layer.clone().embeddings);
    let proj_len = select!(u64, layer.clone().projection.byte_len);
    let projection_pages = select!(List<BytesPage>, layer.projection.pages);
    let hidden = select!(u32, params.clone().hidden_size);
    let ple_width = select!(u32, params.clone().ple_width);
    call!(assert_matrix_bytes, proj_len, ple_width, hidden)?;

    let tokens = select!(List<ActivationRow>, embedded.rows);
    let layer_idx = select!(u32, params.clone().layer_idx);
    let draft = call!(begin_ple_layer, layer_idx);

    let prepared = call_recur_seq!(
        sequence = prepare_ple_row,
        input = tokens,
        output = draft,
        args = (embeddings, projection_pages, params, page_size)
    );
    raster::println!("prefill aux pass → {:?}", prepared);

    let errors = select!(List<String>, prepared.clone().errors);
    let summary_seed = call!(begin_error_summary);
    let summary = call_recur!(
        tile = summarise_layer_errors,
        input = errors,
        state = summary_seed,
        args = ()
    );
    let error_count = select!(u32, summary.clone().count);
    let first_error = select!(String, summary.first);
    call!(assert_layer_complete, error_count, first_error)?;

    Ok(prepared)
}
