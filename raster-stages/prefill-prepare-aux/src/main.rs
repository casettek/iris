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
    embeddings: Bytes<195_840>,
    projection_pages: List<BytesPage>,
    params: PleLayerParams,
    page_size: u64,
    table_bytes: u64,
) -> RecurSequenceOutput<PleLayerInputs> {
    let task = call!(begin_ple_task, input);
    let token_id = select!(u32, task.clone().token_id);
    let ple_width = select!(u32, params.clone().ple_width);
    let hidden = select!(u32, params.clone().hidden_size);
    let byte_off = call!(ple_byte_offset, token_id, ple_width.clone(), table_bytes);
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
fn main(
    embedded: ActivationSequence,
    layer: PleLayer,
    witness: SparseWitness,
) -> Result<PleLayerInputs> {
    let declared_params = select!(PleLayerParams, layer.clone().params);
    let params = call!(validate_layer_params, declared_params)?;
    let page_size = select!(u64, layer.clone().embeddings.page_size);
    let table_bytes = select!(u64, layer.clone().embeddings.byte_len);
    let embeddings = select!(Bytes<195_840>, layer.clone().embeddings);
    let proj_len = select!(u64, layer.clone().projection.byte_len);
    let projection_pages = select!(List<BytesPage>, layer.clone().projection.pages);
    let hidden = select!(u32, params.clone().hidden_size);
    let ple_width = select!(u32, params.clone().ple_width);
    call!(assert_matrix_bytes, proj_len, ple_width, hidden)?;

    let model = select!(ModelContext, embedded.clone().model);
    let tokens = select!(List<ActivationRow>, embedded.rows);
    let layer_idx = select!(u32, params.clone().layer_idx);
    let header = select!(SparseHeader, witness.clone().header);
    let paths = select!(List<SparsePath>, witness.paths);
    let root = call_seq!(identity_root, layer, header.clone());
    let model = call!(identity_verify, model, root, layer_idx.clone());
    let model = call_recur_seq!(
        sequence = verify_ple_page,
        input = tokens.clone(),
        state = model,
        args = (
            embeddings.clone(),
            params.clone(),
            page_size.clone(),
            table_bytes.clone(),
            header,
            paths
        )
    );
    let draft = call!(begin_ple_layer, layer_idx, model);

    let prepared = call_recur_seq!(
        sequence = prepare_ple_row,
        input = tokens,
        output = draft,
        args = (embeddings, projection_pages, params, page_size, table_bytes)
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

#[sequence]
fn identity_bytes_184_320(value: Bytes<184_320>) -> BytesPage {
    let len = select!(u64, value.clone().byte_len);
    let size = select!(u64, value.clone().page_size);
    let pages = select!(List<BytesPage>, value.pages);
    let seed = call!(identity_empty);
    let folded = call_recur!(tile = identity_page, input = pages, state = seed, args = ());
    let root = call!(identity_finish, folded);
    call!(identity_bytes, len, size, root)
}
#[sequence]
fn identity_root(value: PleLayer, header: SparseHeader) -> BytesPage {
    let params = select!(PleLayerParams, value.clone().params);
    let params = call!(identity_hash_plelayerparams, params);
    let embeddings = select!(Bytes<195_840>, value.clone().embeddings);
    let len = select!(u64, embeddings.clone().byte_len);
    let size = select!(u64, embeddings.page_size);
    let embeddings = call!(identity_sparse_bytes, len, size, header);
    let projection = select!(Bytes<184_320>, value.clone().projection);
    let projection = call_seq!(identity_bytes_184_320, projection);
    call!(identity_component, params, embeddings, projection)
}

#[sequence(kind=recur)]
fn verify_ple_page(
    input: RecurSequenceInput<ActivationRow>,
    state: RecurSequenceState<ModelContext>,
    embeddings: Bytes<195_840>,
    params: PleLayerParams,
    page_size: u64,
    table_bytes: u64,
    header: SparseHeader,
    paths: List<SparsePath>,
) -> RecurSequenceState<ModelContext> {
    let task = call!(begin_ple_task, input);
    let token = select!(u32, task.token_id);
    let width = select!(u32, params.ple_width);
    let offset = call!(ple_byte_offset, token, width, table_bytes);
    let index = call!(page_of, offset, page_size);
    let page = select!(BytesPage, embeddings[index]);
    let path = select!(SparsePath, paths[index]);
    call!(identity_sparse_page, state, page, index, header, path)
}
