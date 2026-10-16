//! Phase 4 — `prefill-range`: sequences.
//!
//! ```text
//!   rows    ──recur seq──▶ [ begin · mac Q · mac K · mac V · finish ] ──▶ KvSequence
//!   queries ──recur seq──▶ [ attend keys · mac O · mlp matvecs     ] ──▶ ActivationSequence
//!                            (key sweeps run `chunk = 64`)
//!   errors  ──recur─────▶ summarise ──▶ assert none
//! ```

use raster::prelude::*;

use prefill_range::input::*;
use prefill_range::*;

/// One prompt position: RMS-norm, project Q/K/V by paging each matrix, stamp
/// the loop-carried position, append key and query rows.
#[sequence(kind = recur)]
fn project_token(
    input: RecurSequenceInput<ActivationRow>,
    state: RecurSequenceState<TokenCursor>,
    output: RecurSequenceOutput<KvSequence>,
    params: LayerParams,
    w_q: List<BytesPage>,
    w_k: List<BytesPage>,
    w_v: List<BytesPage>,
) -> (
    RecurSequenceState<TokenCursor>,
    RecurSequenceOutput<KvSequence>,
) {
    let prep = call!(begin_qkv, input, params.clone());
    let hidden = select!(u32, params.clone().hidden_size);
    let q_len = call!(q_out_len, params.clone());
    let kv_len = call!(kv_out_len, params.clone());
    let normed = call!(qkv_normed, prep.clone());
    let acc_q = call!(zero_accum, q_len);
    let q = call_recur!(
        tile = mac_weight_page,
        input = w_q,
        state = acc_q,
        args = (normed.clone(), hidden.clone())
    );
    let acc_k = call!(zero_accum, kv_len.clone());
    let k = call_recur!(
        tile = mac_weight_page,
        input = w_k,
        state = acc_k,
        args = (normed.clone(), hidden.clone())
    );
    let acc_v = call!(zero_accum, kv_len);
    let v = call_recur!(
        tile = mac_weight_page,
        input = w_v,
        state = acc_v,
        args = (normed, hidden)
    );
    let output = call!(finish_qkv, output, state.clone(), prep, q, k, v, params);
    let state = call!(advance_cursor, state);
    (state, output)
}

/// One query: attend over keys, project W_o, run the MLP matvecs, append the
/// layer output row.
#[sequence(kind = recur)]
fn attend_token(
    input: RecurSequenceInput<QueryRow>,
    output: RecurSequenceOutput<ActivationSequence>,
    prior_keys: List<KeyRow>,
    keys: List<KeyRow>,
    donor_a_keys: List<KeyRow>,
    donor_b_keys: List<KeyRow>,
    params: LayerParams,
    w_o: List<BytesPage>,
    w_gate: List<BytesPage>,
    w_up: List<BytesPage>,
    w_down: List<BytesPage>,
    ple_rows: List<PleRow>,
    ple_input_gate: List<BytesPage>,
    ple_layer_projection: List<BytesPage>,
) -> RecurSequenceOutput<ActivationSequence> {
    let query = call!(begin_attention, input);
    // Attention in two passes, matching the reference: score every visible key,
    // softmax once over the whole window, then accumulate the weighted values in
    // `Acc` precision and requantize once.
    //
    // Each pass sweeps the own-cache lists and both committed donor candidates.
    // Exactly one source contributes: tiles compare the source layer against
    // the exact `kv_donor_layer` model fact instead of trusting manifest
    // topology alone.
    //
    // Sweep order is deliberately not load-bearing. Both `score_key` and
    // `accumulate_context` address by `key.position - window_start`, so a key
    // lands in the right slot whichever list it arrived on and whichever
    // iteration saw it. That is what lets a decode stage's inherited cache be
    // pruned to the window without the scores silently shifting.
    let scores = call!(zero_scores, query.clone(), params.clone());
    let donor_a_layer = select!(i32, params.clone().donor_a_layer);
    let donor_b_layer = select!(i32, params.clone().donor_b_layer);
    let scores = call_recur!(
        tile = score_key,
        input = prior_keys.clone(),
        chunk = 64,
        state = scores,
        args = (query.clone(), params.clone(), -1)
    );
    let scores = call_recur!(
        tile = score_key,
        input = keys.clone(),
        chunk = 64,
        state = scores,
        args = (query.clone(), params.clone(), -1)
    );
    let scores = call_recur!(
        tile = score_key,
        input = donor_a_keys.clone(),
        chunk = 64,
        state = scores,
        args = (query.clone(), params.clone(), donor_a_layer.clone())
    );
    let scores = call_recur!(
        tile = score_key,
        input = donor_b_keys.clone(),
        chunk = 64,
        state = scores,
        args = (query.clone(), params.clone(), donor_b_layer.clone())
    );
    let weights = call!(softmax_scores, scores, params.clone());

    let context_acc = call!(zero_context, params.clone());
    let context_acc = call_recur!(
        tile = accumulate_context,
        input = prior_keys,
        chunk = 64,
        state = context_acc,
        args = (query.clone(), weights.clone(), params.clone(), -1)
    );
    let context_acc = call_recur!(
        tile = accumulate_context,
        input = keys.clone(),
        chunk = 64,
        state = context_acc,
        args = (query.clone(), weights.clone(), params.clone(), -1)
    );
    let context_acc = call_recur!(
        tile = accumulate_context,
        input = donor_a_keys,
        chunk = 64,
        state = context_acc,
        args = (
            query.clone(),
            weights.clone(),
            params.clone(),
            donor_a_layer
        )
    );
    let context_acc = call_recur!(
        tile = accumulate_context,
        input = donor_b_keys,
        chunk = 64,
        state = context_acc,
        args = (query.clone(), weights, params.clone(), donor_b_layer)
    );
    let context = call!(finish_context, context_acc);
    let context_values = call!(packed_values, context.clone());
    let q_len = call!(q_out_len, params.clone());
    let hidden = select!(u32, params.clone().hidden_size);
    let acc_o = call!(zero_accum, hidden.clone());
    let attn_proj = call_recur!(
        tile = mac_weight_page,
        input = w_o,
        state = acc_o,
        args = (context_values, q_len)
    );
    let residual = call!(
        attn_residual,
        query.clone(),
        context,
        attn_proj,
        params.clone()
    );
    let ff_in = call!(pre_ff_norm, residual.clone(), params.clone());
    let ff_in_values = call!(packed_values, ff_in.clone());
    let ffn = select!(u32, params.clone().ffn_size);
    let acc_gate = call!(zero_accum, ffn.clone());
    let gate = call_recur!(
        tile = mac_weight_page,
        input = w_gate,
        state = acc_gate,
        args = (ff_in_values.clone(), hidden.clone())
    );
    let acc_up = call!(zero_accum, ffn.clone());
    let up = call_recur!(
        tile = mac_weight_page,
        input = w_up,
        state = acc_up,
        args = (ff_in_values, hidden.clone())
    );
    let gated = call!(gelu_mul, gate, up);
    let gated_values = call!(packed_values, gated);
    let acc_down = call!(zero_accum, hidden.clone());
    let ff = call_recur!(
        tile = mac_weight_page,
        input = w_down,
        state = acc_down,
        args = (gated_values, ffn)
    );
    let xs = call!(finish_mlp, residual, ff_in, ff, params.clone());

    // Gemma 3n's per-layer-embedding block, applied after the MLP residual and
    // before `layer_scalar`. The aux stage built one row per token *this stage*
    // processed, so the row for this token is at its stage-local index — not at
    // its absolute position, which for a decode stage runs past the end of a
    // one-row list. An out-of-range dynamic index aborts the run with no
    // output, so this is the difference between a wrong answer and no answer.
    let local = select!(u32, query.clone().local);
    let ple_row = select!(PleRow, ple_rows[local]);
    let xs_values = call!(packed_values, xs.clone());
    let ple_width = select!(u32, params.clone().ple_width);
    let gate_acc = call!(zero_accum, ple_width.clone());
    let gate = call_recur!(
        tile = mac_weight_page,
        input = ple_input_gate,
        state = gate_acc,
        args = (xs_values, hidden.clone())
    );
    let gated = call!(ple_gate_mul, gate, ple_row);
    let gated_values = call!(packed_values, gated);
    let proj_acc = call!(zero_accum, hidden);
    let projected = call_recur!(
        tile = mac_weight_page,
        input = ple_layer_projection,
        state = proj_acc,
        args = (gated_values, ple_width)
    );

    // This token's own cache entry, fetched by an authorized dynamic index.
    // `keys` holds one row per token pass 1 processed, in order, so it is
    // aligned with the stage-local index for the same reason `ple_rows` is.
    let own_key = select!(KeyRow, keys[local]);
    call!(finish_layer, output, query, xs, projected, params, own_key)
}

/// Phase 4 entrypoint, for one transformer layer.
#[sequence]
fn main(
    activations: ActivationSequence,
    layer: TransformerLayer,
    prior_kv: ActivationSequence,
    donor_a_kv: ActivationSequence,
    donor_b_kv: ActivationSequence,
    ple: PleLayerInputs,
) -> Result<ActivationSequence> {
    let declared_params = select!(LayerParams, layer.clone().params);
    let params = call!(validate_layer_params, declared_params)?;

    let hidden = select!(u32, params.clone().hidden_size);
    let q_len = call!(q_out_len, params.clone());
    let kv_len = call!(kv_out_len, params.clone());
    let ffn = select!(u32, params.clone().ffn_size);

    let w_q_len = select!(u64, layer.clone().w_q.byte_len);
    call!(assert_matrix_bytes, w_q_len, q_len.clone(), hidden.clone())?;
    let w_k_len = select!(u64, layer.clone().w_k.byte_len);
    call!(assert_matrix_bytes, w_k_len, kv_len.clone(), hidden.clone())?;
    let w_v_len = select!(u64, layer.clone().w_v.byte_len);
    call!(assert_matrix_bytes, w_v_len, kv_len, hidden.clone())?;
    let w_o_len = select!(u64, layer.clone().w_o.byte_len);
    call!(assert_matrix_bytes, w_o_len, hidden.clone(), q_len)?;
    let w_gate_len = select!(u64, layer.clone().w_gate.byte_len);
    call!(assert_matrix_bytes, w_gate_len, ffn.clone(), hidden.clone())?;
    let w_up_len = select!(u64, layer.clone().w_up.byte_len);
    call!(assert_matrix_bytes, w_up_len, ffn.clone(), hidden.clone())?;
    let w_down_len = select!(u64, layer.clone().w_down.byte_len);
    call!(assert_matrix_bytes, w_down_len, hidden, ffn)?;

    let w_q = select!(List<BytesPage>, layer.clone().w_q.pages);
    let w_k = select!(List<BytesPage>, layer.clone().w_k.pages);
    let w_v = select!(List<BytesPage>, layer.clone().w_v.pages);
    let w_o = select!(List<BytesPage>, layer.clone().w_o.pages);
    let w_gate = select!(List<BytesPage>, layer.clone().w_gate.pages);
    let w_up = select!(List<BytesPage>, layer.clone().w_up.pages);
    let w_down = select!(List<BytesPage>, layer.clone().w_down.pages);

    // Where this stage sits in the full sequence. Prefill starts at zero; a
    // decode stage starts after the prompt and every token generated so far,
    // and RoPE and window visibility are both defined over that absolute value.
    let start_position = select!(u32, activations.clone().start_position);
    let sliding_window = select!(u32, params.clone().sliding_window);

    let rows = select!(List<ActivationRow>, activations.rows);
    let cursor0 = call!(begin_cursor, clone!(start_position));
    let kv = call_recur_seq!(
        sequence = project_token,
        input = rows,
        state = cursor0,
        output,
        args = (params.clone(), w_q, w_k, w_v)
    );
    let keys = select!(List<KeyRow>, kv.clone().keys);
    let queries = select!(List<QueryRow>, kv.clone().queries);
    // This layer's cache as it stood before this stage — empty for every
    // prefill stage, which binds it to `input_embedding` exactly as the
    // non-sharing layers already bind `donor_kv`.
    let prior_keys = select!(List<KeyRow>, prior_kv.kv);
    // Both candidate donor caches are bound uniformly. Own-cache layers bind
    // both to the empty `input_embedding`; sharing layers bind the two actual
    // candidates. The tile accepts only the candidate whose layer number
    // matches committed `kv_donor_layer`.
    let donor_a_keys = select!(List<KeyRow>, donor_a_kv.kv);
    let donor_b_keys = select!(List<KeyRow>, donor_b_kv.kv);

    let ple_rows = select!(List<PleRow>, ple.rows);
    let ple_input_gate = select!(List<BytesPage>, layer.clone().ple_input_gate.pages);
    let ple_layer_projection = select!(List<BytesPage>, layer.ple_layer_projection.pages);

    // The stage's output is built by two push-only sites deriving in turn from
    // the opened object: this sweep carries the inherited cache forward, then
    // `attend_token` appends one row and one key per token. Each derivation is
    // a new object extending its base — `rows` ends up 1 entry on a decode
    // stage while `kv` ends up `prior + 1`, which no single sweep could build.
    let base = call!(begin_layer_output, clone!(start_position));
    let carried = call_recur!(
        tile = carry_cached_key,
        input = prior_keys.clone(),
        chunk = 64,
        output = base,
        args = (start_position, sliding_window)
    );
    let prepared = call_recur_seq!(
        sequence = attend_token,
        input = queries,
        output = carried,
        args = (
            prior_keys,
            keys,
            donor_a_keys,
            donor_b_keys,
            params,
            w_o,
            w_gate,
            w_up,
            w_down,
            ple_rows,
            ple_input_gate,
            ple_layer_projection
        )
    );
    raster::println!("prefill range pass → {:?}", prepared);

    let projection_errors = select!(List<String>, kv.errors);
    let projection_summary_seed = call!(begin_error_summary);
    let projection_summary = call_recur!(
        tile = summarise_errors,
        input = projection_errors,
        state = projection_summary_seed,
        args = ()
    );
    let attention_errors = select!(List<String>, prepared.clone().errors);
    let attention_summary_seed = call!(begin_error_summary);
    let attention_summary = call_recur!(
        tile = summarise_errors,
        input = attention_errors,
        state = attention_summary_seed,
        args = ()
    );

    let projection_count = select!(u32, projection_summary.clone().count);
    let first_projection_error = select!(String, projection_summary.first);
    let attention_count = select!(u32, attention_summary.clone().count);
    let first_attention_error = select!(String, attention_summary.first);
    call!(
        assert_layer_complete,
        projection_count,
        first_projection_error,
        attention_count,
        first_attention_error
    )?;

    Ok(prepared)
}
