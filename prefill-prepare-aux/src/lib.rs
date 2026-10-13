//! Phase 3 — `prefill-prepare-aux`: tiles.
//!
//! Gemma's per-layer-embedding (PLE) prefill inputs, for **one** layer.
//! The PLE table is addressed by token id; the projection is streamed as
//! pages. The layer loop lives in the chain manifest.
//!
//! `#![no_std]` so the same tiles compile into RISC0 replay guests.

#![no_std]

extern crate alloc;

use alloc::string::String;
use alloc::vec;
use raster::prelude::*;

pub mod input;

use input::*;

/// Seeds the output draft with the layer it belongs to.
#[tile(kind = iter, description = "Open this layer's prefill-input draft")]
pub fn begin_ple_layer(layer_idx: u32) -> Draft<PleLayerInputs> {
    let mut output = Draft::<PleLayerInputs>::new();
    output.layer_idx().set(layer_idx);
    output
}

/// The activation page on a PLE task. `select!(BytesPage, …)` is only for
/// indexing a `Bytes` region, so a field that *is* a page is extracted here.
#[tile(kind = iter, description = "Activation page of a PLE task")]
pub fn ple_activation(task: PleTask) -> BytesPage {
    task.activation
}

/// Materializes one embedded prompt position into a task.
#[tile(kind = iter, description = "Turn an embedded token into a PLE task")]
pub fn begin_ple_task(token: ActivationRow) -> PleTask {
    PleTask {
        token_id: token.token_id,
        activation: token.values,
    }
}

/// `token_id * ple_width * 4`.
#[tile(kind = iter, description = "Byte offset of a PLE embedding row")]
pub fn ple_byte_offset(token_id: u32, ple_width: u32) -> u64 {
    token_id as u64 * ple_width as u64 * 4
}

/// Page index containing a byte offset.
#[tile(kind = iter, description = "Page containing a byte offset", estimated_cycles = 200)]
pub fn page_of(byte_offset: u64, page_size: u64) -> u64 {
    if page_size == 0 {
        0
    } else {
        byte_offset / page_size
    }
}

/// Zero-filled matvec accumulator of `out_len` i32s.
#[tile(kind = iter, description = "Zero a matvec accumulator")]
pub fn zero_accum(out_len: u32) -> ProjAccum {
    ProjAccum {
        values: pack_i32s(&vec![0; out_len as usize]),
        error: String::new(),
    }
}

/// `byte_len` must equal `out_len * in_width * 4`.
#[tile(kind = iter, description = "Reject a weight region of the wrong byte length")]
pub fn assert_matrix_bytes(byte_len: u64, out_len: u32, in_width: u32) -> Result<u32> {
    let expected = out_len as u64 * in_width as u64 * 4;
    if byte_len == expected {
        Ok(0)
    } else {
        Err(alloc::format!(
            "matrix has {byte_len} bytes, expected {expected} ({out_len} x {in_width} i32s)"
        ))
    }
}

/// One page of a row-major matrix: each output row is `in_width` i32s, dotted
/// with `row` and written at `page.offset() / stride`.
#[tile(
    kind = recur,
    description = "MAC one weight page into the output accumulator",
    estimated_cycles = 800_000
)]
pub fn mac_weight_page(
    input: RecurInput<BytesPage>,
    state: RecurState<ProjAccum>,
    row: BytesPage,
    in_width: u32,
) -> RecurState<ProjAccum> {
    let mut state = state;
    if !state.error.is_empty() {
        return state;
    }
    let page = input.into_value();
    if let Err(error) = mac_page(&mut state, &page, &row, in_width) {
        state.error = error;
    }
    state
}

fn mac_page(
    state: &mut ProjAccum,
    page: &BytesPage,
    row: &BytesPage,
    in_width: u32,
) -> core::result::Result<(), String> {
    let row = unpack_i32s(row)?;
    if row.len() != in_width as usize {
        return Err(alloc::format!(
            "activation has {} values, expected in_width {in_width}",
            row.len()
        ));
    }
    if in_width == 0 {
        return Err(String::from("matvec in_width is zero"));
    }
    let weights = unpack_i32s(page)?;
    if weights.len() % row.len() != 0 {
        return Err(String::from("weight page does not contain whole output rows"));
    }
    let stride = in_width as u64 * 4;
    if page.offset() % stride != 0 {
        return Err(String::from("weight page offset is not row-aligned"));
    }
    let mut out = unpack_i32s(&state.values)?;
    let start_row = (page.offset() / stride) as usize;
    let rows_in_page = weights.len() / row.len();
    for local in 0..rows_in_page {
        let out_idx = start_row + local;
        if out_idx >= out.len() {
            return Err(String::from("weight page writes past the accumulator"));
        }
        let w = &weights[local * row.len()..(local + 1) * row.len()];
        out[out_idx] = dot(&row, w)?;
    }
    state.values = pack_i32s(&out);
    Ok(())
}

/// Scale, RMS-norm the projected vector.
#[tile(kind = iter, description = "Scale and normalise a PLE projection")]
pub fn finish_ple_projection(projected: ProjAccum, params: PleLayerParams) -> ProjectedRow {
    if !projected.error.is_empty() {
        return ProjectedRow {
            values: pack_i32s(&[]),
            error: projected.error,
        };
    }
    match finish_projection_inner(&projected, &params) {
        Ok(values) => ProjectedRow {
            values: pack_i32s(&values),
            error: String::new(),
        },
        Err(error) => ProjectedRow {
            values: pack_i32s(&[]),
            error,
        },
    }
}

fn finish_projection_inner(
    projected: &ProjAccum,
    params: &PleLayerParams,
) -> core::result::Result<alloc::vec::Vec<i32>, String> {
    let mut values = unpack_i32s(&projected.values)?;
    if values.len() != params.ple_width as usize {
        return Err(alloc::format!(
            "projection has {} values, expected ple_width {}",
            values.len(),
            params.ple_width
        ));
    }
    scale_row(&mut values, params.projection_scalar);
    rms_norm(
        &mut values,
        &unpack_i32s(&params.norm_weights)?,
        params.norm_eps,
    )?;
    Ok(values)
}

/// Combines the scaled PLE embedding with the projected row and appends it.
#[tile(kind = iter, description = "Combine the PLE embedding with the projected row")]
pub fn combine_ple_row(
    output: Draft<PleLayerInputs>,
    task: PleTask,
    emb_page: BytesPage,
    byte_off: u64,
    projected: ProjectedRow,
    params: PleLayerParams,
) -> Draft<PleLayerInputs> {
    let mut output = output;
    match combine_row(&task, &emb_page, byte_off, &projected, &params) {
        Ok(values) => output.rows().push(PleRow {
            values: pack_i32s(&values),
        }),
        Err(error) => output.errors().push(error),
    }
    output
}

fn combine_row(
    task: &PleTask,
    emb_page: &BytesPage,
    byte_off: u64,
    projected: &ProjectedRow,
    params: &PleLayerParams,
) -> core::result::Result<alloc::vec::Vec<i32>, String> {
    if !projected.error.is_empty() {
        return Err(projected.error.clone());
    }
    let mut embedded = unpack_i32s_at(emb_page, byte_off, params.ple_width)?;
    let projected = unpack_i32s(&projected.values)?;
    if embedded.len() != projected.len() {
        return Err(alloc::format!(
            "PLE row width mismatch for token {}: {} embedded vs {} projected",
            task.token_id,
            embedded.len(),
            projected.len()
        ));
    }

    scale_row(&mut embedded, params.embedding_scale);
    for (value, projected_value) in embedded.iter_mut().zip(projected) {
        *value = add_sat(*value, projected_value);
    }
    scale_row(&mut embedded, params.input_scale);
    Ok(embedded)
}

/// The failure fold's opening state: nothing counted.
///
/// A stored seed, so the fold's state chain starts at a committed value
/// rather than a literal nothing pins.
#[tile(kind = iter, description = "Open the failure summary with nothing counted")]
pub fn begin_error_summary() -> ErrorSummary {
    ErrorSummary {
        count: 0,
        first: String::new(),
    }
}

/// Folds the recorded failures into a count plus the first message.
#[tile(kind = recur, description = "Summarise the rows this layer could not produce")]
pub fn summarise_layer_errors(
    input: RecurInput<String>,
    state: RecurState<ErrorSummary>,
) -> RecurState<ErrorSummary> {
    let mut state = state;
    let message = input.into_value();
    if state.count == 0 {
        state.first = message;
    }
    state.count += 1;
    state
}

/// Fails the program when any row failed.
#[tile(kind = iter, description = "Reject an incomplete layer input")]
pub fn assert_layer_complete(error_count: u32, first_error: String) -> Result<u32> {
    if error_count == 0 {
        Ok(error_count)
    } else {
        Err(alloc::format!(
            "{error_count} PLE row(s) failed; first: {first_error}"
        ))
    }
}

/// Guards the layer external before any of it is used.
#[tile(kind = iter, description = "Validate this layer's declared widths")]
pub fn validate_layer_params(params: PleLayerParams) -> Result<PleLayerParams> {
    if params.hidden_size == 0 || params.ple_width == 0 {
        return Err(String::from(
            "PLE layer requires non-zero hidden_size and ple_width",
        ));
    }
    let expected_norm = params.ple_width as usize * 4;
    if params.norm_weights.len() != expected_norm {
        return Err(alloc::format!(
            "layer {} norm weights have {} bytes, expected {expected_norm}",
            params.layer_idx,
            params.norm_weights.len()
        ));
    }
    if params.norm_eps < 0 {
        return Err(String::from("PLE rms-norm epsilon must be non-negative"));
    }
    Ok(params)
}
