use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use inference_generation::inference_weights::MmapDetwgt;
use inference_generation::model::{
    DirectFinalHead, DirectMatrixView, DirectQ8View, DirectTransformerLayer,
};
use inference_generation::view_kernels::{run_prefill_range_view, score_prefill_logits_view};
use inference_kernels::tensor::{dot_bits, matvec_from_source, Matrix, MatrixSource};
use prefill_range::input::{
    pack_i32s, unpack_i32s, ActivationRow, ActivationSequence, LayerParams, PleLayerInputs, PleRow,
    TransformerLayer,
};
use raster::{Bytes, List};
use sequence_check::input::FinalHeadParams;

const ONE: i32 = 1 << 16;
/// Hidden width of the fixtures: one `Q8_0` block, the narrowest int8 row.
const W: usize = 32;

struct TensorFixture {
    name: &'static str,
    dims: Vec<u64>,
    element_width: u32,
    values: Vec<i32>,
}

#[test]
fn mmap_detwgt_parses_directory_and_sign_extends_i16() {
    let dir = temp_dir("loader");
    fs::create_dir_all(&dir).unwrap();
    let detwgt = dir.join("model.detwgt");
    write_detwgt(
        &detwgt,
        &[TensorFixture {
            name: "tiny",
            dims: vec![2],
            element_width: 16,
            values: vec![-1, 2],
        }],
    );
    let model = MmapDetwgt::open(&detwgt).unwrap();
    assert_eq!(model.values("tiny").unwrap(), vec![-1, 2]);
    assert_eq!(model.slice("tiny").unwrap().value_at(0).unwrap(), -1);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn detwgt_matrix_view_matches_eager_matrix_matvec() {
    let dir = temp_dir("matrix-view");
    fs::create_dir_all(&dir).unwrap();
    let detwgt = dir.join("model.detwgt");
    let values = vec![ONE, 0, 0, 0, ONE, ONE];
    write_detwgt(
        &detwgt,
        &[TensorFixture {
            name: "matrix",
            dims: vec![2, 3],
            element_width: 32,
            values: values.clone(),
        }],
    );
    let model = MmapDetwgt::open(&detwgt).unwrap();
    let view = DirectMatrixView::new(model.matrix("matrix", 2, 3).unwrap());
    let eager = Matrix::from_region("matrix", &paged(&values), 2, 3).unwrap();
    let input = [ONE, 2 * ONE, 3 * ONE];

    assert_eq!(
        matvec_from_source(&view, &input).unwrap(),
        matvec_from_source(&eager, &input).unwrap()
    );

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn row_dot_matches_dot_bits_for_i16_and_i32() {
    let dir = temp_dir("row-dot");
    fs::create_dir_all(&dir).unwrap();
    let detwgt = dir.join("model.detwgt");
    write_detwgt(
        &detwgt,
        &[
            TensorFixture {
                name: "i16_matrix",
                dims: vec![2, 2],
                element_width: 16,
                values: vec![1, -2, 3, 4],
            },
            TensorFixture {
                name: "i32_matrix",
                dims: vec![2, 2],
                element_width: 32,
                values: vec![ONE, 0, 0, ONE],
            },
        ],
    );
    let model = MmapDetwgt::open(&detwgt).unwrap();

    let i16_view = DirectMatrixView::new(model.matrix("i16_matrix", 2, 2).unwrap());
    assert_eq!(
        i16_view.row_dot(0, &[5, 6]).unwrap(),
        dot_bits(&[5, 6], &[1, -2])
    );
    let i32_view = DirectMatrixView::new(model.matrix("i32_matrix", 2, 2).unwrap());
    assert_eq!(
        i32_view.row_dot(1, &[7, 8]).unwrap(),
        dot_bits(&[7, 8], &[0, ONE])
    );

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn final_head_logits_feed_shared_decode_select_tie_break() {
    let dir = temp_dir("stream-final-head");
    fs::create_dir_all(&dir).unwrap();
    let detwgt = dir.join("model.detwgt");
    write_detwgt(
        &detwgt,
        &[TensorFixture {
            name: "projection",
            dims: vec![3, W as u64],
            element_width: 8,
            values: one_hot_rows(&[Some(0), None, Some(0)]),
        }],
    );
    let model = MmapDetwgt::open(&detwgt).unwrap();
    let head = DirectFinalHead {
        params: FinalHeadParams {
            hidden_size: W as u32,
            norm_eps: 0,
            softcap: 0,
            norm_weights: pack_i32_page(&[ONE; W]),
        },
        projection: DirectQ8View::new(model.q8_matrix("projection").unwrap()),
    };
    let activations = prefill_range::input::ActivationSequence {
        model: Default::default(),
        rows: List::from(vec![prefill_range::input::ActivationRow {
            token_id: 1,
            values: pack_i32_page(&one_hot_rows(&[Some(0)])),
        }]),
        errors: List::new(),
        kv: List::new(),
        start_position: 0,
    };
    let logits = score_prefill_logits_view(&activations, &head).unwrap();
    let prior = inference_kernels::kernels::origin::empty_decode_edge();
    let edge = inference_generation::generation::select_next_token(&logits, &prior).unwrap();

    assert_eq!(logits.decode_position, 1);
    assert_eq!(logits.logits.len(), 3);
    assert_eq!(edge.token_id, 0, "equal scores keep the earliest token");
    assert_eq!(
        edge.generated_token_ids.iter().copied().collect::<Vec<_>>(),
        vec![0]
    );

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_decode_rotary_outputs_match_prefill_across_chunks() {
    let dir = temp_dir("decode-rope");
    fs::create_dir_all(&dir).unwrap();
    let detwgt = dir.join("model.detwgt");
    let identity = one_hot_rows(&(0..W).map(Some).collect::<Vec<_>>());
    write_detwgt(
        &detwgt,
        &[
            TensorFixture {
                name: "identity",
                dims: vec![W as u64, W as u64],
                element_width: 32,
                values: identity.clone(),
            },
            TensorFixture {
                name: "q8_identity",
                dims: vec![W as u64, W as u64],
                element_width: 8,
                values: identity.clone(),
            },
        ],
    );
    let model = MmapDetwgt::open(&detwgt).unwrap();
    let matrix = DirectMatrixView::new(model.matrix("identity", W, W).unwrap());
    let q8_matrix = DirectQ8View::new(model.q8_matrix("q8_identity").unwrap());
    let norm = pack_i32s(&[ONE; W]);
    let unit = pack_i32s(&one_hot_rows(&[Some(0)]));
    let params = LayerParams {
        layer_idx: 0,
        hidden_size: W as u32,
        ffn_size: W as u32,
        num_heads: 1,
        num_kv_heads: 1,
        head_dim: W as u32,
        sliding_window: 0,
        attn_scale: ONE,
        layer_scalar: 0,
        norm_eps: 0,
        rope_base: 10_000_i64 << 32,
        rotary_dim: 2,
        rope_freq_base_dim: 2,
        kv_donor_layer: -1,
        donor_a_layer: -2,
        donor_b_layer: -3,
        norm_input: norm.clone(),
        norm_post_attn: norm.clone(),
        norm_pre_ffw: norm.clone(),
        norm_post_ffw: norm.clone(),
        q_norm: unit.clone(),
        k_norm: unit,
        ple_width: W as u32,
        ple_post_norm: norm,
    };
    let view = DirectTransformerLayer {
        params: params.clone(),
        w_q: q8_matrix,
        w_k: q8_matrix,
        w_v: q8_matrix,
        w_o: q8_matrix,
        w_gate: q8_matrix,
        w_up: q8_matrix,
        w_down: q8_matrix,
        ple_input_gate: matrix,
        ple_layer_projection: matrix,
    };
    let layer = TransformerLayer {
        params,
        w_q: q8_paged(&identity),
        w_k: q8_paged(&identity),
        w_v: q8_paged(&identity),
        w_o: q8_paged(&identity),
        w_gate: q8_paged(&identity),
        w_up: q8_paged(&identity),
        w_down: q8_paged(&identity),
        ple_input_gate: paged(&identity),
        ple_layer_projection: paged(&identity),
    };
    let empty = ActivationSequence {
        model: Default::default(),
        rows: List::new(),
        errors: List::new(),
        kv: List::new(),
        start_position: 0,
    };
    let activations = ActivationSequence {
        rows: List::from(
            (0..5)
                .map(|token_id| ActivationRow {
                    token_id,
                    // Distinct V rows make attention outputs sensitive to Q.
                    values: pack_i32s(&{
                        // Every lane at magnitude ONE keeps the RMS at ONE.
                        let mut row = [ONE; W];
                        row[1] = if token_id % 2 == 0 { ONE } else { -ONE };
                        row
                    }),
                })
                .collect::<Vec<_>>(),
        ),
        ..empty.clone()
    };
    let ple = PleLayerInputs {
        model: Default::default(),
        layer_idx: 0,
        rows: List::from(vec![
            PleRow {
                values: pack_i32s(&[ONE; W])
            };
            5
        ]),
        errors: List::new(),
    };
    let run_staged = |input: &ActivationSequence, prior: &ActivationSequence| {
        inference_kernels::run_prefill_range_direct(inference_kernels::PrefillRangeDirectInputs {
            activations: input,
            layer: &layer,
            layer_cache_key: None,
            prior_kv: prior,
            donor_a_kv: &empty,
            donor_b_kv: &empty,
            ple: &ple,
        })
        .unwrap()
    };
    let reference = run_staged(&activations, &empty);
    assert_eq!(
        unpack_i32s(&reference.kv[1].k).unwrap()[..2],
        [35408, 55146]
    );

    let mut prior = empty.clone();
    // Prefill one token, decode twice, then process a two-token chunk.
    for (start, end) in [(0, 1), (1, 2), (2, 3), (3, 5)] {
        let chunk = ActivationSequence {
            rows: List::from(activations.rows.as_slice()[start..end].to_vec()),
            start_position: start as u32,
            ..empty.clone()
        };
        let direct = run_prefill_range_view(&chunk, &view, &prior, &empty, &empty, &ple).unwrap();
        let staged = run_staged(&chunk, &prior);
        for output in [&direct, &staged] {
            assert!(output.errors.is_empty());
            assert_eq!(output.start_position, start as u32);
            assert_eq!(
                serde_json::to_value(&output.rows).unwrap(),
                serde_json::to_value(&reference.rows.as_slice()[start..end]).unwrap(),
                "activation rows at start_position {start}"
            );
            assert_eq!(
                serde_json::to_value(&output.kv).unwrap(),
                serde_json::to_value(&reference.kv.as_slice()[..end]).unwrap(),
                "all cached K/V rows at start_position {start}"
            );
        }
        prior = direct;
    }

    fs::remove_dir_all(dir).unwrap();
}

fn write_detwgt(path: &Path, tensors: &[TensorFixture]) {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"DNWGTV0\0");
    bytes.extend_from_slice(&3_u32.to_le_bytes());
    bytes.extend_from_slice(&1_u32.to_le_bytes());
    bytes.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    for tensor in tensors {
        bytes.extend_from_slice(&(tensor.name.len() as u32).to_le_bytes());
        bytes.extend_from_slice(tensor.name.as_bytes());
        bytes.extend_from_slice(&(tensor.dims.len() as u32).to_le_bytes());
        for dim in &tensor.dims {
            bytes.extend_from_slice(&dim.to_le_bytes());
        }
        bytes.extend_from_slice(&(tensor.values.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&tensor.element_width.to_le_bytes());
        let payload_len = match tensor.element_width {
            8 => q8_exact(&tensor.values).len(),
            width => tensor.values.len() * (width as usize / 8),
        };
        bytes.extend_from_slice(&(payload_len as u64).to_le_bytes());
        bytes.extend_from_slice(&0_u64.to_le_bytes());
        while bytes.len() % 64 != 0 {
            bytes.push(0);
        }
        match tensor.element_width {
            8 => bytes.extend(q8_exact(&tensor.values)),
            16 => {
                for value in &tensor.values {
                    bytes.extend_from_slice(&(*value as i16).to_le_bytes());
                }
            }
            32 => {
                for value in &tensor.values {
                    bytes.extend_from_slice(&value.to_le_bytes());
                }
            }
            other => panic!("unsupported fixture width {other}"),
        }
    }
    fs::write(path, bytes).unwrap();
}

/// Rows `W` wide: `Some(col)` is `ONE` at `col`, `None` is all zero.
fn one_hot_rows(hot: &[Option<usize>]) -> Vec<i32> {
    hot.iter()
        .flat_map(|col| (0..W).map(move |i| if Some(i) == *col { ONE } else { 0 }))
        .collect()
}

/// Q16.16 `values` as exact `Q8_0` blocks: each block takes the largest
/// power-of-two scale that represents all its values exactly.
fn q8_exact(values: &[i32]) -> Vec<u8> {
    let mut out = Vec::new();
    for block in values.chunks(32) {
        assert_eq!(block.len(), 32, "Q8_0 rows are whole blocks");
        let k = (0..=14)
            .find(|k| {
                let step = ONE >> k;
                block
                    .iter()
                    .all(|v| v % step == 0 && (v / step).abs() <= 127)
            })
            .expect("values are not exactly representable as Q8_0");
        out.extend_from_slice(&(((15 - k) as u16) << 10).to_le_bytes());
        out.extend(block.iter().map(|v| (v / (ONE >> k)) as i8 as u8));
    }
    out
}

fn q8_paged<const P: u64>(values: &[i32]) -> Bytes<P> {
    Bytes::<P>::paged(q8_exact(values)).unwrap()
}

fn paged(values: &[i32]) -> Bytes<184_320> {
    Bytes::<184_320>::paged(bytes_of_i32s(values)).unwrap()
}

fn pack_i32_page(values: &[i32]) -> raster::BytesPage {
    raster::BytesPage::__from_parts(0, 0, bytes_of_i32s(values))
}

fn bytes_of_i32s(values: &[i32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn temp_dir(label: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "direct-infer-{label}-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ))
}
