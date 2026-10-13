use prefill_range::input::{
    pack_i32s, ActivationRow, ActivationSequence, KeyRow, LayerParams, PleLayerInputs, PleRow,
    TransformerLayer, PAGE_SIZE,
};
use raster::{Bytes, List};
use staged_infer::{routines, run_prefill_range_direct, PrefillRangeDirectInputs};

const ONE: i32 = 1 << 16;

fn bytes_of_i32s(values: &[i32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn paged(values: &[i32]) -> Bytes<196_608> {
    Bytes::<PAGE_SIZE>::paged(bytes_of_i32s(values)).unwrap()
}

fn identity_2() -> Vec<i32> {
    vec![ONE, 0, 0, ONE]
}

fn tiny_layer(kv_donor_layer: i32) -> TransformerLayer {
    let norm = pack_i32s(&[ONE, ONE]);
    TransformerLayer {
        params: LayerParams {
            layer_idx: 0,
            hidden_size: 2,
            ffn_size: 2,
            num_heads: 1,
            num_kv_heads: 1,
            head_dim: 2,
            sliding_window: 0,
            attn_scale: 0,
            layer_scalar: 0,
            norm_eps: 0,
            rope_base: 1_i64 << 32,
            rotary_dim: 0,
            rope_freq_base_dim: 2,
            kv_donor_layer,
            donor_a_layer: if kv_donor_layer >= 0 {
                kv_donor_layer
            } else {
                -2
            },
            donor_b_layer: -3,
            norm_input: norm.clone(),
            norm_post_attn: norm.clone(),
            norm_pre_ffw: norm.clone(),
            norm_post_ffw: norm.clone(),
            q_norm: norm.clone(),
            k_norm: norm.clone(),
            ple_width: 2,
            ple_post_norm: norm,
        },
        w_q: paged(&identity_2()),
        w_k: paged(&identity_2()),
        w_v: paged(&identity_2()),
        w_o: paged(&identity_2()),
        w_gate: paged(&identity_2()),
        w_up: paged(&identity_2()),
        w_down: paged(&identity_2()),
        ple_input_gate: paged(&identity_2()),
        ple_layer_projection: paged(&identity_2()),
    }
}

fn activation_sequence(values: &[i32]) -> ActivationSequence {
    ActivationSequence {
        rows: List::from(vec![ActivationRow {
            token_id: 7,
            values: pack_i32s(values),
        }]),
        errors: List::new(),
        kv: List::new(),
        start_position: 0,
    }
}

fn donor_sequence() -> ActivationSequence {
    ActivationSequence {
        rows: List::new(),
        errors: List::new(),
        kv: List::from(vec![KeyRow {
            position: 0,
            k: pack_i32s(&[ONE, 0]),
            v: pack_i32s(&[ONE, 0]),
        }]),
        start_position: 0,
    }
}

fn ple_inputs() -> PleLayerInputs {
    PleLayerInputs {
        layer_idx: 0,
        rows: List::from(vec![PleRow {
            values: pack_i32s(&[ONE, ONE]),
        }]),
        errors: List::new(),
    }
}

#[test]
fn direct_prefill_returns_activation_and_own_kv() {
    let activations = activation_sequence(&[ONE, 0]);
    let layer = tiny_layer(-1);
    let donor = activation_sequence(&[]);
    let ple = ple_inputs();

    let output = run_prefill_range_direct(PrefillRangeDirectInputs {
        activations: &activations,
        layer: &layer,
        layer_cache_key: None,
        prior_kv: &donor,
        donor_a_kv: &donor,
        donor_b_kv: &donor,
        ple: &ple,
    })
    .unwrap();

    assert_eq!(output.rows.len(), 1);
    assert_eq!(output.rows[0].token_id, 7);
    assert_eq!(output.kv.len(), 1);
    assert!(output.errors.is_empty());

    let wrapper_output = routines::prefill_range::run_direct(&routines::prefill_range::Inputs {
        activations,
        layer: layer.into(),
        layer_cache_key: None,
        prior_kv: donor.clone(),
        donor_a_kv: donor.clone(),
        donor_b_kv: donor,
        ple,
    })
    .unwrap();
    assert_eq!(
        serde_json::to_value(&output).unwrap(),
        serde_json::to_value(&wrapper_output).unwrap()
    );
}

#[test]
fn donor_layer_carries_prior_kv_and_publishes_own_kv() {
    let activations = activation_sequence(&[ONE, 0]);
    let layer = tiny_layer(0);
    let donor = donor_sequence();
    let ple = ple_inputs();

    let output = run_prefill_range_direct(PrefillRangeDirectInputs {
        activations: &activations,
        layer: &layer,
        layer_cache_key: None,
        prior_kv: &donor,
        donor_a_kv: &donor,
        donor_b_kv: &donor,
        ple: &ple,
    })
    .unwrap();

    assert_eq!(output.rows.len(), 1);
    assert_eq!(output.kv.len(), 2);
    assert_eq!(output.kv[0].position, 0);
    assert_eq!(output.kv[1].position, 0);

    let wrapper_output = routines::prefill_range::run_direct(&routines::prefill_range::Inputs {
        activations,
        layer: layer.into(),
        layer_cache_key: None,
        prior_kv: donor.clone(),
        donor_a_kv: donor.clone(),
        donor_b_kv: donor,
        ple,
    })
    .unwrap();
    assert_eq!(
        serde_json::to_value(&output).unwrap(),
        serde_json::to_value(&wrapper_output).unwrap()
    );
}
