//! Self-contained deterministic checks for the vendored fixed-point kernels.

use det_num::ops as det;
use det_num::{f32_to_acc, f32_to_act, Acc, Act, Wgt};

/// Deterministic pseudo-random activations; no rand dependency, and the same
/// sequence on every run so a failure is reproducible.
fn sample_bits(seed: u64, len: usize) -> Vec<i32> {
    let mut state = seed | 1;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 24) as i32) % (8 << 16)
        })
        .collect()
}

#[test]
fn config_float_conversions_use_expected_fixed_widths() {
    assert_eq!(f32_to_act(1.0).to_bits(), 1 << 16);
    assert_eq!(f32_to_act(-0.5).to_bits(), -(1 << 15));
    assert_eq!(f32_to_acc(1.0).to_bits(), 1_i64 << 32);
    assert_eq!(f32_to_acc(10_000.0).to_bits(), 10_000_i64 << 32);
    assert_eq!(f32_to_acc(1_000_000.0).to_bits(), 1_000_000_i64 << 32);
}

#[test]
fn rms_norm_and_value_rms_norm_are_shape_stable() {
    for width in [4usize, 16, 256] {
        let vals: Vec<Act> = sample_bits(0x5EED ^ width as u64, width)
            .into_iter()
            .map(Act::from_bits)
            .collect();
        let weights: Vec<Wgt> = sample_bits(0xB1CE ^ width as u64, width)
            .into_iter()
            .map(Wgt::from_bits)
            .collect();
        let eps = f32_to_acc(1e-6);

        assert_eq!(det::rms_norm(&vals, &weights, eps).len(), width);
        assert_eq!(det::value_rms_norm(&vals, eps).len(), width);
    }
}

#[test]
fn partial_rotary_leaves_the_tail_untouched() {
    let (rotary_dim, freq_base_dim) = (128usize, 512usize);
    let bits = sample_bits(0xABCD, freq_base_dim);
    let input: Vec<Act> = bits.iter().copied().map(Act::from_bits).collect();
    let out = det::rope_rotate_pairs(
        &input,
        rotary_dim,
        freq_base_dim,
        Acc::from_bits(1_000_000_i64 << 32),
        7,
    );

    for i in rotary_dim..freq_base_dim {
        assert_eq!(
            out[i].to_bits(),
            bits[i],
            "lane {i} past rotary_dim was modified"
        );
    }
}
