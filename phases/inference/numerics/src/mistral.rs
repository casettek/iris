//! The integer functions only `mistral-dense` uses.
//!
//! Each one is defined by the arithmetic written here, with every rounding
//! fixed, so the native layer and a stage program that calls the same code
//! agree bit for bit. They use 64-bit integers only, apart from the table of
//! rotation frequencies, which an import computes once.
//!
//! They are here, and not beside the family's native layer, so that the
//! family's stage program compiles the code direct inference runs.

use alloc::vec::Vec;

const ONE_Q32: u64 = 1 << 32;
/// `ln 2` in Q32, rounded to nearest.
const LN_2_Q32: u64 = 2_977_044_472;
/// `2 pi` and its quarters in Q32.32, the constants `det-num` reduces angles by.
const HALF_PI_Q32: i64 = 6_746_518_852;
const PI_Q32: i64 = 13_493_037_705;
const THREE_HALF_PI_Q32: i64 = 20_239_556_557;
const TWO_PI_Q32: i64 = 26_986_075_409;

/// `x / 2^shift`, rounded to nearest with ties to even.
fn shift_round_ties_even(value: i64, shift: u32) -> i64 {
    let floor = value >> shift;
    let remainder = value - (floor << shift);
    let half = 1_i64 << (shift - 1);
    if remainder > half || (remainder == half && floor & 1 == 1) {
        floor + 1
    } else {
        floor
    }
}

fn saturate(value: i64) -> i32 {
    value.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32
}

/// `e^-a` in Q32 for `a >= 0` in Q32.
///
/// `a = k ln 2 + r` with `0 <= r < ln 2`; `e^-r` is its Taylor series to
/// fourteen terms in Horner form, each product rounded down, and the result is
/// shifted right by `k`.
fn exp_neg_q32(a: u64) -> u64 {
    let k = a / LN_2_Q32;
    if k >= 32 {
        return 0;
    }
    let r = a % LN_2_Q32;
    let mut acc = ONE_Q32;
    let mut n = 14_u64;
    while n >= 1 {
        acc = ONE_Q32 - ((r * acc) >> 32) / n;
        n -= 1;
    }
    acc >> k
}

/// The logistic function of a Q16.16 value, in Q32 (`0 ..= 2^32`).
pub fn sigmoid_q32(x: i32) -> u64 {
    let magnitude = u64::from(x.unsigned_abs());
    // `1 / (1 + e^-|x|)`: the divisor is in `2^32 ..= 2^33`.
    let divisor = ONE_Q32 + exp_neg_q32(magnitude << 16);
    let positive = ((1_u64 << 63) / divisor) << 1;
    if x < 0 {
        ONE_Q32 - positive
    } else {
        positive
    }
}

/// `x * sigmoid(x)` for a Q16.16 value, rounded once to Q16.16.
pub fn silu(x: i32) -> i32 {
    // `|x| <= 2^31` and the sigmoid is at most `2^32`, and it is `0` where
    // `x` is `i32::MIN`, so the product fits.
    let product = i64::from(x) * sigmoid_q32(x) as i64;
    saturate(shift_round_ties_even(product, 32))
}

/// Sine and cosine of a Q32.32 angle, in Q30.
///
/// The angle is reduced to a quarter turn as `det-num` reduces it. Each
/// series is in Horner form over the square of the reduced angle, every
/// product rounded down.
fn sin_cos_q30(angle: i64) -> (i64, i64) {
    const ONE_Q30: i64 = 1 << 30;
    let reduced = angle.rem_euclid(TWO_PI_Q32);
    let (quarter, sin_sign, cos_sign) = if reduced <= HALF_PI_Q32 {
        (reduced, 1, 1)
    } else if reduced <= PI_Q32 {
        (PI_Q32 - reduced, 1, -1)
    } else if reduced <= THREE_HALF_PI_Q32 {
        (reduced - PI_Q32, -1, -1)
    } else {
        (TWO_PI_Q32 - reduced, -1, 1)
    };
    // At most `pi / 2`, so under `2^31` in Q30.
    let t = quarter >> 2;
    let u = (t * t) >> 30;
    let mut sin = ONE_Q30;
    for divisor in [210_i64, 156, 110, 72, 42, 20, 6] {
        sin = ONE_Q30 - ((u * sin) >> 30) / divisor;
    }
    let sin = (t * sin) >> 30;
    let mut cos = ONE_Q30;
    for divisor in [240_i64, 182, 132, 90, 56, 30, 12, 2] {
        cos = ONE_Q30 - ((u * cos) >> 30) / divisor;
    }
    (sin_sign * sin, cos_sign * cos)
}

/// Rotates consecutive pairs of a head's values: pair `j` is values `2j` and
/// `2j + 1`, turned by `position * inverse_frequencies[j]`. Each output value
/// is rounded once to Q16.16.
pub fn rotate_consecutive_pairs(values: &mut [i32], inverse_frequencies: &[i64], position: u32) {
    if position == 0 {
        return;
    }
    for (pair, inverse_frequency) in values.chunks_exact_mut(2).zip(inverse_frequencies) {
        // A frequency is at most one and a position under `2^31`.
        let angle = i64::from(position) * inverse_frequency;
        let (sin, cos) = sin_cos_q30(angle);
        let (a, b) = (i64::from(pair[0]), i64::from(pair[1]));
        pair[0] = saturate(shift_round_ties_even(a * cos - b * sin, 30));
        pair[1] = saturate(shift_round_ties_even(a * sin + b * cos, 30));
    }
}

/// An attention score: the wrapping Q32.32 sum of a query head's products
/// with a key head, times `scale_q32 / 2^32`, rounded once to Q16.16.
pub fn scaled_score(query: &[i32], key: &[i32], scale_q32: u32) -> i32 {
    let sum = query.iter().zip(key).fold(0_i64, |sum, (q, k)| {
        sum.wrapping_add(i64::from(*q) * i64::from(*k))
    });
    let scaled = (i128::from(sum) * i128::from(scale_q32)) >> 32;
    // Q32.32 to Q16.16.
    let half = 1_i128 << 15;
    let floor = scaled >> 16;
    let remainder = scaled - (floor << 16);
    let rounded = if remainder > half || (remainder == half && floor & 1 == 1) {
        floor + 1
    } else {
        floor
    };
    rounded.clamp(i128::from(i32::MIN), i128::from(i32::MAX)) as i32
}

/// `2^32 / sqrt(head_dim)`, rounded down.
pub fn attention_scale_q32(head_dim: u32) -> u32 {
    let target = (1_u128 << 64) / u128::from(head_dim.max(1));
    let (mut low, mut high) = (0_u128, u128::from(u32::MAX));
    while low < high {
        let mid = (low + high + 1) / 2;
        if mid * mid <= target {
            low = mid;
        } else {
            high = mid - 1;
        }
    }
    low as u32
}

fn integer_sqrt(value: u128) -> u128 {
    if value < 2 {
        return value;
    }
    let (mut low, mut high) = (1_u128, 1_u128 << 64);
    while low < high {
        let mid = (low + high + 1) / 2;
        if mid <= value / mid {
            low = mid;
        } else {
            high = mid - 1;
        }
    }
    low
}

/// The angle per position of each consecutive pair under YaRN scaling, in
/// Q32.32.
///
/// Pair `j` of `pairs` starts from `base^(-j / pairs)`. Pairs below `low` keep
/// it, pairs from `high` up have it divided by `factor`, and those between are
/// mixed linearly. `pairs` must be a power of two and `base` a whole number,
/// so that the root is a chain of integer square roots, each rounded down.
pub fn yarn_inverse_frequencies(
    base: u64,
    pairs: u32,
    factor: u32,
    low: u32,
    high: u32,
) -> Option<Vec<i64>> {
    if !pairs.is_power_of_two() || base == 0 || factor == 0 || high <= low {
        return None;
    }
    // `base^(1 / pairs)` in Q32.32.
    let mut root = u128::from(base) << 32;
    for _ in 0..pairs.trailing_zeros() {
        root = integer_sqrt(root << 32);
    }
    // Its inverse, rounded to nearest.
    let step = ((1_u128 << 64) + root / 2) / root;
    let span = u128::from(high - low);
    let mut plain = 1_u128 << 32;
    let mut out = Vec::with_capacity(pairs as usize);
    for pair in 0..pairs {
        // `ramp = kept / span` is the share of the unscaled frequency.
        let kept = u128::from(high.saturating_sub(pair)).min(span);
        let numerator = plain * (span + u128::from(factor - 1) * kept);
        let denominator = u128::from(factor) * span;
        out.push(((numerator + denominator / 2) / denominator) as i64);
        plain = (plain * step + (1 << 31)) >> 32;
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn real(bits: i32) -> f64 {
        f64::from(bits) / 65_536.0
    }

    #[test]
    fn silu_is_within_two_units_of_the_real_function() {
        let mut worst = 0.0_f64;
        let mut x = -(24 << 16);
        while x <= 24 << 16 {
            let exact = real(x) / (1.0 + (-real(x)).exp());
            worst = worst.max((real(silu(x)) - exact).abs());
            x += 977;
        }
        assert!(worst <= 2.0 / 65_536.0, "{worst}");
        assert_eq!(silu(0), 0);
        assert_eq!(silu(i32::MIN), 0);
        assert_eq!(silu(i32::MAX), i32::MAX);
    }

    #[test]
    fn rotation_is_within_two_units_of_the_real_one() {
        let frequencies = yarn_inverse_frequencies(1_000_000, 64, 16, 20, 37).unwrap();
        for position in [1_u32, 7, 513, 16_383] {
            let mut values: Vec<i32> = (0..128).map(|i| (i - 64) * 9_173).collect();
            let before = values.clone();
            rotate_consecutive_pairs(&mut values, &frequencies, position);
            for pair in 0..64 {
                let angle = f64::from(position) * frequencies[pair] as f64 / 4_294_967_296.0;
                let (a, b) = (real(before[2 * pair]), real(before[2 * pair + 1]));
                let expected = (
                    a * angle.cos() - b * angle.sin(),
                    a * angle.sin() + b * angle.cos(),
                );
                assert!((real(values[2 * pair]) - expected.0).abs() <= 2.0 / 65_536.0);
                assert!((real(values[2 * pair + 1]) - expected.1).abs() <= 2.0 / 65_536.0);
            }
        }
    }

    #[test]
    fn yarn_frequencies_follow_the_formula() {
        let frequencies = yarn_inverse_frequencies(1_000_000, 64, 16, 20, 37).unwrap();
        assert_eq!(frequencies[0], 1 << 32);
        for (pair, bits) in frequencies.iter().enumerate() {
            let plain = 1_000_000_f64.powf(-(pair as f64) / 64.0);
            let ramp = 1.0 - ((pair as f64 - 20.0) / 17.0).clamp(0.0, 1.0);
            let expected = plain * (1.0 / 16.0 + 15.0 / 16.0 * ramp);
            let got = *bits as f64 / 4_294_967_296.0;
            assert!(
                (got - expected).abs() <= 1e-8,
                "pair {pair}: {got} vs {expected}"
            );
        }
        assert!(yarn_inverse_frequencies(1_000_000, 48, 16, 20, 37).is_none());
    }

    #[test]
    fn the_score_scale_is_the_inverse_root() {
        assert_eq!(attention_scale_q32(128), 379_625_062);
        assert_eq!(attention_scale_q32(1), u32::MAX);
        let ones = vec![1 << 16; 128];
        // 128 / sqrt(128) = sqrt(128).
        let score = scaled_score(&ones, &ones, attention_scale_q32(128));
        assert!((real(score) - 128_f64.sqrt()).abs() <= 2.0 / 65_536.0);
    }
}
