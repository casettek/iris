//! `Q8_0` block weights: 32 signed 8-bit values sharing one IEEE
//! half-precision scale, laid out as GGUF stores them (34 bytes per block:
//! the scale's little-endian bits, then the 32 values).
//!
//! Local to this repo, not vendored, and pure integer arithmetic like the rest
//! of the crate. Every half-precision value is an integer multiple of `2^-24`,
//! so a scale decodes *exactly* to a Q·.24 integer; a block's weights are then
//! exact too, and a dot product against Q16.16 activations is exact until its
//! single final rounding. Results depend only on the bytes, never on
//! summation order or platform.

/// Values per block.
pub const BLOCK: usize = 32;
/// Bytes per block: a 2-byte scale and 32 one-byte values.
pub const BLOCK_BYTES: usize = 2 + BLOCK;
/// Fractional bits of a decoded scale.
pub const SCALE_FRACTIONAL_BITS: u32 = 24;

/// Bytes of one `Q8_0` row of `cols` weights, or `None` unless `cols` is a
/// positive multiple of the block size.
pub fn row_bytes(cols: usize) -> Option<usize> {
    (cols > 0 && cols % BLOCK == 0).then(|| cols / BLOCK * BLOCK_BYTES)
}

/// A half-precision scale's exact value in Q·.24, or `None` for infinities
/// and NaN. The largest finite magnitude, 65504, is below `2^40`.
pub fn scale_q24(bits: u16) -> Option<i64> {
    let exponent = (bits >> 10) & 0x1f;
    let mantissa = i64::from(bits & 0x3ff);
    let magnitude = match exponent {
        0x1f => return None,
        // Subnormal: mantissa · 2^-24.
        0 => mantissa,
        // Normal: (1024 + mantissa) · 2^(exponent - 25).
        _ => (1024 + mantissa) << (exponent - 1),
    };
    Some(if bits & 0x8000 != 0 {
        -magnitude
    } else {
        magnitude
    })
}

/// One block's scale.
fn block_scale(block: &[u8]) -> Result<i64, &'static str> {
    scale_q24(u16::from_le_bytes([block[0], block[1]])).ok_or("Q8_0 block scale is not finite")
}

/// One output of a `Q8_0` matrix-vector product: the exact dot product of
/// Q16.16 activations with the row's weights, rounded once to Q16.16 (ties
/// to even) and saturated.
///
/// Within a block the products sum exactly in `i64` (at most `2^43`); each
/// block sum times its scale sums in `i128`, which no realistic row width can
/// overflow.
pub fn dot_row(activations: &[i32], row: &[u8]) -> Result<i32, &'static str> {
    if row_bytes(activations.len()) != Some(row.len()) {
        return Err("Q8_0 row does not match the activation width");
    }
    let mut total: i128 = 0;
    for (block, values) in row
        .chunks_exact(BLOCK_BYTES)
        .zip(activations.chunks_exact(BLOCK))
    {
        let scale = block_scale(block)?;
        let sum = block[2..].iter().zip(values).fold(0_i64, |sum, (q, a)| {
            sum + i64::from(*q as i8) * i64::from(*a)
        });
        total = total.wrapping_add(i128::from(sum) * i128::from(scale));
    }
    Ok(round_to_i32(total, SCALE_FRACTIONAL_BITS))
}

/// Dequantizes a `Q8_0` row into Q16.16 values, each `q · scale` rounded once
/// (ties to even) and saturated.
pub fn dequantize_row(row: &[u8], out: &mut [i32]) -> Result<(), &'static str> {
    if row_bytes(out.len()) != Some(row.len()) {
        return Err("Q8_0 row does not match the output width");
    }
    for (block, values) in row
        .chunks_exact(BLOCK_BYTES)
        .zip(out.chunks_exact_mut(BLOCK))
    {
        let scale = block_scale(block)?;
        for (q, value) in block[2..].iter().zip(values) {
            let exact = i128::from(*q as i8) * i128::from(scale);
            *value = round_to_i32(exact, SCALE_FRACTIONAL_BITS - 16);
        }
    }
    Ok(())
}

/// `value / 2^shift`, rounded to nearest with ties to even, saturated to `i32`.
fn round_to_i32(value: i128, shift: u32) -> i32 {
    let floor = value >> shift;
    let remainder = value - (floor << shift);
    let half = 1_i128 << (shift - 1);
    let rounded = if remainder > half || (remainder == half && floor & 1 == 1) {
        floor + 1
    } else {
        floor
    };
    rounded.clamp(i128::from(i32::MIN), i128::from(i32::MAX)) as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn block(scale_bits: u16, values: [i8; BLOCK]) -> [u8; BLOCK_BYTES] {
        let mut out = [0_u8; BLOCK_BYTES];
        out[..2].copy_from_slice(&scale_bits.to_le_bytes());
        for (byte, value) in out[2..].iter_mut().zip(values) {
            *byte = value as u8;
        }
        out
    }

    #[test]
    fn half_precision_scales_decode_exactly() {
        assert_eq!(scale_q24(0x3c00), Some(1 << 24)); // 1.0
        assert_eq!(scale_q24(0xbc00), Some(-(1 << 24))); // -1.0
        assert_eq!(scale_q24(0x0001), Some(1)); // smallest subnormal, 2^-24
        assert_eq!(scale_q24(0x0400), Some(1 << 10)); // smallest normal, 2^-14
        assert_eq!(scale_q24(0x7bff), Some(65504 << 24)); // largest finite
        assert_eq!(scale_q24(0x3555), Some((1024 + 0x155) << 12)); // 0.333…
        assert_eq!(scale_q24(0x7c00), None);
        assert_eq!(scale_q24(0x7e00), None);
    }

    #[test]
    fn dot_row_is_the_exact_product_rounded_once() {
        // Scale 2^-7, so each weight is q / 128.
        let mut values = [0_i8; BLOCK];
        values[0] = 3;
        values[1] = -5;
        let row = block(0x2000, values);
        let mut activations = [0_i32; BLOCK];
        activations[0] = 1 << 16; // 1.0
        activations[1] = 3 << 15; // 1.5
                                  // 3/128 - 7.5/128 = -4.5/128 = -0.03515625 = -2304 in Q16.16.
        assert_eq!(dot_row(&activations, &row), Ok(-2304));
        // Ties go to even: 1 · 2^-17 is exactly half a Q16.16 step.
        let mut one = [0_i8; BLOCK];
        one[0] = 1;
        let tiny = block(0x0200, one); // 2^-15
        let mut a = [0_i32; BLOCK];
        a[0] = 1 << 14; // 0.25 → product 2^-17
        assert_eq!(dot_row(&a, &tiny), Ok(0));
        a[0] = 3 << 14; // 0.75 → 3 · 2^-17 = 1.5 steps → 2
        assert_eq!(dot_row(&a, &tiny), Ok(2));
        a[0] = -(3 << 14);
        assert_eq!(dot_row(&a, &tiny), Ok(-2));
    }

    #[test]
    fn dequantized_rows_match_their_dot_products() {
        let mut values = [0_i8; BLOCK];
        for (index, value) in values.iter_mut().enumerate() {
            *value = (index as i8 - 16) * 7;
        }
        let row = block(0x1e66, values); // ~0.0062
        let mut weights = [0_i32; BLOCK];
        dequantize_row(&row, &mut weights).unwrap();
        for index in 0..BLOCK {
            let mut a = [0_i32; BLOCK];
            a[index] = 1 << 16;
            assert_eq!(dot_row(&a, &row), Ok(weights[index]));
        }
    }

    #[test]
    fn malformed_rows_are_rejected() {
        assert_eq!(row_bytes(33), None);
        assert_eq!(row_bytes(64), Some(68));
        let row = block(0x7c00, [1; BLOCK]);
        assert!(dot_row(&[0; BLOCK], &row).is_err());
        assert!(dot_row(&[0; 2 * BLOCK], &row).is_err());
    }
}
