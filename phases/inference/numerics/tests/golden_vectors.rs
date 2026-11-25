//! Literal expectations calculated with integer arithmetic, not another kernel.
use det_num::ops::{
    acc_combine, add_sat, mac, mac_bits, requantize, rshift_round_ties_even, sub_sat,
};
use det_num::{Acc, Act, Wgt};

#[test]
fn signed_rounding_ties_and_neighbors() {
    // At shift 16, one half is 32768. Ties choose the even integer on both
    // sides of zero; adjacent values must fall on opposite sides of the tie.
    for (input, expected) in [
        (32767, 0),
        (32768, 0),
        (32769, 1),
        (98303, 1),
        (98304, 2),
        (163840, 2),
        (-32767, 0),
        (-32768, 0),
        (-32769, -1),
        (-98303, -1),
        (-98304, -2),
        (-163840, -2),
    ] {
        assert_eq!(
            rshift_round_ties_even(Acc::from_bits(input), 16).to_bits(),
            expected
        );
        assert_eq!(requantize(Acc::from_bits(input)).to_bits(), expected as i32);
    }
}

#[test]
fn wrapping_mac_and_partial_accumulators() {
    for (acc, a, b, expected) in [
        (i64::MAX, 1, 1, i64::MIN),
        (i64::MIN, -1, 1, i64::MAX),
        (7, -3, 5, -8),
        (0, i32::MIN, i32::MIN, 4_611_686_018_427_387_904),
        (4_611_686_018_427_387_904, i32::MIN, i32::MIN, i64::MIN),
    ] {
        assert_eq!(mac_bits(acc, a, b), expected);
        assert_eq!(
            mac(Acc::from_bits(acc), Act::from_bits(a), Wgt::from_bits(b)).to_bits(),
            expected
        );
    }
    assert_eq!(
        acc_combine(Acc::from_bits(i64::MAX), Acc::from_bits(1)).to_bits(),
        i64::MIN
    );
}

#[test]
fn activation_saturation_is_distinct_from_mac_wrapping() {
    for (input, expected) in [
        (i64::MAX, i32::MAX),
        (i64::MIN, i32::MIN),
        (140_737_488_289_792, i32::MAX),
        (-140_737_488_355_328, i32::MIN),
    ] {
        assert_eq!(requantize(Acc::from_bits(input)).to_bits(), expected);
    }
    assert_eq!(
        add_sat(Act::from_bits(i32::MAX), Act::from_bits(1)).to_bits(),
        i32::MAX
    );
    assert_eq!(
        add_sat(Act::from_bits(i32::MIN), Act::from_bits(-1)).to_bits(),
        i32::MIN
    );
    assert_eq!(
        sub_sat(Act::from_bits(i32::MIN), Act::from_bits(1)).to_bits(),
        i32::MIN
    );
    assert_eq!(
        sub_sat(Act::from_bits(i32::MAX), Act::from_bits(-1)).to_bits(),
        i32::MAX
    );
}
