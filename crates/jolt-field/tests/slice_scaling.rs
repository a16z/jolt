#![cfg(feature = "solinas")]

use jolt_field::{Field, Prime128OffsetA7F7 as F, Ring};

#[test]
fn scale_slice_respects_empty_short_and_long_slice_boundaries() {
    let source = [2, 3, 5].map(F::from_u64);
    let mut destination = [F::from_u64(99); 4];
    F::scale_slice_into(F::from_i64(-1), &source, &mut destination);
    assert_eq!(destination, [-2, -3, -5, 99].map(F::from_i64));
    F::scale_slice_into(F::from_u64(7), &source, &mut destination[..2]);
    assert_eq!(destination, [14, 21, -5, 99].map(F::from_i64));
    F::scale_slice_into(F::from_u64(0), &source, &mut destination[..1]);
    assert_eq!(destination, [0, 21, -5, 99].map(F::from_i64));
    F::scale_slice_into(F::from_u64(7), &[], &mut destination);
    F::scale_slice_into(F::from_u64(7), &source, &mut []);
    assert_eq!(destination, [0, 21, -5, 99].map(F::from_i64));
}
