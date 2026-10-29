#![cfg(all(feature = "signed-byte-sum", feature = "solinas"))]

use jolt_field::{Field, Prime128Offset275, Prime128OffsetA7F7};

fn signed_byte_boundaries<F: Field>() {
    let values = [2, 3, 5, 7, 11].map(F::from_u64);
    let coefficients = [i8::MIN, -2, 0, 1, i8::MAX];
    assert_eq!(
        F::sum_signed_bytes(values.iter().zip(coefficients)),
        F::from_i64(1142)
    );
    assert_eq!(F::sum_signed_bytes([]), F::zero());
    let minus_one = F::from_i64(-1);
    for coefficient in i8::MIN..=i8::MAX {
        assert_eq!(
            F::sum_signed_bytes([(&minus_one, coefficient)]),
            F::from_i64(-i64::from(coefficient))
        );
    }
}

#[test]
fn signed_byte_sum_integer_contract() {
    signed_byte_boundaries::<Prime128OffsetA7F7>();
    signed_byte_boundaries::<Prime128Offset275>();
}
