#![cfg(all(feature = "terminal-contraction", feature = "solinas"))]

use jolt_field::{Field, Prime128Offset275, Prime128OffsetA7F7, Ring, Zero};

fn signed_boundaries<F: Field>() {
    let weights = [2, 3, 5, 7, 11].map(F::from_u64);
    let coefficients = [i16::MIN, -1, 0, 1, i16::MAX];
    // 2*(-32768) - 3 + 7 + 11*32767, evaluated as a signed integer.
    assert_eq!(
        F::dot_product_i16(&weights, coefficients),
        F::from_i64(294_905)
    );
    assert_eq!(F::dot_product_i16(&weights, []), F::zero());
    assert_eq!(F::dot_product_i16(&[], coefficients), F::zero());
    assert_eq!(F::dot_product_i16(&weights[..1], [-2, 99]), F::from_i64(-4));
}

#[test]
fn signed_integer_boundary_vectors() {
    signed_boundaries::<Prime128OffsetA7F7>();
    signed_boundaries::<Prime128Offset275>();
}

#[test]
fn weighted_rectangular_polynomial_vector() {
    type F = Prime128OffsetA7F7;
    let first = [1, 2, 3].map(F::from_u64);
    let second = [4, 5, 6].map(F::from_u64);
    let powers = [1, 7, 49].map(F::from_u64);
    let weights = [F::from_u64(2), F::from_i64(-3)];
    // 2*(1+14+147) - 3*(4+35+294).
    assert_eq!(
        F::weighted_dot_product(&[&first, &second], &weights, &powers),
        F::from_i64(-675)
    );
    assert_eq!(F::weighted_dot_product(&[], &[], &powers), F::zero());
}

#[test]
#[should_panic(expected = "one weight")]
fn weighted_rows_reject_weight_count_mismatch() {
    type F = Prime128OffsetA7F7;
    let row = [F::from_u64(1)];
    let _ = F::weighted_dot_product(&[&row], &[], &row);
}
