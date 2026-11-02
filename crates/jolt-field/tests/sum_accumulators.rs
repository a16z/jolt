#![cfg(any(feature = "bn254", feature = "solinas"))]

#[cfg(feature = "bn254")]
use jolt_field::Fr;
#[cfg(feature = "solinas")]
use jolt_field::Prime128OffsetA7F7;
use jolt_field::{Accumulator, JoltField};
use num_bigint::BigUint;

fn check_sum_accumulator<F: JoltField>() {
    let modulus = BigUint::from_bytes_le(&(-F::one()).to_bytes_le_vec()) + 1u8;
    let mut left = F::SumAccumulator::default();
    let mut right = F::SumAccumulator::default();
    let mut expected = BigUint::ZERO;
    assert_eq!(left.reduce(), F::zero());
    for index in 0..128u64 {
        let value = -F::from_u64(index + 1);
        let other = -F::from_u64(3 * index + 1);
        let accumulator = if index % 2 == 0 {
            &mut left
        } else {
            &mut right
        };
        accumulator.add(value);
        let integer = BigUint::from_bytes_le(&value.to_bytes_le_vec());
        expected += &integer;
        if index % 4 == 0 {
            accumulator.fmadd(value, other);
            expected += integer * BigUint::from_bytes_le(&other.to_bytes_le_vec());
        }
    }
    left.merge(right);
    // At most 160 * 2^55 field additions, within the documented slot bound.
    // Doubling exercises high carries without a correspondingly long loop.
    for _ in 0..55 {
        left.merge(left);
        expected <<= 1usize;
    }
    assert_eq!(
        BigUint::from_bytes_le(&left.reduce().to_bytes_le_vec()),
        expected % modulus
    );
}

#[cfg(feature = "solinas")]
#[test]
fn fp128_sum_accumulator_obeys_field_sum_and_merge_laws() {
    check_sum_accumulator::<Prime128OffsetA7F7>();
}

#[cfg(feature = "bn254")]
#[test]
fn fr_sum_accumulator_obeys_field_sum_and_merge_laws() {
    check_sum_accumulator::<Fr>();
}
