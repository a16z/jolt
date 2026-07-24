//! Semantics of the naive scalar/product accumulators and the primitive-int
//! multiplication helpers, checked against additive-chain and byte-level
//! oracles rather than the trait defaults themselves.

#![cfg(feature = "bn254")]

use jolt_field::signed::S256;
use jolt_field::{
    Fr, FromPrimitiveInt, MulPrimitiveInt, NaiveSignedProductAccumulator,
    NaiveSignedScalarAccumulator, SignedProductAccumulator, SignedScalarAccumulator,
};
use num_traits::Zero;

fn f(value: u64) -> Fr {
    Fr::from_u64(value)
}

/// n-fold addition — the independent oracle for every multiplication helper.
fn additive_chain(value: Fr, times: u64) -> Fr {
    let mut acc = Fr::zero();
    for _ in 0..times {
        acc += value;
    }
    acc
}

#[test]
fn mul_primitive_int_matches_additive_chains_and_signs() {
    let x = f(12_345);

    assert_eq!(x.mul_u64(7), additive_chain(x, 7));
    assert_eq!(x.mul_u64(0), Fr::zero());
    assert_eq!(x.mul_i64(5), additive_chain(x, 5));
    assert_eq!(x.mul_i64(-5), -additive_chain(x, 5));
    assert_eq!(x.mul_i64(i64::MIN + 1), -x.mul_u64((i64::MAX) as u64));
    assert_eq!(x.mul_u128(11), additive_chain(x, 11));
    // 2^64 * x == 2^32-fold chain of (2^32 * x): checks the u128 path above
    // the u64 range without 2^64 additions.
    let x_shift32 = x.mul_u64(1u64 << 32);
    assert_eq!(x.mul_u128(1u128 << 64), x_shift32.mul_u64(1u64 << 32));
    assert_eq!(x.mul_i128(-9), -additive_chain(x, 9));
    assert_eq!(x.mul_i128(1i128 << 64), x.mul_u128(1u128 << 64));
}

#[test]
fn naive_scalar_accumulator_folds_signed_multiples() {
    let mut acc = NaiveSignedScalarAccumulator::<Fr>::default();
    acc.add(f(10));
    acc.fmadd_u64(f(3), 4);
    acc.fmadd_i64(f(2), -5);
    // 10 + 3*4 - 2*5 = 12
    assert_eq!(acc.reduce(), f(12));

    let empty = NaiveSignedScalarAccumulator::<Fr>::default();
    assert_eq!(empty.reduce(), Fr::zero());
}

/// A minimal accumulator that does NOT override `fmadd_i64`, so the trait's
/// default sign-splitting body is what runs.
#[derive(Clone, Copy, Default)]
struct DefaultBodyAccumulator(Fr);

impl SignedScalarAccumulator for DefaultBodyAccumulator {
    type Element = Fr;

    fn add(&mut self, value: Fr) {
        self.0 += value;
    }

    fn fmadd_u64(&mut self, value: Fr, scalar: u64) {
        self.0 += value.mul_u64(scalar);
    }

    fn reduce(self) -> Fr {
        self.0
    }
}

#[test]
fn scalar_accumulator_default_fmadd_i64_splits_signs_correctly() {
    let mut acc = DefaultBodyAccumulator::default();
    acc.fmadd_i64(f(7), 3);
    acc.fmadd_i64(f(11), -2);
    acc.fmadd_i64(f(999), 0);
    // 7*3 - 11*2 = -1
    assert_eq!(acc.reduce(), f(21) - f(22));

    // i64::MIN's magnitude does not fit in i64 — the unsigned_abs split must
    // still produce the exact negative multiple.
    let mut acc = DefaultBodyAccumulator::default();
    acc.fmadd_i64(f(1), i64::MIN);
    assert_eq!(acc.reduce(), -f(1).mul_u64(1u64 << 63));
}

#[test]
fn naive_product_accumulator_applies_s256_magnitude_and_sign() {
    let mut acc = NaiveSignedProductAccumulator::<Fr>::default();

    // Zero scalar is skipped entirely.
    acc.fmadd_s256(f(5), &S256::new([0, 0, 0, 0], true));
    assert_eq!(acc.reduce(), Fr::zero());

    // Positive single-limb magnitude: 5 * 42.
    let mut acc = NaiveSignedProductAccumulator::<Fr>::default();
    acc.fmadd_s256(f(5), &S256::from_u64_with_sign(42, true));
    assert_eq!(acc.reduce(), f(5).mul_u64(42));

    // Negative sign flips the term; magnitude spans a second limb:
    // value * (2^64 + 3) with byte-level oracle via mul_u128.
    let mut acc = NaiveSignedProductAccumulator::<Fr>::default();
    acc.fmadd_s256(f(9), &S256::new([3, 1, 0, 0], false));
    assert_eq!(acc.reduce(), -f(9).mul_u128((1u128 << 64) + 3));

    // Terms accumulate across calls: 5*42 - 9*(2^64+3) + 1*1.
    let mut acc = NaiveSignedProductAccumulator::<Fr>::default();
    acc.fmadd_s256(f(5), &S256::from_u64_with_sign(42, true));
    acc.fmadd_s256(f(9), &S256::new([3, 1, 0, 0], false));
    acc.fmadd_s256(f(1), &S256::from_u64_with_sign(1, true));
    assert_eq!(
        acc.reduce(),
        f(5).mul_u64(42) - f(9).mul_u128((1u128 << 64) + 3) + f(1)
    );
}
