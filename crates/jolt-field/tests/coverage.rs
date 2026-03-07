//! Targeted tests to improve code coverage across the jolt-field crate.
//!
//! Covers: NaiveAccumulator, WideAccumulator, Mont254BitChallenge,
//! MontU128Challenge Display/Debug, OptimizedMul blanket impl,
//! Field default methods, SignedBigInt uncovered paths,
//! SignedBigIntHi32 uncovered paths, and macro-generated operator variants.

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::test_rng;
use jolt_field::challenge::{Mont254BitChallenge, MontU128Challenge};
use jolt_field::signed::*;
use jolt_field::{Field, FieldAccumulator, Fr, Limbs, NaiveAccumulator, OptimizedMul};
use num_traits::{One, Zero};

// =========================================================================
// 1. NaiveAccumulator
// =========================================================================

#[test]
fn naive_accumulator_fmadd() {
    let a = <Fr as Field>::from_u64(7);
    let b = <Fr as Field>::from_u64(11);
    let c = <Fr as Field>::from_u64(3);
    let d = <Fr as Field>::from_u64(5);

    let mut acc = NaiveAccumulator::<Fr>::default();
    acc.fmadd(a, b);
    acc.fmadd(c, d);
    // 7*11 + 3*5 = 77 + 15 = 92
    assert_eq!(acc.reduce(), <Fr as Field>::from_u64(92));
}

#[test]
fn naive_accumulator_merge() {
    let mut acc1 = NaiveAccumulator::<Fr>::default();
    acc1.fmadd(<Fr as Field>::from_u64(2), <Fr as Field>::from_u64(3));

    let mut acc2 = NaiveAccumulator::<Fr>::default();
    acc2.fmadd(<Fr as Field>::from_u64(4), <Fr as Field>::from_u64(5));

    acc1.merge(acc2);
    // 2*3 + 4*5 = 6 + 20 = 26
    assert_eq!(acc1.reduce(), <Fr as Field>::from_u64(26));
}

#[test]
fn naive_accumulator_reduce_empty() {
    let acc = NaiveAccumulator::<Fr>::default();
    assert!(acc.reduce().is_zero());
}

// =========================================================================
// 2. WideAccumulator
// =========================================================================

#[test]
fn wide_accumulator_fmadd() {
    use jolt_field::WideAccumulator;

    let a = <Fr as Field>::from_u64(13);
    let b = <Fr as Field>::from_u64(17);

    let mut acc = WideAccumulator::default();
    acc.fmadd(a, b);
    assert_eq!(acc.reduce(), <Fr as Field>::from_u64(13 * 17));
}

#[test]
fn wide_accumulator_merge() {
    use jolt_field::WideAccumulator;

    let mut acc1 = WideAccumulator::default();
    acc1.fmadd(<Fr as Field>::from_u64(10), <Fr as Field>::from_u64(20));

    let mut acc2 = WideAccumulator::default();
    acc2.fmadd(<Fr as Field>::from_u64(30), <Fr as Field>::from_u64(40));

    acc1.merge(acc2);
    // 10*20 + 30*40 = 200 + 1200 = 1400
    assert_eq!(acc1.reduce(), <Fr as Field>::from_u64(1400));
}

#[test]
fn wide_accumulator_reduce_empty() {
    use jolt_field::WideAccumulator;

    let acc = WideAccumulator::default();
    assert!(acc.reduce().is_zero());
}

#[test]
fn wide_accumulator_many_fmadds() {
    use jolt_field::WideAccumulator;

    let mut acc = WideAccumulator::default();
    let mut expected = Fr::zero();
    let mut rng = test_rng();
    for _ in 0..500 {
        let a: Fr = Field::random(&mut rng);
        let b: Fr = Field::random(&mut rng);
        acc.fmadd(a, b);
        expected += a * b;
    }
    assert_eq!(acc.reduce(), expected);
}

// =========================================================================
// 3. Mont254BitChallenge (unconditional tests)
// =========================================================================

#[test]
fn mont254_construction_from_u128() {
    let c = Mont254BitChallenge::<Fr>::from(42u128);
    let f: Fr = c.into();
    assert_eq!(f, <Fr as Field>::from_u128(42));
}

#[test]
fn mont254_construction_from_field() {
    let f = <Fr as Field>::from_u64(99);
    let c = Mont254BitChallenge::<Fr>::from(f);
    assert_eq!(c.value(), f);
}

#[test]
fn mont254_new_and_value() {
    let f: Fr = Field::random(&mut test_rng());
    let c = Mont254BitChallenge::new(f);
    assert_eq!(c.value(), f);
}

#[test]
fn mont254_random() {
    let c = Mont254BitChallenge::<Fr>::random(&mut test_rng());
    let f: Fr = c.into();
    // Random element should almost never be zero
    let _ = f;
}

#[test]
fn mont254_display() {
    let c = Mont254BitChallenge::<Fr>::from(123u128);
    let s = format!("{c}");
    assert!(s.starts_with("Mont254BitChallenge("));
}

#[test]
fn mont254_debug() {
    let c = Mont254BitChallenge::<Fr>::from(456u128);
    let s = format!("{c:?}");
    // Debug is derived, should contain field name
    assert!(s.contains("Mont254BitChallenge"));
}

#[test]
fn mont254_neg() {
    let f = <Fr as Field>::from_u64(10);
    let c = Mont254BitChallenge::new(f);
    let neg_f: Fr = -c;
    assert_eq!(neg_f, -f);
}

#[test]
fn mont254_into_fr_from_ref() {
    let c = Mont254BitChallenge::<Fr>::from(77u128);
    let f1: Fr = c.into();
    let f2: Fr = Fr::from(&c);
    assert_eq!(f1, f2);
}

#[test]
fn mont254_uniform_rand() {
    use ark_ff::UniformRand;
    let c = Mont254BitChallenge::<Fr>::rand(&mut test_rng());
    let _f: Fr = c.into();
}

#[test]
fn mont254_arithmetic() {
    let mut rng = test_rng();
    let a = Mont254BitChallenge::<Fr>::from(100u128);
    let b = Mont254BitChallenge::<Fr>::from(200u128);
    let c: Fr = Field::random(&mut rng);

    // Challenge + Challenge
    let sum: Fr = a + b;
    assert_eq!(sum, Into::<Fr>::into(a) + Into::<Fr>::into(b));

    // Challenge - Challenge
    let diff: Fr = a - b;
    assert_eq!(diff, Into::<Fr>::into(a) - Into::<Fr>::into(b));

    // Challenge * Challenge
    let prod: Fr = a * b;
    assert_eq!(prod, Into::<Fr>::into(a) * Into::<Fr>::into(b));

    // Challenge + Field
    let sum2: Fr = a + c;
    assert_eq!(sum2, Into::<Fr>::into(a) + c);

    // Field + Challenge
    let sum3: Fr = c + a;
    assert_eq!(sum3, c + Into::<Fr>::into(a));

    // Challenge - Field
    let diff2: Fr = a - c;
    assert_eq!(diff2, Into::<Fr>::into(a) - c);

    // Field - Challenge
    let diff3: Fr = c - a;
    assert_eq!(diff3, c - Into::<Fr>::into(a));

    // Challenge * Field
    let prod2: Fr = a * c;
    assert_eq!(prod2, Into::<Fr>::into(a) * c);

    // Field * Challenge
    let prod3: Fr = c * a;
    assert_eq!(prod3, c * Into::<Fr>::into(a));
}

#[test]
#[allow(clippy::op_ref)]
fn mont254_ref_arithmetic_variants() {
    let a = Mont254BitChallenge::<Fr>::from(10u128);
    let b = Mont254BitChallenge::<Fr>::from(20u128);
    let f = <Fr as Field>::from_u64(5);

    // ref-ref: Challenge & Challenge
    let _: Fr = &a + &b;
    let _: Fr = &a - &b;
    let _: Fr = &a * &b;

    // val-ref: Challenge & &Challenge
    let _: Fr = a + &b;
    let _: Fr = a - &b;
    let _: Fr = a * &b;

    // ref-val: &Challenge & Challenge
    let _: Fr = &a + b;
    let _: Fr = &a - b;
    let _: Fr = &a * b;

    // ref-ref: Challenge & Field
    let _: Fr = &a + &f;
    let _: Fr = &a - &f;
    let _: Fr = &a * &f;

    // ref-ref: Field & Challenge
    let _: Fr = &f + &a;
    let _: Fr = &f - &a;
    let _: Fr = &f * &a;

    // val-ref mixed
    let _: Fr = a + &f;
    let _: Fr = f + &a;
    let _: Fr = a * &f;
    let _: Fr = f * &a;
}

#[test]
fn mont254_optimized_mul() {
    let c = Mont254BitChallenge::<Fr>::from(42u128);
    let f: Fr = Field::random(&mut test_rng());

    assert!(c.mul_0_optimized(Fr::zero()).is_zero());
    assert_eq!(c.mul_1_optimized(Fr::one()), c.into());
    assert!(c.mul_01_optimized(Fr::zero()).is_zero());
    assert_eq!(c.mul_01_optimized(Fr::one()), c.into());
    // Non-trivial path
    assert_eq!(c.mul_0_optimized(f), Into::<Fr>::into(c) * f);
    assert_eq!(c.mul_1_optimized(f), Into::<Fr>::into(c) * f);
    assert_eq!(c.mul_01_optimized(f), Into::<Fr>::into(c) * f);
}

// =========================================================================
// 4. MontU128Challenge Display/Debug
// =========================================================================

#[test]
fn mont_u128_display() {
    let c = MontU128Challenge::<Fr>::new(0xDEAD_BEEF_CAFE_BABEu128);
    let s = format!("{c}");
    // Display goes through Limbs Display which shows hex
    assert!(!s.is_empty());
}

#[test]
fn mont_u128_debug() {
    let c = MontU128Challenge::<Fr>::new(0xDEAD_BEEF_CAFE_BABEu128);
    let s = format!("{c:?}");
    assert!(!s.is_empty());
}

#[test]
fn mont_u128_display_zero() {
    let c = MontU128Challenge::<Fr>::new(0u128);
    let s = format!("{c}");
    assert!(!s.is_empty());
}

#[test]
#[allow(clippy::op_ref)]
fn mont_u128_ref_arithmetic_variants() {
    let a = MontU128Challenge::<Fr>::new(10u128);
    let b = MontU128Challenge::<Fr>::new(20u128);
    let f = <Fr as Field>::from_u64(5);

    // ref-ref
    let _: Fr = &a + &b;
    let _: Fr = &a - &b;
    let _: Fr = &a * &b;

    // val-ref
    let _: Fr = a + &b;
    let _: Fr = a - &b;
    let _: Fr = a * &b;

    // ref-val
    let _: Fr = &a + b;
    let _: Fr = &a - b;
    let _: Fr = &a * b;

    // Mixed ref variants
    let _: Fr = &a + &f;
    let _: Fr = &a - &f;
    let _: Fr = &a * &f;
    let _: Fr = &f + &a;
    let _: Fr = &f - &a;
    let _: Fr = &f * &a;
    let _: Fr = a + &f;
    let _: Fr = f + &a;
    let _: Fr = a * &f;
    let _: Fr = f * &a;
}

// =========================================================================
// 5. OptimizedMul<Fr, Fr> blanket impl
// =========================================================================

#[test]
fn optimized_mul_blanket_impl() {
    let mut rng = test_rng();
    let a: Fr = Field::random(&mut rng);
    let b: Fr = Field::random(&mut rng);

    // mul_0_optimized: both nonzero
    assert_eq!(a.mul_0_optimized(b), a * b);

    // mul_0_optimized: first is zero
    assert!(Fr::zero().mul_0_optimized(b).is_zero());

    // mul_0_optimized: second is zero
    assert!(a.mul_0_optimized(Fr::zero()).is_zero());

    // mul_1_optimized: first is one
    assert_eq!(Fr::one().mul_1_optimized(b), b);

    // mul_1_optimized: second is one
    assert_eq!(a.mul_1_optimized(Fr::one()), a);

    // mul_1_optimized: neither is one
    assert_eq!(a.mul_1_optimized(b), a * b);

    // mul_01_optimized: zero path
    assert!(Fr::zero().mul_01_optimized(b).is_zero());
    assert!(a.mul_01_optimized(Fr::zero()).is_zero());

    // mul_01_optimized: one path
    assert_eq!(Fr::one().mul_01_optimized(b), b);
    assert_eq!(a.mul_01_optimized(Fr::one()), a);

    // mul_01_optimized: general path
    assert_eq!(a.mul_01_optimized(b), a * b);
}

// =========================================================================
// 6. Field default methods (from_u8, from_u16, from_u32, from_bool, mul_*)
//    Already covered in field_operations.rs, but adding edge cases
// =========================================================================

#[test]
fn field_from_bool_edge() {
    assert_eq!(<Fr as Field>::from_bool(true), Fr::one());
    assert_eq!(<Fr as Field>::from_bool(false), Fr::zero());
}

#[test]
fn field_from_small_types_boundary() {
    assert_eq!(<Fr as Field>::from_u8(0), Fr::zero());
    assert_eq!(<Fr as Field>::from_u8(255), <Fr as Field>::from_u64(255));
    assert_eq!(<Fr as Field>::from_u16(0), Fr::zero());
    assert_eq!(
        <Fr as Field>::from_u16(65535),
        <Fr as Field>::from_u64(65535)
    );
    assert_eq!(<Fr as Field>::from_u32(0), Fr::zero());
    assert_eq!(
        <Fr as Field>::from_u32(u32::MAX),
        <Fr as Field>::from_u64(u32::MAX as u64)
    );
}

#[test]
fn field_mul_pow_2_boundary() {
    let f = <Fr as Field>::from_u64(1);
    // pow=0 -> f * 1 = f
    assert_eq!(<Fr as Field>::mul_pow_2(&f, 0), f);
    // pow=1 -> f * 2
    assert_eq!(
        <Fr as Field>::mul_pow_2(&f, 1),
        <Fr as Field>::from_u64(2)
    );
    // pow=64 -> goes through while loop at least once
    let result = <Fr as Field>::mul_pow_2(&f, 64);
    let mut expected = f;
    for _ in 0..64 {
        expected = expected + expected;
    }
    assert_eq!(result, expected);
}

#[test]
#[should_panic(expected = "pow > 255")]
fn field_mul_pow_2_overflow() {
    let f = <Fr as Field>::from_u64(1);
    let _ = <Fr as Field>::mul_pow_2(&f, 256);
}

// =========================================================================
// 7. SignedBigInt — uncovered paths
// =========================================================================

#[test]
fn signed_bigint_neg() {
    let a = S64::from_i64(42);
    let b = -a;
    assert!(!b.is_positive);
    assert_eq!(b.magnitude_as_u64(), 42);

    let c = -b;
    assert!(c.is_positive);
}

#[test]
fn signed_bigint_from_u128() {
    let v = 0xDEAD_BEEF_CAFE_BABEu128;
    let s = S128::from_u128(v);
    assert!(s.is_positive);
    assert_eq!(s.magnitude_as_u128(), v);
}

#[test]
fn signed_bigint_from_i128_positive() {
    let v = 123_456_789_012_345_678i128;
    let s = S128::from_i128(v);
    assert!(s.is_positive);
    assert_eq!(s.to_i128(), Some(v));
}

#[test]
fn signed_bigint_from_i128_negative() {
    let v = -123_456_789_012_345_678i128;
    let s = S128::from_i128(v);
    assert!(!s.is_positive);
    assert_eq!(s.to_i128(), Some(v));
}

#[test]
fn signed_bigint_from_u128_trait() {
    let v = 42u128;
    let s: S128 = v.into();
    assert!(s.is_positive);
    assert_eq!(s.magnitude_as_u128(), 42);
}

#[test]
fn signed_bigint_from_i128_trait() {
    let s: S128 = (-99i128).into();
    assert!(!s.is_positive);
    assert_eq!(s.to_i128(), Some(-99));
}

#[test]
fn signed_bigint_sub_trunc() {
    // Same sign, |self| > |rhs|
    let a = S128::from_i128(100);
    let b = S128::from_i128(30);
    let c: S128 = a.sub_trunc::<2>(&b);
    assert_eq!(c.to_i128(), Some(70));

    // Same sign, |self| < |rhs| => sign flips
    let d: S128 = b.sub_trunc::<2>(&a);
    assert_eq!(d.to_i128(), Some(-70));

    // Different signs: positive - negative = add magnitudes
    let e = S128::from_i128(50);
    let f = S128::from_i128(-30);
    let g: S128 = e.sub_trunc::<2>(&f);
    assert_eq!(g.to_i128(), Some(80));
}

#[test]
fn signed_bigint_sub_trunc_mixed() {
    // Same sign, |self| > |rhs|
    let a = S128::from_i128(100);
    let b = S64::from_i64(30);
    let c: S128 = a.sub_trunc_mixed::<1, 2>(&b);
    assert_eq!(c.to_i128(), Some(70));

    // Same sign, |self| < |rhs|
    let d = S64::from_i64(30);
    let e = S128::from_i128(100);
    let f: S128 = d.sub_trunc_mixed::<2, 2>(&e);
    assert_eq!(f.to_i128(), Some(-70));

    // Different signs
    let g = S128::from_i128(50);
    let h = S64::from_i64(-20);
    let i: S128 = g.sub_trunc_mixed::<1, 2>(&h);
    assert_eq!(i.to_i128(), Some(70));
}

#[test]
fn signed_bigint_mul_trunc_widths() {
    // S64 * S128 -> S128
    let a = S64::from_i64(-7);
    let b = S128::from_i128(11);
    let c: S128 = a.mul_trunc::<2, 2>(&b);
    assert_eq!(c.to_i128(), Some(-77));

    // S128 * S128 -> S256
    let d = S128::from_i128(1_000_000);
    let e = S128::from_i128(-2_000_000);
    let f: S256 = d.mul_trunc::<2, 4>(&e);
    assert!(!f.is_positive);
}

#[test]
fn signed_bigint_s256_serialization() {
    let val = S256::new([1, 2, 3, 4], false);
    let mut bytes = Vec::new();
    val.serialize_compressed(&mut bytes).unwrap();
    let restored = S256::deserialize_compressed(&bytes[..]).unwrap();
    assert_eq!(val, restored);
}

#[test]
fn signed_bigint_s192_serialization() {
    let val = S192::new([u64::MAX, 0, 42], true);
    let mut bytes = Vec::new();
    val.serialize_compressed(&mut bytes).unwrap();
    let restored = S192::deserialize_compressed(&bytes[..]).unwrap();
    assert_eq!(val, restored);
}

#[test]
fn signed_bigint_from_u64_mul_i64() {
    let r = S128::from_u64_mul_i64(100, -7);
    assert_eq!(r.to_i128(), Some(-700));

    let r2 = S128::from_u64_mul_i64(100, 7);
    assert_eq!(r2.to_i128(), Some(700));
}

#[test]
fn signed_bigint_from_i64_mul_u64() {
    let r = S128::from_i64_mul_u64(-3, 100);
    assert_eq!(r.to_i128(), Some(-300));
}

#[test]
fn signed_bigint_ordering_negative_magnitudes() {
    // Both negative: larger magnitude = smaller value
    let a = S64::from_i64(-10);
    let b = S64::from_i64(-5);
    assert!(a < b);

    // Both positive: larger magnitude = larger value
    let c = S64::from_i64(10);
    let d = S64::from_i64(5);
    assert!(c > d);
}

// =========================================================================
// 8. SignedBigIntHi32 — uncovered paths
// =========================================================================

#[test]
fn s96_arithmetic() {
    let a = S96::from(10i64);
    let b = S96::from(3i64);

    let sum = a + b;
    assert!(sum.is_positive());
    assert_eq!(sum.magnitude_lo()[0], 13);

    let diff = a - b;
    assert!(diff.is_positive());
    assert_eq!(diff.magnitude_lo()[0], 7);

    let prod = a * b;
    assert!(prod.is_positive());
    assert_eq!(prod.magnitude_lo()[0], 30);
}

#[test]
fn s96_from_negative() {
    let a = S96::from(-5i64);
    assert!(!a.is_positive());
    assert_eq!(a.magnitude_lo()[0], 5);
}

#[test]
fn s96_from_s64() {
    let s = S64::from_i64(-42);
    let wide = S96::from(s);
    assert!(!wide.is_positive());
    assert_eq!(wide.magnitude_lo()[0], 42);
}

#[test]
fn s224_operations() {
    let a = S224::new([1, 0, 0], 0, true);
    let b = S224::new([2, 0, 0], 0, true);
    let sum = a + b;
    assert!(sum.is_positive());
    assert_eq!(sum.magnitude_lo()[0], 3);

    let diff = a - b;
    assert!(!diff.is_positive());
    assert_eq!(diff.magnitude_lo()[0], 1);

    let prod = a * b;
    assert!(prod.is_positive());
    assert_eq!(prod.magnitude_lo()[0], 2);
}

#[test]
fn s224_to_limbs4() {
    let v = S224::new([0xAAAA, 0xBBBB, 0xCCCC], 0xDD, true);
    let limbs: Limbs<4> = v.into();
    assert_eq!(limbs.0[0], 0xAAAA);
    assert_eq!(limbs.0[1], 0xBBBB);
    assert_eq!(limbs.0[2], 0xCCCC);
    assert_eq!(limbs.0[3], 0xDD);
}

#[test]
fn magnitude_as_limbs_nplus1_s96() {
    let v = S96::new([42], 7, true);
    let limbs: Limbs<2> = v.magnitude_as_limbs_nplus1::<2>();
    assert_eq!(limbs.0[0], 42);
    assert_eq!(limbs.0[1], 7);
}

#[test]
fn magnitude_as_limbs_nplus1_s160() {
    let v = S160::new([1, 2], 3, false);
    let limbs: Limbs<3> = v.magnitude_as_limbs_nplus1::<3>();
    assert_eq!(limbs.0[0], 1);
    assert_eq!(limbs.0[1], 2);
    assert_eq!(limbs.0[2], 3);
}

#[test]
fn magnitude_as_limbs_nplus1_s224() {
    let v = S224::new([10, 20, 30], 40, true);
    let limbs: Limbs<4> = v.magnitude_as_limbs_nplus1::<4>();
    assert_eq!(limbs.0[0], 10);
    assert_eq!(limbs.0[1], 20);
    assert_eq!(limbs.0[2], 30);
    assert_eq!(limbs.0[3], 40);
}

#[test]
fn zero_extend_from_s96_to_s160() {
    let s = S96::new([42], 7, false);
    let wide: S160 = SignedBigIntHi32::zero_extend_from(&s);
    assert!(!wide.is_positive());
    // When N > M, hi32 is placed into limb M as u64, new hi32 = 0
    assert_eq!(wide.magnitude_lo()[0], 42);
    assert_eq!(wide.magnitude_lo()[1], 7);
    assert_eq!(wide.magnitude_hi(), 0);
}

#[test]
fn zero_extend_from_s96_to_s96() {
    // N == M case
    let s = S96::new([42], 7, true);
    let same: S96 = SignedBigIntHi32::zero_extend_from(&s);
    assert_eq!(same.magnitude_lo()[0], 42);
    assert_eq!(same.magnitude_hi(), 7);
    assert!(same.is_positive());
}

#[test]
fn s160_ordering() {
    let a = S160::from(100u64);
    let b = S160::from(200u64);
    assert!(a < b);

    let c = S160::new([0, 0], 0, true); // positive zero
    let d = S160::new([0, 0], 0, false); // negative zero
    assert_eq!(c.cmp(&d), std::cmp::Ordering::Equal);

    // Positive > Negative
    let pos = S160::from(1u64);
    let neg = S160::new([1, 0], 0, false);
    assert!(pos > neg);
}

#[test]
fn s160_ordering_negative_magnitudes() {
    // Both negative: larger magnitude = smaller value
    let a = S160::new([10, 0], 0, false);
    let b = S160::new([5, 0], 0, false);
    assert!(a < b);
}

#[test]
fn s160_ordering_hi32_tiebreak() {
    let a = S160::new([0, 0], 1, true);
    let b = S160::new([0, 0], 2, true);
    assert!(a < b);
}

#[test]
fn s160_from_sum_u128() {
    let a = u128::MAX / 2;
    let b = u128::MAX / 2;
    let s = S160::from_sum_u128(a, b);
    assert!(s.is_positive());
    // No overflow into hi32 for this case
    assert_eq!(s.magnitude_hi(), 0);

    // Force carry into hi32
    let s2 = S160::from_sum_u128(u128::MAX, 1);
    assert!(s2.is_positive());
    assert_eq!(s2.magnitude_hi(), 1);
    assert_eq!(s2.magnitude_lo()[0], 0);
    assert_eq!(s2.magnitude_lo()[1], 0);
}

#[test]
fn s160_from_diff_u128() {
    let a = S160::from_diff_u128(100, 200);
    assert!(!a.is_positive());
    assert_eq!(a.magnitude_lo()[0], 100);

    let b = S160::from_diff_u128(200, 100);
    assert!(b.is_positive());
    assert_eq!(b.magnitude_lo()[0], 100);
}

#[test]
fn s160_from_magnitude_u128() {
    let s = S160::from_magnitude_u128(0xDEAD_BEEF_CAFE_BABEu128, false);
    assert!(!s.is_positive());
    assert_eq!(s.magnitude_lo()[0], 0xDEAD_BEEF_CAFE_BABEu128 as u64);
    assert_eq!(
        s.magnitude_lo()[1],
        (0xDEAD_BEEF_CAFE_BABEu128 >> 64) as u64
    );
}

#[test]
fn s160_from_u128_minus_i128_negative_i() {
    // u - (-i) = u + |i| (sum path)
    let v = S160::from_u128_minus_i128(100, -50);
    assert!(v.is_positive());
    assert_eq!(v.magnitude_lo()[0], 150);
}

#[test]
fn s160_from_u128_minus_i128_positive_i_larger() {
    // u - i where i > u (diff path, negative result)
    let v = S160::from_u128_minus_i128(10, 100);
    assert!(!v.is_positive());
    assert_eq!(v.magnitude_lo()[0], 90);
}

#[test]
fn s160_serialization_roundtrip() {
    let val = S160::new([u64::MAX, 123_456], 0xABCD, false);
    let mut bytes = Vec::new();
    val.serialize_compressed(&mut bytes).unwrap();
    let restored = S160::deserialize_compressed(&bytes[..]).unwrap();
    assert_eq!(val, restored);
}

#[test]
fn s96_serialization_roundtrip() {
    let val = S96::new([42], 7, true);
    let mut bytes = Vec::new();
    val.serialize_compressed(&mut bytes).unwrap();
    let restored = S96::deserialize_compressed(&bytes[..]).unwrap();
    assert_eq!(val, restored);
}

#[test]
fn s224_serialization_roundtrip() {
    let val = S224::new([1, 2, 3], 4, false);
    let mut bytes = Vec::new();
    val.serialize_compressed(&mut bytes).unwrap();
    let restored = S224::deserialize_compressed(&bytes[..]).unwrap();
    assert_eq!(val, restored);
}

#[test]
fn signed_bigint_hi32_neg() {
    let a = S160::from(42u64);
    let b = -a;
    assert!(!b.is_positive());
    assert_eq!(b.magnitude_lo()[0], 42);

    // Neg for &SignedBigIntHi32
    let c = -(&a);
    assert!(!c.is_positive());
    assert_eq!(c.magnitude_lo()[0], 42);
}

#[test]
fn signed_bigint_hi32_one() {
    let one = S96::one();
    assert!(one.is_positive());
    assert_eq!(one.magnitude_lo()[0], 1);
    assert_eq!(one.magnitude_hi(), 0);
}

#[test]
fn signed_bigint_hi32_is_zero() {
    let z = S160::zero();
    assert!(z.is_zero());

    let nz = S160::from(1u64);
    assert!(!nz.is_zero());
}

#[test]
fn s160_from_i128() {
    let pos: S160 = 42i128.into();
    assert!(pos.is_positive());
    assert_eq!(pos.magnitude_lo()[0], 42);

    let neg: S160 = (-42i128).into();
    assert!(!neg.is_positive());
    assert_eq!(neg.magnitude_lo()[0], 42);
}

#[test]
fn s160_from_u128() {
    let v: S160 = 0xDEAD_BEEF_CAFE_BABEu128.into();
    assert!(v.is_positive());
    assert_eq!(v.magnitude_lo()[0], 0xDEAD_BEEF_CAFE_BABEu128 as u64);
}

#[test]
fn s160_from_s128() {
    let s = S128::from_i128(-999);
    let wide = S160::from(s);
    assert!(!wide.is_positive());
    assert_eq!(wide.magnitude_lo()[0], 999);
}

// =========================================================================
// 9. Macro-generated operator variants (signed/mod.rs)
// =========================================================================

#[test]
#[allow(clippy::op_ref)]
fn signed_bigint_operator_variants() {
    let a = S64::from_i64(10);
    let b = S64::from_i64(3);

    // val-val
    let _ = a + b;
    let _ = a - b;
    let _ = a * b;

    // val-ref
    let _ = a + &b;
    let _ = a - &b;
    let _ = a * &b;

    // ref-ref
    let _ = &a + &b;
    let _ = &a - &b;
    let _ = &a * &b;

    // OpAssign-val
    let mut c = a;
    c += b;
    assert_eq!(c, S64::from_i64(13));
    c -= b;
    assert_eq!(c, S64::from_i64(10));
    c *= b;
    assert_eq!(c, S64::from_i64(30));

    // OpAssign-ref
    let mut d = a;
    d += &b;
    assert_eq!(d, S64::from_i64(13));
    d -= &b;
    assert_eq!(d, S64::from_i64(10));
    d *= &b;
    assert_eq!(d, S64::from_i64(30));
}

#[test]
#[allow(clippy::op_ref)]
fn signed_bigint_hi32_operator_variants() {
    let a = S160::from(10u64);
    let b = S160::from(3u64);

    // val-val
    let _ = a + b;
    let _ = a - b;
    let _ = a * b;

    // val-ref
    let _ = a + &b;
    let _ = a - &b;
    let _ = a * &b;

    // ref-ref
    let _ = &a + &b;
    let _ = &a - &b;
    let _ = &a * &b;

    // OpAssign-val
    let mut c = a;
    c += b;
    assert!(c.is_positive());
    assert_eq!(c.magnitude_lo()[0], 13);
    c -= b;
    assert_eq!(c.magnitude_lo()[0], 10);
    c *= b;
    assert_eq!(c.magnitude_lo()[0], 30);

    // OpAssign-ref
    let mut d = a;
    d += &b;
    assert_eq!(d.magnitude_lo()[0], 13);
    d -= &b;
    assert_eq!(d.magnitude_lo()[0], 10);
    d *= &b;
    assert_eq!(d.magnitude_lo()[0], 30);
}

// =========================================================================
// 10. SignedBigIntHi32 mul_magnitudes specializations
// =========================================================================

#[test]
fn s96_mul_magnitudes_n1() {
    // N=1 specialization: single lo limb + hi32
    let a = S96::new([u64::MAX], 0, true);
    let b = S96::new([2], 0, true);
    let prod = a * b;
    // u64::MAX * 2 = 0x1_FFFF_FFFE, truncated to 96 bits
    assert!(prod.is_positive());
}

#[test]
fn s160_mul_magnitudes_n2() {
    // N=2 specialization
    let a = S160::new([3, 0], 0, true);
    let b = S160::new([7, 0], 0, true);
    let prod = a * b;
    assert_eq!(prod.magnitude_lo()[0], 21);
    assert!(prod.is_positive());
}

#[test]
fn s224_mul_magnitudes_n3_general() {
    // N=3: general path (N >= 3)
    let a = S224::new([2, 0, 0], 0, true);
    let b = S224::new([3, 0, 0], 0, true);
    let prod = a * b;
    assert_eq!(prod.magnitude_lo()[0], 6);
    assert!(prod.is_positive());
}

#[test]
fn s160_sub_smaller_from_larger() {
    // Tests the sign-flip path in sub_assign_in_place (via add_assign_in_place with neg)
    let a = S160::from(3u64);
    let b = S160::from(10u64);
    let c = a - b;
    assert!(!c.is_positive());
    assert_eq!(c.magnitude_lo()[0], 7);
}

#[test]
fn s160_add_opposite_signs_self_larger() {
    let a = S160::new([10, 0], 0, true);
    let b = S160::new([3, 0], 0, false);
    let c = a + b;
    assert!(c.is_positive());
    assert_eq!(c.magnitude_lo()[0], 7);
}

#[test]
fn s160_add_opposite_signs_rhs_larger() {
    let a = S160::new([3, 0], 0, true);
    let b = S160::new([10, 0], 0, false);
    let c = a + b;
    assert!(!c.is_positive());
    assert_eq!(c.magnitude_lo()[0], 7);
}

#[test]
fn s160_mul_mixed_signs() {
    let a = S160::new([5, 0], 0, true);
    let b = S160::new([3, 0], 0, false);
    let prod = a * b;
    assert!(!prod.is_positive());
    assert_eq!(prod.magnitude_lo()[0], 15);

    let prod2 = b * b;
    assert!(prod2.is_positive());
    assert_eq!(prod2.magnitude_lo()[0], 9);
}

// =========================================================================
// 11. S160 from conversions (i64, u64, S64, S128) — ensure full coverage
// =========================================================================

#[test]
fn s160_from_i64() {
    let v: S160 = (-7i64).into();
    assert!(!v.is_positive());
    assert_eq!(v.magnitude_lo()[0], 7);
    assert_eq!(v.magnitude_lo()[1], 0);
    assert_eq!(v.magnitude_hi(), 0);
}

#[test]
fn s160_from_u64() {
    let v: S160 = 42u64.into();
    assert!(v.is_positive());
    assert_eq!(v.magnitude_lo()[0], 42);
}

#[test]
fn s160_from_s64() {
    let s = S64::from_i64(-99);
    let v = S160::from(s);
    assert!(!v.is_positive());
    assert_eq!(v.magnitude_lo()[0], 99);
}

// =========================================================================
// 12. SignedBigInt add/sub with opposite signs exercising all branches
// =========================================================================

#[test]
fn signed_bigint_add_opposite_signs_self_smaller() {
    let a = S64::from_i64(3);
    let b = S64::from_i64(-10);
    let c = a + b;
    assert!(!c.is_positive);
    assert_eq!(c.magnitude_as_u64(), 7);
}

#[test]
fn signed_bigint_sub_opposite_signs() {
    // positive - negative = add magnitudes
    let a = S64::from_i64(5);
    let b = S64::from_i64(-3);
    let c = a - b;
    assert!(c.is_positive);
    assert_eq!(c.magnitude_as_u64(), 8);
}

#[test]
fn signed_bigint_sub_same_sign_smaller_magnitude() {
    // Same sign, |self| < |rhs| => sign flips
    let a = S64::from_i64(3);
    let b = S64::from_i64(10);
    let c = a - b;
    assert!(!c.is_positive);
    assert_eq!(c.magnitude_as_u64(), 7);
}

// =========================================================================
// 13. SignedBigInt add_trunc_mixed — more branch coverage
// =========================================================================

#[test]
fn signed_bigint_add_trunc_mixed_opposite_signs_self_smaller() {
    let a = S64::from_i64(3);
    let b = S128::from_i128(-100);
    let c: S128 = a.add_trunc_mixed::<2, 2>(&b);
    assert_eq!(c.to_i128(), Some(-97));
}

#[test]
fn signed_bigint_add_trunc_mixed_opposite_signs_self_larger() {
    let a = S128::from_i128(100);
    let b = S64::from_i64(-3);
    let c: S128 = a.add_trunc_mixed::<1, 2>(&b);
    assert_eq!(c.to_i128(), Some(97));
}

// =========================================================================
// 14. SignedBigIntHi32 add with carry into hi32
// =========================================================================

#[test]
fn s96_add_with_carry_into_hi32() {
    let a = S96::new([u64::MAX], 0, true);
    let b = S96::new([1], 0, true);
    let c = a + b;
    assert!(c.is_positive());
    assert_eq!(c.magnitude_lo()[0], 0);
    assert_eq!(c.magnitude_hi(), 1);
}

#[test]
fn s96_sub_with_borrow() {
    let a = S96::new([0], 1, true);
    let b = S96::new([1], 0, true);
    let c = a - b;
    assert!(c.is_positive());
    assert_eq!(c.magnitude_lo()[0], u64::MAX);
    assert_eq!(c.magnitude_hi(), 0);
}

#[test]
fn signed_bigint_hi32_default() {
    let d = S160::default();
    assert!(d.is_zero());
    assert!(d.is_positive());
}

// =========================================================================
// 15. SignedBigInt zero_extend_from to wider
// =========================================================================

#[test]
fn signed_bigint_zero_extend_s64_to_s256() {
    let s = S64::from_i64(-7);
    let wide: S256 = SignedBigInt::zero_extend_from(&s);
    assert!(!wide.is_positive);
    assert_eq!(wide.magnitude.0[0], 7);
    assert_eq!(wide.magnitude.0[1], 0);
    assert_eq!(wide.magnitude.0[2], 0);
    assert_eq!(wide.magnitude.0[3], 0);
}

// =========================================================================
// 16. SignedBigInt default, accessors
// =========================================================================

#[test]
fn signed_bigint_default_is_zero() {
    let d = S64::default();
    assert!(d.is_zero());
}

#[test]
fn signed_bigint_one() {
    let o = S64::one();
    assert!(o.is_positive);
    assert_eq!(o.magnitude_as_u64(), 1);
}

#[test]
fn signed_bigint_accessors() {
    let s = S128::from_i128(-42);
    assert!(!s.sign());
    assert_eq!(s.magnitude_slice(), &[42, 0]);
    assert_eq!(s.magnitude_limbs(), [42, 0]);
    let _ = s.as_magnitude();
}

#[test]
fn signed_bigint_negate() {
    let s = S64::from_i64(10);
    let n = s.negate();
    assert!(!n.is_positive);
    assert_eq!(n.magnitude_as_u64(), 10);
}
