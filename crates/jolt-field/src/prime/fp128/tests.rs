use super::*;
use crate::{FieldCore, PseudoMersenneField};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_core::RngCore;

type F = Prime128Offset275;

#[test]
fn to_limbs_roundtrip() {
    let mut rng = StdRng::seed_from_u64(0xdead_beef_cafe_1234);
    for _ in 0..1000 {
        let a: F = FieldCore::random(&mut rng);
        assert_eq!(Fp128(a.to_limbs()), a);
    }
}

#[test]
fn mul_wide_u64_matches_full_mul() {
    let mut rng = StdRng::seed_from_u64(0x1122_3344_5566_7788);
    for _ in 0..1000 {
        let a: F = FieldCore::random(&mut rng);
        let b = rng.next_u64();
        let expected = a * F::from_u64(b);
        let reduced = F::solinas_reduce(&a.mul_wide_u64(b));
        assert_eq!(reduced, expected);
    }
}

#[test]
fn mul_wide_matches_full_mul() {
    let mut rng = StdRng::seed_from_u64(0xaabb_ccdd_eeff_0011);
    for _ in 0..1000 {
        let a: F = FieldCore::random(&mut rng);
        let b: F = FieldCore::random(&mut rng);
        let expected = a * b;
        let reduced = F::solinas_reduce(&a.mul_wide(b));
        assert_eq!(reduced, expected);
    }
}

#[test]
fn mul_add_matches_mul_then_add() {
    let mut rng = StdRng::seed_from_u64(0x3141_5926_5358_9793);
    for _ in 0..1000 {
        let a: F = FieldCore::random(&mut rng);
        let b: F = FieldCore::random(&mut rng);
        let c: F = FieldCore::random(&mut rng);
        assert_eq!(a.mul_add(b, c), a * b + c);
    }

    let near = -F::one();
    assert_eq!(near.mul_add(near, near), near * near + near);
}

#[test]
fn mul_wide_u128_matches_full_mul() {
    let mut rng = StdRng::seed_from_u64(0x9988_7766_5544_3322);
    for _ in 0..1000 {
        let a: F = FieldCore::random(&mut rng);
        let b = rng.next_u64() as u128 | ((rng.next_u64() as u128) << 64);
        let expected = a * F::from_canonical_u128_reduced(b);
        let reduced = F::solinas_reduce(&a.mul_wide_u128(b));
        assert_eq!(reduced, expected);
    }
}

#[test]
fn mul_wide_limbs_roundtrips_through_reduction() {
    let mut rng = StdRng::seed_from_u64(0x1bad_f00d_0ddc_afe1);
    for _ in 0..1000 {
        let a: F = FieldCore::random(&mut rng);
        let b3 = [rng.next_u64(), rng.next_u64(), rng.next_u64()];
        let b4 = [
            rng.next_u64(),
            rng.next_u64(),
            rng.next_u64(),
            rng.next_u64(),
        ];

        let got3_full = a.mul_wide_limbs::<3, 5>(b3);
        let got3_trunc = a.mul_wide_limbs::<3, 4>(b3);
        assert_eq!(
            got3_trunc,
            [got3_full[0], got3_full[1], got3_full[2], got3_full[3]]
        );
        let exp3 = a * F::solinas_reduce(&b3);
        assert_eq!(F::solinas_reduce(&got3_full), exp3);

        let got4_full = a.mul_wide_limbs::<4, 6>(b4);
        let got4_trunc = a.mul_wide_limbs::<4, 4>(b4);
        assert_eq!(
            got4_trunc,
            [got4_full[0], got4_full[1], got4_full[2], got4_full[3]]
        );
        let exp4 = a * F::solinas_reduce(&b4);
        assert_eq!(F::solinas_reduce(&got4_full), exp4);
    }
}

#[test]
fn solinas_reduce_small_inputs() {
    assert_eq!(F::solinas_reduce(&[]), F::zero());
    assert_eq!(F::solinas_reduce(&[42]), F::from_u64(42));
    let one_shifted = F::from_canonical_u128_reduced(1u128 << 64);
    assert_eq!(F::solinas_reduce(&[0, 1]), one_shifted);
}

#[test]
fn solinas_reduce_4_limbs_max() {
    // 2^256 - 1 ≡ C² - 1 (mod P), since 2^128 ≡ C
    let c = F::from_canonical_u128_reduced(<F as PseudoMersenneField>::MODULUS_OFFSET);
    let expected = c * c - F::one();
    assert_eq!(F::solinas_reduce(&[u64::MAX; 4]), expected);
}

#[test]
fn solinas_reduce_9_limbs() {
    // 1 + 2^512 = 1 + (2^128)^4 ≡ 1 + C^4
    let c = F::from_canonical_u128_reduced(<F as PseudoMersenneField>::MODULUS_OFFSET);
    let expected = F::one() + c * c * c * c;
    assert_eq!(F::solinas_reduce(&[1, 0, 0, 0, 0, 0, 0, 0, 1]), expected);
}

#[test]
fn solinas_reduce_accumulated_products() {
    let mut rng = StdRng::seed_from_u64(0xfeed_face_0bad_c0de);
    let mut acc = [0u64; 5];
    let mut expected = F::zero();

    for _ in 0..200 {
        let a: F = FieldCore::random(&mut rng);
        let b = rng.next_u64();
        let wide = a.mul_wide_u64(b);

        let mut carry: u64 = 0;
        for j in 0..5 {
            let addend = if j < 3 { wide[j] } else { 0 };
            let sum = acc[j] as u128 + addend as u128 + carry as u128;
            acc[j] = sum as u64;
            carry = (sum >> 64) as u64;
        }
        assert_eq!(carry, 0);
        expected += a * F::from_u64(b);
    }

    assert_eq!(F::solinas_reduce(&acc), expected);
}

#[test]
fn solinas_reduce_cross_prime() {
    type G = Prime128Offset275;
    let c = G::from_canonical_u128_reduced(<G as PseudoMersenneField>::MODULUS_OFFSET);
    let expected = c * c - G::one();
    assert_eq!(G::solinas_reduce(&[u64::MAX; 4]), expected);
}

#[test]
fn from_i64_handles_min_without_overflow() {
    let x = F::from_i64(i64::MIN);
    let y = F::from_u64(i64::MIN.unsigned_abs());
    assert_eq!(x + y, F::zero());
}

#[test]
fn prime128_offset_a7f7_constants() {
    // p = 2^128 − 2^32 + 22537, so C = 2^32 − 22537 = 0xFFFFA7F7.
    assert_eq!(
        <Prime128OffsetA7F7 as PseudoMersenneField>::MODULUS_OFFSET,
        0xFFFFA7F7,
    );
    assert_eq!(Prime128OffsetA7F7::C, 0xFFFFA7F7);
    assert_eq!(Prime128OffsetA7F7::C_LO, 0xFFFFA7F7);
    // Round-trip through the field arithmetic: p ≡ 0 (mod p), so
    // Fp(2^128 − C) + Fp(C) = 0.
    let neg_c = -Prime128OffsetA7F7::from_canonical_u128_reduced(0xFFFFA7F7);
    assert_eq!(
        neg_c + Prime128OffsetA7F7::from_canonical_u128_reduced(0xFFFFA7F7),
        Prime128OffsetA7F7::zero()
    );
}

#[test]
fn prime128_offset_a7f7_mul_wide_matches_full_mul() {
    type G = Prime128OffsetA7F7;
    let mut rng = StdRng::seed_from_u64(0xa7f7_a7f7_a7f7_a7f7);
    for _ in 0..1000 {
        let a: G = FieldCore::random(&mut rng);
        let b: G = FieldCore::random(&mut rng);
        let expected = a * b;
        let reduced = G::solinas_reduce(&a.mul_wide(b));
        assert_eq!(reduced, expected);
    }
}
