//! Algebraic group law tests for BN254 G1 and G2.

use jolt_crypto::{Bn254, Bn254G1, Bn254G2, JoltGroup, PairingGroup};
use jolt_field::{Field, Fr};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

fn random_g1(rng: &mut ChaCha20Rng) -> Bn254G1 {
    Bn254::random_g1(rng)
}

#[test]
fn g1_identity() {
    let mut rng = ChaCha20Rng::seed_from_u64(0);
    let a = random_g1(&mut rng);
    assert_eq!(a + Bn254G1::zero(), a);
    assert_eq!(Bn254G1::zero() + a, a);
}

#[test]
fn g1_inverse() {
    let mut rng = ChaCha20Rng::seed_from_u64(1);
    let a = random_g1(&mut rng);
    assert_eq!(a + (-a), Bn254G1::zero());
    assert!(Bn254G1::zero().is_zero());
}

#[test]
fn g1_associativity() {
    let mut rng = ChaCha20Rng::seed_from_u64(2);
    let a = random_g1(&mut rng);
    let b = random_g1(&mut rng);
    let c = random_g1(&mut rng);
    assert_eq!((a + b) + c, a + (b + c));
}

#[test]
fn g1_double_equals_add_self() {
    let mut rng = ChaCha20Rng::seed_from_u64(3);
    let a = random_g1(&mut rng);
    assert_eq!(a.double(), a + a);
}

#[test]
fn g1_scalar_mul_two_equals_double() {
    let mut rng = ChaCha20Rng::seed_from_u64(4);
    let a = random_g1(&mut rng);
    let two = Fr::from_u64(2);
    assert_eq!(a.scalar_mul(&two), a.double());
}

#[test]
fn g1_scalar_mul_zero_is_identity() {
    let mut rng = ChaCha20Rng::seed_from_u64(5);
    let a = random_g1(&mut rng);
    let zero = Fr::from_u64(0);
    assert!(a.scalar_mul(&zero).is_zero());
}

#[test]
fn g1_scalar_mul_one_is_identity() {
    let mut rng = ChaCha20Rng::seed_from_u64(6);
    let a = random_g1(&mut rng);
    let one = Fr::from_u64(1);
    assert_eq!(a.scalar_mul(&one), a);
}

#[test]
fn g1_msm_matches_naive() {
    let mut rng = ChaCha20Rng::seed_from_u64(7);
    let g1 = random_g1(&mut rng);
    let g2 = random_g1(&mut rng);
    let s1 = Fr::random(&mut rng);
    let s2 = Fr::random(&mut rng);

    let msm_result = Bn254G1::msm(&[g1, g2], &[s1, s2]);
    let naive = g1.scalar_mul(&s1) + g2.scalar_mul(&s2);
    assert_eq!(msm_result, naive);
}

#[test]
fn g1_sub_and_sub_assign() {
    let mut rng = ChaCha20Rng::seed_from_u64(8);
    let a = random_g1(&mut rng);
    let b = random_g1(&mut rng);

    let diff = a - b;
    assert_eq!(diff + b, a);

    let mut c = a;
    c -= b;
    assert_eq!(c, diff);
}

#[test]
fn g1_add_assign() {
    let mut rng = ChaCha20Rng::seed_from_u64(9);
    let a = random_g1(&mut rng);
    let b = random_g1(&mut rng);

    let mut c = a;
    c += b;
    assert_eq!(c, a + b);
}

#[test]
fn g2_identity_and_inverse() {
    let g = Bn254::g2_generator();
    assert_eq!(g + Bn254G2::zero(), g);
    assert_eq!(g + (-g), Bn254G2::zero());
}

#[test]
fn g2_double_equals_add_self() {
    let g = Bn254::g2_generator();
    assert_eq!(g.double(), g + g);
}

#[test]
fn g2_scalar_mul_and_msm() {
    let mut rng = ChaCha20Rng::seed_from_u64(10);
    let g = Bn254::g2_generator();
    let s1 = Fr::random(&mut rng);
    let s2 = Fr::random(&mut rng);

    let g2 = g.scalar_mul(&s1);
    let msm_result = Bn254G2::msm(&[g, g2], &[s1, s2]);
    let naive = g.scalar_mul(&s1) + g2.scalar_mul(&s2);
    assert_eq!(msm_result, naive);
}
