//! Pairing bilinearity and consistency tests for BN254.

use jolt_crypto::{Bn254, Bn254G2, Bn254GT, JoltGroup, PairingGroup};
use jolt_field::{Field, Fr};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

#[test]
fn pairing_bilinearity() {
    // e(aG, bH) == e(abG, H) == e(G, abH)
    let mut rng = ChaCha20Rng::seed_from_u64(42);
    let a = Fr::random(&mut rng);
    let b = Fr::random(&mut rng);

    let g1 = Bn254::g1_generator();
    let g2 = Bn254::g2_generator();

    let ab = a * b;

    let lhs = Bn254::pairing(&g1.scalar_mul(&a), &g2.scalar_mul(&b));
    let rhs1 = Bn254::pairing(&g1.scalar_mul(&ab), &g2);
    let rhs2 = Bn254::pairing(&g1, &g2.scalar_mul(&ab));

    assert_eq!(lhs, rhs1, "e(aG, bH) != e(abG, H)");
    assert_eq!(lhs, rhs2, "e(aG, bH) != e(G, abH)");
}

#[test]
fn pairing_with_identity_gives_gt_identity() {
    let g1 = Bn254::g1_generator();
    let g2_id = Bn254G2::identity();

    let result = Bn254::pairing(&g1, &g2_id);
    assert_eq!(
        result,
        Bn254GT::identity(),
        "e(G, O) should be identity in GT"
    );
}

#[test]
fn multi_pairing_matches_sum_of_individual() {
    // multi_pairing([(a,b), (c,d)]) == e(a,b) + e(c,d)  (additive notation)
    let mut rng = ChaCha20Rng::seed_from_u64(99);
    let g1 = Bn254::g1_generator();
    let g2 = Bn254::g2_generator();

    let a = Fr::random(&mut rng);
    let b = Fr::random(&mut rng);

    let g1a = g1.scalar_mul(&a);
    let g1b = g1.scalar_mul(&b);

    let multi = Bn254::multi_pairing(&[g1a, g1b], &[g2, g2]);
    // In additive notation, "sum" is GT addition (which is Fq12 multiplication).
    let sum = Bn254::pairing(&g1a, &g2) + Bn254::pairing(&g1b, &g2);

    assert_eq!(
        multi, sum,
        "multi_pairing should equal sum of individual pairings"
    );
}

#[test]
fn single_multi_pairing_matches_pairing() {
    let g1 = Bn254::g1_generator();
    let g2 = Bn254::g2_generator();

    let single = Bn254::pairing(&g1, &g2);
    let multi = Bn254::multi_pairing(&[g1], &[g2]);
    assert_eq!(single, multi);
}

#[test]
fn generators_are_not_identity() {
    let g1 = Bn254::g1_generator();
    let g2 = Bn254::g2_generator();
    assert!(!g1.is_identity());
    assert!(!g2.is_identity());
}

#[test]
fn gt_add_identity() {
    let g1 = Bn254::g1_generator();
    let g2 = Bn254::g2_generator();
    let e = Bn254::pairing(&g1, &g2);
    let id = Bn254GT::identity();

    assert_eq!(e + id, e, "e + identity should be e");
    assert_eq!(id + e, e, "identity + e should be e");
}

#[test]
fn gt_add_assign() {
    let g1 = Bn254::g1_generator();
    let g2 = Bn254::g2_generator();
    let e = Bn254::pairing(&g1, &g2);

    let mut acc = Bn254GT::identity();
    acc += e;
    assert_eq!(acc, e);
    acc += e;
    assert_eq!(acc, e + e);
}

#[test]
fn gt_sub_is_inverse_of_add() {
    let g1 = Bn254::g1_generator();
    let g2 = Bn254::g2_generator();
    let e = Bn254::pairing(&g1, &g2);

    assert_eq!(e - e, Bn254GT::identity());
    assert_eq!((e + e) - e, e);
}

#[test]
fn gt_neg_is_additive_inverse() {
    let g1 = Bn254::g1_generator();
    let g2 = Bn254::g2_generator();
    let e = Bn254::pairing(&g1, &g2);

    assert_eq!(e + (-e), Bn254GT::identity());
}

#[test]
fn gt_double_equals_add_self() {
    let g1 = Bn254::g1_generator();
    let g2 = Bn254::g2_generator();
    let e = Bn254::pairing(&g1, &g2);

    assert_eq!(e.double(), e + e);
}

#[test]
fn gt_scalar_mul_two_equals_double() {
    let g1 = Bn254::g1_generator();
    let g2 = Bn254::g2_generator();
    let e = Bn254::pairing(&g1, &g2);
    let two = Fr::from_u64(2);

    assert_eq!(e.scalar_mul(&two), e.double());
}

#[test]
fn gt_mul_convenience_matches_add() {
    let g1 = Bn254::g1_generator();
    let g2 = Bn254::g2_generator();
    let e = Bn254::pairing(&g1, &g2);

    // Mul and Add should behave identically (both map to Fq12 multiplication).
    assert_eq!(e * e, e + e);
}

#[test]
fn gt_scalar_mul_zero_is_identity() {
    let g1 = Bn254::g1_generator();
    let g2 = Bn254::g2_generator();
    let e = Bn254::pairing(&g1, &g2);
    let zero = Fr::from_u64(0);

    assert!(e.scalar_mul(&zero).is_identity());
}

#[test]
fn gt_scalar_mul_one_is_noop() {
    let g1 = Bn254::g1_generator();
    let g2 = Bn254::g2_generator();
    let e = Bn254::pairing(&g1, &g2);
    let one = Fr::from_u64(1);

    assert_eq!(e.scalar_mul(&one), e);
}

#[test]
fn gt_msm_matches_naive() {
    let mut rng = ChaCha20Rng::seed_from_u64(55);
    let g1 = Bn254::g1_generator();
    let g2 = Bn254::g2_generator();

    let s1 = Fr::random(&mut rng);
    let s2 = Fr::random(&mut rng);

    let e1 = Bn254::pairing(&g1.scalar_mul(&s1), &g2);
    let e2 = Bn254::pairing(&g1.scalar_mul(&s2), &g2);

    let ms1 = Fr::random(&mut rng);
    let ms2 = Fr::random(&mut rng);

    let msm_result = Bn254GT::msm(&[e1, e2], &[ms1, ms2]);
    let naive = e1.scalar_mul(&ms1) + e2.scalar_mul(&ms2);
    assert_eq!(msm_result, naive);
}

#[test]
fn gt_default_is_identity() {
    assert_eq!(Bn254GT::default(), Bn254GT::identity());
}
