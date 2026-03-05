//! Pairing bilinearity and consistency tests for BN254.

use jolt_crypto::{Bn254, Bn254G2, JoltGroup, PairingGroup};
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
fn pairing_with_identity_gives_one() {
    let g1 = Bn254::g1_generator();
    let g2_zero: Bn254G2 = JoltGroup::zero();

    let result = Bn254::pairing(&g1, &g2_zero);
    assert_eq!(result, Bn254::gt_one(), "e(G, O) should be 1_GT");
}

#[test]
fn multi_pairing_matches_product_of_individual() {
    // multi_pairing([(a,b), (c,d)]) == e(a,b) * e(c,d)
    let mut rng = ChaCha20Rng::seed_from_u64(99);
    let g1 = Bn254::g1_generator();
    let g2 = Bn254::g2_generator();

    let a = Fr::random(&mut rng);
    let b = Fr::random(&mut rng);

    let g1a = g1.scalar_mul(&a);
    let g1b = g1.scalar_mul(&b);

    let multi = Bn254::multi_pairing(&[g1a, g1b], &[g2, g2]);
    let product = Bn254::pairing(&g1a, &g2) * Bn254::pairing(&g1b, &g2);

    assert_eq!(
        multi, product,
        "multi_pairing should equal product of individual pairings"
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
    assert!(!g1.is_zero());
    assert!(!g2.is_zero());
}

#[test]
fn gt_mul_identity() {
    let g1 = Bn254::g1_generator();
    let g2 = Bn254::g2_generator();
    let e = Bn254::pairing(&g1, &g2);
    let one = Bn254::gt_one();

    assert_eq!(e * one, e, "e * 1 should be e");
}

#[test]
fn gt_mul_assign() {
    let g1 = Bn254::g1_generator();
    let g2 = Bn254::g2_generator();
    let e = Bn254::pairing(&g1, &g2);

    let mut acc = Bn254::gt_one();
    acc *= e;
    assert_eq!(acc, e);
    acc *= e;
    assert_eq!(acc, e * e);
}
