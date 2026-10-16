use jolt_crypto::{Bn254, Bn254G1, Bn254G2, JoltGroup};
use jolt_field::{Field, Fr};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

fn random_g1(rng: &mut ChaCha20Rng) -> Bn254G1 {
    Bn254::random_g1(rng)
}

#[test]
fn g1_inverse() {
    let mut rng = ChaCha20Rng::seed_from_u64(1);
    let a = random_g1(&mut rng);
    assert_eq!(a + (-a), Bn254G1::identity());
    assert!(Bn254G1::identity().is_identity());
}

#[test]
fn g1_double_equals_add_self() {
    let mut rng = ChaCha20Rng::seed_from_u64(3);
    let a = random_g1(&mut rng);
    assert_eq!(a.double(), a + a);
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
#[expect(clippy::op_ref)]
fn g1_add_ref() {
    let mut rng = ChaCha20Rng::seed_from_u64(12);
    let a = random_g1(&mut rng);
    let b = random_g1(&mut rng);
    let expected = a + b;
    assert_eq!(a + &b, expected);
}

#[test]
#[expect(clippy::op_ref)]
fn g1_sub_ref() {
    let mut rng = ChaCha20Rng::seed_from_u64(13);
    let a = random_g1(&mut rng);
    let b = random_g1(&mut rng);
    let expected = a - b;
    assert_eq!(a - &b, expected);
}

#[test]
fn g1_msm_empty() {
    let result = Bn254G1::msm(&[], &([] as [Fr; 0]));
    assert!(result.is_identity());
}

#[test]
fn g1_default_is_identity() {
    assert_eq!(Bn254G1::default(), Bn254G1::identity());
}

#[test]
#[should_panic(expected = "msm: bases/scalars length mismatch")]
fn g1_msm_length_mismatch_panics() {
    let mut rng = ChaCha20Rng::seed_from_u64(15);
    let g = random_g1(&mut rng);
    let s = Fr::random(&mut rng);
    let _ = Bn254G1::msm(&[g, g], &[s]);
}

#[test]
#[should_panic(expected = "msm: bases/scalars length mismatch")]
fn g2_msm_length_mismatch_panics() {
    let mut rng = ChaCha20Rng::seed_from_u64(16);
    let g = Bn254::g2_generator();
    let s = Fr::random(&mut rng);
    let _ = Bn254G2::msm(&[g], &[s, s]);
}
