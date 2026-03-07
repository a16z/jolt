//! Integration tests for HyperKZG commit → open → verify pipeline with BN254.

use jolt_crypto::Bn254;
use jolt_field::{Field, Fr};
use jolt_hyperkzg::{HyperKZGProverSetup, HyperKZGScheme, HyperKZGVerifierSetup};
use jolt_openings::{AdditivelyHomomorphic, CommitmentScheme};
use jolt_poly::Polynomial;
use jolt_transcript::{Blake2bTranscript, Transcript};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

type KzgPCS = HyperKZGScheme<Bn254>;

fn make_setup(max_degree: usize) -> (HyperKZGProverSetup<Bn254>, HyperKZGVerifierSetup<Bn254>) {
    let mut rng = ChaCha20Rng::seed_from_u64(0xdead_beef);
    let g1 = Bn254::g1_generator();
    let g2 = Bn254::g2_generator();
    let pk = KzgPCS::setup(&mut rng, max_degree, g1, g2);
    let vk = KzgPCS::verifier_setup(&pk);
    (pk, vk)
}

fn commit_open_verify(
    poly: &Polynomial<Fr>,
    point: &[Fr],
    pk: &HyperKZGProverSetup<Bn254>,
    vk: &HyperKZGVerifierSetup<Bn254>,
    label: &'static [u8],
) {
    let eval = poly.evaluate(point);
    let (commitment, _) = <KzgPCS as CommitmentScheme>::commit(poly.evaluations(), pk);

    let mut t_p = Blake2bTranscript::new(label);
    let proof =
        <KzgPCS as CommitmentScheme>::open(poly, point, eval, pk, None, &mut t_p);

    let mut t_v = Blake2bTranscript::new(label);
    <KzgPCS as CommitmentScheme>::verify(&commitment, point, eval, &proof, vk, &mut t_v)
        .expect("verification should succeed");
}

// ---------------------------------------------------------------------------
// Basic roundtrip for various polynomial sizes
// ---------------------------------------------------------------------------

#[test]
fn roundtrip_num_vars_1_to_8() {
    let mut rng = ChaCha20Rng::seed_from_u64(1000);
    for nv in 1..=8 {
        let (pk, vk) = make_setup(1 << nv);
        let poly = Polynomial::<Fr>::random(nv, &mut rng);
        let point: Vec<Fr> = (0..nv).map(|_| Fr::random(&mut rng)).collect();
        commit_open_verify(&poly, &point, &pk, &vk, b"kzg-sizes");
    }
}

// ---------------------------------------------------------------------------
// Edge cases
// ---------------------------------------------------------------------------

/// All-zero polynomial commits to identity point and still verifies.
#[test]
fn zero_polynomial_roundtrip() {
    let nv = 3;
    let (pk, vk) = make_setup(1 << nv);
    let poly = Polynomial::<Fr>::zeros(nv);
    let point = vec![Fr::from_u64(42); nv];
    commit_open_verify(&poly, &point, &pk, &vk, b"kzg-zero");
}

/// Single-variable polynomial (2 evaluations).
#[test]
fn single_variable_polynomial() {
    let mut rng = ChaCha20Rng::seed_from_u64(2000);
    let (pk, vk) = make_setup(2);
    let poly = Polynomial::<Fr>::random(1, &mut rng);
    let point = vec![Fr::random(&mut rng)];
    commit_open_verify(&poly, &point, &pk, &vk, b"kzg-single-var");
}

/// Constant polynomial (all evaluations are the same value).
#[test]
fn constant_polynomial() {
    let nv = 3;
    let (pk, vk) = make_setup(1 << nv);
    let val = Fr::from_u64(42);
    let poly = Polynomial::new(vec![val; 1 << nv]);
    let mut rng = ChaCha20Rng::seed_from_u64(2001);
    let point: Vec<Fr> = (0..nv).map(|_| Fr::random(&mut rng)).collect();
    commit_open_verify(&poly, &point, &pk, &vk, b"kzg-constant");
}

// ---------------------------------------------------------------------------
// Wrong evaluation rejection
// ---------------------------------------------------------------------------

#[test]
fn wrong_eval_rejected() {
    let mut rng = ChaCha20Rng::seed_from_u64(3000);
    let nv = 4;
    let (pk, vk) = make_setup(1 << nv);
    let poly = Polynomial::<Fr>::random(nv, &mut rng);
    let point: Vec<Fr> = (0..nv).map(|_| Fr::random(&mut rng)).collect();

    let correct_eval = poly.evaluate(&point);
    let wrong_eval = correct_eval + Fr::one();
    let (commitment, _) = <KzgPCS as CommitmentScheme>::commit(poly.evaluations(), &pk);

    // Prover opens with correct eval
    let mut t_p = Blake2bTranscript::new(b"kzg-wrong");
    let proof = <KzgPCS as CommitmentScheme>::open(&poly, &point, correct_eval, &pk, None, &mut t_p);

    // Verifier checks with wrong eval
    let mut t_v = Blake2bTranscript::new(b"kzg-wrong");
    let result = <KzgPCS as CommitmentScheme>::verify(
        &commitment, &point, wrong_eval, &proof, &vk, &mut t_v,
    );
    assert!(result.is_err(), "wrong evaluation must be rejected");
}

// ---------------------------------------------------------------------------
// Homomorphic properties
// ---------------------------------------------------------------------------

/// combine([C_a, C_b], [1, 1]) == commit(a + b).
#[test]
fn homomorphic_sum() {
    let mut rng = ChaCha20Rng::seed_from_u64(4000);
    let nv = 4;
    let (pk, vk) = make_setup(1 << nv);
    let a = Polynomial::<Fr>::random(nv, &mut rng);
    let b = Polynomial::<Fr>::random(nv, &mut rng);

    let (com_a, _) = <KzgPCS as CommitmentScheme>::commit(a.evaluations(), &pk);
    let (com_b, _) = <KzgPCS as CommitmentScheme>::commit(b.evaluations(), &pk);
    let combined_com = <KzgPCS as AdditivelyHomomorphic>::combine(
        &[com_a, com_b],
        &[Fr::one(), Fr::one()],
    );

    let sum_poly = a + b;
    let point: Vec<Fr> = (0..nv).map(|_| Fr::random(&mut rng)).collect();
    let eval = sum_poly.evaluate(&point);

    let mut t_p = Blake2bTranscript::new(b"kzg-homo");
    let proof = <KzgPCS as CommitmentScheme>::open(&sum_poly, &point, eval, &pk, None, &mut t_p);

    let mut t_v = Blake2bTranscript::new(b"kzg-homo");
    <KzgPCS as CommitmentScheme>::verify(&combined_com, &point, eval, &proof, &vk, &mut t_v)
        .expect("homomorphic sum must verify");
}

/// combine with arbitrary scalars: s_a·C_a + s_b·C_b == commit(s_a·a + s_b·b).
#[test]
fn homomorphic_weighted_combination() {
    let mut rng = ChaCha20Rng::seed_from_u64(4001);
    let nv = 3;
    let (pk, vk) = make_setup(1 << nv);
    let a = Polynomial::<Fr>::random(nv, &mut rng);
    let b = Polynomial::<Fr>::random(nv, &mut rng);
    let s_a = Fr::random(&mut rng);
    let s_b = Fr::random(&mut rng);

    let (com_a, _) = <KzgPCS as CommitmentScheme>::commit(a.evaluations(), &pk);
    let (com_b, _) = <KzgPCS as CommitmentScheme>::commit(b.evaluations(), &pk);
    let combined_com = <KzgPCS as AdditivelyHomomorphic>::combine(
        &[com_a, com_b],
        &[s_a, s_b],
    );

    let weighted_poly = a * s_a + b * s_b;
    let point: Vec<Fr> = (0..nv).map(|_| Fr::random(&mut rng)).collect();
    let eval = weighted_poly.evaluate(&point);

    let mut t_p = Blake2bTranscript::new(b"kzg-weighted");
    let proof = <KzgPCS as CommitmentScheme>::open(
        &weighted_poly, &point, eval, &pk, None, &mut t_p,
    );

    let mut t_v = Blake2bTranscript::new(b"kzg-weighted");
    <KzgPCS as CommitmentScheme>::verify(&combined_com, &point, eval, &proof, &vk, &mut t_v)
        .expect("weighted combination must verify");
}

// ---------------------------------------------------------------------------
// Deterministic setup
// ---------------------------------------------------------------------------

#[test]
fn deterministic_setup_from_secret() {
    let g1 = Bn254::g1_generator();
    let g2 = Bn254::g2_generator();
    let beta = Fr::from_u64(12345);

    let pk1 = KzgPCS::setup_from_secret(beta, 16, g1, g2);
    let pk2 = KzgPCS::setup_from_secret(beta, 16, g1, g2);
    let vk1 = KzgPCS::verifier_setup(&pk1);
    let vk2 = KzgPCS::verifier_setup(&pk2);

    // Same setup yields same commitments
    let poly = Polynomial::new(vec![Fr::from_u64(1), Fr::from_u64(2)]);
    let (com1, _) = <KzgPCS as CommitmentScheme>::commit(poly.evaluations(), &pk1);
    let (com2, _) = <KzgPCS as CommitmentScheme>::commit(poly.evaluations(), &pk2);
    assert_eq!(com1, com2, "deterministic setups must produce same commitments");

    // Verify with either setup
    let point = vec![Fr::from_u64(7)];
    let eval = poly.evaluate(&point);
    let mut t = Blake2bTranscript::new(b"det-setup");
    let proof = <KzgPCS as CommitmentScheme>::open(&poly, &point, eval, &pk1, None, &mut t);
    let mut t = Blake2bTranscript::new(b"det-setup");
    <KzgPCS as CommitmentScheme>::verify(&com1, &point, eval, &proof, &vk2, &mut t)
        .expect("cross-setup verification must work");
}

// ---------------------------------------------------------------------------
// Property test: random polynomials always verify
// ---------------------------------------------------------------------------

#[test]
fn property_random_polynomials_always_verify() {
    for seed in 5000..5010 {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        let nv = 2 + (seed as usize % 5); // 2..6
        let (pk, vk) = make_setup(1 << nv);
        let poly = Polynomial::<Fr>::random(nv, &mut rng);
        let point: Vec<Fr> = (0..nv).map(|_| Fr::random(&mut rng)).collect();
        commit_open_verify(&poly, &point, &pk, &vk, b"kzg-property");
    }
}
