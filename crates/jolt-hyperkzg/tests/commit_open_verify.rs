//! End-to-end HyperKZG commit/open/verify tests.

#![expect(clippy::expect_used, reason = "tests may panic on assertion failures")]

mod common;

use common::{make_setup, KzgPCS};
use jolt_crypto::Bn254;
use jolt_field::{Fr, FromPrimitiveInt, RandomSampling};
use jolt_hyperkzg::{HyperKZGProverSetup, HyperKZGVerifierSetup};
use jolt_openings::CommitmentScheme;
use jolt_poly::Polynomial;
use jolt_transcript::{Blake2bTranscript, Transcript};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

fn random_point(num_vars: usize, rng: &mut ChaCha20Rng) -> Vec<Fr> {
    (0..num_vars).map(|_| Fr::random(rng)).collect()
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

    let mut prover_transcript = Blake2bTranscript::new(label);
    let proof =
        <KzgPCS as CommitmentScheme>::open(poly, point, eval, pk, None, &mut prover_transcript);

    let mut verifier_transcript = Blake2bTranscript::new(label);
    <KzgPCS as CommitmentScheme>::verify(
        &commitment,
        point,
        eval,
        &proof,
        vk,
        &mut verifier_transcript,
    )
    .expect("verification should succeed");
}

#[test]
fn roundtrip_num_vars_1_to_8() {
    let mut rng = ChaCha20Rng::seed_from_u64(1000);
    for num_vars in 1..=8 {
        let (pk, vk) = make_setup(1 << num_vars);
        let poly = Polynomial::<Fr>::random(num_vars, &mut rng);
        let point = random_point(num_vars, &mut rng);
        commit_open_verify(&poly, &point, &pk, &vk, b"kzg-sizes");
    }
}

#[test]
fn zero_polynomial_roundtrip() {
    let num_vars = 3;
    let (pk, vk) = make_setup(1 << num_vars);
    let poly = Polynomial::<Fr>::zeros(num_vars);
    let point = vec![Fr::from_u64(42); num_vars];
    commit_open_verify(&poly, &point, &pk, &vk, b"kzg-zero");
}

#[test]
fn single_variable_polynomial() {
    let mut rng = ChaCha20Rng::seed_from_u64(2000);
    let (pk, vk) = make_setup(2);
    let poly = Polynomial::<Fr>::random(1, &mut rng);
    let point = random_point(1, &mut rng);
    commit_open_verify(&poly, &point, &pk, &vk, b"kzg-single-var");
}

#[test]
fn constant_polynomial() {
    let num_vars = 3;
    let (pk, vk) = make_setup(1 << num_vars);
    let value = Fr::from_u64(42);
    let poly = Polynomial::new(vec![value; 1 << num_vars]);
    let mut rng = ChaCha20Rng::seed_from_u64(2001);
    let point = random_point(num_vars, &mut rng);
    commit_open_verify(&poly, &point, &pk, &vk, b"kzg-constant");
}

#[test]
fn property_random_polynomials_always_verify() {
    for seed in 5000..5010 {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        let num_vars = 2 + (seed as usize % 5);
        let (pk, vk) = make_setup(1 << num_vars);
        let poly = Polynomial::<Fr>::random(num_vars, &mut rng);
        let point = random_point(num_vars, &mut rng);
        commit_open_verify(&poly, &point, &pk, &vk, b"kzg-property");
    }
}
