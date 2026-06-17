//! Negative HyperKZG verification tests.

#![expect(clippy::expect_used, reason = "tests may panic on assertion failures")]

mod common;

use common::{make_setup, KzgPCS};
use jolt_crypto::Bn254;
use jolt_field::{Fr, FromPrimitiveInt, RandomSampling};
use jolt_hyperkzg::{error::HyperKZGError, HyperKZGProofPayload};
use jolt_openings::CommitmentScheme;
use jolt_poly::Polynomial;
use jolt_transcript::{Blake2bTranscript, Transcript};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

fn random_point(num_vars: usize, rng: &mut ChaCha20Rng) -> Vec<Fr> {
    (0..num_vars).map(|_| Fr::random(rng)).collect()
}

#[test]
fn wrong_eval_rejected() {
    let mut rng = ChaCha20Rng::seed_from_u64(3000);
    let num_vars = 4;
    let (pk, vk) = make_setup(1 << num_vars);
    let poly = Polynomial::<Fr>::random(num_vars, &mut rng);
    let point = random_point(num_vars, &mut rng);

    let correct_eval = poly.evaluate(&point);
    let wrong_eval = correct_eval + Fr::from_u64(1);
    let (commitment, _) = <KzgPCS as CommitmentScheme>::commit(poly.evaluations(), &pk);

    let mut prover_transcript = Blake2bTranscript::new(b"kzg-wrong");
    let proof = <KzgPCS as CommitmentScheme>::open(
        &poly,
        &point,
        correct_eval,
        &pk,
        None,
        &mut prover_transcript,
    );

    let mut verifier_transcript = Blake2bTranscript::new(b"kzg-wrong");
    let result = <KzgPCS as CommitmentScheme>::verify(
        &commitment,
        &point,
        wrong_eval,
        &proof,
        &vk,
        &mut verifier_transcript,
    );
    assert!(result.is_err(), "wrong evaluation must be rejected");
}

#[test]
fn clear_verify_rejects_zk_payload_discriminant() {
    let mut rng = ChaCha20Rng::seed_from_u64(3100);
    let num_vars = 4;
    let (pk, vk) = make_setup(1 << num_vars);
    let poly = Polynomial::<Fr>::random(num_vars, &mut rng);
    let point = random_point(num_vars, &mut rng);
    let eval = poly.evaluate(&point);
    let (commitment, _) = <KzgPCS as CommitmentScheme>::commit(poly.evaluations(), &pk);

    let mut prover_transcript = Blake2bTranscript::new(b"kzg-payload-kind");
    let mut proof =
        <KzgPCS as CommitmentScheme>::open(&poly, &point, eval, &pk, None, &mut prover_transcript);
    let empty = Vec::new();
    proof.payload = HyperKZGProofPayload::Zk {
        y: [empty.clone(), empty.clone(), empty],
        y_out: Bn254::g1_generator(),
    };

    let mut verifier_transcript = Blake2bTranscript::new(b"kzg-payload-kind");
    let result = KzgPCS::verify(
        &vk,
        &commitment,
        &point,
        &eval,
        &proof,
        &mut verifier_transcript,
    );
    assert!(matches!(
        result,
        Err(HyperKZGError::WrongProofPayload { .. })
    ));
}

#[test]
fn missing_intermediate_commitment_rejects() {
    let mut rng = ChaCha20Rng::seed_from_u64(3200);
    let num_vars = 4;
    let (pk, vk) = make_setup(1 << num_vars);
    let poly = Polynomial::<Fr>::random(num_vars, &mut rng);
    let point = random_point(num_vars, &mut rng);
    let eval = poly.evaluate(&point);
    let (commitment, _) = <KzgPCS as CommitmentScheme>::commit(poly.evaluations(), &pk);

    let mut prover_transcript = Blake2bTranscript::new(b"kzg-missing-com");
    let mut proof =
        <KzgPCS as CommitmentScheme>::open(&poly, &point, eval, &pk, None, &mut prover_transcript);
    let _ = proof.com.pop();

    let mut verifier_transcript = Blake2bTranscript::new(b"kzg-missing-com");
    let result = KzgPCS::verify(
        &vk,
        &commitment,
        &point,
        &eval,
        &proof,
        &mut verifier_transcript,
    );
    assert!(matches!(
        result,
        Err(HyperKZGError::WrongCommitmentCount { .. })
    ));
}

#[test]
fn tampered_proof_rejects() {
    let mut rng = ChaCha20Rng::seed_from_u64(3300);
    let num_vars = 4;
    let (pk, vk) = make_setup(1 << num_vars);
    let poly = Polynomial::<Fr>::random(num_vars, &mut rng);
    let point = random_point(num_vars, &mut rng);
    let eval = poly.evaluate(&point);
    let (commitment, _) = <KzgPCS as CommitmentScheme>::commit(poly.evaluations(), &pk);

    let mut prover_transcript = Blake2bTranscript::new(b"kzg-tamper");
    let mut proof =
        <KzgPCS as CommitmentScheme>::open(&poly, &point, eval, &pk, None, &mut prover_transcript);

    let v = proof
        .clear_evaluations_mut()
        .expect("transparent proof must have clear evaluations");
    let v1 = v[1].clone();
    v[0].clone_from(&v1);

    let mut verifier_transcript = Blake2bTranscript::new(b"kzg-tamper");
    let result = <KzgPCS as CommitmentScheme>::verify(
        &commitment,
        &point,
        eval,
        &proof,
        &vk,
        &mut verifier_transcript,
    );
    assert!(result.is_err(), "tampered proof must be rejected");
}
