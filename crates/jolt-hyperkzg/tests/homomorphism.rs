//! HyperKZG additive homomorphism tests.

#![expect(clippy::expect_used, reason = "tests may panic on assertion failures")]

mod common;

use common::{make_setup, KzgPCS};
use jolt_field::{Fr, FromPrimitiveInt, RandomSampling};
use jolt_openings::{AdditivelyHomomorphic, CommitmentScheme};
use jolt_poly::Polynomial;
use jolt_transcript::{Blake2bTranscript, Transcript};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

fn random_point(num_vars: usize, rng: &mut ChaCha20Rng) -> Vec<Fr> {
    (0..num_vars).map(|_| Fr::random(rng)).collect()
}

#[test]
fn homomorphic_sum() {
    let mut rng = ChaCha20Rng::seed_from_u64(4000);
    let num_vars = 4;
    let (pk, vk) = make_setup(1 << num_vars);
    let a = Polynomial::<Fr>::random(num_vars, &mut rng);
    let b = Polynomial::<Fr>::random(num_vars, &mut rng);

    let (com_a, _) = <KzgPCS as CommitmentScheme>::commit(a.evaluations(), &pk);
    let (com_b, _) = <KzgPCS as CommitmentScheme>::commit(b.evaluations(), &pk);
    let combined_com = <KzgPCS as AdditivelyHomomorphic>::combine(
        &[com_a, com_b],
        &[Fr::from_u64(1), Fr::from_u64(1)],
    );

    let sum_poly = a + b;
    let point = random_point(num_vars, &mut rng);
    let eval = sum_poly.evaluate(&point);

    let mut prover_transcript = Blake2bTranscript::new(b"kzg-homo");
    let proof = <KzgPCS as CommitmentScheme>::open(
        &sum_poly,
        &point,
        eval,
        &pk,
        None,
        &mut prover_transcript,
    );

    let mut verifier_transcript = Blake2bTranscript::new(b"kzg-homo");
    <KzgPCS as CommitmentScheme>::verify(
        &combined_com,
        &point,
        eval,
        &proof,
        &vk,
        &mut verifier_transcript,
    )
    .expect("homomorphic sum must verify");
}

#[test]
fn homomorphic_weighted_combination() {
    let mut rng = ChaCha20Rng::seed_from_u64(4001);
    let num_vars = 3;
    let (pk, vk) = make_setup(1 << num_vars);
    let a = Polynomial::<Fr>::random(num_vars, &mut rng);
    let b = Polynomial::<Fr>::random(num_vars, &mut rng);
    let s_a = Fr::random(&mut rng);
    let s_b = Fr::random(&mut rng);

    let (com_a, _) = <KzgPCS as CommitmentScheme>::commit(a.evaluations(), &pk);
    let (com_b, _) = <KzgPCS as CommitmentScheme>::commit(b.evaluations(), &pk);
    let combined_com = <KzgPCS as AdditivelyHomomorphic>::combine(&[com_a, com_b], &[s_a, s_b]);

    let weighted_poly = a * s_a + b * s_b;
    let point = random_point(num_vars, &mut rng);
    let eval = weighted_poly.evaluate(&point);

    let mut prover_transcript = Blake2bTranscript::new(b"kzg-weighted");
    let proof = <KzgPCS as CommitmentScheme>::open(
        &weighted_poly,
        &point,
        eval,
        &pk,
        None,
        &mut prover_transcript,
    );

    let mut verifier_transcript = Blake2bTranscript::new(b"kzg-weighted");
    <KzgPCS as CommitmentScheme>::verify(
        &combined_com,
        &point,
        eval,
        &proof,
        &vk,
        &mut verifier_transcript,
    )
    .expect("weighted combination must verify");
}

#[test]
fn clear_hints_combine_to_clear_hint() {
    let mut rng = ChaCha20Rng::seed_from_u64(4100);
    let num_vars = 3;
    let (pk, _) = make_setup(1 << num_vars);
    let a = Polynomial::<Fr>::random(num_vars, &mut rng);
    let b = Polynomial::<Fr>::random(num_vars, &mut rng);

    let (_, hint_a) = <KzgPCS as CommitmentScheme>::commit(a.evaluations(), &pk);
    let (_, hint_b) = <KzgPCS as CommitmentScheme>::commit(b.evaluations(), &pk);
    assert!(!hint_a.is_zk());
    assert!(!hint_b.is_zk());

    let combined = <KzgPCS as AdditivelyHomomorphic>::combine_hints(
        vec![hint_a, hint_b],
        &[Fr::from_u64(3), Fr::from_u64(5)],
    );
    assert!(!combined.is_zk());
}
