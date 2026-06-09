//! ZK HyperKZG additive homomorphism tests.

#![cfg(feature = "zk")]
#![expect(clippy::expect_used, reason = "tests may panic on assertion failures")]

use jolt_crypto::{Bn254, JoltGroup};
use jolt_field::{Fr, FromPrimitiveInt, RandomSampling};
use jolt_hyperkzg::HyperKZGScheme;
use jolt_openings::{AdditivelyHomomorphic, CommitmentScheme, ZkOpeningScheme};
use jolt_poly::Polynomial;
use jolt_transcript::{Blake2bTranscript, Transcript};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

type KzgPCS = HyperKZGScheme<Bn254>;

fn make_zk_setup(max_degree: usize) -> jolt_hyperkzg::HyperKZGProverSetup<Bn254> {
    let g1 = Bn254::g1_generator();
    let hiding_g1 = g1.scalar_mul(&Fr::from_u64(17));
    let g2 = Bn254::g2_generator();
    KzgPCS::setup_zk_from_secret(Fr::from_u64(12345), max_degree, g1, hiding_g1, g2)
}

fn random_point(num_vars: usize, rng: &mut ChaCha20Rng) -> Vec<Fr> {
    (0..num_vars).map(|_| Fr::random(rng)).collect()
}

#[test]
fn homomorphic_zk_weighted_combination_verifies_with_combined_hint() {
    let mut rng = ChaCha20Rng::seed_from_u64(7400);
    let num_vars = 4;
    let pk = make_zk_setup(1 << num_vars);
    let vk = KzgPCS::verifier_setup(&pk);
    let poly_a = Polynomial::<Fr>::random(num_vars, &mut rng);
    let poly_b = Polynomial::<Fr>::random(num_vars, &mut rng);
    let scalar_a = Fr::random(&mut rng);
    let scalar_b = Fr::random(&mut rng);

    let (commitment_a, hint_a) = <KzgPCS as ZkOpeningScheme>::commit_zk(poly_a.evaluations(), &pk);
    let (commitment_b, hint_b) = <KzgPCS as ZkOpeningScheme>::commit_zk(poly_b.evaluations(), &pk);
    assert!(hint_a.is_zk());
    assert!(hint_b.is_zk());

    let combined_commitment = <KzgPCS as AdditivelyHomomorphic>::combine(
        &[commitment_a, commitment_b],
        &[scalar_a, scalar_b],
    );
    let combined_hint = <KzgPCS as AdditivelyHomomorphic>::combine_hints(
        vec![hint_a, hint_b],
        &[scalar_a, scalar_b],
    );
    assert!(combined_hint.is_zk());

    let weighted_poly = poly_a * scalar_a + poly_b * scalar_b;
    let point = random_point(num_vars, &mut rng);
    let eval = weighted_poly.evaluate(&point);

    let mut prover_transcript = Blake2bTranscript::new(b"hyperkzg-zk-combine");
    let (proof, y_out, output_blind) = KzgPCS::open_zk(
        &weighted_poly,
        &point,
        eval,
        &pk,
        combined_hint,
        &mut prover_transcript,
    );

    let mut verifier_transcript = Blake2bTranscript::new(b"hyperkzg-zk-combine");
    let verified_y_out = KzgPCS::verify_zk(
        &combined_commitment,
        &point,
        &proof,
        &vk,
        &mut verifier_transcript,
    )
    .expect("combined ZK opening should verify");

    let hiding_g1 = Bn254::g1_generator().scalar_mul(&Fr::from_u64(17));
    let expected_y_out =
        Bn254::g1_generator().scalar_mul(&eval) + hiding_g1.scalar_mul(&output_blind);
    assert_eq!(verified_y_out, y_out);
    assert_eq!(verified_y_out, expected_y_out);
}

#[test]
#[should_panic(expected = "cannot combine mixed transparent and ZK")]
fn combining_mixed_clear_and_zk_hints_panics() {
    let mut rng = ChaCha20Rng::seed_from_u64(7410);
    let num_vars = 3;
    let pk = make_zk_setup(1 << num_vars);
    let poly = Polynomial::<Fr>::random(num_vars, &mut rng);

    let (_, clear_hint) =
        <KzgPCS as jolt_openings::CommitmentScheme>::commit(poly.evaluations(), &pk);
    let (_, zk_hint) = <KzgPCS as ZkOpeningScheme>::commit_zk(poly.evaluations(), &pk);

    let _ = <KzgPCS as AdditivelyHomomorphic>::combine_hints(
        vec![clear_hint, zk_hint],
        &[Fr::from_u64(1), Fr::from_u64(1)],
    );
}
