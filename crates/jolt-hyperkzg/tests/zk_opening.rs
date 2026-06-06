//! ZK HyperKZG opening tests.

#![cfg(feature = "zk")]
#![expect(clippy::expect_used, reason = "tests may panic on assertion failures")]

use jolt_crypto::{Bn254, JoltGroup};
use jolt_field::{Fr, FromPrimitiveInt, RandomSampling};
use jolt_hyperkzg::HyperKZGScheme;
use jolt_openings::{CommitmentScheme, ZkOpeningScheme};
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
fn zk_roundtrip_returns_hidden_evaluation_commitment() {
    let mut rng = ChaCha20Rng::seed_from_u64(7000);
    let num_vars = 4;
    let pk = make_zk_setup(1 << num_vars);
    let vk = KzgPCS::verifier_setup(&pk);
    let poly = Polynomial::<Fr>::random(num_vars, &mut rng);
    let point = random_point(num_vars, &mut rng);
    let eval = poly.evaluate(&point);

    let (commitment, hint) = <KzgPCS as ZkOpeningScheme>::commit_zk(poly.evaluations(), &pk);
    assert!(hint.is_zk());

    let mut prover_transcript = Blake2bTranscript::new(b"hyperkzg-zk-roundtrip");
    let (proof, y_out, output_blind) =
        KzgPCS::open_zk(&poly, &point, eval, &pk, hint, &mut prover_transcript);

    let mut verifier_transcript = Blake2bTranscript::new(b"hyperkzg-zk-roundtrip");
    let verified_y_out =
        KzgPCS::verify_zk(&commitment, &point, &proof, &vk, &mut verifier_transcript)
            .expect("ZK opening should verify");

    let hiding_g1 = Bn254::g1_generator().scalar_mul(&Fr::from_u64(17));
    let expected_y_out =
        Bn254::g1_generator().scalar_mul(&eval) + hiding_g1.scalar_mul(&output_blind);
    assert_eq!(verified_y_out, y_out);
    assert_eq!(verified_y_out, expected_y_out);
}

#[test]
fn zk_commitment_uses_fresh_blinding() {
    let mut rng = ChaCha20Rng::seed_from_u64(7100);
    let num_vars = 3;
    let pk = make_zk_setup(1 << num_vars);
    let poly = Polynomial::<Fr>::random(num_vars, &mut rng);

    let (commitment_a, _) = <KzgPCS as ZkOpeningScheme>::commit_zk(poly.evaluations(), &pk);
    let (commitment_b, _) = <KzgPCS as ZkOpeningScheme>::commit_zk(poly.evaluations(), &pk);

    assert_ne!(
        commitment_a, commitment_b,
        "ZK commitments should use fresh scalar blinds"
    );
}
