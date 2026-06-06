//! Negative ZK HyperKZG verification and precondition tests.

#![cfg(feature = "zk")]
#![expect(clippy::expect_used, reason = "tests may panic on assertion failures")]

use jolt_crypto::{Bn254, Bn254G1, JoltGroup};
use jolt_field::{Fr, FromPrimitiveInt, RandomSampling};
use jolt_hyperkzg::{HyperKZGCommitment, HyperKZGProof, HyperKZGScheme, HyperKZGVerifierSetup};
use jolt_openings::{CommitmentScheme, ZkOpeningScheme};
use jolt_poly::Polynomial;
use jolt_transcript::{Blake2bTranscript, Transcript};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

type KzgPCS = HyperKZGScheme<Bn254>;

struct ZkCase {
    commitment: HyperKZGCommitment<Bn254>,
    point: Vec<Fr>,
    proof: HyperKZGProof<Bn254>,
    verifier_setup: HyperKZGVerifierSetup<Bn254>,
}

fn make_zk_setup(max_degree: usize) -> jolt_hyperkzg::HyperKZGProverSetup<Bn254> {
    let g1 = Bn254::g1_generator();
    let hiding_g1 = g1.scalar_mul(&Fr::from_u64(17));
    let g2 = Bn254::g2_generator();
    KzgPCS::setup_zk_from_secret(Fr::from_u64(12345), max_degree, g1, hiding_g1, g2)
}

fn random_point(num_vars: usize, rng: &mut ChaCha20Rng) -> Vec<Fr> {
    (0..num_vars).map(|_| Fr::random(rng)).collect()
}

fn make_case(seed: u64, label: &'static [u8]) -> ZkCase {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let num_vars = 4;
    let prover_setup = make_zk_setup(1 << num_vars);
    let verifier_setup = KzgPCS::verifier_setup(&prover_setup);
    let poly = Polynomial::<Fr>::random(num_vars, &mut rng);
    let point = random_point(num_vars, &mut rng);
    let eval = poly.evaluate(&point);
    let (commitment, hint) =
        <KzgPCS as ZkOpeningScheme>::commit_zk(poly.evaluations(), &prover_setup);

    let mut prover_transcript = Blake2bTranscript::new(label);
    let (proof, _y_out, _output_blind) = KzgPCS::open_zk(
        &poly,
        &point,
        eval,
        &prover_setup,
        hint,
        &mut prover_transcript,
    );

    ZkCase {
        commitment,
        point,
        proof,
        verifier_setup,
    }
}

fn verify_case(
    case: &ZkCase,
    label: &'static [u8],
) -> Result<Bn254G1, jolt_openings::OpeningsError> {
    let mut verifier_transcript = Blake2bTranscript::new(label);
    KzgPCS::verify_zk(
        &case.commitment,
        &case.point,
        &case.proof,
        &case.verifier_setup,
        &mut verifier_transcript,
    )
}

#[test]
fn tampered_zk_evaluation_commitment_rejects() {
    let label = b"hyperkzg-zk-tamper-y";
    let mut case = make_case(7500, label);
    let (y, _) = case
        .proof
        .hidden_evaluation_commitments_mut()
        .expect("ZK proof should expose hidden evaluation commitments");
    y[0][0] += Bn254::g1_generator();

    assert!(
        verify_case(&case, label).is_err(),
        "tampered ZK evaluation must reject"
    );
}

#[test]
fn tampered_y_out_rejects() {
    let label = b"hyperkzg-zk-tamper-out";
    let mut case = make_case(7510, label);
    let (_, y_out) = case
        .proof
        .hidden_evaluation_commitments_mut()
        .expect("ZK proof should expose hidden evaluation commitments");
    *y_out += Bn254::g1_generator();

    assert!(
        verify_case(&case, label).is_err(),
        "tampered ZK output commitment must reject"
    );
}

#[test]
fn tampered_witness_rejects() {
    let label = b"hyperkzg-zk-tamper-w";
    let mut case = make_case(7520, label);
    case.proof.w[0] += Bn254::g1_generator();

    assert!(
        verify_case(&case, label).is_err(),
        "tampered ZK witness must reject"
    );
}

#[test]
fn tampered_fold_commitment_rejects() {
    let label = b"hyperkzg-zk-tamper-com";
    let mut case = make_case(7530, label);
    case.proof.com[0] += Bn254::g1_generator();

    assert!(
        verify_case(&case, label).is_err(),
        "tampered ZK fold commitment must reject"
    );
}

#[test]
fn wrong_zk_commitment_rejects() {
    let label = b"hyperkzg-zk-wrong-com";
    let mut case = make_case(7540, label);
    let mut rng = ChaCha20Rng::seed_from_u64(7541);
    let prover_setup = make_zk_setup(1 << case.point.len());
    let wrong_poly = Polynomial::<Fr>::random(case.point.len(), &mut rng);
    let (wrong_commitment, _) =
        <KzgPCS as ZkOpeningScheme>::commit_zk(wrong_poly.evaluations(), &prover_setup);
    case.commitment = wrong_commitment;

    assert!(
        verify_case(&case, label).is_err(),
        "wrong ZK commitment must reject"
    );
}

#[test]
fn wrong_claimed_eval_in_zk_open_rejects() {
    let mut rng = ChaCha20Rng::seed_from_u64(7545);
    let label = b"hyperkzg-zk-wrong-eval";
    let num_vars = 4;
    let prover_setup = make_zk_setup(1 << num_vars);
    let verifier_setup = KzgPCS::verifier_setup(&prover_setup);
    let poly = Polynomial::<Fr>::random(num_vars, &mut rng);
    let point = random_point(num_vars, &mut rng);
    let wrong_eval = poly.evaluate(&point) + Fr::from_u64(1);
    let (commitment, hint) =
        <KzgPCS as ZkOpeningScheme>::commit_zk(poly.evaluations(), &prover_setup);

    let mut prover_transcript = Blake2bTranscript::new(label);
    let (proof, _y_out, _output_blind) = KzgPCS::open_zk(
        &poly,
        &point,
        wrong_eval,
        &prover_setup,
        hint,
        &mut prover_transcript,
    );

    let mut verifier_transcript = Blake2bTranscript::new(label);
    let result = KzgPCS::verify_zk(
        &commitment,
        &point,
        &proof,
        &verifier_setup,
        &mut verifier_transcript,
    );
    assert!(
        result.is_err(),
        "wrong hidden output evaluation must reject"
    );
}

#[test]
fn malformed_zk_evaluation_width_rejects() {
    let label = b"hyperkzg-zk-bad-width";
    let mut case = make_case(7550, label);
    let (y, _) = case
        .proof
        .hidden_evaluation_commitments_mut()
        .expect("ZK proof should expose hidden evaluation commitments");
    let _ = y[0].pop();

    assert!(
        verify_case(&case, label).is_err(),
        "malformed ZK evaluation row must reject"
    );
}

#[test]
fn verify_zk_rejects_clear_payload() {
    let mut rng = ChaCha20Rng::seed_from_u64(7560);
    let label = b"hyperkzg-zk-clear-payload";
    let num_vars = 3;
    let prover_setup = make_zk_setup(1 << num_vars);
    let verifier_setup = KzgPCS::verifier_setup(&prover_setup);
    let poly = Polynomial::<Fr>::random(num_vars, &mut rng);
    let point = random_point(num_vars, &mut rng);
    let eval = poly.evaluate(&point);
    let (commitment, _) = <KzgPCS as CommitmentScheme>::commit(poly.evaluations(), &prover_setup);

    let mut prover_transcript = Blake2bTranscript::new(label);
    let proof = <KzgPCS as CommitmentScheme>::open(
        &poly,
        &point,
        eval,
        &prover_setup,
        None,
        &mut prover_transcript,
    );

    let mut verifier_transcript = Blake2bTranscript::new(label);
    let result = KzgPCS::verify_zk(
        &commitment,
        &point,
        &proof,
        &verifier_setup,
        &mut verifier_transcript,
    );
    assert!(
        result.is_err(),
        "ZK verifier should reject a clear proof payload"
    );
}

#[test]
#[should_panic(expected = "ZK SRS must contain hiding bases")]
fn commit_zk_with_plain_srs_panics() {
    let mut rng = ChaCha20Rng::seed_from_u64(7570);
    let num_vars = 3;
    let g1 = Bn254::g1_generator();
    let g2 = Bn254::g2_generator();
    let plain_setup = KzgPCS::setup_from_secret(Fr::from_u64(12345), 1 << num_vars, g1, g2);
    let poly = Polynomial::<Fr>::random(num_vars, &mut rng);

    let _ = <KzgPCS as ZkOpeningScheme>::commit_zk(poly.evaluations(), &plain_setup);
}

#[test]
#[should_panic(expected = "ZK HyperKZG opening requires a ZK opening hint")]
fn open_zk_with_clear_hint_panics() {
    let mut rng = ChaCha20Rng::seed_from_u64(7580);
    let num_vars = 3;
    let prover_setup = make_zk_setup(1 << num_vars);
    let poly = Polynomial::<Fr>::random(num_vars, &mut rng);
    let point = random_point(num_vars, &mut rng);
    let eval = poly.evaluate(&point);
    let (_, clear_hint) = <KzgPCS as CommitmentScheme>::commit(poly.evaluations(), &prover_setup);

    let mut prover_transcript = Blake2bTranscript::new(b"hyperkzg-zk-clear-hint");
    let _ = KzgPCS::open_zk(
        &poly,
        &point,
        eval,
        &prover_setup,
        clear_hint,
        &mut prover_transcript,
    );
}
