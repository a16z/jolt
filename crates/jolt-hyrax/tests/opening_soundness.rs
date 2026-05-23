#![expect(
    clippy::expect_used,
    reason = "integration tests may panic on setup failures"
)]

use jolt_crypto::{Bn254, Bn254G1, Grumpkin, GrumpkinPoint, JoltGroup, Pedersen, PedersenSetup};
use jolt_field::{Fq, Fr, FromPrimitiveInt};
use jolt_hyrax::{
    HyraxCommitment, HyraxDimensions, HyraxOpeningProof, HyraxProverSetup, HyraxScheme,
    HyraxSetupParams, HyraxVerifierSetup,
};
use jolt_openings::{AdditivelyHomomorphic, CommitmentScheme, OpeningsError};
use jolt_poly::{EqPolynomial, Polynomial};
use jolt_transcript::{Blake2bTranscript, Transcript};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

type TestVc = Pedersen<Bn254G1>;
type TestHyrax = HyraxScheme<TestVc>;
type GrumpkinHyrax = HyraxScheme<Pedersen<GrumpkinPoint>>;

struct OpeningCase {
    verifier_setup: HyraxVerifierSetup<TestVc>,
    commitment: HyraxCommitment<Bn254G1>,
    proof: HyraxOpeningProof<Fr>,
    eval: Fr,
}

fn setup(
    row_vars: usize,
    col_vars: usize,
) -> (HyraxProverSetup<TestVc>, HyraxVerifierSetup<TestVc>) {
    let mut rng = ChaCha20Rng::seed_from_u64(10_000 + row_vars as u64 * 131 + col_vars as u64);
    let row_len = 1usize << col_vars;
    let generators = (0..row_len).map(|_| Bn254::random_g1(&mut rng)).collect();
    let blinding_generator = Bn254::random_g1(&mut rng);
    let vc_setup = PedersenSetup::new(generators, blinding_generator);
    let dimensions =
        HyraxDimensions::new(row_vars + col_vars, row_vars, col_vars).expect("valid dimensions");
    TestHyrax::setup(HyraxSetupParams {
        dimensions,
        vc_setup,
    })
}

fn grumpkin_setup(
    row_vars: usize,
    col_vars: usize,
) -> (
    HyraxProverSetup<Pedersen<GrumpkinPoint>>,
    HyraxVerifierSetup<Pedersen<GrumpkinPoint>>,
) {
    let generator = Grumpkin::generator();
    let row_len = 1usize << col_vars;
    let generators = (1..=row_len)
        .map(|index| generator.scalar_mul(&Fq::from_u64(index as u64)))
        .collect();
    let opening_generator = generator.scalar_mul(&Fq::from_u64(99));
    let vc_setup = PedersenSetup::new(generators, opening_generator);
    let dimensions =
        HyraxDimensions::new(row_vars + col_vars, row_vars, col_vars).expect("valid dimensions");
    GrumpkinHyrax::setup(HyraxSetupParams {
        dimensions,
        vc_setup,
    })
}

fn polynomial(num_vars: usize, salt: u64) -> Polynomial<Fr> {
    let evals: Vec<Fr> = (0..(1usize << num_vars))
        .map(|index| {
            let x = index as u64 + 1;
            Fr::from_u64((x * x * 17) + (x * (salt + 29)) + salt * 41 + 5)
        })
        .collect();
    Polynomial::from(evals)
}

fn point(num_vars: usize, salt: u64) -> Vec<Fr> {
    (0..num_vars)
        .map(|index| Fr::from_u64((index as u64 + 3) * (salt + 11)))
        .collect()
}

fn fq_polynomial(num_vars: usize, salt: u64) -> Polynomial<Fq> {
    let evals: Vec<Fq> = (0..(1usize << num_vars))
        .map(|index| {
            let x = index as u64 + 1;
            Fq::from_u64((x * x * 23) + (x * (salt + 31)) + salt * 43 + 7)
        })
        .collect();
    Polynomial::from(evals)
}

fn fq_point(num_vars: usize, salt: u64) -> Vec<Fq> {
    (0..num_vars)
        .map(|index| Fq::from_u64((index as u64 + 5) * (salt + 13)))
        .collect()
}

fn open_case(row_vars: usize, col_vars: usize, salt: u64) -> OpeningCase {
    let (prover_setup, verifier_setup) = setup(row_vars, col_vars);
    let poly = polynomial(row_vars + col_vars, salt);
    let point = point(poly.num_vars(), salt);
    let eval = poly.evaluate(&point);
    let (commitment, hint) = TestHyrax::commit(&poly, &prover_setup);
    let mut transcript = Blake2bTranscript::new(b"hyrax-integration-open");
    let proof = TestHyrax::open(
        &poly,
        &point,
        eval,
        &prover_setup,
        Some(hint),
        &mut transcript,
    );
    OpeningCase {
        verifier_setup,
        commitment,
        proof,
        eval,
    }
}

fn verify(
    verifier_setup: &HyraxVerifierSetup<TestVc>,
    commitment: &HyraxCommitment<Bn254G1>,
    point: &[Fr],
    eval: Fr,
    proof: &HyraxOpeningProof<Fr>,
) -> Result<(), OpeningsError> {
    let mut transcript = Blake2bTranscript::new(b"hyrax-integration-open");
    TestHyrax::verify(
        commitment,
        point,
        eval,
        proof,
        verifier_setup,
        &mut transcript,
    )
}

fn column_eval(combined_row: &[Fr], col_point: &[Fr]) -> Fr {
    EqPolynomial::new(col_point.to_vec())
        .evaluations()
        .iter()
        .zip(combined_row.iter())
        .map(|(weight, value)| *weight * *value)
        .sum()
}

fn expected_combined_row(
    poly: &Polynomial<Fr>,
    dimensions: HyraxDimensions,
    row_point: &[Fr],
) -> Vec<Fr> {
    let row_weights = EqPolynomial::new(row_point.to_vec()).evaluations();
    let row_len = dimensions.row_len().expect("valid row len");
    let mut combined = vec![Fr::from_u64(0); row_len];
    for (row, weight) in poly.evaluations().chunks(row_len).zip(row_weights.iter()) {
        for (dst, value) in combined.iter_mut().zip(row.iter()) {
            *dst += *weight * *value;
        }
    }
    combined
}

fn assert_rejects(result: Result<(), OpeningsError>) {
    let err = result.expect_err("tampered opening must reject");
    assert!(matches!(err, OpeningsError::VerificationFailed));
}

#[test]
fn trait_and_native_openings_verify_across_dimension_splits() {
    for (row_vars, col_vars, salt) in [(0, 0, 1), (0, 3, 2), (3, 0, 3), (1, 4, 4), (4, 1, 5)] {
        let (prover_setup, verifier_setup) = setup(row_vars, col_vars);
        let dimensions = prover_setup.dimensions;
        let poly = polynomial(row_vars + col_vars, salt);
        let point = point(poly.num_vars(), salt);
        let eval = poly.evaluate(&point);
        let (commitment, hint) = TestHyrax::commit(&poly, &prover_setup);

        let (native_proof, native_eval) =
            TestHyrax::opening_proof(&prover_setup, &poly, &point).expect("native opening proof");
        assert_eq!(native_eval, eval);
        TestHyrax::verify_opening_proof(&verifier_setup, &commitment, &point, eval, &native_proof)
            .expect("native opening verifies");

        let mut prover_transcript = Blake2bTranscript::new(b"hyrax-integration-open");
        let trait_proof = TestHyrax::open(
            &poly,
            &point,
            eval,
            &prover_setup,
            Some(hint),
            &mut prover_transcript,
        );
        verify(&verifier_setup, &commitment, &point, eval, &trait_proof)
            .expect("trait opening verifies");

        let (row_point, _) = dimensions.split_point(&point).expect("valid point");
        assert_eq!(
            trait_proof.combined_row,
            expected_combined_row(&poly, dimensions, row_point),
        );
        assert_eq!(trait_proof.combined_row_opening_scalar, Fr::from_u64(0));
    }
}

#[test]
fn wrong_evaluation_rejects() {
    let case = open_case(3, 2, 20);
    let point = point(5, 20);
    assert_rejects(verify(
        &case.verifier_setup,
        &case.commitment,
        &point,
        case.eval + Fr::from_u64(1),
        &case.proof,
    ));
}

#[test]
fn tampered_combined_row_rejects_even_with_matching_evaluation() {
    let case = open_case(3, 2, 30);
    let point = point(5, 30);
    let (_, col_point) = case
        .verifier_setup
        .dimensions
        .split_point(&point)
        .expect("valid point");
    let mut proof = case.proof;
    proof.combined_row[0] += Fr::from_u64(9);
    let matching_tampered_eval = column_eval(&proof.combined_row, col_point);

    assert_rejects(verify(
        &case.verifier_setup,
        &case.commitment,
        &point,
        matching_tampered_eval,
        &proof,
    ));
}

#[test]
fn tampered_combined_row_opening_scalar_rejects() {
    let case = open_case(2, 3, 40);
    let point = point(5, 40);
    let mut proof = case.proof;
    proof.combined_row_opening_scalar += Fr::from_u64(1);

    assert_rejects(verify(
        &case.verifier_setup,
        &case.commitment,
        &point,
        case.eval,
        &proof,
    ));
}

#[test]
fn truncated_combined_row_rejects() {
    let case = open_case(2, 3, 50);
    let point = point(5, 50);
    let mut proof = case.proof;
    let _ = proof.combined_row.pop();

    assert_rejects(verify(
        &case.verifier_setup,
        &case.commitment,
        &point,
        case.eval,
        &proof,
    ));
}

#[test]
fn proof_for_one_commitment_rejects_against_another_commitment() {
    let (prover_setup, verifier_setup) = setup(3, 2);
    let point = point(5, 60);
    let poly_a = polynomial(5, 60);
    let poly_b = polynomial(5, 61);
    let eval_a = poly_a.evaluate(&point);
    let (_, hint_a) = TestHyrax::commit(&poly_a, &prover_setup);
    let (commitment_b, ()) = TestHyrax::commit(&poly_b, &prover_setup);
    let mut transcript = Blake2bTranscript::new(b"hyrax-integration-open");
    let proof_a = TestHyrax::open(
        &poly_a,
        &point,
        eval_a,
        &prover_setup,
        Some(hint_a),
        &mut transcript,
    );

    assert_rejects(verify(
        &verifier_setup,
        &commitment_b,
        &point,
        eval_a,
        &proof_a,
    ));
}

#[test]
fn proof_for_one_row_point_rejects_at_another_row_point() {
    let case = open_case(3, 2, 70);
    let mut wrong_point = point(5, 70);
    wrong_point[0] += Fr::from_u64(5);
    let wrong_eval = polynomial(5, 70).evaluate(&wrong_point);

    assert_rejects(verify(
        &case.verifier_setup,
        &case.commitment,
        &wrong_point,
        wrong_eval,
        &case.proof,
    ));
}

#[test]
fn tampered_row_commitment_rejects() {
    let mut case = open_case(3, 2, 80);
    let point = point(5, 80);
    case.commitment.rows[1] += Bn254::g1_generator();

    assert_rejects(verify(
        &case.verifier_setup,
        &case.commitment,
        &point,
        case.eval,
        &case.proof,
    ));
}

#[test]
fn row_commitment_count_mismatch_rejects() {
    let mut case = open_case(3, 2, 90);
    let point = point(5, 90);
    let _ = case.commitment.rows.pop();

    assert_rejects(verify(
        &case.verifier_setup,
        &case.commitment,
        &point,
        case.eval,
        &case.proof,
    ));
}

#[test]
fn additive_homomorphic_combination_verifies_against_combined_polynomial() {
    let (prover_setup, verifier_setup) = setup(3, 2);
    let point = point(5, 100);
    let poly_a = polynomial(5, 100);
    let poly_b = polynomial(5, 101);
    let scalar_a = Fr::from_u64(7);
    let scalar_b = Fr::from_u64(13);
    let (commitment_a, ()) = TestHyrax::commit(&poly_a, &prover_setup);
    let (commitment_b, ()) = TestHyrax::commit(&poly_b, &prover_setup);
    let combined_commitment =
        TestHyrax::combine(&[commitment_a, commitment_b], &[scalar_a, scalar_b]);
    let combined_poly = Polynomial::from(
        poly_a
            .evaluations()
            .iter()
            .zip(poly_b.evaluations().iter())
            .map(|(a, b)| scalar_a * *a + scalar_b * *b)
            .collect::<Vec<_>>(),
    );
    let combined_eval = combined_poly.evaluate(&point);
    let (_, hint) = TestHyrax::commit(&combined_poly, &prover_setup);
    let mut transcript = Blake2bTranscript::new(b"hyrax-integration-open");
    let proof = TestHyrax::open(
        &combined_poly,
        &point,
        combined_eval,
        &prover_setup,
        Some(hint),
        &mut transcript,
    );

    verify(
        &verifier_setup,
        &combined_commitment,
        &point,
        combined_eval,
        &proof,
    )
    .expect("opening verifies against homomorphically combined commitment");
}

#[test]
fn grumpkin_backed_hyrax_verifies_and_rejects_tampering() {
    let (prover_setup, verifier_setup) = grumpkin_setup(2, 3);
    let poly = fq_polynomial(5, 110);
    let point = fq_point(5, 110);
    let eval = poly.evaluate(&point);
    let (commitment, hint) = GrumpkinHyrax::commit(&poly, &prover_setup);
    let mut prover_transcript = Blake2bTranscript::new(b"hyrax-grumpkin-open");
    let proof = GrumpkinHyrax::open(
        &poly,
        &point,
        eval,
        &prover_setup,
        Some(hint),
        &mut prover_transcript,
    );

    let mut verifier_transcript = Blake2bTranscript::new(b"hyrax-grumpkin-open");
    GrumpkinHyrax::verify(
        &commitment,
        &point,
        eval,
        &proof,
        &verifier_setup,
        &mut verifier_transcript,
    )
    .expect("Grumpkin-backed Hyrax verifies");

    let mut tampered_proof = proof;
    tampered_proof.combined_row[0] += Fq::from_u64(1);
    let mut verifier_transcript = Blake2bTranscript::new(b"hyrax-grumpkin-open");
    let err = GrumpkinHyrax::verify(
        &commitment,
        &point,
        eval,
        &tampered_proof,
        &verifier_setup,
        &mut verifier_transcript,
    )
    .expect_err("tampered Grumpkin-backed opening rejects");
    assert!(matches!(err, OpeningsError::VerificationFailed));
}
