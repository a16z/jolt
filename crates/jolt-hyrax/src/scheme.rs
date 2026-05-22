use std::marker::PhantomData;

use jolt_crypto::VectorCommitmentOpening;
use jolt_crypto::{Commitment, HomomorphicCommitment, VectorCommitment};
use jolt_openings::{AdditivelyHomomorphic, CommitmentScheme, OpeningsError};
use jolt_poly::{MultilinearPoly, Polynomial};
use jolt_transcript::{AppendToTranscript, Label, LabelWithCount, Transcript};
use serde::{de::DeserializeOwned, Deserialize, Serialize};

use crate::{
    HyraxCommitment, HyraxDimensions, HyraxError, HyraxOpeningProof, HyraxProverSetup,
    HyraxSetupParams, HyraxVerifierSetup,
};

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct HyraxScheme<VC: VectorCommitment> {
    _marker: PhantomData<VC>,
}

impl<VC> HyraxScheme<VC>
where
    VC: VectorCommitment,
    VC::Output: HomomorphicCommitment<VC::Field>,
{
    pub fn opening_proof(
        setup: &HyraxProverSetup<VC>,
        poly: &Polynomial<VC::Field>,
        point: &[VC::Field],
    ) -> Result<(HyraxOpeningProof<VC::Field>, VC::Field), HyraxError> {
        validate_dimensions_for_poly(&setup.dimensions, poly.num_vars())?;
        validate_capacity::<VC>(setup)?;
        let (row_point, col_point) = setup.dimensions.split_point(point)?;
        let row_blindings = zero_row_blindings::<VC>(&setup.dimensions)?;
        let (row_opening, opened_eval) = VC::open_committed_rows(
            poly.evaluations(),
            &row_blindings,
            setup.dimensions.row_len()?,
            row_point,
            col_point,
        )?;
        Ok((HyraxOpeningProof { row_opening }, opened_eval))
    }

    pub fn verify_opening_proof(
        setup: &HyraxVerifierSetup<VC>,
        commitment: &HyraxCommitment<VC::Output>,
        point: &[VC::Field],
        eval: VC::Field,
        proof: &HyraxOpeningProof<VC::Field>,
    ) -> Result<(), HyraxError> {
        setup.dimensions.validate()?;
        validate_verifier_capacity::<VC>(setup)?;
        let row_count = setup.dimensions.row_count()?;
        if commitment.rows.len() != row_count {
            return Err(HyraxError::RowCommitmentCountMismatch {
                expected: row_count,
                got: commitment.rows.len(),
            });
        }
        let (row_point, col_point) = setup.dimensions.split_point(point)?;
        let opened_eval = VC::verify_committed_rows(
            &setup.vc_setup,
            &commitment.rows,
            row_point,
            col_point,
            &proof.row_opening,
        )?;
        if opened_eval != eval {
            return Err(HyraxError::EvaluationMismatch);
        }
        Ok(())
    }
}

impl<VC> Commitment for HyraxScheme<VC>
where
    VC: VectorCommitment,
{
    type Output = HyraxCommitment<VC::Output>;
}

impl<VC> CommitmentScheme for HyraxScheme<VC>
where
    VC: VectorCommitment,
    VC::Field: AppendToTranscript,
    VC::Output: HomomorphicCommitment<VC::Field>,
    VC::Setup: Serialize + DeserializeOwned,
{
    type Field = VC::Field;
    type Proof = HyraxOpeningProof<VC::Field>;
    type ProverSetup = HyraxProverSetup<VC>;
    type VerifierSetup = HyraxVerifierSetup<VC>;
    type Polynomial = Polynomial<VC::Field>;
    type OpeningHint = ();
    type SetupParams = HyraxSetupParams<VC>;

    fn setup(params: Self::SetupParams) -> (Self::ProverSetup, Self::VerifierSetup) {
        assert_valid_dimensions(&params.dimensions);
        let prover = HyraxProverSetup {
            dimensions: params.dimensions,
            vc_setup: params.vc_setup,
        };
        let verifier = Self::verifier_setup(&prover);
        (prover, verifier)
    }

    fn verifier_setup(prover_setup: &Self::ProverSetup) -> Self::VerifierSetup {
        prover_setup.into()
    }

    fn commit<P: MultilinearPoly<Self::Field> + ?Sized>(
        poly: &P,
        setup: &Self::ProverSetup,
    ) -> (Self::Output, Self::OpeningHint) {
        assert_valid_dimensions(&setup.dimensions);
        assert_valid_poly(&setup.dimensions, poly.num_vars());
        assert_valid_capacity::<VC>(setup);

        let row_len = setup.dimensions.row_len().unwrap_or_default();
        let row_count = setup.dimensions.row_count().unwrap_or_default();
        let zero_blinding = VC::Field::default();
        let mut rows = Vec::with_capacity(row_count);

        poly.for_each_row(setup.dimensions.col_vars, &mut |row_index, row| {
            assert_eq!(
                row_index,
                rows.len(),
                "Hyrax row iterator produced rows out of order",
            );
            assert_eq!(row.len(), row_len, "Hyrax row length mismatch");
            rows.push(VC::commit(&setup.vc_setup, row, &zero_blinding));
        });

        assert_eq!(rows.len(), row_count, "Hyrax row count mismatch");
        (HyraxCommitment { rows }, ())
    }

    fn open(
        poly: &Self::Polynomial,
        point: &[Self::Field],
        _eval: Self::Field,
        setup: &Self::ProverSetup,
        _hint: Option<Self::OpeningHint>,
        _transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Self::Proof {
        let result = Self::opening_proof(setup, poly, point);
        assert!(result.is_ok(), "Hyrax open failed");
        match result {
            Ok((proof, _)) => proof,
            Err(_) => HyraxOpeningProof {
                row_opening: VectorCommitmentOpening {
                    combined_vector: Vec::new(),
                    combined_blinding: Self::Field::default(),
                },
            },
        }
    }

    fn verify(
        commitment: &Self::Output,
        point: &[Self::Field],
        eval: Self::Field,
        proof: &Self::Proof,
        setup: &Self::VerifierSetup,
        _transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Result<(), OpeningsError> {
        Self::verify_opening_proof(setup, commitment, point, eval, proof)
            .map_err(|_| OpeningsError::VerificationFailed)
    }

    fn bind_opening_inputs(
        transcript: &mut impl Transcript<Challenge = Self::Field>,
        point: &[Self::Field],
        eval: &Self::Field,
    ) {
        transcript.append(&LabelWithCount(b"hyrax_opening_point", point.len() as u64));
        for scalar in point {
            scalar.append_to_transcript(transcript);
        }
        transcript.append(&Label(b"hyrax_opening_eval"));
        eval.append_to_transcript(transcript);
    }
}

impl<VC> AdditivelyHomomorphic for HyraxScheme<VC>
where
    VC: VectorCommitment,
    VC::Field: AppendToTranscript,
    VC::Output: HomomorphicCommitment<VC::Field>,
    VC::Setup: Serialize + DeserializeOwned,
{
    fn combine(commitments: &[Self::Output], scalars: &[Self::Field]) -> Self::Output {
        assert_eq!(commitments.len(), scalars.len());
        commitments
            .iter()
            .zip(scalars.iter())
            .fold(HyraxCommitment::default(), |acc, (commitment, scalar)| {
                HyraxCommitment::linear_combine(&acc, commitment, scalar)
            })
    }
}

fn validate_dimensions_for_poly(
    dimensions: &HyraxDimensions,
    poly_num_vars: usize,
) -> Result<(), HyraxError> {
    dimensions.validate()?;
    if poly_num_vars != dimensions.num_vars {
        return Err(HyraxError::PolynomialVariableMismatch {
            expected: dimensions.num_vars,
            got: poly_num_vars,
        });
    }
    Ok(())
}

fn validate_capacity<VC: VectorCommitment>(setup: &HyraxProverSetup<VC>) -> Result<(), HyraxError> {
    let row_len = setup.dimensions.row_len()?;
    let capacity = VC::capacity(&setup.vc_setup);
    if row_len > capacity {
        return Err(HyraxError::CommitmentCapacityExceeded { capacity, row_len });
    }
    Ok(())
}

fn validate_verifier_capacity<VC: VectorCommitment>(
    setup: &HyraxVerifierSetup<VC>,
) -> Result<(), HyraxError> {
    let row_len = setup.dimensions.row_len()?;
    let capacity = VC::capacity(&setup.vc_setup);
    if row_len > capacity {
        return Err(HyraxError::CommitmentCapacityExceeded { capacity, row_len });
    }
    Ok(())
}

fn zero_row_blindings<VC: VectorCommitment>(
    dimensions: &HyraxDimensions,
) -> Result<Vec<VC::Field>, HyraxError> {
    Ok(vec![VC::Field::default(); dimensions.row_count()?])
}

fn assert_valid_dimensions(dimensions: &HyraxDimensions) {
    assert!(dimensions.validate().is_ok(), "invalid Hyrax dimensions");
}

fn assert_valid_poly(dimensions: &HyraxDimensions, poly_num_vars: usize) {
    assert!(
        validate_dimensions_for_poly(dimensions, poly_num_vars).is_ok(),
        "invalid Hyrax polynomial",
    );
}

fn assert_valid_capacity<VC: VectorCommitment>(setup: &HyraxProverSetup<VC>) {
    assert!(
        validate_capacity::<VC>(setup).is_ok(),
        "invalid Hyrax setup",
    );
}

#[cfg(test)]
#[expect(clippy::expect_used, reason = "tests may panic on assertion failures")]
mod tests {
    use jolt_crypto::{Bn254, Bn254G1, Pedersen, PedersenSetup};
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_openings::{AdditivelyHomomorphic, CommitmentScheme};
    use jolt_poly::Polynomial;
    use jolt_transcript::Blake2bTranscript;
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    use super::*;

    type TestHyrax = HyraxScheme<Pedersen<Bn254G1>>;

    fn setup(
        row_vars: usize,
        col_vars: usize,
    ) -> (
        HyraxProverSetup<Pedersen<Bn254G1>>,
        HyraxVerifierSetup<Pedersen<Bn254G1>>,
    ) {
        let mut rng = ChaCha20Rng::seed_from_u64(10 + row_vars as u64 * 100 + col_vars as u64);
        let row_len = 1usize << col_vars;
        let generators = (0..row_len).map(|_| Bn254::random_g1(&mut rng)).collect();
        let blinding_generator = Bn254::random_g1(&mut rng);
        let vc_setup = PedersenSetup::new(generators, blinding_generator);
        let dimensions = HyraxDimensions::new(row_vars + col_vars, row_vars, col_vars)
            .expect("valid dimensions");
        TestHyrax::setup(HyraxSetupParams {
            dimensions,
            vc_setup,
        })
    }

    fn polynomial(num_vars: usize) -> Polynomial<Fr> {
        let evals: Vec<Fr> = (0..(1usize << num_vars))
            .map(|index| Fr::from_u64((index as u64 + 3) * 7))
            .collect();
        Polynomial::from(evals)
    }

    fn point(num_vars: usize) -> Vec<Fr> {
        (0..num_vars)
            .map(|index| Fr::from_u64(index as u64 + 2))
            .collect()
    }

    fn prove_and_verify(row_vars: usize, col_vars: usize) {
        let (prover_setup, verifier_setup) = setup(row_vars, col_vars);
        let poly = polynomial(row_vars + col_vars);
        let point = point(poly.num_vars());
        let eval = poly.evaluate(&point);
        let (commitment, hint) = TestHyrax::commit(&poly, &prover_setup);

        let mut prover_transcript = Blake2bTranscript::new(b"hyrax-test");
        let proof = TestHyrax::open(
            &poly,
            &point,
            eval,
            &prover_setup,
            Some(hint),
            &mut prover_transcript,
        );

        let mut verifier_transcript = Blake2bTranscript::new(b"hyrax-test");
        TestHyrax::verify(
            &commitment,
            &point,
            eval,
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
        )
        .expect("valid opening verifies");
    }

    #[test]
    fn round_trip_small_dimensions() {
        prove_and_verify(0, 1);
        prove_and_verify(1, 1);
        prove_and_verify(2, 3);
        prove_and_verify(3, 2);
    }

    #[test]
    fn wrong_evaluation_rejects() {
        let (prover_setup, verifier_setup) = setup(2, 2);
        let poly = polynomial(4);
        let point = point(4);
        let eval = poly.evaluate(&point);
        let (commitment, hint) = TestHyrax::commit(&poly, &prover_setup);
        let mut prover_transcript = Blake2bTranscript::new(b"hyrax-wrong-eval");
        let proof = TestHyrax::open(
            &poly,
            &point,
            eval,
            &prover_setup,
            Some(hint),
            &mut prover_transcript,
        );

        let mut verifier_transcript = Blake2bTranscript::new(b"hyrax-wrong-eval");
        let err = TestHyrax::verify(
            &commitment,
            &point,
            eval + Fr::from_u64(1),
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
        )
        .expect_err("wrong evaluation must reject");
        assert!(matches!(err, OpeningsError::VerificationFailed));
    }

    #[test]
    fn wrong_point_rejects() {
        let (prover_setup, verifier_setup) = setup(2, 2);
        let poly = polynomial(4);
        let point = point(4);
        let eval = poly.evaluate(&point);
        let (commitment, hint) = TestHyrax::commit(&poly, &prover_setup);
        let mut prover_transcript = Blake2bTranscript::new(b"hyrax-wrong-point");
        let proof = TestHyrax::open(
            &poly,
            &point,
            eval,
            &prover_setup,
            Some(hint),
            &mut prover_transcript,
        );

        let mut wrong_point = point;
        wrong_point[0] += Fr::from_u64(5);
        let mut verifier_transcript = Blake2bTranscript::new(b"hyrax-wrong-point");
        let err = TestHyrax::verify(
            &commitment,
            &wrong_point,
            eval,
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
        )
        .expect_err("wrong point must reject");
        assert!(matches!(err, OpeningsError::VerificationFailed));
    }

    #[test]
    fn wrong_row_commitment_rejects() {
        let (prover_setup, verifier_setup) = setup(2, 2);
        let poly = polynomial(4);
        let point = point(4);
        let eval = poly.evaluate(&point);
        let (mut commitment, hint) = TestHyrax::commit(&poly, &prover_setup);
        commitment.rows[0] += Bn254::g1_generator();
        let mut prover_transcript = Blake2bTranscript::new(b"hyrax-wrong-commitment");
        let proof = TestHyrax::open(
            &poly,
            &point,
            eval,
            &prover_setup,
            Some(hint),
            &mut prover_transcript,
        );

        let mut verifier_transcript = Blake2bTranscript::new(b"hyrax-wrong-commitment");
        let err = TestHyrax::verify(
            &commitment,
            &point,
            eval,
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
        )
        .expect_err("wrong commitment must reject");
        assert!(matches!(err, OpeningsError::VerificationFailed));
    }

    #[test]
    fn additive_homomorphic_combines_rows() {
        let (prover_setup, _) = setup(2, 2);
        let poly_a = polynomial(4);
        let evals_b: Vec<Fr> = (0..poly_a.len())
            .map(|index| Fr::from_u64(index as u64 * 11 + 9))
            .collect();
        let poly_b = Polynomial::from(evals_b);
        let scalar_a = Fr::from_u64(3);
        let scalar_b = Fr::from_u64(5);

        let (commitment_a, ()) = TestHyrax::commit(&poly_a, &prover_setup);
        let (commitment_b, ()) = TestHyrax::commit(&poly_b, &prover_setup);
        let combined = TestHyrax::combine(&[commitment_a, commitment_b], &[scalar_a, scalar_b]);

        let combined_evals: Vec<Fr> = poly_a
            .evaluations()
            .iter()
            .zip(poly_b.evaluations().iter())
            .map(|(left, right)| scalar_a * *left + scalar_b * *right)
            .collect();
        let combined_poly = Polynomial::from(combined_evals);
        let (expected, ()) = TestHyrax::commit(&combined_poly, &prover_setup);
        assert_eq!(combined, expected);
    }

    #[test]
    fn verifier_setup_serializes() {
        let (_, verifier_setup) = setup(2, 2);
        let json = serde_json::to_vec(&verifier_setup).expect("serialize verifier setup");
        let recovered: HyraxVerifierSetup<Pedersen<Bn254G1>> =
            serde_json::from_slice(&json).expect("deserialize verifier setup");
        assert_eq!(recovered.dimensions, verifier_setup.dimensions);
        assert_eq!(
            recovered.vc_setup.message_generators,
            verifier_setup.vc_setup.message_generators,
        );
        assert_eq!(
            recovered.vc_setup.blinding_generator,
            verifier_setup.vc_setup.blinding_generator,
        );
    }

    #[test]
    fn derives_pedersen_hyrax_setup_from_dory_setup() {
        let dory_setup = jolt_dory::DoryScheme::setup_prover(8);
        let dimensions = HyraxDimensions::new(4, 2, 2).expect("valid dimensions");
        let prover_setup =
            HyraxProverSetup::<Pedersen<Bn254G1>>::derive_from(&dory_setup, dimensions)
                .expect("derive Hyrax setup from Dory setup");
        let verifier_setup = TestHyrax::verifier_setup(&prover_setup);

        let poly = polynomial(4);
        let point = point(4);
        let eval = poly.evaluate(&point);
        let (commitment, hint) = TestHyrax::commit(&poly, &prover_setup);

        let mut prover_transcript = Blake2bTranscript::new(b"hyrax-dory-derived");
        let proof = TestHyrax::open(
            &poly,
            &point,
            eval,
            &prover_setup,
            Some(hint),
            &mut prover_transcript,
        );

        let mut verifier_transcript = Blake2bTranscript::new(b"hyrax-dory-derived");
        TestHyrax::verify(
            &commitment,
            &point,
            eval,
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
        )
        .expect("Dory-derived Hyrax setup verifies");
    }
}
