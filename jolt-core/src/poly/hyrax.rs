use super::dense_mlpoly::DensePolynomial;
use super::pedersen::{PedersenCommitment, PedersenGenerators};
use crate::poly::eq_poly::EqPolynomial;
use crate::utils::errors::ProofVerifyError;
use crate::utils::math::Math;
use crate::utils::transcript::{AppendToTranscript, ProofTranscript};
use crate::utils::{compute_dotproduct, mul_0_1_optimized};
use ark_ec::CurveGroup;
use ark_serialize::*;
use ark_std::{One, Zero};
use merlin::Transcript;
use rayon::prelude::*;

#[cfg(feature = "ark-msm")]
use ark_ec::VariableBaseMSM;

#[cfg(not(feature = "ark-msm"))]
use crate::msm::VariableBaseMSM;

pub fn matrix_dimensions(num_vars: usize) -> (usize, usize) {
    (num_vars / 2, num_vars - num_vars / 2)
}

pub struct HyraxGenerators<G: CurveGroup> {
    pub gens: PedersenGenerators<G>,
}

impl<G: CurveGroup> HyraxGenerators<G> {
    // the number of variables in the multilinear polynomial
    pub fn new(num_vars: usize, pedersen_generators: &PedersenGenerators<G>) -> Self {
        let (_left, right) = matrix_dimensions(num_vars);
        let gens = pedersen_generators.clone_n(right.pow2());
        HyraxGenerators { gens }
    }
}

pub struct HyraxCommitment<G: CurveGroup> {
    row_commitments: Vec<G>,
}

impl<G: CurveGroup> HyraxCommitment<G> {
    #[tracing::instrument(skip_all, name = "HyraxCommitment::commit")]
    pub fn commit(poly: &DensePolynomial<G::ScalarField>, gens: &HyraxGenerators<G>) -> Self {
        let n = poly.len();
        let ell = poly.get_num_vars();
        assert_eq!(n, ell.pow2());

        let (left_num_vars, right_num_vars) = matrix_dimensions(ell);
        let L_size = left_num_vars.pow2();
        let R_size = right_num_vars.pow2();
        assert_eq!(L_size * R_size, n);

        let gens = CurveGroup::normalize_batch(&gens.gens.generators);
        let row_commitments = poly
            .evals_ref()
            .par_chunks(R_size)
            .map(|row| PedersenCommitment::commit_vector(row, &gens))
            .collect();
        Self { row_commitments }
    }
}

impl<G: CurveGroup> AppendToTranscript<G> for HyraxCommitment<G> {
    fn append_to_transcript<T: ProofTranscript<G>>(
        &self,
        label: &'static [u8],
        transcript: &mut T,
    ) {
        transcript.append_message(label, b"poly_commitment_begin");
        for i in 0..self.row_commitments.len() {
            transcript.append_point(b"poly_commitment_share", &self.row_commitments[i]);
        }
        transcript.append_message(label, b"poly_commitment_end");
    }
}

#[derive(Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct HyraxOpeningProof<G: CurveGroup> {
    vector_matrix_product: Vec<G::ScalarField>,
}

/// See Section 14.3 of Thaler's Proofs, Arguments, and Zero-Knowledge
impl<G: CurveGroup> HyraxOpeningProof<G> {
    fn protocol_name() -> &'static [u8] {
        b"Hyrax opening proof"
    }

    #[tracing::instrument(skip_all, name = "HyraxOpeningProof::prove")]
    pub fn prove(
        poly: &DensePolynomial<G::ScalarField>,
        opening_point: &[G::ScalarField], // point at which the polynomial is evaluated
        transcript: &mut Transcript,
    ) -> HyraxOpeningProof<G> {
        <Transcript as ProofTranscript<G>>::append_protocol_name(
            transcript,
            HyraxOpeningProof::<G>::protocol_name(),
        );

        // assert vectors are of the right size
        assert_eq!(poly.get_num_vars(), opening_point.len());

        let (left_num_vars, right_num_vars) = matrix_dimensions(opening_point.len());
        let L_size = left_num_vars.pow2();
        let R_size = right_num_vars.pow2();

        // compute the L and R vectors
        let eq = EqPolynomial::new(opening_point.to_vec());
        let (L, R) = eq.compute_factored_evals();
        assert_eq!(L.len(), L_size);
        assert_eq!(R.len(), R_size);

        // compute vector-matrix product between L and Z viewed as a matrix
        let vector_matrix_product = Self::vector_matrix_product(poly, &L);

        HyraxOpeningProof {
            vector_matrix_product,
        }
    }

    pub fn verify(
        &self,
        gens: &HyraxGenerators<G>,
        transcript: &mut Transcript,
        opening_point: &[G::ScalarField], // point at which the polynomial is evaluated
        opening: &G::ScalarField,         // evaluation \widetilde{Z}(r)
        commitment: &HyraxCommitment<G>,
    ) -> Result<(), ProofVerifyError> {
        <Transcript as ProofTranscript<G>>::append_protocol_name(
            transcript,
            HyraxOpeningProof::<G>::protocol_name(),
        );

        // compute L and R
        let eq: EqPolynomial<_> = EqPolynomial::new(opening_point.to_vec());
        let (L, R) = eq.compute_factored_evals();

        // Verifier-derived commitment to u * a = \prod Com(u_j)^{a_j}
        let homomorphically_derived_commitment: G =
            VariableBaseMSM::msm_u64(&G::normalize_batch(&commitment.row_commitments), &L).unwrap();

        let product_commitment = VariableBaseMSM::msm_u64(
            &G::normalize_batch(&gens.gens.generators),
            &self.vector_matrix_product,
        )
        .unwrap();

        let dot_product = compute_dotproduct(&self.vector_matrix_product, &R);

        if (homomorphically_derived_commitment == product_commitment) && (dot_product == *opening) {
            Ok(())
        } else {
            Err(ProofVerifyError::InternalError)
        }
    }

    #[tracing::instrument(skip_all, name = "HyraxOpeningProof::vector_matrix_product")]
    fn vector_matrix_product(
        poly: &DensePolynomial<G::ScalarField>,
        L: &[G::ScalarField],
    ) -> Vec<G::ScalarField> {
        let (_left_num_vars, right_num_vars) = super::hyrax::matrix_dimensions(poly.get_num_vars());
        let R_size = right_num_vars.pow2();

        poly.evals_ref()
            .par_chunks(R_size)
            .enumerate()
            .map(|(i, row)| {
                row.iter()
                    .map(|x| mul_0_1_optimized(&L[i], x))
                    .collect::<Vec<G::ScalarField>>()
            })
            .reduce(
                || vec![G::ScalarField::zero(); R_size],
                |mut acc: Vec<_>, row| {
                    acc.iter_mut().zip(row).for_each(|(x, y)| *x += y);
                    acc
                },
            )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_curve25519::EdwardsProjective as G1Projective;
    use ark_std::One;

    #[test]
    fn check_polynomial_commit() {
        check_polynomial_commit_helper::<G1Projective>()
    }

    fn check_polynomial_commit_helper<G: CurveGroup>() {
        let Z = vec![
            G::ScalarField::one(),
            G::ScalarField::from(2u64),
            G::ScalarField::one(),
            G::ScalarField::from(4u64),
        ];
        let poly = DensePolynomial::new(Z);

        // r = [4,3]
        let r = vec![G::ScalarField::from(4u64), G::ScalarField::from(3u64)];
        let eval = poly.evaluate(&r);
        assert_eq!(eval, G::ScalarField::from(28u64));

        let pedersen_generators = PedersenGenerators::new(1 << 8, b"test-two");
        let gens = HyraxGenerators::<G>::new(poly.get_num_vars(), &pedersen_generators);
        let poly_commitment = HyraxCommitment::commit(&poly, &gens);

        let mut prover_transcript = Transcript::new(b"example");
        let proof = HyraxOpeningProof::prove(&poly, &r, &mut prover_transcript);

        let mut verifier_transcript = Transcript::new(b"example");

        assert!(proof
            .verify(&gens, &mut verifier_transcript, &r, &eval, &poly_commitment)
            .is_ok());
    }
}
