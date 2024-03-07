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
use num_integer::Roots;
use rayon::prelude::*;

#[cfg(feature = "ark-msm")]
use ark_ec::VariableBaseMSM;

#[cfg(not(feature = "ark-msm"))]
use crate::msm::VariableBaseMSM;

pub fn matrix_dimensions(num_vars: usize, matrix_aspect_ratio: usize) -> (usize, usize) {
    let mut row_size = (num_vars / 2).pow2();
    row_size = (row_size * matrix_aspect_ratio.sqrt()).next_power_of_two();

    let right_num_vars = row_size.log_2();
    let left_num_vars = num_vars - right_num_vars;
    let col_size = left_num_vars.pow2();

    (col_size, row_size)
}

pub struct HyraxGenerators<const RATIO: usize, G: CurveGroup> {
    pub gens: PedersenGenerators<G>,
}

impl<const RATIO: usize, G: CurveGroup> HyraxGenerators<RATIO, G> {
    // the number of variables in the multilinear polynomial
    pub fn new(num_vars: usize, pedersen_generators: &PedersenGenerators<G>) -> Self {
        let (_left, right) = matrix_dimensions(num_vars, RATIO);
        let gens = pedersen_generators.clone_n(right);
        HyraxGenerators { gens }
    }
}

#[derive(Debug)]
pub struct HyraxCommitment<const RATIO: usize, G: CurveGroup> {
    row_commitments: Vec<G>,
}

impl<const RATIO: usize, G: CurveGroup> HyraxCommitment<RATIO, G> {
    #[tracing::instrument(skip_all, name = "HyraxCommitment::commit")]
    pub fn commit(
        poly: &DensePolynomial<G::ScalarField>,
        gens: &HyraxGenerators<RATIO, G>,
    ) -> Self {
        let n = poly.len();
        let ell = poly.get_num_vars();
        assert_eq!(n, ell.pow2());

        let (L_size, R_size) = matrix_dimensions(ell, RATIO);
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

impl<const RATIO: usize, G: CurveGroup> AppendToTranscript<G> for HyraxCommitment<RATIO, G> {
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
pub struct HyraxOpeningProof<const RATIO: usize, G: CurveGroup> {
    vector_matrix_product: Vec<G::ScalarField>,
}

/// See Section 14.3 of Thaler's Proofs, Arguments, and Zero-Knowledge
impl<const RATIO: usize, G: CurveGroup> HyraxOpeningProof<RATIO, G> {
    fn protocol_name() -> &'static [u8] {
        b"Hyrax opening proof"
    }

    #[tracing::instrument(skip_all, name = "HyraxOpeningProof::prove")]
    pub fn prove(
        poly: &DensePolynomial<G::ScalarField>,
        opening_point: &[G::ScalarField], // point at which the polynomial is evaluated
        transcript: &mut Transcript,
    ) -> HyraxOpeningProof<RATIO, G> {
        <Transcript as ProofTranscript<G>>::append_protocol_name(transcript, Self::protocol_name());

        // assert vectors are of the right size
        assert_eq!(poly.get_num_vars(), opening_point.len());

        // compute the L and R vectors
        let (L_size, _R_size) = matrix_dimensions(poly.get_num_vars(), RATIO);
        let eq = EqPolynomial::new(opening_point.to_vec());
        let (L, _R) = eq.compute_factored_evals(L_size);

        // compute vector-matrix product between L and Z viewed as a matrix
        let vector_matrix_product = Self::vector_matrix_product(poly, &L);

        HyraxOpeningProof {
            vector_matrix_product,
        }
    }

    pub fn verify(
        &self,
        gens: &HyraxGenerators<RATIO, G>,
        transcript: &mut Transcript,
        opening_point: &[G::ScalarField], // point at which the polynomial is evaluated
        opening: &G::ScalarField,         // evaluation \widetilde{Z}(r)
        commitment: &HyraxCommitment<RATIO, G>,
    ) -> Result<(), ProofVerifyError> {
        <Transcript as ProofTranscript<G>>::append_protocol_name(transcript, Self::protocol_name());

        // compute L and R
        let (L_size, _R_size) = matrix_dimensions(opening_point.len(), RATIO);
        let eq: EqPolynomial<_> = EqPolynomial::new(opening_point.to_vec());
        let (L, R) = eq.compute_factored_evals(L_size);

        // Verifier-derived commitment to u * a = \prod Com(u_j)^{a_j}
        let homomorphically_derived_commitment: G =
            VariableBaseMSM::msm(&G::normalize_batch(&commitment.row_commitments), &L).unwrap();

        let product_commitment = VariableBaseMSM::msm(
            &G::normalize_batch(&gens.gens.generators),
            &self.vector_matrix_product,
        )
        .unwrap();

        let dot_product = compute_dotproduct(&self.vector_matrix_product, &R);

        if (homomorphically_derived_commitment == product_commitment) && (dot_product == *opening) {
            Ok(())
        } else {
            assert!(false, "hyrax verify error");
            Err(ProofVerifyError::InternalError)
        }
    }

    #[tracing::instrument(skip_all, name = "HyraxOpeningProof::vector_matrix_product")]
    fn vector_matrix_product(
        poly: &DensePolynomial<G::ScalarField>,
        L: &[G::ScalarField],
    ) -> Vec<G::ScalarField> {
        let (_, R_size) = super::hyrax::matrix_dimensions(poly.get_num_vars(), RATIO);

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

#[derive(Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct BatchedHyraxOpeningProof<const RATIO: usize, G: CurveGroup> {
    joint_proof: HyraxOpeningProof<RATIO, G>,
}

/// See Section 16.1 of Thaler's Proofs, Arguments, and Zero-Knowledge
impl<const RATIO: usize, G: CurveGroup> BatchedHyraxOpeningProof<RATIO, G> {
    #[tracing::instrument(skip_all, name = "BatchedHyraxOpeningProof::prove")]
    pub fn prove(
        polynomials: &[&DensePolynomial<G::ScalarField>],
        opening_point: &[G::ScalarField],
        openings: &[G::ScalarField],
        transcript: &mut Transcript,
    ) -> Self {
        <Transcript as ProofTranscript<G>>::append_protocol_name(transcript, Self::protocol_name());

        // append the claimed evaluations to transcript
        <Transcript as ProofTranscript<G>>::append_scalars(transcript, b"evals_ops_val", &openings);

        let rlc_coefficients: Vec<_> = <Transcript as ProofTranscript<G>>::challenge_vector(
            transcript,
            b"challenge_combine_n_to_one",
            polynomials.len(),
        );

        let poly_len = polynomials[0].len();

        let rlc_poly = rlc_coefficients
            .par_iter()
            .zip(polynomials.par_iter())
            .map(|(coeff, poly)| poly.evals_ref().iter().map(|eval| *coeff * eval).collect())
            .reduce(
                || vec![G::ScalarField::zero(); poly_len],
                |running, new| {
                    debug_assert_eq!(running.len(), new.len());
                    running
                        .iter()
                        .zip(new.iter())
                        .map(|(r, n)| *r + n)
                        .collect()
                },
            );

        let joint_proof =
            HyraxOpeningProof::prove(&DensePolynomial::new(rlc_poly), opening_point, transcript);
        Self { joint_proof }
    }

    pub fn verify(
        &self,
        gens: &HyraxGenerators<RATIO, G>,
        opening_point: &[G::ScalarField],
        openings: &[G::ScalarField],
        commitments: &[&HyraxCommitment<RATIO, G>],
        transcript: &mut Transcript,
    ) -> Result<(), ProofVerifyError> {
        let (L_size, _R_size) = matrix_dimensions(opening_point.len(), RATIO);

        <Transcript as ProofTranscript<G>>::append_protocol_name(transcript, Self::protocol_name());

        // append the claimed evaluations to transcript
        <Transcript as ProofTranscript<G>>::append_scalars(transcript, b"evals_ops_val", &openings);

        let rlc_coefficients: Vec<_> = <Transcript as ProofTranscript<G>>::challenge_vector(
            transcript,
            b"challenge_combine_n_to_one",
            openings.len(),
        );

        let rlc_eval = compute_dotproduct(&rlc_coefficients, openings);

        let rlc_commitment = rlc_coefficients
            .par_iter()
            .zip(commitments.par_iter())
            .map(|(coeff, commitment)| {
                commitment
                    .row_commitments
                    .iter()
                    .map(|commitment| *commitment * coeff)
                    .collect()
            })
            .reduce(
                || vec![G::zero(); L_size],
                |running, new| {
                    debug_assert_eq!(running.len(), new.len());
                    running
                        .iter()
                        .zip(new.iter())
                        .map(|(r, n)| *r + n)
                        .collect()
                },
            );

        self.joint_proof.verify(
            gens,
            transcript,
            opening_point,
            &rlc_eval,
            &HyraxCommitment {
                row_commitments: rlc_commitment,
            },
        )
    }

    fn protocol_name() -> &'static [u8] {
        b"Jolt BatchedHyraxOpeningProof"
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
        let gens = HyraxGenerators::<1, G>::new(poly.get_num_vars(), &pedersen_generators);
        let poly_commitment = HyraxCommitment::commit(&poly, &gens);

        let mut prover_transcript = Transcript::new(b"example");
        let proof = HyraxOpeningProof::prove(&poly, &r, &mut prover_transcript);

        let mut verifier_transcript = Transcript::new(b"example");

        assert!(proof
            .verify(&gens, &mut verifier_transcript, &r, &eval, &poly_commitment)
            .is_ok());
    }
}
