//! This file implements the Hyrax polynomial commitment scheme.
//! Note that we choose not to implement the `CommitmentScheme` trait, as
//! Hyrax is not intended to be used as the core PCS for Jolt itself (in fact,
//! there are some incompatibilities with the batched opening proof protocol
//! used in Jolt).
//!
//! Instead, Hyrax will be used as the PCS in Spartan to prove the verification
//! of a Jolt proof (i.e. SNARK composition).
use super::pedersen::{PedersenCommitment, PedersenGenerators};
use crate::field::JoltField;
use crate::poly::dense_mlpoly::DensePolynomial;
use crate::poly::eq_poly::EqPolynomial;
use crate::poly::multilinear_polynomial::MultilinearPolynomial;
use crate::utils::errors::ProofVerifyError;
use crate::utils::math::Math;
use crate::utils::transcript::{AppendToTranscript, Transcript};
use crate::utils::{compute_dotproduct, mul_0_1_optimized};
use ark_ec::CurveGroup;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use num_integer::Roots;
use rayon::prelude::*;
use tracing::trace_span;

use crate::msm::{Icicle, VariableBaseMSM};

/// Hyrax commits to a multilinear polynomial by interpreting its coefficients as a
/// matrix. Given the number of variables in the polynomial, and the desired "aspect
/// ratio", returns the column and row size of that matrix.
pub fn matrix_dimensions(num_vars: usize, matrix_aspect_ratio: usize) -> (usize, usize) {
    let mut row_size = (num_vars / 2).pow2();
    row_size = (row_size * matrix_aspect_ratio.sqrt()).next_power_of_two();

    let right_num_vars = std::cmp::min(row_size.log_2(), num_vars - 1);
    row_size = right_num_vars.pow2();
    let left_num_vars = num_vars - right_num_vars;
    let col_size = left_num_vars.pow2();

    (col_size, row_size)
}

#[derive(Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct HyraxGenerators<const RATIO: usize, G: CurveGroup> {
    pub gens: PedersenGenerators<G>,
}

impl<const RATIO: usize, G: CurveGroup> HyraxGenerators<RATIO, G> {
    pub fn new(num_vars: usize) -> Self {
        let (_left, right) = matrix_dimensions(num_vars, RATIO);
        let gens = PedersenGenerators::new(right, b"Jolt v1 Hyrax generators");
        HyraxGenerators { gens }
    }
}

#[derive(Default, Clone, Debug, PartialEq, CanonicalSerialize, CanonicalDeserialize)]
pub struct HyraxCommitment<const RATIO: usize, G: CurveGroup> {
    pub row_commitments: Vec<G>,
}

impl<const RATIO: usize, F: JoltField, G: CurveGroup<ScalarField = F> + Icicle>
    HyraxCommitment<RATIO, G>
{
    #[tracing::instrument(skip_all, name = "HyraxCommitment::commit")]
    pub fn commit(
        poly: &DensePolynomial<G::ScalarField>,
        generators: &PedersenGenerators<G>,
    ) -> Self {
        let n = poly.len();
        let ell = n.log_2();

        let (L_size, R_size) = matrix_dimensions(ell, 1);
        assert_eq!(L_size * R_size, n);

        let gens = CurveGroup::normalize_batch(&generators.generators[..R_size]);
        let row_commitments = poly
            .Z
            .par_chunks(R_size)
            .map(|row| PedersenCommitment::commit_vector(row, &gens))
            .collect();
        Self { row_commitments }
    }

    /// Same result as committing to each polynomial in the batch individually,
    /// but tends to have better parallelism.
    #[tracing::instrument(skip_all, name = "HyraxCommitment::batch_commit")]
    pub fn batch_commit(
        batch: &[&[G::ScalarField]],
        generators: &PedersenGenerators<G>,
    ) -> Vec<Self> {
        let n = batch[0].len();
        batch.iter().for_each(|poly| assert_eq!(poly.len(), n));
        let ell = n.log_2();

        let (L_size, R_size) = matrix_dimensions(ell, RATIO);
        assert_eq!(L_size * R_size, n);

        let gens = CurveGroup::normalize_batch(&generators.generators[..R_size]);

        let rows = batch.par_iter().flat_map(|poly| poly.par_chunks(R_size));
        let row_commitments: Vec<G> = rows
            .map(|row| PedersenCommitment::commit_vector(row, &gens))
            .collect();

        row_commitments
            .par_chunks(L_size)
            .map(|chunk| Self {
                row_commitments: chunk.to_vec(),
            })
            .collect()
    }
}

impl<const RATIO: usize, G: CurveGroup> AppendToTranscript for HyraxCommitment<RATIO, G> {
    fn append_to_transcript<ProofTranscript: Transcript>(&self, transcript: &mut ProofTranscript) {
        for i in 0..self.row_commitments.len() {
            transcript.append_point(&self.row_commitments[i]);
        }
    }
}

/// A Hyrax opening proof for a single polynomial opened at a single point.
#[derive(Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct HyraxOpeningProof<const RATIO: usize, G: CurveGroup> {
    pub vector_matrix_product: Vec<G::ScalarField>,
}

/// See Section 14.3 of Thaler's Proofs, Arguments, and Zero-Knowledge
impl<const RATIO: usize, F: JoltField, G: CurveGroup<ScalarField = F> + Icicle>
    HyraxOpeningProof<RATIO, G>
{
    #[tracing::instrument(skip_all, name = "HyraxOpeningProof::prove")]
    pub fn prove(
        poly: &DensePolynomial<G::ScalarField>,
        opening_point: &[G::ScalarField], // point at which the polynomial is evaluated
        ratio: usize,
    ) -> HyraxOpeningProof<RATIO, G> {
        // assert vectors are of the right size
        assert_eq!(poly.get_num_vars(), opening_point.len());

        // compute the L and R vectors
        let (L_size, _R_size) = matrix_dimensions(poly.get_num_vars(), ratio);
        let L = EqPolynomial::evals(&opening_point[..L_size.log_2()]);

        // compute vector-matrix product between L and Z viewed as a matrix
        let vector_matrix_product = Self::vector_matrix_product(poly, &L, ratio);

        HyraxOpeningProof {
            vector_matrix_product,
        }
    }

    pub fn verify(
        &self,
        pedersen_generators: &PedersenGenerators<G>,
        opening_point: &[G::ScalarField], // point at which the polynomial is evaluated
        opening: &G::ScalarField,         // evaluation \widetilde{Z}(r)
        commitment: &HyraxCommitment<RATIO, G>,
    ) -> Result<(), ProofVerifyError> {
        // compute L and R
        let (L_size, R_size) = matrix_dimensions(opening_point.len(), RATIO);
        let L = EqPolynomial::evals(&opening_point[..L_size.log_2()]);
        let R = EqPolynomial::evals(&opening_point[L_size.log_2()..]);

        // Verifier-derived commitment to u * a = \prod Com(u_j)^{a_j}
        let homomorphically_derived_commitment: G = VariableBaseMSM::msm(
            &G::normalize_batch(&commitment.row_commitments),
            None,
            &MultilinearPolynomial::from(L),
            None,
        )
        .unwrap();

        let product_commitment = VariableBaseMSM::msm_field_elements(
            &G::normalize_batch(&pedersen_generators.generators[..R_size]),
            None,
            &self.vector_matrix_product,
            None,
            false,
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
        ratio: usize,
    ) -> Vec<G::ScalarField> {
        let (_, R_size) = matrix_dimensions(poly.get_num_vars(), ratio);

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

/// A Hyrax opening proof for multiple polynomials opened at the same point.
#[derive(Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct BatchedHyraxOpeningProof<const RATIO: usize, G: CurveGroup> {
    pub joint_proof: HyraxOpeningProof<RATIO, G>,
}

/// See Section 16.1 of Thaler's Proofs, Arguments, and Zero-Knowledge
impl<const RATIO: usize, F: JoltField, G: CurveGroup<ScalarField = F> + Icicle>
    BatchedHyraxOpeningProof<RATIO, G>
{
    #[tracing::instrument(skip_all, name = "BatchedHyraxOpeningProof::prove")]
    pub fn prove<ProofTranscript: Transcript>(
        polynomials: &[&DensePolynomial<G::ScalarField>],
        opening_point: &[G::ScalarField],
        openings: &[G::ScalarField],
        transcript: &mut ProofTranscript,
    ) -> Self {
        // append the claimed evaluations to transcript
        transcript.append_scalars(openings);

        let rlc_coefficients: Vec<_> = transcript.challenge_vector(polynomials.len());

        let _span = trace_span!("Compute RLC of polynomials");
        let _enter = _span.enter();

        let poly_len = polynomials[0].len();

        let num_chunks = rayon::current_num_threads().next_power_of_two();
        let chunk_size = poly_len / num_chunks;

        let rlc_poly = if chunk_size > 0 {
            (0..num_chunks)
                .into_par_iter()
                .flat_map_iter(|chunk_index| {
                    let mut chunk = vec![G::ScalarField::zero(); chunk_size];
                    for (coeff, poly) in rlc_coefficients.iter().zip(polynomials.iter()) {
                        for (rlc, poly_eval) in chunk
                            .iter_mut()
                            .zip(poly.evals_ref()[chunk_index * chunk_size..].iter())
                        {
                            *rlc += mul_0_1_optimized(poly_eval, coeff);
                        }
                    }
                    chunk
                })
                .collect::<Vec<_>>()
        } else {
            rlc_coefficients
                .par_iter()
                .zip(polynomials.par_iter())
                .map(|(coeff, poly)| poly.evals_ref().iter().map(|eval| *coeff * *eval).collect())
                .reduce(
                    || vec![G::ScalarField::zero(); poly_len],
                    |running, new| {
                        debug_assert_eq!(running.len(), new.len());
                        running
                            .iter()
                            .zip(new.iter())
                            .map(|(r, n)| *r + *n)
                            .collect()
                    },
                )
        };

        drop(_enter);
        drop(_span);

        let joint_proof =
            HyraxOpeningProof::prove(&DensePolynomial::new(rlc_poly), opening_point, RATIO);

        Self { joint_proof }
    }

    #[tracing::instrument(skip_all, name = "BatchedHyraxOpeningProof::verify")]
    pub fn verify<ProofTranscript: Transcript>(
        &self,
        pedersen_generators: &PedersenGenerators<G>,
        opening_point: &[G::ScalarField],
        openings: &[G::ScalarField],
        commitments: &[&HyraxCommitment<RATIO, G>],
        transcript: &mut ProofTranscript,
    ) -> Result<(), ProofVerifyError> {
        assert_eq!(openings.len(), commitments.len());
        let (L_size, _R_size) = matrix_dimensions(opening_point.len(), RATIO);
        commitments.iter().enumerate().for_each(|(i, commitment)| {
            assert_eq!(
                L_size,
                commitment.row_commitments.len(),
                "Row commitment {}/{} wrong length.",
                i,
                commitments.len()
            )
        });

        // append the claimed evaluations to transcript
        transcript.append_scalars(openings);

        let rlc_coefficients: Vec<_> = transcript.challenge_vector(openings.len());

        let rlc_eval = compute_dotproduct(&rlc_coefficients, openings);

        let rlc_commitment = rlc_coefficients
            .par_iter()
            .zip(commitments.par_iter())
            .map(|(coeff, commitment)| {
                commitment
                    .row_commitments
                    .iter()
                    .map(|row_commitment| *row_commitment * coeff)
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
            pedersen_generators,
            opening_point,
            &rlc_eval,
            &HyraxCommitment {
                row_commitments: rlc_commitment,
            },
        )
    }
}
