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
use crate::msm::VariableBaseMSM;
use crate::poly::dense_mlpoly::DensePolynomial;
use crate::poly::eq_poly::EqPolynomial;
use crate::poly::multilinear_polynomial::MultilinearPolynomial;
use crate::transcripts::{AppendToTranscript, Transcript};
use crate::utils::errors::ProofVerifyError;
use crate::utils::math::Math;
use crate::utils::{compute_dotproduct, mul_0_1_optimized};
use ark_ec::CurveGroup;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use num_integer::Roots;
use rayon::prelude::*;
use tracing::trace_span;

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

impl<const RATIO: usize, F: JoltField, G: CurveGroup<ScalarField = F>> HyraxCommitment<RATIO, G> {
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
impl<const RATIO: usize, F: JoltField, G: CurveGroup<ScalarField = F>> HyraxOpeningProof<RATIO, G> {
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
            &MultilinearPolynomial::from(L),
        )
        .unwrap();

        let product_commitment = VariableBaseMSM::msm_field_elements(
            &G::normalize_batch(&pedersen_generators.generators[..R_size]),
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
impl<const RATIO: usize, F: JoltField, G: CurveGroup<ScalarField = F>>
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

// A non-impl of StreamingCommitmentScheme
impl<const RATIO: usize, F: JoltField, G: CurveGroup<ScalarField = F>> HyraxCommitment<RATIO, G> {
    // type State<'a> = HyraxCommitmentState<G>;

    pub fn initialize(n: usize, generators: &PedersenGenerators<G>) -> HyraxCommitmentState<G> {
        let ell = n.log_2();

        let (L_size, R_size) = matrix_dimensions(ell, RATIO);
        assert_eq!(L_size * R_size, n);

        let generators = CurveGroup::normalize_batch(&generators.generators[..R_size]);

        let row_commitments = Vec::with_capacity(L_size);
        let current_row = Vec::with_capacity(R_size);

        HyraxCommitmentState {
            row_commitments,
            generators,
            current_row,
            L_size,
            R_size,
        }
    }

    pub fn process(mut state: HyraxCommitmentState<G>, eval: F) -> HyraxCommitmentState<G> {
        state.current_row.push(eval);

        if state.current_row.len() == state.R_size {
            let commitment =
                PedersenCommitment::commit_vector(&state.current_row, &state.generators);
            state.row_commitments.push(commitment);

            state.current_row.clear();
        }

        state
    }

    pub fn finalize(state: HyraxCommitmentState<G>) -> HyraxCommitment<RATIO, G> {
        assert_eq!(
            state.current_row.len(),
            0,
            "Incorrect number of elements processed."
        );
        assert_eq!(
            state.row_commitments.len(),
            state.L_size,
            "Incorrect number of elements processed."
        );

        HyraxCommitment {
            row_commitments: state.row_commitments,
        }
    }
}

pub struct HyraxCommitmentState<G: CurveGroup> {
    row_commitments: Vec<G>,
    generators: Vec<<G as CurveGroup>::Affine>,
    current_row: Vec<G::ScalarField>,
    L_size: usize,
    R_size: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transcripts::Blake2bTranscript;
    use ark_bn254::{Fr, G1Projective};
    use ark_std::test_rng;
    use ark_std::UniformRand;

    #[test]
    fn test_hyrax_batch_pcs_random_polys() {
        const NUM_VARS: usize = 10; // 2^10 = 1024 coefficients per poly
        const NUM_POLYS: usize = 5;
        const RATIO: usize = 1;

        type F = Fr;
        type G = G1Projective;
        type Transcript = Blake2bTranscript;

        let mut rng = test_rng();

        let polys: Vec<DensePolynomial<F>> = (0..NUM_POLYS)
            .map(|_| {
                let coeffs: Vec<F> = (0..1 << NUM_VARS).map(|_| F::rand(&mut rng)).collect();
                DensePolynomial::new(coeffs)
            })
            .collect();

        // Setup generators
        let (L_size, R_size) = matrix_dimensions(NUM_VARS, RATIO);
        let gens = PedersenGenerators::<G>::new(R_size, b"test hyrax batch");

        // Commit to all polynomials
        let poly_refs: Vec<&[F]> = polys.iter().map(|p| p.evals_ref()).collect();
        let commitments = HyraxCommitment::<RATIO, G>::batch_commit(&poly_refs, &gens);

        // Generate random opening point
        let opening_point: Vec<F> = (0..NUM_VARS).map(|_| F::rand(&mut rng)).collect();

        // Compute openings (evaluations at the point)
        let openings: Vec<F> = polys
            .iter()
            .map(|poly| poly.evaluate(&opening_point))
            .collect();

        // Create proof
        let mut prover_transcript = Transcript::new(b"test_batch_hyrax");
        let poly_refs_for_proof: Vec<&DensePolynomial<F>> = polys.iter().collect();
        let proof = BatchedHyraxOpeningProof::<RATIO, G>::prove(
            &poly_refs_for_proof,
            &opening_point,
            &openings,
            &mut prover_transcript,
        );

        // Verify proof
        let mut verifier_transcript = Transcript::new(b"test_batch_hyrax");
        let commitment_refs: Vec<&HyraxCommitment<RATIO, G>> = commitments.iter().collect();
        let result = proof.verify(
            &gens,
            &opening_point,
            &openings,
            &commitment_refs,
            &mut verifier_transcript,
        );

        assert!(result.is_ok(), "Batch Hyrax verification failed");
    }

    #[test]
    fn test_hyrax_batch_pcs_edge_cases() {
        const RATIO: usize = 1;
        type F = Fr;
        type G = G1Projective;
        type Transcript = Blake2bTranscript;

        let mut rng = test_rng();

        // Test with single polynomial
        let num_vars = 4; // 2^4 = 16 coefficients
        let coeffs: Vec<F> = (0..16).map(|_| F::rand(&mut rng)).collect();
        let poly = DensePolynomial::new(coeffs);

        let (_, R_size) = matrix_dimensions(num_vars, RATIO);
        let gens = PedersenGenerators::<G>::new(R_size, b"test single");

        let poly_slice = &[poly.evals_ref()];
        let commitments = HyraxCommitment::<RATIO, G>::batch_commit(poly_slice, &gens);

        let opening_point: Vec<F> = (0..num_vars).map(|_| F::rand(&mut rng)).collect();
        let opening = poly.evaluate(&opening_point);

        let mut prover_transcript = Transcript::new(b"test_single");
        let proof = BatchedHyraxOpeningProof::<RATIO, G>::prove(
            &[&poly],
            &opening_point,
            &[opening],
            &mut prover_transcript,
        );

        let mut verifier_transcript = Transcript::new(b"test_single");
        let result = proof.verify(
            &gens,
            &opening_point,
            &[opening],
            &[&commitments[0]],
            &mut verifier_transcript,
        );

        assert!(result.is_ok(), "Single polynomial verification failed");

        // Test with wrong opening value (should fail)
        let wrong_opening = opening + F::from(1u64);
        let mut verifier_transcript_wrong = Transcript::new(b"test_single");
        let result_wrong = proof.verify(
            &gens,
            &opening_point,
            &[wrong_opening],
            &[&commitments[0]],
            &mut verifier_transcript_wrong,
        );

        assert!(
            result_wrong.is_err(),
            "Verification should fail with wrong opening"
        );
    }

    #[test]
    fn test_hyrax_fp12_multilinear_completeness() {
        use ark_bn254::{Fq, Fq12};
        use ark_grumpkin::{Fr as GrumpkinFr, Projective as GrumpkinProjective};
        use jolt_optimizations::fq12_to_multilinear_evals;

        const RATIO: usize = 1;
        const NUM_FP12_ELEMENTS: usize = 4;
        type G = GrumpkinProjective; // Grumpkin's scalar field is BN254's Fq
        type Transcript = Blake2bTranscript;

        let mut rng = test_rng();

        // Generate random Fp12 elements
        let fp12_elements: Vec<Fq12> = (0..NUM_FP12_ELEMENTS)
            .map(|_| Fq12::rand(&mut rng))
            .collect();

        // Convert each Fp12 to multilinear evaluations (16 Fq elements)
        let multilinear_evals_fq: Vec<Vec<Fq>> = fp12_elements
            .iter()
            .map(|fp12| fq12_to_multilinear_evals(fp12))
            .collect();

        // Create DensePolynomials from the Fq evaluations
        let polys: Vec<DensePolynomial<Fq>> = multilinear_evals_fq
            .iter()
            .map(|evals| DensePolynomial::new(evals.clone()))
            .collect();

        // Setup generators for Hyrax (16 = 2^4 elements per poly)
        let num_vars = 4; // 2^4 = 16
        let (_, R_size) = matrix_dimensions(num_vars, RATIO);
        let gens = PedersenGenerators::<G>::new(R_size, b"test fp12 hyrax");

        // Commit to all polynomials
        let poly_refs: Vec<&[Fq]> = polys.iter().map(|p| p.evals_ref()).collect();
        let commitments = HyraxCommitment::<RATIO, G>::batch_commit(&poly_refs[..], &gens);

        // Generate random opening point (4 variables)
        let opening_point: Vec<Fq> = (0..num_vars).map(|_| Fq::rand(&mut rng)).collect();

        // Compute evaluations at the opening point
        let openings: Vec<Fq> = polys
            .iter()
            .map(|poly| poly.evaluate(&opening_point))
            .collect();

        // Create batched opening proof
        let mut prover_transcript = Transcript::new(b"test_fp12_batch");
        let poly_refs_for_proof: Vec<&DensePolynomial<Fq>> = polys.iter().collect();
        let proof = BatchedHyraxOpeningProof::<RATIO, G>::prove(
            &poly_refs_for_proof[..],
            &opening_point,
            &openings,
            &mut prover_transcript,
        );

        // Verify the proof
        let mut verifier_transcript = Transcript::new(b"test_fp12_batch");
        let commitment_refs: Vec<&HyraxCommitment<RATIO, G>> = commitments.iter().collect();
        let result = proof.verify(
            &gens,
            &opening_point,
            &openings,
            &commitment_refs[..],
            &mut verifier_transcript,
        );

        assert!(result.is_ok(), "Fp12 multilinear Hyrax verification failed");

        assert_ne!(
            commitments[0].row_commitments, commitments[1].row_commitments,
            "Different Fp12 elements should produce different commitments"
        );

        assert_eq!(
            polys[0].len(),
            16,
            "Each Fp12 should produce 16 coefficients"
        );
        assert_ne!(
            polys[0].Z, polys[1].Z,
            "Different Fp12 elements should produce different polynomials"
        );
    }

    #[test]
    fn test_hyrax_fp12_soundness() {
        use ark_bn254::{Fq, Fq12};
        use ark_grumpkin::Projective as GrumpkinProjective;
        use jolt_optimizations::fq12_to_multilinear_evals;

        const RATIO: usize = 1;
        type G = GrumpkinProjective;
        type Transcript = Blake2bTranscript;

        let mut rng = test_rng();

        let fp12_element = Fq12::rand(&mut rng);

        let multilinear_evals = fq12_to_multilinear_evals(&fp12_element);

        // Create polynomial from the evaluations
        let poly = DensePolynomial::new(multilinear_evals.clone());

        // Setup generators for Hyrax
        let num_vars = 4; // 2^4 = 16
        let (_, R_size) = matrix_dimensions(num_vars, RATIO);
        let gens = PedersenGenerators::<G>::new(R_size, b"test cheating");

        // Commit to the polynomial
        let poly_slice = &[poly.evals_ref()];
        let commitments = HyraxCommitment::<RATIO, G>::batch_commit(&poly_slice[..], &gens);

        // Generate random opening point
        let opening_point: Vec<Fq> = (0..num_vars).map(|_| Fq::rand(&mut rng)).collect();

        // Compute the correct evaluation
        let correct_opening = poly.evaluate(&opening_point);

        // Create a proof for the correct opening
        let mut prover_transcript = Transcript::new(b"test_cheating");
        let proof = BatchedHyraxOpeningProof::<RATIO, G>::prove(
            &[&poly],
            &opening_point,
            &[correct_opening],
            &mut prover_transcript,
        );

        // Test 1: Try to verify with a wrong opening value (should fail)
        let wrong_opening = correct_opening + Fq::from(1u64);
        let mut verifier_transcript_wrong = Transcript::new(b"test_cheating");
        let result_wrong_opening = proof.verify(
            &gens,
            &opening_point,
            &[wrong_opening],
            &[&commitments[0]],
            &mut verifier_transcript_wrong,
        );
        assert!(
            result_wrong_opening.is_err(),
            "Verification should fail with wrong opening value"
        );

        // Test 2: Create a cheating polynomial that doesn't come from Fp12
        // but try to use a proof from the real Fp12 polynomial
        let mut cheating_evals = multilinear_evals.clone();
        cheating_evals[12] = Fq::rand(&mut rng);
        cheating_evals[13] = Fq::rand(&mut rng);
        cheating_evals[14] = Fq::rand(&mut rng);
        cheating_evals[15] = Fq::rand(&mut rng);

        let cheating_poly = DensePolynomial::new(cheating_evals);

        // Commit to the cheating polynomial
        let cheating_slice = &[cheating_poly.evals_ref()];
        let cheating_commitments =
            HyraxCommitment::<RATIO, G>::batch_commit(&cheating_slice[..], &gens);

        // The commitment should be different
        assert_ne!(
            commitments[0].row_commitments, cheating_commitments[0].row_commitments,
            "Cheating polynomial should have different commitment"
        );

        // Try to use the proof from the original polynomial with the cheating commitment
        // This should fail because the commitment doesn't match the proof
        let mut verifier_transcript_cheat = Transcript::new(b"test_cheating");
        let result_wrong_commitment = proof.verify(
            &gens,
            &opening_point,
            &[correct_opening],
            &[&cheating_commitments[0]],
            &mut verifier_transcript_cheat,
        );
        assert!(
            result_wrong_commitment.is_err(),
            "Verification should fail when commitment doesn't match the proof"
        );

        // Test 3: Create a proof for a modified Fp12 element but claim it's the original
        let different_fp12 = Fq12::rand(&mut rng);
        let different_evals = fq12_to_multilinear_evals(&different_fp12);
        let different_poly = DensePolynomial::new(different_evals);
        let different_opening = different_poly.evaluate(&opening_point);

        // Create proof for the different polynomial
        let mut prover_transcript_different = Transcript::new(b"test_different");
        let different_proof = BatchedHyraxOpeningProof::<RATIO, G>::prove(
            &[&different_poly],
            &opening_point,
            &[different_opening],
            &mut prover_transcript_different,
        );

        // Try to verify this proof against the original commitment (should fail)
        let mut verifier_transcript_mismatch = Transcript::new(b"test_different");
        let result_mismatch = different_proof.verify(
            &gens,
            &opening_point,
            &[different_opening],
            &[&commitments[0]], // Using original commitment
            &mut verifier_transcript_mismatch,
        );
        assert!(
            result_mismatch.is_err(),
            "Verification should fail when proof doesn't match commitment"
        );

        // Verify that the correct proof still works
        let mut verifier_transcript_correct = Transcript::new(b"test_cheating");
        let result_correct = proof.verify(
            &gens,
            &opening_point,
            &[correct_opening],
            &[&commitments[0]],
            &mut verifier_transcript_correct,
        );
        assert!(result_correct.is_ok(), "Correct proof should still verify");
    }

    #[test]
    fn test_hyrax_fp12_batch_eval_performance() {
        use ark_bn254::{Fq, Fq12};
        use ark_ff::BigInteger;
        use ark_ff::Field;
        use ark_grumpkin::Projective as GrumpkinProjective;
        use jolt_optimizations::fq12_poly::{fq12_to_multilinear_evals, fq12_to_poly12_coeffs};
        use std::time::Instant;
        const RATIO: usize = 1;
        const NUM_FP12_ELEMENTS: usize = 12800;
        type G = GrumpkinProjective;
        type Transcript = Blake2bTranscript;

        let mut rng = test_rng();
        println!("\n=== Hyrax Fq12 Batch Evaluation Performance Test ===");
        println!("Number of Fq12 elements: {}", NUM_FP12_ELEMENTS);

        // Generate random Fq12 elements
        let start = Instant::now();
        let fp12_elements: Vec<Fq12> = (0..NUM_FP12_ELEMENTS)
            .map(|_| Fq12::rand(&mut rng))
            .collect();
        println!(
            "Time to generate {} Fq12 elements: {:?}",
            NUM_FP12_ELEMENTS,
            start.elapsed()
        );

        // Convert each Fq12 to multilinear evaluations
        let start = Instant::now();
        let multilinear_evals_fq: Vec<Vec<Fq>> = fp12_elements
            .iter()
            .map(|fp12| fq12_to_multilinear_evals(fp12))
            .collect();
        println!("Time to convert to multilinear form: {:?}", start.elapsed());

        // Create DensePolynomials
        let start = Instant::now();
        let polys: Vec<DensePolynomial<Fq>> = multilinear_evals_fq
            .iter()
            .map(|evals| DensePolynomial::new(evals.clone()))
            .collect();
        println!("Time to create dense polynomials: {:?}", start.elapsed());

        // Setup generators
        let num_vars = 4; // 2^4 = 16
        let (_, R_size) = matrix_dimensions(num_vars, RATIO);
        let gens = PedersenGenerators::<G>::new(R_size, b"test fp12 batch perf");

        // Commit to each polynomial individually (to avoid thread pool issues)
        let start = Instant::now();
        let commitments: Vec<HyraxCommitment<RATIO, G>> = polys
            .iter()
            .map(|poly| HyraxCommitment::<RATIO, G>::commit(poly, &gens))
            .collect();
        let commit_time = start.elapsed();
        println!(
            "\nTime to commit {} polynomials individually: {:?}",
            NUM_FP12_ELEMENTS, commit_time
        );

        // Generate random opening point
        let opening_point: Vec<Fq> = (0..num_vars).map(|_| Fq::rand(&mut rng)).collect();

        // Compute all evaluations
        let start = Instant::now();
        let openings: Vec<Fq> = polys
            .iter()
            .map(|poly| poly.evaluate(&opening_point))
            .collect();
        let eval_time = start.elapsed();
        println!(
            "Time to evaluate {} polynomials: {:?}",
            NUM_FP12_ELEMENTS, eval_time
        );

        // Create batched opening proof
        let start = Instant::now();
        let mut prover_transcript = Transcript::new(b"test_fp12_batch_perf");
        let poly_ptrs: Vec<&DensePolynomial<Fq>> = polys.iter().collect();
        let proof = BatchedHyraxOpeningProof::<RATIO, G>::prove(
            &poly_ptrs,
            &opening_point,
            &openings,
            &mut prover_transcript,
        );
        let prove_time = start.elapsed();
        println!("\nBatch proof generation time: {:?}", prove_time);

        // Verify the batched proof
        let start = Instant::now();
        let mut verifier_transcript = Transcript::new(b"test_fp12_batch_perf");
        let commitment_refs: Vec<&HyraxCommitment<RATIO, G>> = commitments.iter().collect();
        let result = proof.verify(
            &gens,
            &opening_point,
            &openings,
            &commitment_refs,
            &mut verifier_transcript,
        );
        let verify_time = start.elapsed();
        println!("Batch proof verification time: {:?}", verify_time);

        assert!(result.is_ok(), "Batch proof verification failed");

        // Test soundness: tamper with one evaluation
        let mut tampered_openings = openings.clone();
        let tamper_index = NUM_FP12_ELEMENTS / 2;
        tampered_openings[tamper_index] = tampered_openings[tamper_index] + Fq::from(1u64);

        let mut verifier_transcript_tampered = Transcript::new(b"test_fp12_batch_perf");
        let tampered_result = proof.verify(
            &gens,
            &opening_point,
            &tampered_openings,
            &commitment_refs,
            &mut verifier_transcript_tampered,
        );
        assert!(
            tampered_result.is_err(),
            "Verification should fail when one evaluation is tampered"
        );

        // For comparison, time a single proof (just for reference)
        let start = Instant::now();
        let mut single_transcript = Transcript::new(b"single");
        let single_proof = BatchedHyraxOpeningProof::<RATIO, G>::prove(
            &[&polys[0]],
            &opening_point,
            &[openings[0]],
            &mut single_transcript,
        );
        let single_prove_time = start.elapsed();

        let start = Instant::now();
        let mut single_verifier_transcript = Transcript::new(b"single");
        let single_result = single_proof.verify(
            &gens,
            &opening_point,
            &[openings[0]],
            &[&commitments[0]],
            &mut single_verifier_transcript,
        );
        let single_verify_time = start.elapsed();
        assert!(single_result.is_ok());

        println!("\n=== Performance Comparison ===");
        println!("Single proof generation: {:?}", single_prove_time);
        println!(
            "Batch proof generation ({} polys): {:?}",
            NUM_FP12_ELEMENTS, prove_time
        );
        println!(
            "Batch speedup factor: {:.2}x",
            (single_prove_time.as_nanos() as f64 * NUM_FP12_ELEMENTS as f64)
                / prove_time.as_nanos() as f64
        );

        println!("\nSingle proof verification: {:?}", single_verify_time);
        println!(
            "Batch proof verification ({} polys): {:?}",
            NUM_FP12_ELEMENTS, verify_time
        );
        println!(
            "Verification speedup factor: {:.2}x",
            (single_verify_time.as_nanos() as f64 * NUM_FP12_ELEMENTS as f64)
                / verify_time.as_nanos() as f64
        );

        println!("\n=== Summary ===");
        println!(
            "Successfully proved and verified {} Fq12 polynomial evaluations in a single batch",
            NUM_FP12_ELEMENTS
        );
        println!(
            "Total time (commit + prove + verify): {:?}",
            commit_time + prove_time + verify_time
        );
    }
}
