use std::marker::PhantomData;

use crate::poly::dense_mlpoly::DensePolynomial;
use super::commitment_scheme::CommitmentScheme;
use super::pedersen::{PedersenCommitment, PedersenGenerators};
use crate::poly::eq_poly::EqPolynomial;
use crate::utils::errors::ProofVerifyError;
use crate::utils::math::Math;
use crate::utils::transcript::{AppendToTranscript, ProofTranscript};
use crate::utils::{compute_dotproduct, mul_0_1_optimized};
use ark_ec::CurveGroup;
use ark_ff::PrimeField;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::Zero;
use common::constants::NUM_R1CS_POLYS;
use num_integer::Roots;
use rayon::prelude::*;
use tracing::trace_span;

use crate::msm::VariableBaseMSM;

#[derive(Clone)]
pub struct HyraxScheme<G: CurveGroup> {
    marker: PhantomData<G>,
}

pub fn matrix_dimensions(num_vars: usize, matrix_aspect_ratio: usize) -> (usize, usize) {
    let mut row_size = (num_vars / 2).pow2();
    row_size = (row_size * matrix_aspect_ratio.sqrt()).next_power_of_two();

    let right_num_vars = std::cmp::min(row_size.log_2(), num_vars - 1);
    row_size = right_num_vars.pow2();
    let left_num_vars = num_vars - right_num_vars;
    let col_size = left_num_vars.pow2();

    (col_size, row_size)
}

// impl<F: JoltField, G: CurveGroup<ScalarField = F>> CommitmentScheme for HyraxScheme<G> {
impl<F: PrimeField, G: CurveGroup<ScalarField = F>> CommitmentScheme for HyraxScheme<G> {
    type Field = G::ScalarField; // TODO(sragss): include? or seperate field config?
    type Generators = PedersenGenerators<G>;
    type Commitment = HyraxCommitment<NUM_R1CS_POLYS, G>;
    type Proof = HyraxOpeningProof<NUM_R1CS_POLYS, G>;
    type BatchedProof = BatchedHyraxOpeningProof<NUM_R1CS_POLYS, G>;

    fn generators(max_commit_size: usize) -> Self::Generators {
        PedersenGenerators::new(max_commit_size, b"Jolt v1 Hyrax generators")
    }
    fn commit(poly: &DensePolynomial<Self::Field>, gens: &Self::Generators) -> Self::Commitment {
        HyraxCommitment::commit(poly, gens)
    }
    fn batch_commit(
        evals: &Vec<Vec<Self::Field>>,
        gens: &Self::Generators,
    ) -> Vec<Self::Commitment> {
        HyraxCommitment::batch_commit(evals, gens)
    }
    fn commit_slice(eval_slice: &[Self::Field], generators: &Self::Generators) -> Self::Commitment {
        HyraxCommitment::commit_slice(eval_slice, generators)
    }
    fn batch_commit_polys(
        polys: &Vec<DensePolynomial<Self::Field>>,
        generators: &Self::Generators,
    ) -> Vec<Self::Commitment> {
        let num_vars = polys[0].get_num_vars();
        let n = num_vars.pow2();
        polys
            .iter()
            .for_each(|poly| assert_eq!(poly.as_ref().len(), n));

        let (L_size, R_size) = matrix_dimensions(num_vars, NUM_R1CS_POLYS);
        assert_eq!(L_size * R_size, n);

        let gens = CurveGroup::normalize_batch(&generators.generators[..R_size]);

        let rows = polys
            .par_iter()
            .flat_map(|poly| poly.evals_ref().par_chunks(R_size));
        let row_commitments: Vec<G> = rows
            .map(|row| PedersenCommitment::commit_vector(row, &gens))
            .collect();

        row_commitments
            .par_chunks(L_size)
            .map(|chunk| HyraxCommitment {
                row_commitments: chunk.to_vec(),
            })
            .collect()
    }
    fn batch_commit_polys_ref(
        polys: &Vec<&DensePolynomial<Self::Field>>,
        generators: &Self::Generators,
    ) -> Vec<Self::Commitment> {
        let num_vars = polys[0].get_num_vars();
        let n = num_vars.pow2();
        polys
            .iter()
            .for_each(|poly| assert_eq!(poly.as_ref().len(), n));

        let (L_size, R_size) = matrix_dimensions(num_vars, NUM_R1CS_POLYS);
        assert_eq!(L_size * R_size, n);

        let gens = CurveGroup::normalize_batch(&generators.generators[..R_size]);

        let rows = polys
            .par_iter()
            .flat_map(|poly| poly.evals_ref().par_chunks(R_size));
        let row_commitments: Vec<G> = rows
            .map(|row| PedersenCommitment::commit_vector(row, &gens))
            .collect();

        row_commitments
            .par_chunks(L_size)
            .map(|chunk| HyraxCommitment {
                row_commitments: chunk.to_vec(),
            })
            .collect()
    }
    fn prove(
        poly: &DensePolynomial<Self::Field>,
        opening_point: &[Self::Field], // point at which the polynomial is evaluated
        transcript: &mut ProofTranscript,
    ) -> Self::Proof {
        HyraxOpeningProof::prove(poly, opening_point, transcript)
    }
    fn batch_prove(
        polynomials: &[&DensePolynomial<Self::Field>],
        opening_point: &[Self::Field],
        openings: &[Self::Field],
        transcript: &mut ProofTranscript,
    ) -> Self::BatchedProof {
        BatchedHyraxOpeningProof::prove(polynomials, opening_point, openings, transcript)
    }
    fn verify(
        proof: &Self::Proof,
        generators: &Self::Generators,
        transcript: &mut ProofTranscript,
        opening_point: &[Self::Field], // point at which the polynomial is evaluated
        opening: &Self::Field,         // evaluation \widetilde{Z}(r)
        commitment: &Self::Commitment,
    ) -> Result<(), ProofVerifyError> {
        HyraxOpeningProof::verify(
            proof,
            generators,
            transcript,
            opening_point,
            opening,
            commitment,
        )
    }
    fn batch_verify(
        batch_proof: &Self::BatchedProof,
        generators: &Self::Generators,
        opening_point: &[Self::Field],
        openings: &[Self::Field],
        commitments: &[&Self::Commitment],
        transcript: &mut ProofTranscript,
    ) -> Result<(), ProofVerifyError> {
        BatchedHyraxOpeningProof::verify(
            batch_proof,
            generators,
            opening_point,
            openings,
            commitments,
            transcript,
        )
    }
    fn protocol_name() -> &'static [u8] {
        b"Jolt BatchedHyraxOpeningProof"
    }
}

impl<G: CurveGroup> HyraxScheme<G> {
    #[tracing::instrument(skip_all, name = "HyraxOpeningProof::vector_matrix_product")]
    fn vector_matrix_product(
        poly: &DensePolynomial<G::ScalarField>,
        L: &[G::ScalarField],
    ) -> Vec<G::ScalarField> {
        let (_, R_size) = matrix_dimensions(poly.get_num_vars(), NUM_R1CS_POLYS);

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

#[derive(Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct HyraxGenerators<const RATIO: usize, G: CurveGroup> {
    pub gens: PedersenGenerators<G>,
}

#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct HyraxCommitment<const RATIO: usize, G: CurveGroup> {
    pub row_commitments: Vec<G>,
}

impl<const RATIO: usize, G: CurveGroup> HyraxCommitment<RATIO, G> {
    #[tracing::instrument(skip_all, name = "HyraxCommitment::commit")]
    pub fn commit(
        poly: &DensePolynomial<G::ScalarField>,
        generators: &PedersenGenerators<G>,
    ) -> Self {
        let n = poly.len();
        let ell = poly.get_num_vars();
        assert_eq!(n, ell.pow2());

        Self::commit_slice(poly.evals_ref(), generators)
    }

    #[tracing::instrument(skip_all, name = "HyraxCommitment::commit_slice")]
    pub fn commit_slice(eval_slice: &[G::ScalarField], generators: &PedersenGenerators<G>) -> Self {
        let n = eval_slice.len();
        let ell = n.log_2();

        let (L_size, R_size) = matrix_dimensions(ell, RATIO);
        assert_eq!(L_size * R_size, n);
        println!("NUM_R1CS_POLYS {}", NUM_R1CS_POLYS);
        println!("RATIO          {}", RATIO);

        let gens = CurveGroup::normalize_batch(&generators.generators[..R_size]);
        let row_commitments = eval_slice
            .par_chunks(R_size)
            .map(|row| PedersenCommitment::commit_vector(row, &gens))
            .collect();
        Self { row_commitments }
    }

    #[tracing::instrument(skip_all, name = "HyraxCommitment::batch_commit")]
    pub fn batch_commit(
        batch: &Vec<Vec<G::ScalarField>>,
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

    #[tracing::instrument(skip_all, name = "HyraxCommitment::batch_commit_polys")]
    pub fn batch_commit_polys(
        polys: Vec<&DensePolynomial<G::ScalarField>>,
        generators: &PedersenGenerators<G>,
    ) -> Vec<Self> {
        let num_vars = polys[0].get_num_vars();
        let n = num_vars.pow2();
        polys
            .iter()
            .for_each(|poly| assert_eq!(poly.as_ref().len(), n));

        let (L_size, R_size) = matrix_dimensions(num_vars, RATIO);
        assert_eq!(L_size * R_size, n);

        let gens = CurveGroup::normalize_batch(&generators.generators[..R_size]);

        let rows = polys
            .par_iter()
            .flat_map(|poly| poly.evals_ref().par_chunks(R_size));
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
    fn append_to_transcript(&self, label: &'static [u8], transcript: &mut ProofTranscript) {
        transcript.append_message(label, b"poly_commitment_begin");
        for i in 0..self.row_commitments.len() {
            transcript.append_point(b"poly_commitment_share", &self.row_commitments[i]);
        }
        transcript.append_message(label, b"poly_commitment_end");
    }
}

#[derive(Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct HyraxOpeningProof<const RATIO: usize, G: CurveGroup> {
    pub vector_matrix_product: Vec<G::ScalarField>,
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
        transcript: &mut ProofTranscript,
    ) -> HyraxOpeningProof<RATIO, G> {
        transcript.append_protocol_name(Self::protocol_name());

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
        pedersen_generators: &PedersenGenerators<G>,
        transcript: &mut ProofTranscript,
        opening_point: &[G::ScalarField], // point at which the polynomial is evaluated
        opening: &G::ScalarField,         // evaluation \widetilde{Z}(r)
        commitment: &HyraxCommitment<RATIO, G>,
    ) -> Result<(), ProofVerifyError> {
        transcript.append_protocol_name(Self::protocol_name());

        // compute L and R
        let (L_size, R_size) = matrix_dimensions(opening_point.len(), RATIO);
        let eq: EqPolynomial<_> = EqPolynomial::new(opening_point.to_vec());
        let (L, R) = eq.compute_factored_evals(L_size);

        // Verifier-derived commitment to u * a = \prod Com(u_j)^{a_j}
        let homomorphically_derived_commitment: G =
            VariableBaseMSM::msm(&G::normalize_batch(&commitment.row_commitments), &L).unwrap();

        let product_commitment = VariableBaseMSM::msm(
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
    ) -> Vec<G::ScalarField> {
        let (_, R_size) = matrix_dimensions(poly.get_num_vars(), RATIO);

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
    pub joint_proof: HyraxOpeningProof<RATIO, G>,
}

/// See Section 16.1 of Thaler's Proofs, Arguments, and Zero-Knowledge
impl<const RATIO: usize, G: CurveGroup> BatchedHyraxOpeningProof<RATIO, G> {
    #[tracing::instrument(skip_all, name = "BatchedHyraxOpeningProof::prove")]
    pub fn prove(
        polynomials: &[&DensePolynomial<G::ScalarField>],
        opening_point: &[G::ScalarField],
        openings: &[G::ScalarField],
        transcript: &mut ProofTranscript,
    ) -> Self {
        transcript.append_protocol_name(Self::protocol_name());

        // append the claimed evaluations to transcript
        transcript.append_scalars(b"evals_ops_val", openings);

        let rlc_coefficients: Vec<_> =
            transcript.challenge_vector(b"challenge_combine_n_to_one", polynomials.len());

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
                )
        };

        drop(_enter);
        drop(_span);

        let joint_proof =
            HyraxOpeningProof::prove(&DensePolynomial::new(rlc_poly), opening_point, transcript);
        Self { joint_proof }
    }

    pub fn verify(
        &self,
        pedersen_generators: &PedersenGenerators<G>,
        opening_point: &[G::ScalarField],
        openings: &[G::ScalarField],
        commitments: &[&HyraxCommitment<RATIO, G>],
        transcript: &mut ProofTranscript,
    ) -> Result<(), ProofVerifyError> {
        let (L_size, _R_size) = matrix_dimensions(opening_point.len(), RATIO);

        transcript.append_protocol_name(Self::protocol_name());

        // append the claimed evaluations to transcript
        transcript.append_scalars(b"evals_ops_val", openings);

        let rlc_coefficients: Vec<_> =
            transcript.challenge_vector(b"challenge_combine_n_to_one", openings.len());

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
    use ark_bn254::G1Projective;
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

        let generators: PedersenGenerators<G> = PedersenGenerators::new(1 << 8, b"test-two");
        let poly_commitment: HyraxCommitment<1, G> = HyraxCommitment::commit(&poly, &generators);

        let mut prover_transcript = ProofTranscript::new(b"example");
        let proof = HyraxOpeningProof::prove(&poly, &r, &mut prover_transcript);

        let mut verifier_transcript = ProofTranscript::new(b"example");

        assert!(proof
            .verify(
                &generators,
                &mut verifier_transcript,
                &r,
                &eval,
                &poly_commitment
            )
            .is_ok());
    }
}
