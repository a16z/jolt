use std::marker::PhantomData;

use super::commitment_scheme::{BatchType, CommitShape, CommitmentScheme};
use super::pedersen::{PedersenCommitment, PedersenGenerators};
use crate::field::JoltField;
use crate::poly::dense_mlpoly::DensePolynomial;
use crate::poly::eq_poly::EqPolynomial;
use crate::utils::errors::ProofVerifyError;
use crate::utils::math::Math;
use crate::utils::transcript::{AppendToTranscript, Transcript};
use crate::utils::{compute_dotproduct, mul_0_1_optimized};
use ark_ec::CurveGroup;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use num_integer::Roots;
use rayon::prelude::*;
use tracing::trace_span;

use crate::msm::VariableBaseMSM;

#[derive(Clone)]
pub struct HyraxScheme<G: CurveGroup, ProofTranscript: Transcript> {
    marker: PhantomData<(G, ProofTranscript)>,
}

const TRACE_LEN_R1CS_POLYS_BATCH_RATIO: usize = 64;
const SURGE_RATIO_READ_WRITE: usize = 16;
const SURGE_RATIO_FINAL: usize = 4;

pub fn batch_type_to_ratio(batch_type: &BatchType) -> usize {
    match batch_type {
        BatchType::Big => TRACE_LEN_R1CS_POLYS_BATCH_RATIO,
        BatchType::GrandProduct => TRACE_LEN_R1CS_POLYS_BATCH_RATIO,
        BatchType::Small => 1,
        BatchType::SurgeReadWrite => SURGE_RATIO_READ_WRITE,
        BatchType::SurgeInitFinal => SURGE_RATIO_FINAL,
    }
}

pub fn matrix_dimensions(num_vars: usize, ratio: usize) -> (usize, usize) {
    let mut row_size = (num_vars / 2).pow2();
    row_size = (row_size * ratio.sqrt()).next_power_of_two();

    let right_num_vars = std::cmp::min(row_size.log_2(), num_vars - 1);
    row_size = right_num_vars.pow2();
    let left_num_vars = num_vars - right_num_vars;
    let col_size = left_num_vars.pow2();

    (col_size, row_size)
}

impl<F: JoltField, G: CurveGroup<ScalarField = F>, ProofTranscript: Transcript>
    CommitmentScheme<ProofTranscript> for HyraxScheme<G, ProofTranscript>
{
    type Field = G::ScalarField;
    type Setup = PedersenGenerators<G>;
    type Commitment = HyraxCommitment<G>;
    type Proof = HyraxOpeningProof<G, ProofTranscript>;
    type BatchedProof = BatchedHyraxOpeningProof<G, ProofTranscript>;

    fn setup(shapes: &[CommitShape]) -> Self::Setup {
        let mut max_len: usize = 0;
        for shape in shapes {
            let len = matrix_dimensions(
                shape.input_length.log_2(),
                batch_type_to_ratio(&shape.batch_type),
            )
            .1;
            if len > max_len {
                max_len = len;
            }
        }
        PedersenGenerators::new(max_len, b"Jolt v1 Hyrax generators")
    }
    fn commit(poly: &DensePolynomial<Self::Field>, gens: &Self::Setup) -> Self::Commitment {
        HyraxCommitment::commit(poly, gens)
    }
    fn batch_commit(
        evals: &[&[Self::Field]],
        gens: &Self::Setup,
        batch_type: BatchType,
    ) -> Vec<Self::Commitment> {
        HyraxCommitment::batch_commit(evals, gens, batch_type)
    }
    fn commit_slice(eval_slice: &[Self::Field], generators: &Self::Setup) -> Self::Commitment {
        HyraxCommitment::commit_slice(eval_slice, generators)
    }
    fn prove(
        _setup: &Self::Setup,
        poly: &DensePolynomial<Self::Field>,
        opening_point: &[Self::Field],
        transcript: &mut ProofTranscript,
    ) -> Self::Proof {
        // Implicitly prove is "prove_single", with a ratio = 1
        HyraxOpeningProof::prove(poly, opening_point, 1, transcript)
    }
    fn batch_prove(
        _setup: &Self::Setup,
        polynomials: &[&DensePolynomial<Self::Field>],
        opening_point: &[Self::Field],
        openings: &[Self::Field],
        batch_type: BatchType,
        transcript: &mut ProofTranscript,
    ) -> Self::BatchedProof {
        BatchedHyraxOpeningProof::prove(
            polynomials,
            opening_point,
            openings,
            batch_type,
            transcript,
        )
    }
    fn combine_commitments(
        commitments: &[&Self::Commitment],
        coeffs: &[Self::Field],
    ) -> Self::Commitment {
        let max_size = commitments
            .iter()
            .map(|commitment| commitment.row_commitments.len())
            .max()
            .unwrap();

        let row_commitments = coeffs
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
                || vec![G::zero(); max_size],
                |running, new| {
                    running
                        .iter()
                        .zip(new.iter())
                        .map(|(r, n)| *r + n)
                        .collect()
                },
            );
        HyraxCommitment { row_commitments }
    }

    fn verify(
        proof: &Self::Proof,
        generators: &Self::Setup,
        transcript: &mut ProofTranscript,
        opening_point: &[Self::Field],
        opening: &Self::Field,
        commitment: &Self::Commitment,
    ) -> Result<(), ProofVerifyError> {
        // Implicitly verify is "prove_single", with a ratio = 1
        HyraxOpeningProof::verify(
            proof,
            generators,
            transcript,
            opening_point,
            opening,
            commitment,
            1,
        )
    }
    #[tracing::instrument(skip_all, name = "HyraxScheme::batch_verify")]
    fn batch_verify(
        batch_proof: &Self::BatchedProof,
        generators: &Self::Setup,
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

#[derive(Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct HyraxGenerators<G: CurveGroup> {
    pub gens: PedersenGenerators<G>,
}

#[derive(Default, Clone, Debug, PartialEq, CanonicalSerialize, CanonicalDeserialize)]
pub struct HyraxCommitment<G: CurveGroup> {
    pub row_commitments: Vec<G>,
}

impl<F: JoltField, G: CurveGroup<ScalarField = F>> HyraxCommitment<G> {
    #[tracing::instrument(skip_all, name = "HyraxCommitment::commit")]
    pub fn commit(
        poly: &DensePolynomial<G::ScalarField>,
        generators: &PedersenGenerators<G>,
    ) -> Self {
        Self::commit_slice(poly.evals_ref(), generators)
    }

    #[tracing::instrument(skip_all, name = "HyraxCommitment::commit_slice")]
    pub fn commit_slice(eval_slice: &[G::ScalarField], generators: &PedersenGenerators<G>) -> Self {
        let n = eval_slice.len();
        let ell = n.log_2();

        let (L_size, R_size) = matrix_dimensions(ell, 1);
        assert_eq!(L_size * R_size, n);

        let gens = CurveGroup::normalize_batch(&generators.generators[..R_size]);
        let row_commitments = eval_slice
            .par_chunks(R_size)
            .map(|row| PedersenCommitment::commit_vector(row, &gens))
            .collect();
        Self { row_commitments }
    }

    #[tracing::instrument(skip_all, name = "HyraxCommitment::batch_commit")]
    pub fn batch_commit(
        batch: &[&[G::ScalarField]],
        generators: &PedersenGenerators<G>,
        batch_type: BatchType,
    ) -> Vec<Self> {
        let n = batch[0].len();
        batch.iter().for_each(|poly| assert_eq!(poly.len(), n));
        let ell = n.log_2();

        let ratio = batch_type_to_ratio(&batch_type);

        let (L_size, R_size) = matrix_dimensions(ell, ratio);
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

impl<G: CurveGroup> AppendToTranscript for HyraxCommitment<G> {
    fn append_to_transcript<ProofTranscript: Transcript>(&self, transcript: &mut ProofTranscript) {
        transcript.append_message(b"poly_commitment_begin");
        for i in 0..self.row_commitments.len() {
            transcript.append_point(&self.row_commitments[i]);
        }
        transcript.append_message(b"poly_commitment_end");
    }
}

#[derive(Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct HyraxOpeningProof<G: CurveGroup, ProofTranscript: Transcript> {
    pub vector_matrix_product: Vec<G::ScalarField>,
    _marker: PhantomData<ProofTranscript>,
}

/// See Section 14.3 of Thaler's Proofs, Arguments, and Zero-Knowledge
impl<F, G, ProofTranscript> HyraxOpeningProof<G, ProofTranscript>
where
    F: JoltField,
    G: CurveGroup<ScalarField = F>,
    ProofTranscript: Transcript,
{
    fn protocol_name() -> &'static [u8] {
        b"Hyrax opening proof"
    }

    #[tracing::instrument(skip_all, name = "HyraxOpeningProof::prove")]
    pub fn prove(
        poly: &DensePolynomial<G::ScalarField>,
        opening_point: &[G::ScalarField], // point at which the polynomial is evaluated
        ratio: usize,
        transcript: &mut ProofTranscript,
    ) -> HyraxOpeningProof<G, ProofTranscript> {
        let protocol_name = Self::protocol_name();
        transcript.append_message(protocol_name);

        // assert vectors are of the right size
        assert_eq!(poly.get_num_vars(), opening_point.len());

        // compute the L and R vectors
        let (L_size, _R_size) = matrix_dimensions(poly.get_num_vars(), ratio);
        let eq = EqPolynomial::new(opening_point.to_vec());
        let (L, _R) = eq.compute_factored_evals(L_size);

        // compute vector-matrix product between L and Z viewed as a matrix
        let vector_matrix_product = Self::vector_matrix_product(poly, &L, ratio);

        HyraxOpeningProof {
            vector_matrix_product,
            _marker: PhantomData,
        }
    }

    pub fn verify(
        &self,
        pedersen_generators: &PedersenGenerators<G>,
        transcript: &mut ProofTranscript,
        opening_point: &[G::ScalarField], // point at which the polynomial is evaluated
        opening: &G::ScalarField,         // evaluation \widetilde{Z}(r)
        commitment: &HyraxCommitment<G>,
        ratio: usize,
    ) -> Result<(), ProofVerifyError> {
        let protocol_name = Self::protocol_name();
        transcript.append_message(protocol_name);

        // compute L and R
        let (L_size, R_size) = matrix_dimensions(opening_point.len(), ratio);
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

#[derive(Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct BatchedHyraxOpeningProof<G: CurveGroup, ProofTranscript: Transcript> {
    pub joint_proof: HyraxOpeningProof<G, ProofTranscript>,
    pub ratio: usize,
    _marker: PhantomData<ProofTranscript>,
}

/// See Section 16.1 of Thaler's Proofs, Arguments, and Zero-Knowledge
impl<F: JoltField, G: CurveGroup<ScalarField = F>, ProofTranscript: Transcript>
    BatchedHyraxOpeningProof<G, ProofTranscript>
{
    #[tracing::instrument(skip_all, name = "BatchedHyraxOpeningProof::prove")]
    pub fn prove(
        polynomials: &[&DensePolynomial<G::ScalarField>],
        opening_point: &[G::ScalarField],
        openings: &[G::ScalarField],
        batch_type: BatchType,
        transcript: &mut ProofTranscript,
    ) -> Self {
        let protocol_name = Self::protocol_name();
        transcript.append_message(protocol_name);

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

        let ratio = batch_type_to_ratio(&batch_type);
        let joint_proof = HyraxOpeningProof::prove(
            &DensePolynomial::new(rlc_poly),
            opening_point,
            ratio,
            transcript,
        );

        Self {
            joint_proof,
            ratio,
            _marker: PhantomData,
        }
    }

    #[tracing::instrument(skip_all, name = "BatchedHyraxOpeningProof::verify")]
    pub fn verify(
        &self,
        pedersen_generators: &PedersenGenerators<G>,
        opening_point: &[G::ScalarField],
        openings: &[G::ScalarField],
        commitments: &[&HyraxCommitment<G>],
        transcript: &mut ProofTranscript,
    ) -> Result<(), ProofVerifyError> {
        assert_eq!(openings.len(), commitments.len());
        let (L_size, _R_size) = matrix_dimensions(opening_point.len(), self.ratio);
        commitments.iter().enumerate().for_each(|(i, commitment)| {
            assert_eq!(
                L_size,
                commitment.row_commitments.len(),
                "Row commitment {}/{} wrong length.",
                i,
                commitments.len()
            )
        });

        let protocol_name = Self::protocol_name();
        transcript.append_message(protocol_name);

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
            transcript,
            opening_point,
            &rlc_eval,
            &HyraxCommitment {
                row_commitments: rlc_commitment,
            },
            self.ratio,
        )
    }

    fn protocol_name() -> &'static [u8] {
        b"Jolt BatchedHyraxOpeningProof"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::transcript::{KeccakTranscript, Transcript};
    use ark_bn254::{Fr, G1Projective};

    #[test]
    fn check_polynomial_commit() {
        check_polynomial_commit_helper::<Fr, G1Projective, 1>();
        check_polynomial_commit_helper::<Fr, G1Projective, 4>();
    }

    fn check_polynomial_commit_helper<
        F: JoltField,
        G: CurveGroup<ScalarField = F>,
        const RATIO: usize,
    >() {
        let Z = vec![
            G::ScalarField::one(),
            G::ScalarField::from_u64(2u64).unwrap(),
            G::ScalarField::one(),
            G::ScalarField::from_u64(4u64).unwrap(),
        ];
        let poly = DensePolynomial::new(Z);

        // r = [4,3]
        let r = vec![
            G::ScalarField::from_u64(4u64).unwrap(),
            G::ScalarField::from_u64(3u64).unwrap(),
        ];
        let eval = poly.evaluate(&r);
        assert_eq!(eval, G::ScalarField::from_u64(28u64).unwrap());

        let generators: PedersenGenerators<G> = PedersenGenerators::new(1 << 8, b"test-two");
        let poly_commitment: HyraxCommitment<G> = HyraxCommitment::commit(&poly, &generators);

        let mut prover_transcript = KeccakTranscript::new(b"example");
        let proof = HyraxOpeningProof::prove(&poly, &r, RATIO, &mut prover_transcript);

        let mut verifier_transcript = KeccakTranscript::new(b"example");

        assert!(proof
            .verify(
                &generators,
                &mut verifier_transcript,
                &r,
                &eval,
                &poly_commitment,
                RATIO
            )
            .is_ok());
    }
}
