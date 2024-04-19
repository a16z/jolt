use std::marker::PhantomData;

use ark_ec::CurveGroup;
use ark_ff::PrimeField;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::Zero;
use common::constants::NUM_R1CS_POLYS;
use num_integer::Roots;

use rayon::prelude::*;

use super::{
    dense_mlpoly::DensePolynomial,
    hyrax::{BatchedHyraxOpeningProof, HyraxCommitment, HyraxOpeningProof},
    pedersen::PedersenGenerators,
};
use crate::{
    lasso::memory_checking::NoPreprocessing,
    poly::pedersen::PedersenCommitment,
    utils::{
        errors::ProofVerifyError,
        math::Math,
        mul_0_1_optimized,
        transcript::{AppendToTranscript, ProofTranscript},
    },
};

pub trait CommitmentScheme: Clone + Sync + Send + 'static {
    type Field: PrimeField;
    type Generators: Clone + Sync + Send;
    type Commitment: Sync + Send + CanonicalSerialize + CanonicalDeserialize + AppendToTranscript;
    type Proof: Sync + Send + CanonicalSerialize + CanonicalDeserialize;
    type BatchedProof: Sync + Send + CanonicalSerialize + CanonicalDeserialize;

    fn generators(max_commit_size: usize) -> Self::Generators;
    fn commit(poly: &DensePolynomial<Self::Field>, gens: &Self::Generators) -> Self::Commitment;
    fn batch_commit(
        evals: &Vec<Vec<Self::Field>>,
        gens: &Self::Generators,
    ) -> Vec<Self::Commitment>;
    fn commit_slice(evals: &[Self::Field], gens: &Self::Generators) -> Self::Commitment;
    fn batch_commit_polys(
        polys: &Vec<DensePolynomial<Self::Field>>,
        gens: &Self::Generators,
    ) -> Vec<Self::Commitment>;
    fn batch_commit_polys_ref(
        polys: &Vec<&DensePolynomial<Self::Field>>,
        gens: &Self::Generators,
    ) -> Vec<Self::Commitment>;
    fn prove(
        poly: &DensePolynomial<Self::Field>,
        opening_point: &[Self::Field], // point at which the polynomial is evaluated
        transcript: &mut ProofTranscript,
    ) -> Self::Proof;
    fn batch_prove(
        polynomials: &[&DensePolynomial<Self::Field>],
        opening_point: &[Self::Field],
        openings: &[Self::Field],
        transcript: &mut ProofTranscript,
    ) -> Self::BatchedProof;

    fn verify(
        proof: &Self::Proof,
        generators: &Self::Generators,
        transcript: &mut ProofTranscript,
        opening_point: &[Self::Field], // point at which the polynomial is evaluated
        opening: &Self::Field,         // evaluation \widetilde{Z}(r)
        commitment: &Self::Commitment,
    ) -> Result<(), ProofVerifyError>;

    fn batch_verify(
        batch_proof: &Self::BatchedProof,
        generators: &Self::Generators,
        opening_point: &[Self::Field],
        openings: &[Self::Field],
        commitments: &[&Self::Commitment],
        transcript: &mut ProofTranscript,
    ) -> Result<(), ProofVerifyError>;

    fn protocol_name() -> &'static [u8];
}

// TODO(sragss): Could split out the prove / verify functions to their own traits

#[derive(Clone)]
pub struct HyraxConfig<G: CurveGroup> {
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

impl<G: CurveGroup> CommitmentScheme for HyraxConfig<G> {
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

impl<G: CurveGroup> HyraxConfig<G> {
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

/// Encapsulates the pattern of a collection of related polynomials (e.g. those used to
/// prove instruction lookups in Jolt) that can be "batched" for more efficient
/// commitments/openings.
pub trait StructuredCommitment<C: CommitmentScheme>: Send + Sync + Sized {
    /// The batched commitment to these polynomials.
    type Commitment;

    /// Commits to batched polynomials, typically using `HyraxCommitment::batch_commit_polys`.
    fn commit(&self, generators: &C::Generators) -> Self::Commitment;
}

/// Encapsulates the pattern of opening a batched polynomial commitment at a single point.
/// Note that there may be a one-to-many mapping from `StructuredCommitment` to `StructuredOpeningProof`:
/// different subset of the same polynomials may be opened at different points, resulting in
/// different opening proofs.
pub trait StructuredOpeningProof<F, C, Polynomials>:
    Sync + CanonicalSerialize + CanonicalDeserialize
where
    F: PrimeField,
    C: CommitmentScheme<Field = F>,
    Polynomials: StructuredCommitment<C> + ?Sized,
{
    type Preprocessing = NoPreprocessing;
    type Proof: Sync + CanonicalSerialize + CanonicalDeserialize;

    /// Evaluates each of the given `polynomials` at the given `opening_point`.
    fn open(polynomials: &Polynomials, opening_point: &[F]) -> Self;

    /// Proves that the `polynomials`, evaluated at `opening_point`, output the values given
    /// by `openings`. The polynomials should already be committed by the prover.
    fn prove_openings(
        polynomials: &Polynomials,
        opening_point: &[F],
        openings: &Self,
        transcript: &mut ProofTranscript,
    ) -> Self::Proof;

    /// Often some of the openings do not require an opening proof provided by the prover, and
    /// instead can be efficiently computed by the verifier by itself. This function populates
    /// any such fields in `self`.
    fn compute_verifier_openings(
        &mut self,
        _preprocessing: &Self::Preprocessing,
        _opening_point: &[F],
    ) {
    }

    /// Verifies an opening proof, given the associated polynomial `commitment` and `opening_point`.
    fn verify_openings(
        &self,
        generators: &C::Generators,
        opening_proof: &Self::Proof,
        commitment: &Polynomials::Commitment,
        opening_point: &[F],
        transcript: &mut ProofTranscript,
    ) -> Result<(), ProofVerifyError>;
}
