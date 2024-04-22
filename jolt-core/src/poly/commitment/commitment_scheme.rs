use ark_ff::PrimeField;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

use crate::{poly::dense_mlpoly::DensePolynomial, utils::{errors::ProofVerifyError, transcript::{AppendToTranscript, ProofTranscript}}};

pub trait CommitmentScheme: Clone + Sync + Send + 'static {
    // type Field: JoltField;
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