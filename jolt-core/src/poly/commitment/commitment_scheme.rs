use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use std::fmt::Debug;

use crate::utils::transcript::Transcript;
use crate::{
    field::JoltField,
    poly::multilinear_polynomial::MultilinearPolynomial,
    utils::{errors::ProofVerifyError, transcript::AppendToTranscript},
};

#[derive(Clone, Debug)]
pub struct CommitShape {
    pub input_length: usize,
    pub batch_type: BatchType,
}

impl CommitShape {
    pub fn new(input_length: usize, batch_type: BatchType) -> Self {
        Self {
            input_length,
            batch_type,
        }
    }
}

#[derive(Clone, Debug)]
pub enum BatchType {
    Big,
    Small,
    SurgeInitFinal,
    SurgeReadWrite,
    GrandProduct,
}

pub trait CommitmentScheme<ProofTranscript: Transcript>: Clone + Sync + Send + 'static {
    type Field: JoltField + Sized;
    type Setup: Clone + Sync + Send;
    type Commitment: Default
        + Debug
        + Sync
        + Send
        + PartialEq
        + CanonicalSerialize
        + CanonicalDeserialize
        + AppendToTranscript;
    type Proof: Sync + Send + CanonicalSerialize + CanonicalDeserialize;
    type BatchedProof: Sync + Send + CanonicalSerialize + CanonicalDeserialize;

    fn setup(shapes: &[CommitShape]) -> Self::Setup;
    fn commit(poly: &MultilinearPolynomial<Self::Field>, setup: &Self::Setup) -> Self::Commitment;
    fn batch_commit(
        polys: &[&MultilinearPolynomial<Self::Field>],
        gens: &Self::Setup,
        batch_type: BatchType,
    ) -> Vec<Self::Commitment>;

    /// Homomorphically combines multiple commitments into a single commitment, computed as a
    /// linear combination with the given coefficients.
    fn combine_commitments(
        _commitments: &[&Self::Commitment],
        _coeffs: &[Self::Field],
    ) -> Self::Commitment {
        todo!("`combine_commitments` should be on a separate `AdditivelyHomomorphic` trait")
    }

    fn prove(
        setup: &Self::Setup,
        poly: &MultilinearPolynomial<Self::Field>,
        opening_point: &[Self::Field], // point at which the polynomial is evaluated
        transcript: &mut ProofTranscript,
    ) -> Self::Proof;

    fn verify(
        proof: &Self::Proof,
        setup: &Self::Setup,
        transcript: &mut ProofTranscript,
        opening_point: &[Self::Field], // point at which the polynomial is evaluated
        opening: &Self::Field,         // evaluation \widetilde{Z}(r)
        commitment: &Self::Commitment,
    ) -> Result<(), ProofVerifyError>;

    fn protocol_name() -> &'static [u8];
}
