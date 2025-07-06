use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use std::borrow::Borrow;
use std::fmt::Debug;

use crate::utils::transcript::Transcript;
use crate::{
    field::JoltField,
    poly::multilinear_polynomial::MultilinearPolynomial,
    utils::{errors::ProofVerifyError, transcript::AppendToTranscript},
};

pub trait CommitmentScheme<ProofTranscript: Transcript>: Clone + Sync + Send + 'static {
    type Field: JoltField + Sized;
    type Setup: Clone + Sync + Send + CanonicalSerialize + CanonicalDeserialize;
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

    fn setup(max_len: usize) -> Self::Setup;
    fn srs_size(setup: &Self::Setup) -> usize;
    fn commit(poly: &MultilinearPolynomial<Self::Field>, setup: &Self::Setup) -> Self::Commitment;
    fn batch_commit<U>(polys: &[U], gens: &Self::Setup) -> Vec<Self::Commitment>
    where
        U: Borrow<MultilinearPolynomial<Self::Field>> + Sync;

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

pub trait StreamingCommitmentScheme<ProofTranscript: Transcript>:
    CommitmentScheme<ProofTranscript>
{
    type State<'a>; // : Clone + Debug;

    fn initialize<'a>(size: usize, setup: &'a Self::Setup) -> Self::State<'a>;
    fn process<'a>(state: Self::State<'a>, eval: Self::Field) -> Self::State<'a>;
    fn finalize<'a>(state: Self::State<'a>) -> Self::Commitment;
}
