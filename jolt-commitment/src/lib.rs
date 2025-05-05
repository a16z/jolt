use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use std::borrow::Borrow;
use std::fmt::Debug;

pub mod error;

use crate::error::ProofVerifyError;
use jolt_fiat_shamir::{AppendToTranscript, Transcript};
use jolt_field::JoltField;
use jolt_poly::multilinear_polynomial::MultilinearPolynomial;

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

    fn setup(max_len: usize) -> Self::Setup;
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
