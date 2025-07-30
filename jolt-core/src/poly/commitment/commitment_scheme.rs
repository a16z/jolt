use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use std::borrow::Borrow;
use std::fmt::Debug;

use crate::utils::transcript::Transcript;
use crate::{
    field::JoltField,
    poly::multilinear_polynomial::MultilinearPolynomial,
    utils::{errors::ProofVerifyError, transcript::AppendToTranscript},
};

pub trait CommitmentScheme: Clone + Sync + Send + 'static {
    type Field: JoltField + Sized;
    type ProverSetup: Clone + Sync + Send + Debug + CanonicalSerialize + CanonicalDeserialize;
    type VerifierSetup: Clone + Sync + Send + Debug + CanonicalSerialize + CanonicalDeserialize;
    type Commitment: Default
        + Debug
        + Sync
        + Send
        + PartialEq
        + CanonicalSerialize
        + CanonicalDeserialize
        + AppendToTranscript
        + Clone;
    type Proof: Sync + Send + CanonicalSerialize + CanonicalDeserialize + Clone + Debug;
    type BatchedProof: Sync + Send + CanonicalSerialize + CanonicalDeserialize;
    /// A hint that helps the prover compute an opening proof. Typically some byproduct of
    /// the commitment computation, e.g. for Dory the Pedersen commitments to the rows can be
    /// used as a hint for the opening proof.
    type OpeningProofHint: Sync + Send + Clone + Debug;

    fn setup_prover(max_len: usize) -> Self::ProverSetup;
    fn setup_verifier(setup: &Self::ProverSetup) -> Self::VerifierSetup;
    fn srs_size(setup: &Self::ProverSetup) -> usize;
    fn commit(
        poly: &MultilinearPolynomial<Self::Field>,
        setup: &Self::ProverSetup,
    ) -> (Self::Commitment, Self::OpeningProofHint);
    fn batch_commit<U>(polys: &[U], gens: &Self::ProverSetup) -> Vec<Self::Commitment>
    where
        U: Borrow<MultilinearPolynomial<Self::Field>> + Sync;

    /// Homomorphically combines multiple commitments into a single commitment, computed as a
    /// linear combination with the given coefficients.
    fn combine_commitments<C: Borrow<Self::Commitment>>(
        _commitments: &[C],
        _coeffs: &[Self::Field],
    ) -> Self::Commitment {
        todo!("`combine_commitments` should be on a separate `AdditivelyHomomorphic` trait")
    }

    /// Homomorphically combines multiple opening proof hints into a single hint, computed as a
    /// linear combination with the given coefficients.
    fn combine_hints(
        _hints: Vec<Self::OpeningProofHint>,
        _coeffs: &[Self::Field],
    ) -> Self::OpeningProofHint {
        unimplemented!()
    }

    fn prove<ProofTranscript: Transcript>(
        setup: &Self::ProverSetup,
        poly: &MultilinearPolynomial<Self::Field>,
        opening_point: &[Self::Field], // point at which the polynomial is evaluated
        hint: Self::OpeningProofHint,
        transcript: &mut ProofTranscript,
    ) -> Self::Proof;

    fn verify<ProofTranscript: Transcript>(
        proof: &Self::Proof,
        setup: &Self::VerifierSetup,
        transcript: &mut ProofTranscript,
        opening_point: &[Self::Field], // point at which the polynomial is evaluated
        opening: &Self::Field,         // evaluation \widetilde{Z}(r)
        commitment: &Self::Commitment,
    ) -> Result<(), ProofVerifyError>;

    fn protocol_name() -> &'static [u8];
}

pub trait StreamingCommitmentScheme: CommitmentScheme {
    type State<'a>; // : Clone + Debug;

    fn initialize<'a>(size: usize, setup: &'a Self::ProverSetup) -> Self::State<'a>;
    fn process<'a>(state: Self::State<'a>, eval: Self::Field) -> Self::State<'a>;
    fn finalize<'a>(state: Self::State<'a>) -> Self::Commitment;
}
