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

    /// Generates the prover setup for this PCS. `max_num_vars` is the maximum number of
    /// variables of any polynomial that will be committed using this setup.
    fn setup_prover(max_num_vars: usize) -> Self::ProverSetup;

    /// Generates the verifier setup from the prover setup.
    fn setup_verifier(setup: &Self::ProverSetup) -> Self::VerifierSetup;

    /// Commits to a multilinear polynomial using the provided setup.
    ///
    /// # Arguments
    /// * `poly` - The multilinear polynomial to commit to
    /// * `setup` - The prover setup for the commitment scheme
    ///
    /// # Returns
    /// A tuple containing the commitment to the polynomial and a hint that can be used
    /// to optimize opening proof generation
    fn commit(
        poly: &MultilinearPolynomial<Self::Field>,
        setup: &Self::ProverSetup,
    ) -> (Self::Commitment, Self::OpeningProofHint);

    /// Commits to multiple multilinear polynomials in batch.
    ///
    /// # Arguments
    /// * `polys` - A slice of multilinear polynomials to commit to
    /// * `gens` - The prover setup for the commitment scheme
    ///
    /// # Returns
    /// A vector of commitments, one for each input polynomial
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

    /// Generates a proof of evaluation for a polynomial at a specific point.
    ///
    /// # Arguments
    /// * `setup` - The prover setup for the commitment scheme
    /// * `poly` - The multilinear polynomial being proved
    /// * `opening_point` - The point at which the polynomial is evaluated
    /// * `hint` - A hint that helps optimize the proof generation
    /// * `transcript` - The transcript for Fiat-Shamir transformation
    ///
    /// # Returns
    /// A proof of the polynomial evaluation at the specified point
    fn prove<ProofTranscript: Transcript>(
        setup: &Self::ProverSetup,
        poly: &MultilinearPolynomial<Self::Field>,
        opening_point: &[Self::Field],
        hint: Self::OpeningProofHint,
        transcript: &mut ProofTranscript,
    ) -> Self::Proof;

    /// Verifies a proof of polynomial evaluation at a specific point.
    ///
    /// # Arguments
    /// * `proof` - The proof to be verified
    /// * `setup` - The verifier setup for the commitment scheme
    /// * `transcript` - The transcript for Fiat-Shamir transformation
    /// * `opening_point` - The point at which the polynomial is evaluated
    /// * `opening` - The claimed evaluation value of the polynomial at the opening point
    /// * `commitment` - The commitment to the polynomial
    ///
    /// # Returns
    /// Ok(()) if the proof is valid, otherwise a ProofVerifyError
    fn verify<ProofTranscript: Transcript>(
        proof: &Self::Proof,
        setup: &Self::VerifierSetup,
        transcript: &mut ProofTranscript,
        opening_point: &[Self::Field],
        opening: &Self::Field,
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
