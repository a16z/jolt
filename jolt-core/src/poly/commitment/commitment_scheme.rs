use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use std::borrow::Borrow;
use std::fmt::Debug;

use crate::transcripts::{AppendToTranscript, Transcript};
use crate::{
    field::JoltField,
    poly::multilinear_polynomial::MultilinearPolynomial,
    utils::{errors::ProofVerifyError, small_scalar::SmallScalar},
    zkvm::recursion::witness::GTCombineWitness,
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
    type OpeningProofHint: Sync + Send + Clone + Debug + PartialEq;

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
    fn batch_commit<U>(
        polys: &[U],
        gens: &Self::ProverSetup,
    ) -> Vec<(Self::Commitment, Self::OpeningProofHint)>
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
    /// * `hint` - An optional hint that helps optimize the proof generation.
    ///   When `None`, implementations should compute the hint internally if needed.
    /// * `transcript` - The transcript for Fiat-Shamir transformation
    ///
    /// # Returns
    /// A proof of the polynomial evaluation at the specified point
    fn prove<ProofTranscript: Transcript>(
        setup: &Self::ProverSetup,
        poly: &MultilinearPolynomial<Self::Field>,
        opening_point: &[<Self::Field as JoltField>::Challenge],
        hint: Option<Self::OpeningProofHint>,
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
        opening_point: &[<Self::Field as JoltField>::Challenge],
        opening: &Self::Field,
        commitment: &Self::Commitment,
    ) -> Result<(), ProofVerifyError>;

    fn protocol_name() -> &'static [u8];
}

pub trait StreamingCommitmentScheme: CommitmentScheme {
    /// The type representing chunk state (tier 1 commitments)
    type ChunkState: Send + Sync + Clone + PartialEq + Debug;

    /// Compute tier 1 commitment for a chunk of small scalar values
    fn process_chunk<T: SmallScalar>(setup: &Self::ProverSetup, chunk: &[T]) -> Self::ChunkState;

    /// Compute tier 1 commitment for a chunk of one-hot values
    fn process_chunk_onehot(
        setup: &Self::ProverSetup,
        onehot_k: usize,
        chunk: &[Option<usize>],
    ) -> Self::ChunkState;

    /// Compute tier 2 commitment from accumulated tier 1 commitments
    fn aggregate_chunks(
        setup: &Self::ProverSetup,
        onehot_k: Option<usize>,
        tier1_commitments: &[Self::ChunkState],
    ) -> (Self::Commitment, Self::OpeningProofHint);
}

/// Generic extension trait for commitment schemes that adds recursion support
pub trait RecursionExt<F: JoltField>: CommitmentScheme<Field = F> {
    /// verifier computations
    type Witness;
    /// hints for efficient verification (must be serializable for proof transport)
    type Hint: CanonicalSerialize + CanonicalDeserialize + Clone + Send + Sync;
    /// Hint for combine_commitments offloading (the final combined commitment)
    type CombineHint;

    /// Generate witnesses and convert them to hints
    /// Returns both the full witnesses (for proving) and hints (for verification)
    fn witness_gen<ProofTranscript: Transcript>(
        proof: &Self::Proof,
        setup: &Self::VerifierSetup,
        transcript: &mut ProofTranscript,
        point: &[<F as JoltField>::Challenge],
        evaluation: &F,
        commitment: &Self::Commitment,
    ) -> Result<(Self::Witness, Self::Hint), ProofVerifyError>;

    /// Verify with hint-based approach
    fn verify_with_hint<ProofTranscript: Transcript>(
        proof: &Self::Proof,
        setup: &Self::VerifierSetup,
        transcript: &mut ProofTranscript,
        point: &[<F as JoltField>::Challenge],
        evaluation: &F,
        commitment: &Self::Commitment,
        hint: &Self::Hint,
    ) -> Result<(), ProofVerifyError>;

    /// Generate witness for combine_commitments offloading.
    ///
    /// Computes `result = sum_i(coeff_i * commitment_i)` while capturing
    /// intermediate witnesses for GT exp/mul operations.
    ///
    /// Returns the witness (for recursion proving) and hint (the final result).
    fn generate_combine_witness<C: Borrow<Self::Commitment>>(
        commitments: &[C],
        coeffs: &[F],
    ) -> (GTCombineWitness, Self::CombineHint);

    /// Use precomputed hint instead of computing combine_commitments directly.
    fn combine_with_hint(hint: &Self::CombineHint) -> Self::Commitment;

    /// Extract the underlying Fq12 from the combine hint for serialization.
    fn combine_hint_to_fq12(hint: &Self::CombineHint) -> ark_bn254::Fq12;

    /// Reconstruct commitment from serialized Fq12 hint.
    fn combine_with_hint_fq12(hint: &ark_bn254::Fq12) -> Self::Commitment;
}
