use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use std::borrow::Borrow;
use std::fmt::Debug;

use crate::poly::opening_proof::BatchPolynomialSource;
use crate::transcripts::Transcript;
use crate::{
    field::JoltField,
    poly::multilinear_polynomial::MultilinearPolynomial,
    utils::{errors::ProofVerifyError, small_scalar::SmallScalar},
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
        + Clone;
    type Proof: Sync + Send + CanonicalSerialize + CanonicalDeserialize + Clone + Debug;
    type BatchedProof: Sync + Send + CanonicalSerialize + CanonicalDeserialize;
    /// A hint that helps the prover compute an opening proof. Typically some byproduct of
    /// the commitment computation, e.g. for Dory the Pedersen commitments to the rows can be
    /// used as a hint for the opening proof.
    type OpeningProofHint: Sync + Send + Clone + Debug + PartialEq;

    fn setup_prover(max_num_vars: usize) -> Self::ProverSetup;

    fn setup_verifier(setup: &Self::ProverSetup) -> Self::VerifierSetup;

    fn commit(
        poly: &MultilinearPolynomial<Self::Field>,
        setup: &Self::ProverSetup,
    ) -> (Self::Commitment, Self::OpeningProofHint);

    fn batch_commit<U>(
        polys: &[U],
        gens: &Self::ProverSetup,
    ) -> Vec<(Self::Commitment, Self::OpeningProofHint)>
    where
        U: Borrow<MultilinearPolynomial<Self::Field>> + Sync;

    fn prove<ProofTranscript: Transcript>(
        setup: &Self::ProverSetup,
        poly: &MultilinearPolynomial<Self::Field>,
        opening_point: &[<Self::Field as JoltField>::Challenge],
        hint: Option<Self::OpeningProofHint>,
        transcript: &mut ProofTranscript,
        commitment: &Self::Commitment,
    ) -> Self::Proof;

    fn verify<ProofTranscript: Transcript>(
        proof: &Self::Proof,
        setup: &Self::VerifierSetup,
        transcript: &mut ProofTranscript,
        opening_point: &[<Self::Field as JoltField>::Challenge],
        opening: &Self::Field,
        commitment: &Self::Commitment,
    ) -> Result<(), ProofVerifyError>;

    #[allow(clippy::too_many_arguments)]
    fn batch_prove<ProofTranscript: Transcript, S: BatchPolynomialSource<Self::Field>>(
        setup: &Self::ProverSetup,
        poly_source: &S,
        hints: Vec<Self::OpeningProofHint>,
        commitments: &[&Self::Commitment],
        opening_point: &[<Self::Field as JoltField>::Challenge],
        claims: &[Self::Field],
        coeffs: &[Self::Field],
        transcript: &mut ProofTranscript,
    ) -> Self::BatchedProof;

    /// Verifies a batch opening proof for multiple polynomials evaluated at a single point.
    fn batch_verify<ProofTranscript: Transcript>(
        proof: &Self::BatchedProof,
        setup: &Self::VerifierSetup,
        transcript: &mut ProofTranscript,
        opening_point: &[<Self::Field as JoltField>::Challenge],
        commitments: &[&Self::Commitment],
        claims: &[Self::Field],
        coeffs: &[Self::Field],
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
