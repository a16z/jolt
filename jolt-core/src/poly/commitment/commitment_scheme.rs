use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use std::borrow::Borrow;
use std::fmt::Debug;

use crate::poly::opening_proof::BatchPolynomialSource;
use crate::transcripts::Transcript;
use crate::{
    curve::JoltCurve,
    field::JoltField,
    poly::multilinear_polynomial::MultilinearPolynomial,
    utils::{errors::ProofVerifyError, small_scalar::SmallScalar},
};

pub trait CommitmentScheme: Clone + Sync + Send + Default + 'static {
    type Field: JoltField + Sized;
    /// PCS-specific configuration carried by the instance. Opaque to generic code.
    type Config: Clone + Sync + Send + CanonicalSerialize + CanonicalDeserialize;
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
    /// A hint that helps the prover compute an opening proof. Typically some byproduct of
    /// the commitment computation, e.g. for Dory the Pedersen commitments to the rows can be
    /// used as a hint for the opening proof.
    type OpeningProofHint: Sync + Send + Clone + Debug + PartialEq;

    fn setup_prover(max_num_vars: usize) -> Self::ProverSetup;

    fn setup_verifier(setup: &Self::ProverSetup) -> Self::VerifierSetup;

    /// Reconstruct a PCS instance from a batched proof (e.g. for the verifier to
    /// recover PCS-specific configuration serialized during proving).
    fn from_proof(proof: &Self::Proof) -> Self;

    fn config(&self) -> &Self::Config;

    fn commit(
        &self,
        poly: &MultilinearPolynomial<Self::Field>,
        setup: &Self::ProverSetup,
    ) -> (Self::Commitment, Self::OpeningProofHint);

    fn batch_commit<U>(
        &self,
        polys: &[U],
        gens: &Self::ProverSetup,
    ) -> Vec<(Self::Commitment, Self::OpeningProofHint)>
    where
        U: Borrow<MultilinearPolynomial<Self::Field>> + Sync;

    fn prove<ProofTranscript: Transcript>(
        &self,
        setup: &Self::ProverSetup,
        poly: &MultilinearPolynomial<Self::Field>,
        opening_point: &[<Self::Field as JoltField>::Challenge],
        hint: Option<Self::OpeningProofHint>,
        transcript: &mut ProofTranscript,
        commitment: &Self::Commitment,
    ) -> (Self::Proof, Option<Self::Field>);

    fn verify<ProofTranscript: Transcript>(
        &self,
        proof: &Self::Proof,
        setup: &Self::VerifierSetup,
        transcript: &mut ProofTranscript,
        opening_point: &[<Self::Field as JoltField>::Challenge],
        opening: &Self::Field,
        commitment: &Self::Commitment,
    ) -> Result<(), ProofVerifyError>;

    #[allow(clippy::too_many_arguments)]
    fn batch_prove<ProofTranscript: Transcript, S: BatchPolynomialSource<Self::Field>>(
        &self,
        setup: &Self::ProverSetup,
        poly_source: &S,
        hints: Vec<Self::OpeningProofHint>,
        commitments: &[&Self::Commitment],
        opening_point: &[<Self::Field as JoltField>::Challenge],
        claims: &[Self::Field],
        coeffs: &[Self::Field],
        transcript: &mut ProofTranscript,
    ) -> (Self::Proof, Option<Self::Field>);

    #[allow(clippy::too_many_arguments)]
    fn batch_verify<ProofTranscript: Transcript>(
        &self,
        proof: &Self::Proof,
        setup: &Self::VerifierSetup,
        transcript: &mut ProofTranscript,
        opening_point: &[<Self::Field as JoltField>::Challenge],
        commitments: &[&Self::Commitment],
        claims: &[Self::Field],
        coeffs: &[Self::Field],
    ) -> Result<(), ProofVerifyError>;

    fn protocol_name() -> &'static [u8];

    /// Extracts raw BN254 G1 generators and blinding generator from the prover setup.
    /// Used to derive ZK Pedersen generators from PCS setup.
    /// Returns None for PCS that don't support ZK Pedersen commitments.
    #[cfg(feature = "zk")]
    fn zk_generators_raw(
        _setup: &Self::ProverSetup,
        _count: usize,
    ) -> Option<(Vec<crate::curve::Bn254G1>, crate::curve::Bn254G1)> {
        None
    }
}

pub trait ZkEvalCommitment<C: JoltCurve>: CommitmentScheme {
    /// Returns the evaluation commitment (e.g. y_com) if present in the proof.
    fn eval_commitment(proof: &Self::Proof) -> Option<C::G1>;

    /// Returns the generators used for evaluation commitments in the prover setup.
    fn eval_commitment_gens(setup: &Self::ProverSetup) -> Option<(C::G1, C::G1)>;

    /// Returns the generators used for evaluation commitments in the verifier setup.
    fn eval_commitment_gens_verifier(setup: &Self::VerifierSetup) -> Option<(C::G1, C::G1)>;
}

pub trait StreamingCommitmentScheme: CommitmentScheme {
    /// The type representing chunk state (tier 1 commitments)
    type ChunkState: Send + Sync + Clone + PartialEq + Debug;

    fn process_chunk<T: SmallScalar>(
        &self,
        setup: &Self::ProverSetup,
        chunk: &[T],
    ) -> Self::ChunkState;

    fn process_chunk_onehot(
        &self,
        setup: &Self::ProverSetup,
        onehot_k: usize,
        chunk: &[Option<usize>],
    ) -> Self::ChunkState;

    fn aggregate_chunks(
        &self,
        setup: &Self::ProverSetup,
        onehot_k: Option<usize>,
        tier1_commitments: &[Self::ChunkState],
    ) -> (Self::Commitment, Self::OpeningProofHint);
}
