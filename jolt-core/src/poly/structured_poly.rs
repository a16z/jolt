use crate::field::JoltField;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

use super::commitment::commitment_scheme::CommitmentScheme;
use crate::{
    lasso::memory_checking::NoPreprocessing,
    utils::{errors::ProofVerifyError, transcript::ProofTranscript},
};

/// Encapsulates the pattern of a collection of related polynomials (e.g. those used to
/// prove instruction lookups in Jolt) that can be "batched" for more efficient
/// commitments/openings.
pub trait StructuredCommitment<C: CommitmentScheme>: Send + Sync + Sized {
    /// The batched commitment to these polynomials.
    type Commitment;

    /// Commits to batched polynomials.
    fn commit(&self, generators: &C::Setup) -> Self::Commitment;
}

/// Encapsulates the pattern of opening a batched polynomial commitment at a single point.
/// Note that there may be a one-to-many mapping from `StructuredCommitment` to `StructuredOpeningProof`:
/// different subset of the same polynomials may be opened at different points, resulting in
/// different opening proofs.
pub trait StructuredOpeningProof<F, C, Polynomials>:
    Sync + CanonicalSerialize + CanonicalDeserialize
where
    F: JoltField,
    C: CommitmentScheme<Field = F>,
    Polynomials: StructuredCommitment<C>,
{
    type Preprocessing = NoPreprocessing;
    type Proof: Sync + CanonicalSerialize + CanonicalDeserialize;

    /// Evaluates each of the given `polynomials` at the given `opening_point`.
    fn open(polynomials: &Polynomials, opening_point: &[F]) -> Self;

    /// Proves that the `polynomials`, evaluated at `opening_point`, output the values given
    /// by `openings`. The polynomials should already be committed by the prover.
    fn prove_openings(
        generators: &C::Setup,
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
        generators: &C::Setup,
        opening_proof: &Self::Proof,
        commitment: &Polynomials::Commitment,
        opening_point: &[F],
        transcript: &mut ProofTranscript,
    ) -> Result<(), ProofVerifyError>;
}
