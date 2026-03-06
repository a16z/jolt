//! Abstract commitment scheme trait hierarchy.
//!
//! Three tiers of polynomial commitment scheme (PCS) interfaces:
//!
//! 1. [`CommitmentScheme`] — base trait for commit, open, and verify.
//!    Extends [`jolt_crypto::Commitment`] to inherit the `Output` associated type.
//! 2. [`AdditivelyHomomorphic`] — commitments can be linearly combined.
//! 3. [`StreamingCommitment`] — incremental/chunked commitment.

use jolt_crypto::Commitment;
use jolt_field::Field;
use jolt_transcript::Transcript;
use serde::{de::DeserializeOwned, Serialize};

use crate::error::OpeningsError;

/// Base polynomial commitment scheme for multilinear polynomials.
///
/// A PCS allows a prover to commit to a multilinear polynomial
/// $f : \mathbb{F}^n \to \mathbb{F}$ and later prove that $f(r) = v$
/// for a verifier-chosen point $r \in \mathbb{F}^n$.
///
/// Extends [`Commitment`] from `jolt-crypto` — the `Output` associated type
/// serves as the commitment value, shared with the vector commitment hierarchy.
///
/// No algebraic structure is assumed on commitments — this trait covers
/// both hash-based (FRI, Brakedown) and group-based (KZG, Dory) schemes.
///
/// Setup parameters are provided externally (not created by the trait).
/// Concrete types (e.g., `DoryScheme`) may provide inherent `setup_prover`/
/// `setup_verifier` methods, but these are not part of the generic interface.
pub trait CommitmentScheme: Commitment + Clone + Send + Sync + 'static {
    /// Scalar field of the polynomial.
    type Field: Field;

    /// A proof that a committed polynomial evaluates to a claimed value at a given point.
    type Proof: Clone + Send + Sync + Serialize + DeserializeOwned;

    /// Prover-side structured reference string or setup material.
    type ProverSetup: Clone + Send + Sync;

    /// Verifier-side structured reference string or setup material.
    type VerifierSetup: Clone + Send + Sync + Serialize + DeserializeOwned;

    /// Commits to a multilinear polynomial given its evaluation table over the Boolean hypercube.
    fn commit(evaluations: &[Self::Field], setup: &Self::ProverSetup) -> Self::Output;

    /// Produces an opening proof that the committed polynomial evaluates to `eval` at `point`.
    ///
    /// `evaluations` is the full evaluation table. The transcript must be in the same
    /// state as the verifier's transcript at this protocol step.
    fn open(
        evaluations: &[Self::Field],
        point: &[Self::Field],
        eval: Self::Field,
        setup: &Self::ProverSetup,
        transcript: &mut impl Transcript,
    ) -> Self::Proof;

    /// Verifies that the committed polynomial evaluates to `eval` at `point`.
    ///
    /// # Errors
    ///
    /// Returns [`OpeningsError::VerificationFailed`] if the proof is invalid.
    fn verify(
        commitment: &Self::Output,
        point: &[Self::Field],
        eval: Self::Field,
        proof: &Self::Proof,
        setup: &Self::VerifierSetup,
        transcript: &mut impl Transcript,
    ) -> Result<(), OpeningsError>;
}

/// Additively homomorphic commitment scheme (e.g., Dory, KZG).
///
/// When commitments live in an additive group, they can be linearly combined:
///
/// $$C_{\text{combined}} = \sum_{i=0}^{k-1} s_i \cdot C_i$$
///
/// This algebraic property enables batch reduction strategies like
/// [`RlcReduction`](crate::RlcReduction), but batching itself is not
/// part of this trait — it is a reduction concern.
pub trait AdditivelyHomomorphic: CommitmentScheme {
    /// Computes a linear combination of commitments:
    /// $C = \sum_i s_i \cdot C_i$.
    ///
    /// # Panics
    ///
    /// Panics if `commitments` and `scalars` have different lengths.
    fn combine(commitments: &[Self::Output], scalars: &[Self::Field]) -> Self::Output;
}

/// Streaming (chunked) commitment scheme.
///
/// Allows committing to a polynomial incrementally, one chunk of evaluations
/// at a time, without materializing the full evaluation table. Useful for
/// large polynomials that exceed available memory (e.g., Dory tier-1/tier-2).
///
/// The prover setup is passed explicitly to [`feed`](Self::feed) and
/// [`finish`](Self::finish) — no lifetime or ownership gymnastics required.
pub trait StreamingCommitment: CommitmentScheme {
    /// Intermediate state accumulated during streaming.
    type PartialCommitment: Clone + Send + Sync;

    /// Begins a streaming commitment session.
    fn begin(setup: &Self::ProverSetup) -> Self::PartialCommitment;

    /// Feeds the next chunk of evaluations into the partial commitment.
    fn feed(
        partial: &mut Self::PartialCommitment,
        chunk: &[Self::Field],
        setup: &Self::ProverSetup,
    );

    /// Finalizes the streaming session, producing the full commitment.
    fn finish(partial: Self::PartialCommitment, setup: &Self::ProverSetup) -> Self::Output;
}
