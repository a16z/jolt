//! Abstract commitment scheme trait hierarchy.
//!
//! Three tiers of polynomial commitment scheme (PCS) interfaces:
//!
//! 1. [`CommitmentScheme`] — base trait for commit, open, and verify.
//! 2. [`HomomorphicCommitmentScheme`] — extends base with additive homomorphism,
//!    enabling batch proofs via random linear combination (RLC).
//! 3. [`StreamingCommitmentScheme`] — extends base with incremental/chunked commitment.

use std::fmt::Debug;

use jolt_field::Field;
use jolt_poly::MultilinearPolynomial;
use jolt_transcript::Transcript;
use serde::{de::DeserializeOwned, Serialize};

use crate::error::OpeningsError;

/// Base polynomial commitment scheme for multilinear polynomials.
///
/// A PCS allows a prover to commit to a multilinear polynomial
/// $f : \mathbb{F}^n \to \mathbb{F}$ and later prove that $f(r) = v$
/// for a verifier-chosen point $r \in \mathbb{F}^n$.
///
/// No algebraic structure is assumed on commitments — this trait covers
/// both hash-based (FRI, Brakedown) and group-based (KZG, Dory) schemes.
pub trait CommitmentScheme: Clone + Send + Sync + 'static {
    /// Scalar field of the polynomial.
    type Field: Field;

    /// An opaque binding commitment to a polynomial.
    type Commitment: Clone + Send + Sync + Debug + Serialize + DeserializeOwned + PartialEq;

    /// A proof that a committed polynomial evaluates to a claimed value at a given point.
    type Proof: Clone + Send + Sync + Serialize + DeserializeOwned;

    /// Prover-side structured reference string or setup material.
    type ProverSetup: Clone + Send + Sync;

    /// Verifier-side structured reference string or setup material.
    type VerifierSetup: Clone + Send + Sync + Serialize + DeserializeOwned;

    /// Protocol name for domain separation in transcripts.
    fn protocol_name() -> &'static str;

    /// Generates prover setup parameters supporting polynomials up to $2^{\texttt{max\_size}}$ evaluations.
    fn setup_prover(max_size: usize) -> Self::ProverSetup;

    /// Generates verifier setup parameters supporting polynomials up to $2^{\texttt{max\_size}}$ evaluations.
    fn setup_verifier(max_size: usize) -> Self::VerifierSetup;

    /// Commits to a multilinear polynomial, producing a binding commitment.
    fn commit(
        poly: &impl MultilinearPolynomial<Self::Field>,
        setup: &Self::ProverSetup,
    ) -> Self::Commitment;

    /// Produces an opening proof that the committed polynomial evaluates to `eval` at `point`.
    ///
    /// The transcript must be in the same state as the verifier's transcript at this
    /// protocol step.
    fn prove(
        poly: &impl MultilinearPolynomial<Self::Field>,
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
        commitment: &Self::Commitment,
        point: &[Self::Field],
        eval: Self::Field,
        proof: &Self::Proof,
        setup: &Self::VerifierSetup,
        transcript: &mut impl Transcript,
    ) -> Result<(), OpeningsError>;
}

/// Additively homomorphic commitment scheme (e.g., Dory, KZG).
///
/// When commitments live in an additive group, multiple opening claims at the
/// same evaluation point can be batched into a single proof via random linear
/// combination:
///
/// $$C_{\text{combined}} = \sum_{i=0}^{k-1} \rho^i \cdot C_i, \quad
///   v_{\text{combined}} = \sum_{i=0}^{k-1} \rho^i \cdot v_i$$
///
/// where $\rho$ is a Fiat-Shamir challenge.
pub trait HomomorphicCommitmentScheme: CommitmentScheme {
    /// A batched opening proof covering multiple polynomials.
    type BatchedProof: Clone + Send + Sync + Serialize + DeserializeOwned;

    /// Computes a linear combination of commitments:
    /// $C = \sum_i s_i \cdot C_i$.
    ///
    /// # Panics
    ///
    /// Panics if `commitments` and `scalars` have different lengths.
    fn combine_commitments(
        commitments: &[Self::Commitment],
        scalars: &[Self::Field],
    ) -> Self::Commitment;

    /// Produces a batched opening proof for multiple polynomials at their respective points.
    ///
    /// # Panics
    ///
    /// Panics if `polynomials`, `points`, and `evals` have different lengths.
    fn batch_prove(
        polynomials: &[&dyn MultilinearPolynomial<Self::Field>],
        points: &[Vec<Self::Field>],
        evals: &[Self::Field],
        setup: &Self::ProverSetup,
        transcript: &mut impl Transcript,
    ) -> Self::BatchedProof;

    /// Verifies a batched opening proof.
    ///
    /// # Errors
    ///
    /// Returns [`OpeningsError::VerificationFailed`] if any claim in the batch is invalid.
    fn batch_verify(
        commitments: &[Self::Commitment],
        points: &[Vec<Self::Field>],
        evals: &[Self::Field],
        proof: &Self::BatchedProof,
        setup: &Self::VerifierSetup,
        transcript: &mut impl Transcript,
    ) -> Result<(), OpeningsError>;
}

/// Streaming (chunked) commitment scheme.
///
/// Allows committing to a polynomial incrementally, one chunk of evaluations
/// at a time, without materializing the full evaluation table. Useful for
/// large polynomials that exceed available memory (e.g., Dory tier-1/tier-2).
pub trait StreamingCommitmentScheme: CommitmentScheme {
    /// Intermediate state accumulated during streaming.
    type PartialCommitment: Clone + Send + Sync;

    /// Begins a streaming commitment session.
    fn begin_streaming(setup: &Self::ProverSetup) -> Self::PartialCommitment;

    /// Feeds the next chunk of evaluations into the partial commitment.
    fn stream_chunk(partial: &mut Self::PartialCommitment, chunk: &[Self::Field]);

    /// Finalizes the streaming session, producing the full commitment.
    fn finalize_streaming(partial: Self::PartialCommitment) -> Self::Commitment;
}
