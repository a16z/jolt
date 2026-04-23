//! Backend abstraction for verifier-side **commitment** operations.
//!
//! Where [`FieldBackend`](crate::FieldBackend) lifts scalar arithmetic off
//! the concrete field, [`CommitmentBackend`] lifts the three commitment-
//! shaped operations the verifier performs:
//!
//! 1. **Wrap** a commitment value (from the proof or the verifying key)
//!    into the backend's commitment representation.
//! 2. **Absorb** a commitment into the Fiat-Shamir transcript.
//! 3. **Verify** an opening claim against a commitment, point, evaluation,
//!    and opening proof.
//!
//! The trait is **deliberately minimal** and **PCS-family agnostic**: it
//! never names a curve, a pairing, an MSM, or a linear combination of
//! commitments. Anything PCS-specific (RLC batching, FRI folding, lattice
//! aggregation) lives inside per-PCS implementations of
//! [`OpeningReduction::reduce_verifier_with_backend`](jolt_openings::OpeningReduction)
//! (added in step 2.5 of the `CommitmentBackend` cutover, see
//! `specs/1461`), not on this trait.
//!
//! # Why this lives next to `FieldBackend`
//!
//! `CommitmentBackend` mirrors `FieldBackend` one-for-one:
//!
//! | Backend     | `Scalar`    | `Commitment`              | `verify_opening`           |
//! |-------------|-------------|---------------------------|----------------------------|
//! | [`Native`](crate::Native)   | `F`         | `PCS::Output`             | calls `PCS::verify`        |
//! | `Tracing`   | `AstNodeId` | `AstNodeId`               | records `OpeningCheck` AST |
//! | `R1CSGen`   | `LcId`      | recursion-side group var  | emits in-circuit verifier  |
//!
//! Co-locating the two backends keeps a single `&mut backend` argument
//! threading through `verify_with_backend`, instead of forcing the caller
//! to manage two parallel handles.
//!
//! # Polymorphism over `PCS`
//!
//! The trait is generic over a [`CommitmentScheme`] so the verifier can
//! continue to be parameterised by the PCS family. The bound
//! `PCS: CommitmentScheme<Field = Self::F>` keeps the backend's field
//! aligned with the PCS's field — without it, the transcript challenges
//! would not type-check across the boundary.
//!
//! # No `GroupBackend`
//!
//! The earlier Phase 2 sketch (`GroupBackend` with low-level `MSM` /
//! `pairing` primitives) was rejected as PCS-specific and not
//! representable by hash- or lattice-based schemes. See `specs/1461`
//! "Phase 2 Amendment" for the full rationale.

use jolt_openings::{CommitmentScheme, OpeningsError};
use jolt_transcript::{AppendToTranscript, Transcript};

use crate::backend::{CommitmentOrigin, FieldBackend};
use crate::tracing::SchemeTag;

/// Backend abstraction for verifier-side commitment operations.
///
/// See the module docs for context. All methods take `&mut self` so AST-
/// and constraint-emitting backends can mutate their internal recorders
/// while the verifier code stays oblivious to whether it is running
/// natively, tracing into an AST, or emitting R1CS.
///
/// # Implementation contract
///
/// - `wrap_commitment` is the *only* way the verifier injects a raw
///   `PCS::Output` into a [`CommitmentBackend`]. AST backends record
///   the wrap; native backends pass it through.
/// - `absorb_commitment` MUST keep the underlying transcript bit-
///   identical across backends so squeezed challenges replay correctly.
///   For Tracing this means forwarding to the inner `Blake2bTranscript`
///   *and* recording a [`AstOp::TranscriptAbsorbCommitment`](crate::AstOp::TranscriptAbsorbCommitment)
///   node.
/// - `verify_opening` is the verifier's only entry point into the PCS's
///   `verify` routine. Native invokes it eagerly; Tracing defers it via
///   an [`AstOp::OpeningCheck`](crate::AstOp::OpeningCheck) node + an
///   [`AstAssertion::OpeningHolds`](crate::AstAssertion::OpeningHolds)
///   obligation.
pub trait CommitmentBackend<PCS>: FieldBackend
where
    PCS: CommitmentScheme<Field = <Self as FieldBackend>::F>,
    PCS::Output: AppendToTranscript,
    Self::Transcript: Transcript<Challenge = <Self as FieldBackend>::F>,
{
    /// Backend-side handle for a commitment.
    ///
    /// - `Native::Commitment = PCS::Output` (zero-overhead identity).
    /// - `Tracing::Commitment = AstNodeId` (handle into the recorded
    ///   AST; the raw `PCS::Output` is held in a side vector keyed by
    ///   the [`CommitmentWrap`](crate::AstOp::CommitmentWrap) node id).
    type Commitment: Clone + std::fmt::Debug;

    /// Wraps a raw commitment value into the backend's representation,
    /// labelling it with [`CommitmentOrigin`] for provenance.
    ///
    /// AST backends record a [`AstOp::CommitmentWrap`](crate::AstOp::CommitmentWrap)
    /// node and return its id; native backends return the input by value.
    fn wrap_commitment(
        &mut self,
        value: PCS::Output,
        origin: CommitmentOrigin,
        label: &'static str,
    ) -> Self::Commitment;

    /// Absorbs a commitment into the supplied transcript.
    ///
    /// `label` identifies the absorbed datum within the transcript's
    /// labelled-domain encoding; backends MUST forward this to the
    /// underlying transcript verbatim so squeezed challenges replay
    /// across backends.
    fn absorb_commitment(
        &mut self,
        transcript: &mut Self::Transcript,
        commitment: &Self::Commitment,
        label: &'static [u8],
    );

    /// Verifies a single opening claim against `commitment`.
    ///
    /// `point` and `claim` are backend-wrapped scalars. `proof`, `vk`,
    /// and the live `transcript` are passed through to `<PCS as
    /// CommitmentScheme>::verify` by native backends, or recorded as
    /// an [`AstOp::OpeningCheck`](crate::AstOp::OpeningCheck) node by
    /// AST backends.
    ///
    /// `scheme_tag` is a `&'static str` discriminator (e.g. `"dory"`,
    /// `"hyperkzg"`, `"mock"`) that downstream consumers (Lean export,
    /// recursion circuits) dispatch on. Native backends ignore it; AST
    /// backends record it on the emitted `OpeningCheck` node.
    ///
    /// **Batching is the PCS's responsibility, not this trait's.** The
    /// verifier reduces a batch of claims to a single combined claim
    /// via `OpeningReduction::reduce_verifier_with_backend` (added in
    /// step 2.5 of the cutover) before invoking `verify_opening`.
    #[allow(
        clippy::too_many_arguments,
        reason = "mirrors CommitmentScheme::verify (vk, c, point, claim, proof, transcript) plus scheme_tag for AST consumers; collapsing into a struct buys nothing"
    )]
    fn verify_opening(
        &mut self,
        vk: &PCS::VerifierSetup,
        commitment: &Self::Commitment,
        point: &[Self::Scalar],
        claim: &Self::Scalar,
        proof: &PCS::Proof,
        transcript: &mut Self::Transcript,
        scheme_tag: SchemeTag,
    ) -> Result<(), OpeningsError>;
}
