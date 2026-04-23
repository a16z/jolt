//! Backend abstraction for the Jolt verifier.
//!
//! Two parallel traits lift the verifier off concrete primitives:
//!
//! - [`FieldBackend`] — every scalar operation, every Fiat-Shamir
//!   transcript event, and every equality assertion.
//! - [`CommitmentBackend`] — every PCS-shaped operation (wrap, unwrap,
//!   absorb, opening check). PCS-family agnostic by design: no curves,
//!   pairings, MSMs, or commitment linear combinations on its surface.
//!
//! The traits live in `jolt-openings` (rather than the higher-level
//! `jolt-verifier-backend` crate that ships their concrete implementations)
//! so [`crate::OpeningReduction::reduce_verifier_with_backend`] can name
//! `CommitmentBackend<Self>` directly. Per-PCS reduction impls (Dory,
//! HyperKZG, Mock) thus stay in their own crates without the verifier
//! crate having to expose a parallel "backend opening reduction" trait.
//! Concrete backend implementations (`Native`, `Tracing`) live in
//! `jolt-verifier-backend`.
//!
//! See `crates/jolt-verifier-backend/src/lib.rs` for the public docs that
//! describe how the trait surface is consumed end-to-end.

use jolt_field::Field;
use jolt_transcript::{AppendToTranscript, Transcript};
use thiserror::Error;

use crate::error::OpeningsError;
use crate::schemes::CommitmentScheme;

/// Provenance label for a wrapped scalar.
///
/// Native ignores this. Tracing / R1CS use it to label witness rows so the
/// resulting AST or constraint system can be audited and partitioned into
/// public-input vs. proof-data vs. challenge regions.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ScalarOrigin {
    /// Constant baked into the verification key (R1CS public input).
    Public,
    /// Field element pulled from the proof — must be opened/checked elsewhere.
    Proof,
    /// Field element squeezed from the Fiat-Shamir transcript.
    Challenge,
}

/// Provenance label for a wrapped commitment, mirroring [`ScalarOrigin`].
///
/// Most commitments the verifier sees are `Proof`-origin (sent by the
/// prover and absorbed into the transcript). `Public`-origin is reserved
/// for commitments that are part of the verifying key (e.g., a baked-in
/// commitment to a setup polynomial).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum CommitmentOrigin {
    /// Commitment baked into the verification key.
    Public,
    /// Commitment pulled from the proof.
    Proof,
}

/// Errors raised by a [`FieldBackend`] / [`CommitmentBackend`].
///
/// Most variants only fire from the `Native` backend — non-native backends
/// (Tracing, R1CSGen) typically defer assertions into recorded constraints
/// rather than checking them eagerly.
#[derive(Debug, Error)]
pub enum BackendError {
    /// A [`FieldBackend::assert_eq`] check failed at runtime.
    ///
    /// Carries the `&'static str` context label passed by the caller so the
    /// failing site is identifiable without strings being attached to every
    /// scalar.
    #[error("backend assertion '{0}' failed")]
    AssertionFailed(&'static str),

    /// A [`FieldBackend::inverse`] was requested for a value the backend
    /// determined to be zero.
    #[error("backend inverse of zero (ctx: {0})")]
    InverseOfZero(&'static str),

    /// A `replay`-time PCS opening check failed: `<PCS as
    /// CommitmentScheme>::verify` returned `Err` for the referenced
    /// `OpeningCheck` AST node.
    ///
    /// `ctx` is the caller-supplied label (matching the assertion site);
    /// `source` is the underlying [`OpeningsError`].
    #[error("backend opening check '{ctx}' failed: {source}")]
    OpeningCheckFailed {
        /// Caller-supplied debug context.
        ctx: &'static str,
        /// Underlying PCS verification failure.
        #[source]
        source: OpeningsError,
    },
}

/// Backend abstraction for verifier-side field arithmetic.
///
/// Every method on the verifier's hot path (`add`, `sub`, `mul`, `assert_eq`,
/// ...) is dispatched through this trait so the same code can run as native
/// field ops, as a recorded AST (Tracing), or as emitted R1CS constraints
/// (R1CSGen).
///
/// # Implementation notes
///
/// - `Scalar` must be `Clone` (not `Copy`) so AST-style backends can use
///   reference-counted node handles. The native impl uses `Scalar = F` and
///   pays no clone cost.
/// - All arithmetic methods take `&Self::Scalar` so AST backends can freely
///   share children. Native impls dereference and forward.
/// - `assert_eq` returns `Result` so Native can fail loudly. Tracing /
///   R1CSGen will normally return `Ok(())` after recording the constraint.
/// - The trait is intentionally minimal. Higher-level shapes (eq evaluation,
///   univariate Horner) live as free functions in `jolt-verifier-backend`'s
///   helpers module so they compose with any backend without trait surface
///   bloat.
pub trait FieldBackend {
    /// Underlying prime field that the verifier's claims live in.
    type F: Field;

    /// Backend-specific handle to a field-valued scalar.
    ///
    /// For the `Native` backend this is just `F`. For Tracing it is an
    /// `ExprId`-style node handle. For R1CSGen it is a witness-vector index.
    type Scalar: Clone + std::fmt::Debug;

    /// Fiat-Shamir transcript implementation paired with this backend.
    ///
    /// For the `Native` backend this is the underlying `Blake2bTranscript<F>`
    /// directly. For Tracing it wraps the same transcript but additionally
    /// records every absorb/squeeze into the backend's AST graph, so the
    /// symbolic execution trace covers transcript operations as well as
    /// field arithmetic.
    type Transcript: Transcript<Challenge = Self::F>;

    /// Wraps a native field element into the backend's scalar representation,
    /// labelling it with [`ScalarOrigin`] for provenance.
    fn wrap(&mut self, value: Self::F, origin: ScalarOrigin, label: &'static str) -> Self::Scalar;

    /// Wraps a public verifier-key value. Equivalent to
    /// `wrap(v, ScalarOrigin::Public, label)`.
    #[inline]
    fn wrap_public(&mut self, value: Self::F, label: &'static str) -> Self::Scalar {
        self.wrap(value, ScalarOrigin::Public, label)
    }

    /// Wraps a value extracted from the proof. The verifier has not yet
    /// committed to it via the transcript, so backends should treat it as
    /// untrusted.
    #[inline]
    fn wrap_proof(&mut self, value: Self::F, label: &'static str) -> Self::Scalar {
        self.wrap(value, ScalarOrigin::Proof, label)
    }

    /// Wraps a Fiat-Shamir challenge that has already been squeezed from the
    /// transcript.
    #[inline]
    fn wrap_challenge(&mut self, value: Self::F, label: &'static str) -> Self::Scalar {
        self.wrap(value, ScalarOrigin::Challenge, label)
    }

    /// Wraps an integer literal as a constant scalar.
    fn const_i128(&mut self, v: i128) -> Self::Scalar;

    /// Returns the field's additive identity wrapped as a scalar.
    #[inline]
    fn const_zero(&mut self) -> Self::Scalar {
        self.const_i128(0)
    }

    /// Returns the field's multiplicative identity wrapped as a scalar.
    #[inline]
    fn const_one(&mut self) -> Self::Scalar {
        self.const_i128(1)
    }

    /// Returns `a + b`.
    fn add(&mut self, a: &Self::Scalar, b: &Self::Scalar) -> Self::Scalar;
    /// Returns `a - b`.
    fn sub(&mut self, a: &Self::Scalar, b: &Self::Scalar) -> Self::Scalar;
    /// Returns `a * b`.
    fn mul(&mut self, a: &Self::Scalar, b: &Self::Scalar) -> Self::Scalar;
    /// Returns `-a`.
    fn neg(&mut self, a: &Self::Scalar) -> Self::Scalar;

    /// Default impl is `mul(a, a)`; backends may override for a single-node
    /// AST encoding or a dedicated R1CS shortcut.
    #[inline]
    fn square(&mut self, a: &Self::Scalar) -> Self::Scalar {
        self.mul(a, a)
    }

    /// Multiplicative inverse.
    ///
    /// Native errors with [`BackendError::InverseOfZero`] when the underlying
    /// field element is zero. Non-native backends are free to return a
    /// symbolic inverse and emit the standard `inv * a = 1` constraint.
    fn inverse(
        &mut self,
        a: &Self::Scalar,
        ctx: &'static str,
    ) -> Result<Self::Scalar, BackendError>;

    /// Asserts `a == b`.
    ///
    /// Native compares the underlying field elements. Tracing / R1CSGen
    /// record an `a - b = 0` constraint and return `Ok(())`.
    fn assert_eq(
        &mut self,
        a: &Self::Scalar,
        b: &Self::Scalar,
        ctx: &'static str,
    ) -> Result<(), BackendError>;

    /// Reads back the concrete field element underlying a scalar.
    ///
    /// Implemented only by backends that hold a witness assignment
    /// (Native, R1CSGen). Tracing-only backends should return `None`.
    /// Used by the verifier when it needs to forward a native value into
    /// the next protocol step (e.g., feeding a sumcheck output into an opening
    /// claim that wants the raw `F`).
    fn unwrap(&self, scalar: &Self::Scalar) -> Option<Self::F>;

    /// Construct a fresh transcript bound to this backend.
    ///
    /// For Tracing the returned transcript shares the backend's AST graph so
    /// every absorb and squeeze is recorded into the same DAG that arithmetic
    /// ops land in. For Native the returned transcript is just a bare
    /// `Blake2bTranscript`.
    fn new_transcript(&mut self, label: &'static [u8]) -> Self::Transcript;

    /// Squeeze a Fiat-Shamir challenge through the backend's transcript.
    ///
    /// Returns both the raw [`Self::F`] (so the caller can keep driving the
    /// transcript with native APIs) and a backend [`Self::Scalar`] that
    /// references the AST node carrying the challenge value. For Native
    /// these are the same field element; for Tracing the scalar is a node
    /// handle to the challenge-value node just emitted by the transcript,
    /// so subsequent field arithmetic links to the squeeze instead of
    /// producing a fresh `Wrap` node.
    fn squeeze(
        &mut self,
        transcript: &mut Self::Transcript,
        label: &'static str,
    ) -> (Self::F, Self::Scalar);
}

/// Backend abstraction for verifier-side commitment operations.
///
/// Where [`FieldBackend`] lifts scalar arithmetic off the concrete field,
/// [`CommitmentBackend`] lifts the four commitment-shaped operations the
/// verifier performs:
///
/// 1. **Wrap** a commitment value (from the proof or the verifying key)
///    into the backend's commitment representation.
/// 2. **Unwrap** a backend commitment back to its concrete `PCS::Output`.
/// 3. **Absorb** a commitment into the Fiat-Shamir transcript.
/// 4. **Verify** an opening claim against a commitment, point, evaluation,
///    and opening proof.
///
/// The trait is **deliberately minimal** and **PCS-family agnostic**: it
/// never names a curve, a pairing, an MSM, or a linear combination of
/// commitments. Anything PCS-specific (RLC batching, FRI folding, lattice
/// aggregation) lives inside per-PCS implementations of
/// [`crate::OpeningReduction::reduce_verifier_with_backend`], not on this
/// trait.
///
/// # Implementation contract
///
/// - `wrap_commitment` is the *only* way the verifier injects a raw
///   `PCS::Output` into a [`CommitmentBackend`]. AST backends record the
///   wrap; native backends pass it through.
/// - `unwrap_commitment` is the *only* way an opening-reduction impl reads
///   back a concrete `PCS::Output` from a backend handle. Native clones it
///   directly; AST backends pull it off the originating wrap node.
/// - `absorb_commitment` MUST keep the underlying transcript bit-identical
///   across backends so squeezed challenges replay correctly.
/// - `verify_opening` is the verifier's only entry point into the PCS's
///   `verify` routine. Native invokes it eagerly; Tracing defers it via an
///   `OpeningCheck` AST node + an `OpeningHolds` obligation.
pub trait CommitmentBackend<PCS>: FieldBackend
where
    PCS: CommitmentScheme<Field = <Self as FieldBackend>::F>,
    PCS::Output: AppendToTranscript,
    Self::Transcript: Transcript<Challenge = <Self as FieldBackend>::F>,
{
    /// Backend-side handle for a commitment.
    ///
    /// - `Native::Commitment = PCS::Output` (zero-overhead identity).
    /// - `Tracing::Commitment = AstNodeId` — handle into the recorded
    ///   AST. The raw `PCS::Output` is *inlined* on the originating
    ///   `CommitmentWrap` node.
    type Commitment: Clone + std::fmt::Debug;

    /// Wraps a raw commitment value into the backend's representation,
    /// labelling it with [`CommitmentOrigin`] for provenance.
    fn wrap_commitment(
        &mut self,
        value: PCS::Output,
        origin: CommitmentOrigin,
        label: &'static str,
    ) -> Self::Commitment;

    /// Unwraps a backend commitment back to its concrete `PCS::Output`.
    ///
    /// Native: identity clone. Tracing: looks up the inlined value on the
    /// originating `CommitmentWrap` node.
    ///
    /// This method exists so per-PCS implementations of
    /// [`crate::OpeningReduction::reduce_verifier_with_backend`] can
    /// combine commitments via the PCS algebra (e.g.
    /// `AdditivelyHomomorphic::combine`) and then re-wrap the result with
    /// [`Self::wrap_commitment`]. It is **not** intended for use inside the
    /// verifier proper: the verifier must remain oblivious to commitment
    /// values to preserve symbolic faithfulness.
    fn unwrap_commitment(&self, commitment: &Self::Commitment) -> PCS::Output;

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
    /// `point` and `claim` are backend-wrapped scalars. `proof`, `vk`, and
    /// the live `transcript` are passed through to `<PCS as
    /// CommitmentScheme>::verify` by native backends, or recorded as an
    /// `OpeningCheck` AST node by AST backends. The `PCS` is statically known
    /// via the `CommitmentBackend<PCS>` type parameter, so downstream
    /// consumers walking an `AstGraph<PCS>` learn the scheme from the type
    /// alone.
    ///
    /// **Batching is the PCS's responsibility, not this trait's.** The
    /// verifier reduces a batch of claims to a single combined claim via
    /// [`crate::OpeningReduction::reduce_verifier_with_backend`] before
    /// invoking `verify_opening`.
    fn verify_opening(
        &mut self,
        vk: &PCS::VerifierSetup,
        commitment: &Self::Commitment,
        point: &[Self::Scalar],
        claim: &Self::Scalar,
        proof: &PCS::Proof,
        transcript: &mut Self::Transcript,
    ) -> Result<(), OpeningsError>;
}
