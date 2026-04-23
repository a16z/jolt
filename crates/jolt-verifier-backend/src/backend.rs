use jolt_field::Field;
use jolt_transcript::Transcript;

use crate::error::BackendError;

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

/// Backend abstraction for verifier-side field arithmetic.
///
/// Every method on the verifier's hot path (`add`, `sub`, `mul`, `assert_eq`,
/// ...) is dispatched through this trait so the same code can run as native
/// field ops, as a recorded AST (Tracing), or as emitted R1CS constraints
/// (R1CSGen). See the crate-level docs for the rationale.
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
///   univariate Horner) live as free functions in [`crate::helpers`] so they
///   compose with any backend without trait surface bloat.
pub trait FieldBackend {
    /// Underlying prime field that the verifier's claims live in.
    type F: Field;

    /// Backend-specific handle to a field-valued scalar.
    ///
    /// For [`Native`](crate::Native) this is just `F`. For Tracing it is an
    /// `ExprId`-style node handle. For R1CSGen it is a witness-vector index.
    type Scalar: Clone + std::fmt::Debug;

    /// Fiat-Shamir transcript implementation paired with this backend.
    ///
    /// For [`Native`](crate::Native) this is the underlying
    /// `Blake2bTranscript<F>` directly. For Tracing it wraps the same
    /// transcript but additionally records every absorb/squeeze into the
    /// backend's [`AstGraph`](crate::AstGraph), so the symbolic execution
    /// trace covers transcript operations as well as field arithmetic.
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
    /// For Tracing the returned transcript shares the backend's [`AstGraph`]
    /// so every absorb and squeeze is recorded into the same DAG that
    /// arithmetic ops land in. For Native the returned transcript is just
    /// a bare `Blake2bTranscript`.
    fn new_transcript(&mut self, label: &'static [u8]) -> Self::Transcript;

    /// Squeeze a Fiat-Shamir challenge through the backend's transcript.
    ///
    /// Returns both the raw [`Self::F`] (so the caller can keep driving the
    /// transcript with native APIs) and a backend [`Self::Scalar`] that
    /// references the AST node carrying the challenge value. For Native
    /// these are the same field element; for Tracing the scalar is a node
    /// handle to the [`AstOp::TranscriptChallengeValue`](crate::AstOp::TranscriptChallengeValue)
    /// just emitted by the transcript, so subsequent field arithmetic links
    /// to the squeeze instead of producing a fresh `Wrap` node.
    fn squeeze(
        &mut self,
        transcript: &mut Self::Transcript,
        label: &'static str,
    ) -> (Self::F, Self::Scalar);
}
