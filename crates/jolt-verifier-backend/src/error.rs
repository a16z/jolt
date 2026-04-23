use jolt_openings::OpeningsError;
use thiserror::Error;

/// Errors raised by a [`FieldBackend`](crate::FieldBackend).
///
/// Most variants only fire from [`Native`](crate::Native) — non-native backends
/// (Tracing, R1CSGen) typically defer assertions into recorded constraints
/// rather than checking them eagerly.
#[derive(Debug, Error)]
pub enum BackendError {
    /// A [`FieldBackend::assert_eq`](crate::FieldBackend::assert_eq) check
    /// failed at runtime.
    ///
    /// Carries the `&'static str` context label passed by the caller so the
    /// failing site is identifiable without strings being attached to every
    /// scalar.
    #[error("backend assertion '{0}' failed")]
    AssertionFailed(&'static str),

    /// A [`FieldBackend::inverse`](crate::FieldBackend::inverse) was requested
    /// for a value the backend determined to be zero.
    #[error("backend inverse of zero (ctx: {0})")]
    InverseOfZero(&'static str),

    /// A `replay`-time PCS opening check
    /// ([`AstAssertion::OpeningHolds`](crate::AstAssertion::OpeningHolds))
    /// failed: `<PCS as CommitmentScheme>::verify` returned `Err` for the
    /// referenced [`AstOp::OpeningCheck`](crate::AstOp::OpeningCheck) node.
    ///
    /// `ctx` is the caller-supplied label (matching the assertion site);
    /// `source` is the underlying [`OpeningsError`].
    #[error("backend opening check '{ctx}' failed: {source}")]
    OpeningCheckFailed {
        /// Caller-supplied debug context (`AstAssertion::OpeningHolds.ctx`).
        ctx: &'static str,
        /// Underlying PCS verification failure.
        #[source]
        source: OpeningsError,
    },
}
