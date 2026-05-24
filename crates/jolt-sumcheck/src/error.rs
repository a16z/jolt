//! Error types for sumcheck protocol verification failures.

use jolt_field::FieldCore;

/// Errors that can occur during sumcheck verification.
///
/// Each variant corresponds to a distinct failure mode in the sumcheck
/// protocol, enabling the caller to diagnose exactly where verification
/// diverged.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum SumcheckError<F: FieldCore> {
    /// Round check failed: the domain sum did not match the expected value
    /// carried forward from the previous round.
    #[error("round {round}: expected sum {expected}, got {actual}")]
    RoundCheckFailed {
        /// Zero-indexed round number where the check failed.
        round: usize,
        /// The expected sum.
        expected: F,
        /// The computed domain sum.
        actual: F,
    },

    /// A round polynomial exceeded the declared degree bound.
    #[error("degree bound exceeded: degree {got}, max {max}")]
    DegreeBoundExceeded {
        /// Actual degree of the offending round polynomial.
        got: usize,
        /// Maximum allowed degree from the claim.
        max: usize,
    },

    /// A round polynomial encoded in compressed form had fewer than two
    /// coefficients, so there is no linear term to omit. Any valid
    /// compressed sumcheck round polynomial has degree ≥ 1.
    #[error("round {round}: compressed round polynomial requires >= 2 coefficients, got {got}")]
    CompressedPolynomialTooShort {
        /// Zero-indexed round number where the malformed polynomial appeared.
        round: usize,
        /// Actual number of coefficients received.
        got: usize,
    },

    /// The domain-sum coefficient vector must have exactly one scalar per
    /// round-polynomial coefficient.
    #[error("round {round}: expected {expected} round-sum coefficients, got {got}")]
    RoundSumCoefficientCountMismatch {
        /// Zero-indexed round number where the mismatch appeared.
        round: usize,
        /// Expected number of coefficients, equal to `degree + 1`.
        expected: usize,
        /// Actual number of coefficients supplied by the domain.
        got: usize,
    },

    /// An integer-domain sumcheck round used an invalid domain size.
    #[error("integer sumcheck domain size must be between 1 and i64::MAX, got {domain_size}")]
    InvalidIntegerDomain {
        /// Number of integer points in the domain.
        domain_size: usize,
    },

    /// The number of round polynomials in the proof does not match
    /// the number of variables in the claim.
    #[error("expected {expected} rounds, proof contains {got}")]
    WrongNumberOfRounds {
        /// Expected number of rounds (equal to `num_vars`).
        expected: usize,
        /// Actual number of round polynomials in the proof.
        got: usize,
    },

    /// A round witness did not contain any coefficients.
    #[error("round polynomial must contain at least one coefficient")]
    EmptyRoundCoefficients,

    /// The caller selected a verifier path that is incompatible with the proof
    /// wire encoding.
    #[error("wrong sumcheck proof encoding: expected {expected}, got {got}")]
    WrongProofEncoding {
        /// Expected proof encoding.
        expected: &'static str,
        /// Actual proof encoding.
        got: &'static str,
    },

    /// Batched verification received an empty claims slice.
    #[error("batched verification requires at least one claim")]
    EmptyClaims,

    /// A batched evaluation claim was asked for an impossible point slice.
    #[error("batched point range overflow: offset {offset}, num_vars {num_vars}")]
    BatchedPointRangeOverflow {
        /// Starting index into the batched challenge vector.
        offset: usize,
        /// Number of variables in the requested instance.
        num_vars: usize,
    },

    /// A batched evaluation claim did not contain enough challenges for the requested instance.
    #[error(
        "batched point out of range: offset {offset}, num_vars {num_vars}, total challenges {total}"
    )]
    BatchedPointOutOfRange {
        /// Starting index into the batched challenge vector.
        offset: usize,
        /// Number of variables in the requested instance.
        num_vars: usize,
        /// Total number of available batched challenges.
        total: usize,
    },

    /// The domain did not provide exactly one coefficient for summing a
    /// constant padding round.
    #[error("expected {expected} padding-scale coefficients, got {got}")]
    PaddingScaleCoefficientCountMismatch {
        /// Expected number of coefficients for a degree-0 round sum.
        expected: usize,
        /// Actual number of coefficients supplied by the domain.
        got: usize,
    },
}
