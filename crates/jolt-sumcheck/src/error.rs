//! Error types for sumcheck protocol verification failures.

/// Errors that can occur during sumcheck verification.
///
/// Each variant corresponds to a distinct failure mode in the sumcheck
/// protocol, enabling the caller to diagnose exactly where verification
/// diverged.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum SumcheckError {
    /// Round check failed: the sum $s_i(0) + s_i(1)$ did not match the
    /// expected value carried forward from the previous round.
    #[error("round {round}: expected sum {expected}, got {actual}")]
    RoundCheckFailed {
        /// Zero-indexed round number where the check failed.
        round: usize,
        /// The expected sum (as a display string).
        expected: String,
        /// The computed sum $s_i(0) + s_i(1)$ (as a display string).
        actual: String,
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

    /// The number of round polynomials in the proof does not match
    /// the number of variables in the claim.
    #[error("expected {expected} rounds, proof contains {got}")]
    WrongNumberOfRounds {
        /// Expected number of rounds (equal to `num_vars`).
        expected: usize,
        /// Actual number of round polynomials in the proof.
        got: usize,
    },

    /// Batched verification received an empty claims slice.
    #[error("batched verification requires at least one claim")]
    EmptyClaims,
}
