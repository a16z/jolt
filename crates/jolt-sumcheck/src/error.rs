//! Error types for sumcheck protocol verification failures.

/// Errors that can occur during sumcheck verification.
///
/// Each variant corresponds to a distinct failure mode in the sumcheck
/// protocol, enabling the caller to diagnose exactly where verification
/// diverged.
#[derive(Debug, thiserror::Error)]
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

    /// The final evaluation did not match the oracle query.
    #[error("final evaluation mismatch")]
    FinalEvalMismatch,

    /// A round polynomial exceeded the declared degree bound.
    #[error("degree bound exceeded: degree {got}, max {max}")]
    DegreeBoundExceeded {
        /// Actual degree of the offending round polynomial.
        got: usize,
        /// Maximum allowed degree from the claim.
        max: usize,
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
}
