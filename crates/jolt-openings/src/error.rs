//! Error types for polynomial commitment scheme operations.
//!
//! These errors can arise during setup (invalid or undersized parameters),
//! proving (polynomial exceeds setup capacity), or verification (opening
//! proof rejection, commitment mismatch).

/// Errors arising from polynomial commitment scheme operations.
#[derive(Debug, thiserror::Error)]
pub enum OpeningsError {
    /// An opening proof failed verification against the claimed evaluation.
    #[error("opening proof verification failed")]
    VerificationFailed,

    /// The commitment provided by the verifier does not match the expected value.
    #[error("commitment mismatch: expected {expected}, got {actual}")]
    CommitmentMismatch {
        /// String representation of the expected commitment.
        expected: String,
        /// String representation of the actual commitment.
        actual: String,
    },

    /// The setup parameters are invalid for the requested operation.
    #[error("invalid setup parameters: {0}")]
    InvalidSetup(String),

    /// The polynomial exceeds the maximum size supported by the setup.
    #[error("polynomial size {poly_size} exceeds setup max {setup_max}")]
    PolynomialTooLarge {
        /// Size of the polynomial (number of evaluations).
        poly_size: usize,
        /// Maximum size supported by the setup parameters.
        setup_max: usize,
    },
}
