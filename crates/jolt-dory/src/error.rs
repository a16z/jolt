//! Error types for the Dory commitment scheme.

/// Errors that can occur during Dory commitment scheme operations.
#[derive(Debug, thiserror::Error)]
pub enum DoryError {
    /// The polynomial exceeds the maximum size supported by the current parameters.
    #[error("polynomial has {num_vars} variables but params support at most {max_vars}")]
    PolynomialTooLarge {
        /// Number of variables in the polynomial.
        num_vars: usize,
        /// Maximum number of variables supported by the setup.
        max_vars: usize,
    },

    /// The dory-pcs backend returned an error during commitment.
    #[error("dory-pcs commitment failed: {0}")]
    CommitFailed(String),

    /// The dory-pcs backend returned an error during proof generation.
    #[error("dory-pcs proof generation failed: {0}")]
    ProveFailed(String),

    /// The dory-pcs backend returned an error during verification.
    #[error("dory-pcs verification failed: {0}")]
    VerifyFailed(String),

    /// The setup parameters are missing or invalid.
    #[error("invalid setup: {0}")]
    InvalidSetup(String),
}
