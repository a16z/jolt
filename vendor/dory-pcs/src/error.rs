/// Errors that can occur in Dory PCS operations
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum DoryError {
    /// The proof verification failed
    #[error("Invalid proof")]
    InvalidProof,

    /// Polynomial size is invalid for the given parameters
    #[error("Invalid polynomial size: expected {expected}, got {actual}")]
    InvalidSize {
        /// Expected size
        expected: usize,
        /// Actual size
        actual: usize,
    },

    /// Evaluation point has wrong dimension
    #[error("Invalid evaluation point dimension: expected {expected}, got {actual}")]
    InvalidPointDimension {
        /// Expected dimension
        expected: usize,
        /// Actual dimension
        actual: usize,
    },

    /// Invalid input parameters
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// Setup file not found or corrupted
    #[error("Invalid or missing URS file: {0}")]
    InvalidURS(String),
}
