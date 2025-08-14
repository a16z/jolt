/// Custom error types for Dory
#[derive(Debug, thiserror::Error)]
pub enum DoryError {
    #[error("Invalid proof")]
    InvalidProof,
    #[error("Invalid parameters")]
    InvalidParameters,
    #[error("Invalid commitment")]
    InvalidCommitment,
}
