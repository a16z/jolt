use core::fmt::Debug;
use thiserror::Error;

#[derive(Error, Debug, Default)]
pub enum ProofVerifyError {
    #[error("Invalid input length, expected length {0} but got {1}")]
    InvalidInputLength(usize, usize),
    #[error("Input too large")]
    InputTooLarge,
    #[error("Output too large")]
    OutputTooLarge,
    #[error("Memory layout mismatch")]
    MemoryLayoutMismatch,
    #[error("Proof verification failed")]
    #[default]
    InternalError,
    #[error("Compressed group element failed to decompress: {0:?}")]
    DecompressionError([u8; 32]),
    #[error("R1CS proof verification failed: {0}")]
    SpartanError(String),
    #[error("Length Error: SRS Length: {0}, Key Length: {1}")]
    KeyLengthError(usize, usize),
    #[error("Invalid key length: {0}, expected power of 2")]
    InvalidKeyLength(usize),
    #[error("Invalid opening proof -- the proof failed to verify")]
    InvalidOpeningProof,
    #[error("Dory proof verification failed: {0}")]
    DoryError(String),
    #[error("Sumcheck verification failed")]
    SumcheckVerificationError,
    #[error("Univariate-skip round verification failed")]
    UniSkipVerificationError,
}
