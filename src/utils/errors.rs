use core::fmt::Debug;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ProofVerifyError {
  #[error("Proof verification failed")]
  InternalError,
  #[error("Compressed group element failed to decompress: {0:?}")]
  DecompressionError([u8; 32]),
}

impl Default for ProofVerifyError {
  fn default() -> Self {
    ProofVerifyError::InternalError
  }
}
