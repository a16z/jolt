use core::fmt::Debug;
use thiserror::Error;

#[derive(Error, Debug, Default)]
pub enum ProofVerifyError {
  #[default]
  #[error("Proof verification failed")]
  InternalError,
  #[error("Compressed group element failed to decompress: {0:?}")]
  DecompressionError([u8; 32]),
}