use core::fmt::Debug;
use thiserror::Error;

#[derive(Error, Debug, Default)]
pub enum ProofVerifyError {
    #[error("Invalid input length, expected length {0} but got {1}")]
    InvalidInputLength(usize, usize),
    #[error("Input too large")]
    InputTooLarge,
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
}

#[derive(Error, Debug)]
pub enum ONNXError {
    #[error("Invalid ONNX model: {0}")]
    InvalidModel(String),
    #[error("Invalid ONNX node: {0}")]
    InvalidNode(String),
    #[error("Invalid ONNX input: {0}")]
    InvalidInput(String),
    #[error("Invalid ONNX output: {0}")]
    InvalidOutput(String),
    #[error("Invalid ONNX attribute: {0}")]
    InvalidAttribute(String),
    #[error("Invalid ONNX type: {0}")]
    InvalidType(String),
}
