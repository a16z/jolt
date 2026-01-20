use core::fmt::Debug;
use thiserror::Error;

#[derive(Error, Debug, Default)]
pub enum ProofVerifyError {
    // === Core input / layout ===
    #[error("Invalid input length, expected length {0} but got {1}")]
    InvalidInputLength(usize, usize),
    #[error("Input too large")]
    InputTooLarge,
    #[error("Output too large")]
    OutputTooLarge,
    #[error("Memory layout mismatch")]
    MemoryLayoutMismatch,

    // === Proof structure validation ===
    #[error("Trace length must be non-zero")]
    InvalidTraceLength,
    #[error("RAM K must be non-zero")]
    InvalidRamK,
    #[error("Bytecode K must be non-zero")]
    InvalidBytecodeK,
    #[error("Trace length {0} would overflow in next_power_of_two")]
    TraceLengthOverflow(usize),
    #[error("{context} count mismatch: expected {expected}, got {actual}")]
    CountMismatch {
        context: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("Proof structure invalid: expected {expected} rounds, got {actual}")]
    InvalidRoundCount { expected: usize, actual: usize },
    #[error("{context}: need {needed} elements, have {have}")]
    SliceTooShort {
        context: &'static str,
        needed: usize,
        have: usize,
    },

    // === Claim consistency errors ===
    #[error("Claim mismatch: {0}")]
    ClaimMismatch(&'static str),

    // === Address/memory errors ===
    #[error("Failed to remap address {0:#x}")]
    AddressRemapFailed(u64),

    // === Field arithmetic errors ===
    #[error("Field arithmetic error: {0}")]
    FieldArithmeticError(&'static str),

    // === Missing proof components ===
    #[error("Missing proof component: {0}")]
    MissingProofComponent(&'static str),

    // === Config validation ===
    #[error("Invalid read-write checking configuration: {0}")]
    InvalidReadWriteConfig(String),
    #[error("Invalid one-hot configuration: {0}")]
    InvalidOneHotConfig(String),

    // === Opening accumulator errors ===
    #[error("Missing committed polynomial opening: {0}")]
    MissingCommittedOpening(String),
    #[error("Missing virtual polynomial opening: {0}")]
    MissingVirtualOpening(String),
    #[error("Missing advice opening: {0}")]
    MissingAdviceOpening(String),

    // === Sumcheck errors ===
    #[error("No sumcheck instances provided")]
    NoSumcheckInstances,
    #[error("Sumcheck verification failed")]
    SumcheckVerificationError,
    #[error("Univariate-skip round verification failed")]
    UniSkipVerificationError,

    // === Cryptographic / PCS errors ===
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

    // === Serialization ===
    #[error("Serialization failed: {0}")]
    SerializationError(String),
    #[error("Deserialization failed: {0}")]
    DeserializationError(String),

    // === Catch-all ===
    #[error("Proof verification failed")]
    #[default]
    InternalError,
}
