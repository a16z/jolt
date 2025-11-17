#[cfg(feature = "host")]
pub use jolt_core::host;
#[cfg(feature = "host")]
pub use jolt_core::zkvm::proof_serialization::serialize_and_print_size;

pub use common::jolt_device::{JoltDevice, MemoryConfig, MemoryLayout};
pub use jolt_core::ark_bn254::Fr as F;
pub use jolt_core::field::JoltField;
pub use jolt_core::guest;
pub use jolt_core::poly::commitment::dory::DoryCommitmentScheme as PCS;
pub use jolt_core::zkvm::{
    proof_serialization::JoltProof, prover::JoltProverPreprocessing,
    verifier::JoltVerifierPreprocessing, RV64IMACProof, RV64IMACProver, RV64IMACVerifier,
    Serializable,
};

// Re-exports needed by the provable macro
pub use jolt_core::poly::commitment::commitment_scheme::CommitmentScheme;
pub use jolt_core::poly::commitment::dory::DoryGlobals;
pub use jolt_core::poly::multilinear_polynomial::MultilinearPolynomial;
pub use jolt_core::zkvm::ram::populate_memory_states;
