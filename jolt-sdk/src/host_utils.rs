#[cfg(feature = "host")]
pub use jolt_core::host;
#[cfg(feature = "host")]
pub use jolt_core::zkvm::proof_serialization::serialize_and_print_size;
#[cfg(feature = "host")]
pub use jolt_core::zkvm::{prover::JoltProverPreprocessing, RV64IMACProver};

pub use common::jolt_device::{JoltDevice, MemoryConfig, MemoryLayout};
pub use jolt_core::ark_bn254::Fr as F;
pub use jolt_core::field::JoltField;
pub use jolt_core::guest;
pub use jolt_core::poly::commitment::dory::deserialize_ark_dory_proof_marked;
pub use jolt_core::poly::commitment::dory::deserialize_arkworks_verifier_setup_marked;
pub use jolt_core::poly::commitment::dory::DoryCommitmentScheme as PCS;
pub use jolt_core::zkvm::guest_serde::{GuestDeserialize, GuestSerialize};
pub use jolt_core::zkvm::{
    proof_serialization::JoltProof, verifier::JoltSharedPreprocessing,
    verifier::JoltVerifierPreprocessing, RV64IMACProof, RV64IMACVerifier, Serializable,
};

// Extra re-exports for fine-grained (de)serialization profiling in recursion.
pub use jolt_core::zkvm::bytecode::{BytecodePCMapper, BytecodePreprocessing};
pub use tracer::instruction::Instruction;

// Re-exports needed by the provable macro
pub use jolt_core::poly::commitment::commitment_scheme::CommitmentScheme;
pub use jolt_core::poly::commitment::dory::{DoryContext, DoryGlobals};
pub use jolt_core::poly::multilinear_polynomial::MultilinearPolynomial;
pub use jolt_core::zkvm::ram::populate_memory_states;
