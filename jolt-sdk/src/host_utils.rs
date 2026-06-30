#[cfg(feature = "host")]
pub use jolt_prover_legacy::host;
#[cfg(feature = "host")]
pub use jolt_prover_legacy::zkvm::{prover::JoltProverPreprocessing, RV64IMACProver};
#[cfg(feature = "host")]
pub type ProofTranscript = jolt_prover_legacy::zkvm::RV64IMACSponge;
#[cfg(feature = "host")]
pub use jolt_program::execution::{
    ExecutionBackend, OwnedTrace, TraceError, TraceInputs, TraceOutput, TraceSource,
};
#[cfg(feature = "host")]
pub use tracer::TracerBackend;

pub use common::jolt_device::{JoltDevice, MemoryConfig, MemoryLayout};
pub use jolt_prover_legacy::ark_bn254::Fr as F;
pub use jolt_prover_legacy::curve::Bn254Curve as Curve;
pub use jolt_prover_legacy::curve::JoltCurve;
pub use jolt_prover_legacy::field::JoltField;
pub use jolt_prover_legacy::guest;
pub use jolt_prover_legacy::poly::commitment::dory::DoryCommitmentScheme as PCS;
pub use jolt_prover_legacy::zkvm::{
    bytecode::PreprocessingError,
    preprocessing::JoltSharedPreprocessing,
    program::{CommittedProgramProverData, ProgramPreprocessing},
    Serializable,
};
pub use jolt_prover_legacy::AdviceTape;
pub use jolt_transcript::DEFAULT_JOLT_SESSION;

#[cfg(any(feature = "host", feature = "guest-verifier"))]
pub type VerifierPCS = jolt_dory::DoryScheme;
#[cfg(any(feature = "host", feature = "guest-verifier"))]
pub type VerifierVC = jolt_crypto::Pedersen<jolt_crypto::Bn254G1>;
#[cfg(any(feature = "host", feature = "guest-verifier"))]
pub type VerifierField = jolt_field::Fr;
#[cfg(any(feature = "host", feature = "guest-verifier"))]
pub type VerifierTranscript = jolt_transcript::Blake2b512;
#[cfg(any(feature = "host", feature = "guest-verifier"))]
pub type JoltVerifierPreprocessing =
    jolt_verifier::JoltVerifierPreprocessing<VerifierPCS, VerifierVC>;
#[cfg(any(feature = "host", feature = "guest-verifier"))]
pub type RV64IMACProof = jolt_verifier::JoltProof<VerifierPCS>;
#[cfg(any(feature = "host", feature = "guest-verifier"))]
pub type JoltProof = RV64IMACProof;
#[cfg(any(feature = "host", feature = "guest-verifier"))]
pub type VerifierTrustedAdviceCommitment = jolt_dory::DoryCommitment;

#[cfg(any(feature = "host", feature = "guest-verifier"))]
pub fn serialize_verifier_object<T: serde::Serialize>(
    item: &T,
) -> Result<Vec<u8>, bincode::error::EncodeError> {
    bincode::serde::encode_to_vec(item, bincode::config::standard())
}

#[cfg(any(feature = "host", feature = "guest-verifier"))]
pub fn deserialize_verifier_object<T: serde::de::DeserializeOwned>(
    bytes: &[u8],
) -> Result<T, bincode::error::DecodeError> {
    let (value, consumed) = bincode::serde::decode_from_slice(bytes, bincode::config::standard())?;
    if consumed == bytes.len() {
        Ok(value)
    } else {
        Err(bincode::error::DecodeError::OtherString(
            "trailing bytes in verifier object".to_string(),
        ))
    }
}

#[cfg(feature = "host")]
pub fn serialize_and_print_size(
    item_name: &str,
    file_name: &str,
    item: &impl serde::Serialize,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let data = bincode::serde::encode_to_vec(item, bincode::config::standard())?;
    std::fs::write(file_name, &data)?;
    let file_size_kb = data.len() as f64 / 1024.0;
    println!("{item_name} Written to {file_name}");
    println!("{item_name} size: {file_size_kb:.1} kB");
    Ok(())
}

// Re-exports needed by the provable macro
pub use jolt_prover_legacy::poly::commitment::commitment_scheme::CommitmentScheme;
pub use jolt_prover_legacy::poly::commitment::dory::{DoryContext, DoryGlobals, DoryLayout};
pub use jolt_prover_legacy::poly::multilinear_polynomial::MultilinearPolynomial;
pub use jolt_prover_legacy::zkvm::preprocessing::BlindfoldSetup;
#[cfg(feature = "host")]
pub use jolt_prover_legacy::zkvm::proof::ProofCommitmentScheme;
pub use jolt_prover_legacy::zkvm::ram::populate_memory_states;
