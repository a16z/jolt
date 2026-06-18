#[cfg(feature = "host")]
pub use jolt_core::host;
#[cfg(feature = "host")]
pub use jolt_core::zkvm::proof_serialization::serialize_and_print_size as serialize_core_and_print_size;
#[cfg(feature = "host")]
pub use jolt_core::zkvm::{prover::JoltProverPreprocessing, RV64IMACProver};
#[cfg(feature = "host")]
pub use jolt_program::execution::{
    ExecutionBackend, OwnedTrace, TraceError, TraceInputs, TraceOutput, TraceSource,
};
#[cfg(feature = "host")]
pub use tracer::TracerBackend;

pub use common::jolt_device::{JoltDevice, MemoryConfig, MemoryLayout};
pub use jolt_core::ark_bn254::Fr as F;
pub use jolt_core::curve::Bn254Curve as Curve;
pub use jolt_core::curve::JoltCurve;
pub use jolt_core::field::JoltField;
pub use jolt_core::guest;
pub use jolt_core::poly::commitment::dory::DoryCommitmentScheme as PCS;
pub use jolt_core::zkvm::{
    bytecode::PreprocessingError,
    program::{CommittedProgramProverData, ProgramPreprocessing},
    verifier::JoltSharedPreprocessing,
    Serializable,
};
#[cfg(all(feature = "guest-verifier", not(feature = "host")))]
pub use jolt_core::zkvm::{verifier::JoltVerifierPreprocessing, RV64IMACProof, RV64IMACVerifier};
pub use jolt_core::AdviceTape;

#[cfg(feature = "host")]
pub type CoreJoltVerifierPreprocessing<CoreF = F, CoreC = Curve, CorePCS = PCS> =
    jolt_core::zkvm::verifier::JoltVerifierPreprocessing<CoreF, CoreC, CorePCS>;
#[cfg(feature = "host")]
pub type CoreRV64IMACProof = jolt_core::zkvm::RV64IMACProof;
#[cfg(feature = "host")]
pub type CoreRV64IMACVerifier<'a> = jolt_core::zkvm::RV64IMACVerifier<'a>;
#[cfg(feature = "host")]
pub type JoltProof = CoreRV64IMACProof;
#[cfg(feature = "host")]
pub type JoltVerifierPreprocessing<CoreF = F, CoreC = Curve, CorePCS = PCS> =
    jolt_core_verifier_bridge::ImportedCorePreprocessing<CoreF, CoreC, CorePCS>;
#[cfg(feature = "host")]
pub type RV64IMACProof = jolt_core_verifier_bridge::RV64IMACProof;
#[cfg(feature = "host")]
pub type VerifierPCS = jolt_dory::DoryScheme;
#[cfg(feature = "host")]
pub type VerifierVC = jolt_crypto::Pedersen<jolt_crypto::Bn254G1>;
#[cfg(feature = "host")]
pub type VerifierField = jolt_field::Fr;
#[cfg(feature = "host")]
pub type VerifierTrustedAdviceCommitment =
    jolt_core_verifier_bridge::VerifierTrustedAdviceCommitment;

#[cfg(feature = "host")]
pub fn verifier_preprocessing_from_core(
    preprocessing: &CoreJoltVerifierPreprocessing,
) -> JoltVerifierPreprocessing {
    jolt_core_verifier_bridge::verifier_preprocessing_from_core(preprocessing)
}

#[cfg(feature = "host")]
pub fn verifier_proof_from_core(
    proof: CoreRV64IMACProof,
) -> Result<RV64IMACProof, jolt_verifier::VerifierError> {
    jolt_core_verifier_bridge::proof_into_verifier(proof)
}

#[cfg(feature = "host")]
pub fn verifier_commitment_from_core(
    commitment: <PCS as CommitmentScheme>::Commitment,
) -> VerifierTrustedAdviceCommitment {
    jolt_core_verifier_bridge::commitment_into_verifier::<F, PCS>(commitment)
}

#[cfg(feature = "host")]
pub fn verify_rv64imac(
    preprocessing: &JoltVerifierPreprocessing,
    public_io: &JoltDevice,
    proof: &RV64IMACProof,
    trusted_advice_commitment: Option<&VerifierTrustedAdviceCommitment>,
    zk: bool,
) -> Result<(), jolt_verifier::VerifierError> {
    jolt_verifier::verify::<
        VerifierField,
        VerifierPCS,
        VerifierVC,
        jolt_transcript::LegacyBlake2bTranscript<VerifierField>,
    >(
        preprocessing,
        public_io,
        proof,
        trusted_advice_commitment,
        zk,
    )
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
pub use jolt_core::poly::commitment::commitment_scheme::CommitmentScheme;
pub use jolt_core::poly::commitment::dory::{DoryContext, DoryGlobals, DoryLayout};
pub use jolt_core::poly::multilinear_polynomial::MultilinearPolynomial;
pub use jolt_core::zkvm::ram::populate_memory_states;
pub use jolt_core::zkvm::verifier::BlindfoldSetup;
