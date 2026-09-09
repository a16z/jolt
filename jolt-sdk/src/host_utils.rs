#[cfg(feature = "host")]
use std::{
    error::Error as StdError,
    io::{Error as IoError, ErrorKind},
    sync::Arc,
};

#[cfg(feature = "host")]
use common::constants::RAM_START_ADDRESS;
#[cfg(feature = "host")]
use jolt_crypto::PedersenSetup;
#[cfg(any(feature = "host", feature = "guest-verifier"))]
use jolt_crypto::{Bn254G1, Pedersen};
#[cfg(feature = "host")]
use jolt_dory::DoryHint;
#[cfg(any(feature = "host", feature = "guest-verifier"))]
use jolt_dory::{DoryCommitment, DoryScheme};
#[cfg(any(feature = "host", feature = "guest-verifier"))]
use jolt_field::Fr;
#[cfg(feature = "host")]
use jolt_prover::dory::stages::stage0::TrustedAdviceCommitment;
#[cfg(feature = "host")]
use jolt_prover::{JoltProverPreprocessing as GenericJoltProverPreprocessing, ProverConfig};
#[cfg(all(
    any(feature = "host", feature = "guest-verifier"),
    feature = "transcript-keccak"
))]
use jolt_transcript::KeccakTranscript;
#[cfg(all(
    any(feature = "host", feature = "guest-verifier"),
    any(
        feature = "transcript-blake2b",
        not(any(
            feature = "transcript-poseidon",
            feature = "transcript-keccak",
            feature = "transcript-blake2b"
        ))
    )
))]
use jolt_transcript::LegacyBlake2bTranscript;
#[cfg(all(
    any(feature = "host", feature = "guest-verifier"),
    feature = "transcript-poseidon"
))]
use jolt_transcript::PoseidonTranscript;
#[cfg(feature = "host")]
use jolt_verifier::ProgramPreprocessing as GenericProgramPreprocessing;
#[cfg(feature = "host")]
use jolt_witness::{JoltVmWitnessConfig, JoltVmWitnessInputs, TraceBackend as WitnessTraceBackend};
#[cfg(feature = "host")]
use tracer::TracerInlineExpansionProvider;

#[cfg(feature = "host")]
pub use jolt_host as host;
#[cfg(all(
    any(feature = "host", feature = "guest-verifier"),
    feature = "transcript-poseidon"
))]
pub type ProtocolTranscript = PoseidonTranscript<Fr>;
#[cfg(all(
    any(feature = "host", feature = "guest-verifier"),
    feature = "transcript-keccak"
))]
pub type ProtocolTranscript = KeccakTranscript<Fr>;
#[cfg(all(
    any(feature = "host", feature = "guest-verifier"),
    feature = "transcript-blake2b"
))]
pub type ProtocolTranscript = LegacyBlake2bTranscript<Fr>;
#[cfg(all(
    any(feature = "host", feature = "guest-verifier"),
    not(any(
        feature = "transcript-poseidon",
        feature = "transcript-keccak",
        feature = "transcript-blake2b"
    ))
))]
pub type ProtocolTranscript = LegacyBlake2bTranscript<Fr>;
#[cfg(feature = "host")]
pub type ProofTranscript = ProtocolTranscript;
#[cfg(feature = "host")]
pub use jolt_program::execution::{
    ExecutionBackend, OwnedTrace, TraceError, TraceInputs, TraceOutput, TraceSource,
};
#[cfg(feature = "host")]
pub use tracer::TracerBackend;

pub use common::jolt_device::{JoltDevice, MemoryConfig, MemoryLayout};
#[cfg(feature = "host")]
pub use jolt_dory::DoryScheme as PCS;
#[cfg(feature = "host")]
pub use jolt_field::{Fr as F, JoltField};
#[cfg(feature = "host")]
pub use jolt_openings::CommitmentScheme;
#[cfg(feature = "host")]
pub use jolt_program::preprocess::JoltProgramPreprocessing;
#[cfg(feature = "host")]
pub use jolt_prover::{CommittedProgramProverData, JoltSharedPreprocessing, PreprocessingError};
#[cfg(feature = "host")]
pub type JoltProverPreprocessing = GenericJoltProverPreprocessing<PCS, VerifierVC>;
#[cfg(feature = "host")]
pub type ProgramPreprocessing = GenericProgramPreprocessing<PCS>;
#[cfg(feature = "host")]
pub use tracer::AdviceTape;

#[cfg(feature = "host")]
pub type VerifierPCS = DoryScheme;
#[cfg(feature = "host")]
pub type VerifierVC = Pedersen<Bn254G1>;
#[cfg(feature = "host")]
pub type VerifierField = jolt_field::Fr;
#[cfg(feature = "host")]
pub type VerifierTranscript = ProtocolTranscript;
#[cfg(feature = "host")]
pub type JoltVerifierPreprocessing =
    jolt_verifier::JoltVerifierPreprocessing<VerifierPCS, VerifierVC>;
#[cfg(feature = "host")]
pub type RV64IMACProof = jolt_verifier::JoltProof<VerifierPCS, VerifierVC>;
#[cfg(feature = "host")]
pub type JoltProof = RV64IMACProof;
#[cfg(feature = "host")]
pub type VerifierTrustedAdviceCommitment = DoryCommitment;
#[cfg(feature = "host")]
pub type TrustedAdviceOpeningHint = DoryHint;

#[cfg(feature = "host")]
pub type BlindfoldSetup = PedersenSetup<Bn254G1>;

#[cfg(feature = "host")]
fn program_preprocessing(
    source: &mut dyn host::JoltProgramSource,
    mut memory_config: MemoryConfig,
    max_trace_length: usize,
) -> Result<JoltProgramPreprocessing, PreprocessingError> {
    let (bytecode, memory_init, program_size, entry_address) = source.decode();
    memory_config.program_size = Some(program_size);
    JoltProgramPreprocessing::new(
        bytecode,
        memory_init,
        MemoryLayout::new(&memory_config),
        entry_address,
        max_trace_length,
        source.instruction_profile(),
    )
    .map_err(Into::into)
}

#[cfg(feature = "host")]
pub fn preprocess_shared_program(
    source: &mut dyn host::JoltProgramSource,
    memory_config: MemoryConfig,
    max_trace_length: usize,
) -> Result<JoltSharedPreprocessing, PreprocessingError> {
    JoltSharedPreprocessing::new(program_preprocessing(
        source,
        memory_config,
        max_trace_length,
    )?)
}

#[cfg(feature = "host")]
pub fn preprocess_program(
    source: &mut dyn host::JoltProgramSource,
    memory_config: MemoryConfig,
    max_trace_length: usize,
    bytecode_chunk_count: Option<usize>,
) -> Result<JoltProverPreprocessing, PreprocessingError> {
    let program = program_preprocessing(source, memory_config, max_trace_length)?;
    match bytecode_chunk_count {
        Some(chunk_count) => jolt_prover::dory::preprocess_committed(program, chunk_count),
        None => JoltSharedPreprocessing::new(program).map(jolt_prover::dory::from_shared),
    }
}

#[cfg(feature = "host")]
pub fn verifier_preprocessing_from_prover(
    prover: &JoltProverPreprocessing,
) -> JoltVerifierPreprocessing {
    prover.verifier_preprocessing()
}

#[cfg(all(feature = "guest-verifier", not(feature = "host")))]
pub type VerifierPCS = DoryScheme;
#[cfg(all(feature = "guest-verifier", not(feature = "host")))]
pub type VerifierVC = Pedersen<Bn254G1>;
#[cfg(all(feature = "guest-verifier", not(feature = "host")))]
pub type VerifierField = jolt_field::Fr;
#[cfg(all(feature = "guest-verifier", not(feature = "host")))]
pub type VerifierTranscript = ProtocolTranscript;
#[cfg(all(feature = "guest-verifier", not(feature = "host")))]
pub type JoltVerifierPreprocessing =
    jolt_verifier::JoltVerifierPreprocessing<VerifierPCS, VerifierVC>;
#[cfg(all(feature = "guest-verifier", not(feature = "host")))]
pub type RV64IMACProof = jolt_verifier::JoltProof<VerifierPCS, VerifierVC>;
#[cfg(all(feature = "guest-verifier", not(feature = "host")))]
pub type JoltProof = RV64IMACProof;
#[cfg(all(feature = "guest-verifier", not(feature = "host")))]
pub type VerifierTrustedAdviceCommitment = DoryCommitment;

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
) -> Result<(), Box<dyn StdError + Send + Sync>> {
    let data = bincode::serde::encode_to_vec(item, bincode::config::standard())?;
    std::fs::write(file_name, &data)?;
    let file_size_kb = data.len() as f64 / 1024.0;
    println!("{item_name} Written to {file_name}");
    println!("{item_name} size: {file_size_kb:.1} kB");
    Ok(())
}

#[cfg(feature = "host")]
pub fn compute_advice_tape(
    source: &dyn host::JoltProgramSource,
    inputs: &[u8],
    untrusted_advice: &[u8],
    trusted_advice: &[u8],
    memory_layout: &MemoryLayout,
) -> Result<Option<Vec<u8>>, Box<dyn StdError + Send + Sync>> {
    let Some(elf) = source.get_elf_compute_advice_contents() else {
        return Ok(None);
    };
    let mut inline_provider = TracerInlineExpansionProvider::new();
    let program = jolt_program::build_jolt_program_with_inline_provider(
        &elf,
        &mut inline_provider,
        source.instruction_profile(),
    )?;
    let memory_config = MemoryConfig {
        max_untrusted_advice_size: memory_layout.max_untrusted_advice_size,
        max_trusted_advice_size: memory_layout.max_trusted_advice_size,
        max_input_size: memory_layout.max_input_size,
        max_output_size: memory_layout.max_output_size,
        stack_size: memory_layout.stack_size,
        heap_size: memory_layout.heap_size,
        program_size: Some(program.program_end - RAM_START_ADDRESS),
    };
    let trace = program.trace_with(
        &mut TracerBackend::new(),
        TraceInputs::new(
            inputs.to_vec(),
            untrusted_advice.to_vec(),
            trusted_advice.to_vec(),
            memory_config,
        ),
    )?;
    Ok(trace.advice_tape)
}

#[cfg(feature = "host")]
#[expect(
    clippy::too_many_arguments,
    reason = "SDK proof boundary mirrors guest inputs"
)]
pub fn prove_program(
    source: &dyn host::JoltProgramSource,
    preprocessing: &JoltProverPreprocessing,
    inputs: &[u8],
    untrusted_advice: &[u8],
    trusted_advice: &[u8],
    trusted_advice_commitment: Option<DoryCommitment>,
    trusted_advice_hint: Option<DoryHint>,
    advice_tape: Option<Vec<u8>>,
) -> Result<(RV64IMACProof, JoltDevice), Box<dyn StdError + Send + Sync>> {
    let program = Arc::new(source.build_jolt_program()?);
    let program_preprocessing = preprocessing.program_arc().ok_or_else(|| {
        IoError::new(
            ErrorKind::InvalidInput,
            "prover preprocessing does not retain the full program",
        )
    })?;
    let memory_layout = preprocessing.verifier.program.memory_layout();
    let memory_config = MemoryConfig {
        max_untrusted_advice_size: memory_layout.max_untrusted_advice_size,
        max_trusted_advice_size: memory_layout.max_trusted_advice_size,
        max_input_size: memory_layout.max_input_size,
        max_output_size: memory_layout.max_output_size,
        stack_size: memory_layout.stack_size,
        heap_size: memory_layout.heap_size,
        program_size: Some(memory_layout.program_size),
    };
    let trace_inputs = TraceInputs::new(
        inputs.to_vec(),
        untrusted_advice.to_vec(),
        trusted_advice.to_vec(),
        memory_config,
    )
    .with_advice_tape(advice_tape);
    #[cfg(not(feature = "field-inline"))]
    let trace = TracerBackend::new().trace_compact(
        &program,
        trace_inputs,
        &program_preprocessing.bytecode,
    )?;
    #[cfg(feature = "field-inline")]
    let trace = program.trace_with(&mut TracerBackend::new(), trace_inputs)?;
    let public_io = trace.device.clone();
    #[cfg(not(feature = "field-inline"))]
    let config = ProverConfig::derive_compact::<F>(
        trace.trace.as_slice(),
        memory_layout,
        preprocessing.verifier.program.min_bytecode_address(),
        preprocessing.verifier.program.program_image_len_words(),
        preprocessing.verifier.program.max_padded_trace_length(),
    )?;
    #[cfg(feature = "field-inline")]
    let config = ProverConfig::derive::<F>(
        trace.trace.rows(),
        memory_layout,
        preprocessing.verifier.program.min_bytecode_address(),
        preprocessing.verifier.program.program_image_len_words(),
        preprocessing.verifier.program.max_padded_trace_length(),
    )?;
    let witness_config = JoltVmWitnessConfig::new(
        config.trace_length.ilog2() as usize,
        config.ram_K,
        config.one_hot_config,
    )
    .include_trusted_advice(trusted_advice_commitment.is_some())
    .include_untrusted_advice(!untrusted_advice.is_empty());
    let witness_inputs = JoltVmWitnessInputs::new(&program, &program_preprocessing, trace);
    #[cfg(not(feature = "field-inline"))]
    let witness = WitnessTraceBackend::<OwnedTrace>::from_compact(witness_config, witness_inputs);
    #[cfg(feature = "field-inline")]
    let witness = WitnessTraceBackend::<OwnedTrace>::try_new(witness_config, witness_inputs)?;
    let trusted_advice = match (trusted_advice_commitment, trusted_advice_hint) {
        (Some(commitment), Some(hint)) => Some(TrustedAdviceCommitment { commitment, hint }),
        (None, None) => None,
        _ => {
            return Err(IoError::new(
                ErrorKind::InvalidInput,
                "trusted advice commitment and hint must be supplied together",
            )
            .into())
        }
    };
    let backend = jolt_prover::JoltBackend::<F, PCS>::optimized();
    let proof = jolt_prover::dory::prove::<F, PCS, VerifierVC, ProofTranscript, _>(
        &backend,
        preprocessing,
        &config,
        trusted_advice.as_ref(),
        &witness,
        &public_io,
    )?;
    Ok((proof, public_io))
}
