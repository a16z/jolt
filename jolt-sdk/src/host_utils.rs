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
pub use jolt_program::execution::{ExecutionBackend, TraceError, TraceInputs, TraceOutput};
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
pub use jolt_prover::{
    CommittedProgramProverData, JoltSharedPreprocessing, MemoryBudget, PreprocessingError,
};
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
        MemoryLayout::try_new(&memory_config)?,
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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProgramPreprocessingMode {
    Full,
    Committed,
}

#[cfg(feature = "host")]
pub fn preprocess_program(
    source: &mut dyn host::JoltProgramSource,
    memory_config: MemoryConfig,
    max_trace_length: usize,
    mode: ProgramPreprocessingMode,
) -> Result<JoltProverPreprocessing, PreprocessingError> {
    let program = program_preprocessing(source, memory_config, max_trace_length)?;
    match mode {
        ProgramPreprocessingMode::Committed => jolt_prover::dory::preprocess_committed(program),
        ProgramPreprocessingMode::Full => {
            JoltSharedPreprocessing::new(program).and_then(jolt_prover::dory::from_shared)
        }
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

/// `memory_budget`, when set, selects the trace with
/// [`TracerBackend::limited_for_row_domain`] over the preprocessing's maximum
/// padded trace length. That routing proxy is not a peak: a run far shorter
/// than the domain can take the limited trace although the original trace
/// would fit, and a budget just above the proxy keeps an original trace whose
/// unpriced requests can exceed it. The limited trace rejects parallel tracing
/// (`TRACER_PARALLEL`), fails at the first covered request the budget cannot
/// hold, starting with its setup requests (over 1 GiB by default), and keeps
/// its row reserve in the trace, where the known-eager check counts it.
/// That check, [`MemoryBudget::check_known_eager`], runs after tracing and
/// before the witness is built. Passing does not mean the proof fits in the
/// budget, and [`compute_advice_tape`]'s trace is not budgeted.
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
    memory_budget: Option<MemoryBudget>,
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
    let max_trace_length = preprocessing.verifier.program.max_padded_trace_length();
    let mut tracer = memory_budget
        .and_then(|budget| TracerBackend::limited_for_row_domain(budget.bytes(), max_trace_length))
        .unwrap_or_default();
    let trace_inputs = TraceInputs::new(
        inputs.to_vec(),
        untrusted_advice.to_vec(),
        trusted_advice.to_vec(),
        memory_config,
    )
    .with_advice_tape(advice_tape);
    let trace = program.trace_with(&mut tracer, trace_inputs)?;
    let public_io = trace.device.clone();
    let config = ProverConfig::derive_from_dimensions::<F>(
        trace.dimensions,
        memory_layout,
        preprocessing.verifier.program.min_bytecode_address(),
        preprocessing.verifier.program.program_image_len_words(),
        max_trace_length,
    )?;
    if let Some(budget) = memory_budget {
        budget.check_known_eager::<F>(&config, &trace.trace)?;
    }
    let witness_config = JoltVmWitnessConfig::new(
        config.trace_length.ilog2() as usize,
        config.ram_K,
        config.one_hot_config,
    )
    .include_trusted_advice(trusted_advice_commitment.is_some())
    .include_untrusted_advice(!untrusted_advice.is_empty());
    let witness_inputs = JoltVmWitnessInputs::new(&program, &program_preprocessing, trace);
    let witness = WitnessTraceBackend::try_new(witness_config, witness_inputs)?;
    #[cfg(feature = "field-inline")]
    let witness = witness.with_field_inline()?;
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
    let proof = jolt_prover::dory::prove::<F, PCS, VerifierVC, ProofTranscript>(
        backend.with_witness(&witness),
        preprocessing,
        &config,
        trusted_advice.as_ref(),
        &public_io,
    )?;
    Ok((proof, public_io))
}

#[cfg(all(test, feature = "host"))]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests {
    use std::process::Command;

    use jolt_program::execution::{TraceAllocationError, TraceAllocationPhase};

    use super::*;

    const ROUTE_CHILD: &str = "JOLT_SDK_TRACE_ROUTE_CHILD";
    const ROUTE_TEST: &str =
        "host_utils::tests::prove_program_routes_its_trace_by_its_memory_budget";
    const INPUTS: [u8; 6] = [0xbd, 0xaa, 0xde, 0x5, 0x11, 0x5c];

    type Proved = Result<(RV64IMACProof, JoltDevice), Box<dyn StdError + Send + Sync>>;

    fn prove(
        program: &host::Program,
        preprocessing: &JoltProverPreprocessing,
        budget: Option<usize>,
    ) -> Proved {
        let budget = budget.map(MemoryBudget::from_bytes);
        prove_program(
            program,
            preprocessing,
            &INPUTS,
            &[],
            &[],
            None,
            None,
            None,
            budget,
        )
    }

    fn verify(preprocessing: &JoltProverPreprocessing, proved: Proved) {
        let (proof, io_device) = proved.unwrap();
        jolt_verifier::verify::<VerifierField, VerifierPCS, VerifierVC, VerifierTranscript>(
            &preprocessing.verifier,
            &io_device,
            &proof,
            None,
        )
        .unwrap();
    }

    fn trace_error(proved: Proved) -> TraceError {
        *proved.err().unwrap().downcast::<TraceError>().unwrap()
    }

    /// The budget reaches the tracer's route over the preprocessing domain.
    /// The threshold is the least budget at which the factory, over that
    /// domain, keeps the original trace. With `TRACER_PARALLEL=2`, set only in
    /// a child process of this test binary, no budget, the threshold and 32 GiB
    /// prove through the original two-pass trace, while one byte less selects
    /// the limited trace, which rejects it. Without it, that budget proves
    /// through the limited trace, and a one-byte budget stops at its setup.
    #[test]
    fn prove_program_routes_its_trace_by_its_memory_budget() {
        if let Ok(parallel) = std::env::var(ROUTE_CHILD) {
            let mut program = host::Program::new("muldiv-guest");
            let memory_config = MemoryConfig {
                heap_size: 1 << 15,
                ..Default::default()
            };
            let preprocessing = preprocess_program(
                &mut program,
                memory_config,
                1 << 12,
                ProgramPreprocessingMode::Full,
            )
            .unwrap();
            let domain = preprocessing.verifier.program.max_padded_trace_length();
            let (mut limited, mut original) = (0, usize::MAX);
            assert!(TracerBackend::limited_for_row_domain(limited, domain).is_some());
            assert!(TracerBackend::limited_for_row_domain(original, domain).is_none());
            while original - limited > 1 {
                let budget = limited + (original - limited) / 2;
                match TracerBackend::limited_for_row_domain(budget, domain) {
                    Some(_) => limited = budget,
                    None => original = budget,
                }
            }
            #[cfg(not(feature = "field-inline"))]
            assert_eq!(original, 96 * (1 << 24) + 64 * (1 << 12));
            if parallel == "2" {
                for budget in [None, Some(original), Some(32 << 30)] {
                    verify(&preprocessing, prove(&program, &preprocessing, budget));
                }
                assert!(matches!(
                    trace_error(prove(&program, &preprocessing, Some(limited))),
                    TraceError::UnsupportedTraceAllocationMode
                ));
            } else {
                verify(
                    &preprocessing,
                    prove(&program, &preprocessing, Some(limited)),
                );
                assert!(matches!(
                    trace_error(prove(&program, &preprocessing, Some(1))),
                    TraceError::TraceAllocation(TraceAllocationError::LimitExceeded {
                        phase: TraceAllocationPhase::Initial,
                        limit_bytes: 1,
                        ..
                    })
                ));
            }
            return;
        }
        for parallel in ["2", "unset"] {
            let mut child = Command::new(std::env::current_exe().unwrap());
            child
                .args(["--exact", ROUTE_TEST, "--nocapture"])
                .env(ROUTE_CHILD, parallel)
                .env_remove("JOLT_TRACER_CAPACITY_ROWS");
            match parallel {
                "2" => child.env("TRACER_PARALLEL", parallel),
                _ => child.env_remove("TRACER_PARALLEL"),
            };
            let output = child.output().unwrap();
            let stdout = String::from_utf8_lossy(&output.stdout);
            assert!(
                output.status.success() && stdout.contains("1 passed"),
                "TRACER_PARALLEL={parallel}: {stdout}{}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
    }
}
