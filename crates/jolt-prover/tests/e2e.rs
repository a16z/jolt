#![expect(
    clippy::expect_used,
    reason = "e2e proof tests should fail loudly on guest build, proving, or verification failures"
)]

use std::{
    fmt::Debug,
    sync::{Mutex, MutexGuard, Once},
};

use blake2::{
    digest::{consts::U32, Digest},
    Blake2b,
};
use common::{
    constants::ONEHOT_CHUNK_THRESHOLD_LOG_T,
    jolt_device::{JoltDevice, MemoryLayout},
};
use jolt_backends::cpu::CpuBackend;
use jolt_claims::protocols::jolt::{JoltOneHotConfig, JoltReadWriteConfig, TracePolynomialOrder};
use jolt_crypto::{Bn254G1, Pedersen};
use jolt_dory::DoryScheme;
use jolt_field::Fr;
use jolt_openings::CommitmentScheme;
use jolt_program::{
    execution::{JoltProgram, OwnedTrace, RamAccess, TraceOutput, TraceRow},
    preprocess::JoltProgramPreprocessing,
};
use jolt_prover::{JoltProverPreprocessing, ProofParameters, ProverConfig};
use jolt_sdk::host::Program;
use jolt_transcript::Blake2bTranscript;
use jolt_verifier::{
    compat, zk_vector_commitment_capacity_requirement, JoltVerifierPreprocessing, NoPcsAssist,
};
use jolt_witness::{
    protocols::jolt_vm::{JoltVmWitnessBuilder, JoltVmWitnessConfig, JoltVmWitnessInputs},
    WitnessBuilder,
};
use serde::de::DeserializeOwned;
use tracer::TracerBackend;

type Pcs = DoryScheme;
type Vc = Pedersen<Bn254G1>;

static E2E_LOCK: Mutex<()> = Mutex::new(());
static RAYON_INIT: Once = Once::new();

const TARGET_DIR: &str = "/tmp/jolt-prover-e2e-guests";
const MULDIV_MAX_TRACE_LENGTH: usize = 65_536;
const SHA2_CHAIN_MAX_TRACE_LENGTH: usize = 4_194_304;
const SHA2_CHAIN_ITERS: u32 = 8;
const E2E_STACK_SIZE: usize = 128 * 1024 * 1024;

#[test]
fn modular_muldiv_proof_verifies() {
    let a = 12_031_293u32;
    let b = 17u32;
    let c = 92u32;
    let mut input_bytes = Vec::new();
    append_input(&mut input_bytes, &a);
    append_input(&mut input_bytes, &b);
    append_input(&mut input_bytes, &c);
    let expected = a * b / c;

    run_e2e(
        "muldiv",
        MULDIV_MAX_TRACE_LENGTH,
        muldiv_guest::compile_muldiv,
        input_bytes,
        expected,
    );
}

#[test]
fn modular_sha2_chain_proof_verifies() {
    let input = [5u8; 32];
    let mut input_bytes = Vec::new();
    append_input(&mut input_bytes, &input);
    append_input(&mut input_bytes, &SHA2_CHAIN_ITERS);
    let expected = sha2_chain_guest::sha2_chain(input, SHA2_CHAIN_ITERS);

    run_e2e(
        "sha2-chain",
        SHA2_CHAIN_MAX_TRACE_LENGTH,
        sha2_chain_guest::compile_sha2_chain,
        input_bytes,
        expected,
    );
}

fn run_e2e<Output>(
    name: &'static str,
    max_trace_length: usize,
    compile: fn(&str) -> Program,
    input_bytes: Vec<u8>,
    expected_output: Output,
) where
    Output: Debug + DeserializeOwned + PartialEq + Send + 'static,
{
    std::thread::Builder::new()
        .name(format!("jolt-prover-e2e-{name}"))
        .stack_size(E2E_STACK_SIZE)
        .spawn(move || {
            prove_and_verify(
                name,
                max_trace_length,
                compile,
                input_bytes,
                expected_output,
            );
        })
        .expect("e2e thread should spawn")
        .join()
        .expect("e2e thread should not panic");
}

fn prove_and_verify<Output>(
    name: &'static str,
    max_trace_length: usize,
    compile: fn(&str) -> Program,
    input_bytes: Vec<u8>,
    expected_output: Output,
) where
    Output: Debug + DeserializeOwned + PartialEq,
{
    let _guard = e2e_guard();
    initialize_rayon();
    let mut host_program = compile(TARGET_DIR);
    let program = host_program
        .jolt_program()
        .expect("guest should compile into a modular Jolt program");
    let mut tracer = TracerBackend::new();
    let trace = host_program
        .trace_with_backend(&mut tracer, &input_bytes, &[], &[])
        .expect("guest should execute through the tracer backend");
    let public_io = trace.device.clone();
    let program_preprocessing = program_preprocessing(&program, &public_io, max_trace_length);
    let proof_parameters = proof_parameters(&program_preprocessing, &trace);
    let preprocessing = prover_preprocessing(program_preprocessing, max_trace_length, &program);
    let witness = trace_witness(
        &program,
        &preprocessing.verifier.program,
        trace,
        proof_parameters,
    );
    let mut backend = CpuBackend::default();
    let config = ProverConfig::default().with_proof_parameters(proof_parameters);

    #[cfg(feature = "field-inline")]
    let prover_output = {
        let field_inline_witness = witness
            .field_inline_witness()
            .expect("field-inline witness should build from the traced program");
        jolt_prover::prove_with_components(
            &preprocessing,
            &public_io,
            &witness,
            &field_inline_witness,
            config,
            &mut backend,
        )
        .expect("modular prover should produce a proof")
    };

    #[cfg(not(feature = "field-inline"))]
    let prover_output = jolt_prover::prove_with_components(
        &preprocessing,
        &public_io,
        &witness,
        config,
        &mut backend,
    )
    .expect("modular prover should produce a proof");

    jolt_verifier::verify::<Fr, Pcs, Vc, Blake2bTranscript, NoPcsAssist>(
        &preprocessing.verifier,
        &public_io,
        &prover_output.proof,
        prover_output.trusted_advice_commitment.as_ref(),
        cfg!(feature = "zk"),
    )
    .expect("modular verifier should accept the modular proof");

    let actual_output: Output = decode_output(&public_io);
    assert_eq!(actual_output, expected_output, "{name} output mismatch");
}

fn e2e_guard() -> MutexGuard<'static, ()> {
    E2E_LOCK.lock().expect("e2e mutex should not be poisoned")
}

fn initialize_rayon() {
    RAYON_INIT.call_once(|| {
        rayon::ThreadPoolBuilder::new()
            .stack_size(E2E_STACK_SIZE)
            .build_global()
            .expect("global rayon pool should initialize for e2e proofs");
    });
}

fn append_input<T>(bytes: &mut Vec<u8>, value: &T)
where
    T: serde::Serialize + ?Sized,
{
    bytes.append(
        &mut jolt_sdk::postcard::to_stdvec(value)
            .expect("guest input should serialize with postcard"),
    );
}

fn decode_output<Output>(public_io: &JoltDevice) -> Output
where
    Output: DeserializeOwned,
{
    let mut outputs = public_io.outputs.clone();
    outputs.resize(public_io.memory_layout.max_output_size as usize, 0);
    jolt_sdk::postcard::from_bytes(&outputs).expect("guest output should deserialize with postcard")
}

fn program_preprocessing(
    program: &JoltProgram,
    public_io: &JoltDevice,
    max_trace_length: usize,
) -> JoltProgramPreprocessing {
    JoltProgramPreprocessing::new(
        program.expanded_bytecode.clone(),
        program.memory_init.clone(),
        public_io.memory_layout.clone(),
        program.entry_address,
        max_trace_length,
        program.profile,
    )
    .expect("program preprocessing should be valid")
}

fn prover_preprocessing(
    program: JoltProgramPreprocessing,
    max_trace_length: usize,
    jolt_program: &JoltProgram,
) -> JoltProverPreprocessing<Pcs, Vc> {
    let max_log_t = max_trace_length.next_power_of_two().trailing_zeros() as usize;
    let max_log_k_chunk = if max_log_t < ONEHOT_CHUNK_THRESHOLD_LOG_T {
        4
    } else {
        8
    };
    let (pcs_setup, _) = Pcs::setup(max_log_k_chunk + max_log_t);
    let preprocessing_digest = preprocessing_digest(&program);
    let verifier = JoltVerifierPreprocessing::<Pcs, Vc>::from_pcs_prover_setup(
        program,
        preprocessing_digest,
        &pcs_setup,
        zk_vector_commitment_capacity_requirement(),
    );

    #[cfg(feature = "field-inline")]
    let verifier = {
        let code_size = verifier.program.bytecode.code_size;
        verifier.with_field_inline_bytecode(field_inline_bytecode(jolt_program, code_size))
    };

    #[cfg(not(feature = "field-inline"))]
    let _ = jolt_program;

    JoltProverPreprocessing::new(verifier, pcs_setup)
}

fn preprocessing_digest(program: &JoltProgramPreprocessing) -> [u8; 32] {
    let bytes = bincode::serde::encode_to_vec(program, bincode::config::standard())
        .expect("program preprocessing should serialize into memory");
    Blake2b::<U32>::digest(bytes).into()
}

fn trace_witness<'a>(
    program: &'a JoltProgram,
    preprocessing: &'a JoltProgramPreprocessing,
    trace: TraceOutput<OwnedTrace>,
    proof_parameters: ProofParameters,
) -> <JoltVmWitnessBuilder<OwnedTrace> as WitnessBuilder<Fr>>::Witness<'a> {
    let config = JoltVmWitnessConfig::new(
        proof_parameters.trace_length.trailing_zeros() as usize,
        proof_parameters.ram_k,
        proof_parameters.one_hot_config,
    )
    .retain_trace_rows(true)
    .include_trusted_advice(!trace.device.trusted_advice.is_empty())
    .include_untrusted_advice(!trace.device.untrusted_advice.is_empty());

    let mut builder = JoltVmWitnessBuilder::<OwnedTrace>::new();
    <JoltVmWitnessBuilder<OwnedTrace> as WitnessBuilder<Fr>>::build(
        &mut builder,
        &config,
        JoltVmWitnessInputs::new(program, preprocessing, trace),
    )
    .expect("trace-backed witness should build")
}

fn proof_parameters(
    preprocessing: &JoltProgramPreprocessing,
    trace: &TraceOutput<OwnedTrace>,
) -> ProofParameters {
    let trace_length = padded_trace_length(trace.trace.rows().len(), preprocessing);
    let log_t = trace_length.trailing_zeros() as usize;
    let ram_k = ram_k(
        preprocessing,
        trace.trace.rows(),
        &trace.device.memory_layout,
    );
    let ram_log_k = ram_k.trailing_zeros() as usize;

    let rw_config = compat::config::ReadWriteConfig::try_from((log_t, ram_log_k))
        .expect("read-write config should be valid");
    let one_hot_config = compat::config::OneHotConfig::from(log_t);

    ProofParameters::new(
        trace_length,
        ram_k,
        JoltReadWriteConfig {
            ram_rw_phase1_num_rounds: rw_config.ram_rw_phase1_num_rounds,
            ram_rw_phase2_num_rounds: rw_config.ram_rw_phase2_num_rounds,
            registers_rw_phase1_num_rounds: rw_config.registers_rw_phase1_num_rounds,
            registers_rw_phase2_num_rounds: rw_config.registers_rw_phase2_num_rounds,
        },
        JoltOneHotConfig {
            log_k_chunk: one_hot_config.log_k_chunk,
            lookups_ra_virtual_log_k_chunk: one_hot_config.lookups_ra_virtual_log_k_chunk,
        },
        TracePolynomialOrder::CycleMajor,
    )
}

fn padded_trace_length(
    unpadded_trace_length: usize,
    preprocessing: &JoltProgramPreprocessing,
) -> usize {
    let trace_length = if unpadded_trace_length < 256 {
        256
    } else {
        (unpadded_trace_length + 1).next_power_of_two()
    };
    assert!(
        trace_length <= preprocessing.max_padded_trace_length,
        "trace length {trace_length} exceeds max {}",
        preprocessing.max_padded_trace_length
    );
    trace_length
}

fn ram_k(
    preprocessing: &JoltProgramPreprocessing,
    rows: &[TraceRow],
    layout: &MemoryLayout,
) -> usize {
    let trace_max = rows
        .iter()
        .filter_map(|row| {
            ram_access_address(row).and_then(|address| remap_address(address, layout))
        })
        .max()
        .unwrap_or(0);
    let bytecode_end = remap_address(preprocessing.ram.min_bytecode_address, layout).unwrap_or(0)
        + preprocessing.ram.bytecode_words.len() as u64
        + 1;
    trace_max.max(bytecode_end).next_power_of_two() as usize
}

fn ram_access_address(row: &TraceRow) -> Option<u64> {
    match row.ram_access {
        RamAccess::Read(read) => Some(read.address),
        RamAccess::Write(write) => Some(write.address),
        RamAccess::NoOp => None,
    }
}

fn remap_address(address: u64, layout: &MemoryLayout) -> Option<u64> {
    if address == 0 || address < layout.get_lowest_address() {
        None
    } else {
        Some((address - layout.get_lowest_address()) / 8)
    }
}

#[cfg(feature = "field-inline")]
fn field_inline_bytecode(
    program: &JoltProgram,
    padded_len: usize,
) -> Vec<jolt_claims::protocols::field_inline::formulas::bytecode::FieldInlineBytecodeRow> {
    use jolt_claims::protocols::field_inline::formulas::bytecode as field_bytecode;

    let mut rows = program
        .expanded_bytecode
        .iter()
        .map(field_bytecode::field_inline_bytecode_row)
        .collect::<Vec<_>>();
    rows.resize(padded_len, Default::default());
    rows
}
