//! Shared fixture builder for the `metal_*_cpu_eval` examples: compiles the
//! guest, traces it, derives the packed prover configuration and returns the
//! real witness plane the isolated CPU and Metal arms evaluate against.

use std::sync::Arc;

use common::jolt_device::MemoryConfig;
use jolt_akita::AkitaField;
use jolt_host::{JoltProgramSource as _, Program};
use jolt_program::execution::{JoltProgram, OwnedTrace, TraceInputs, TraceOutput};
use jolt_program::preprocess::{BytecodePreprocessing, JoltProgramPreprocessing};
use jolt_prover::akita::preprocessing::preprocess_full;
use jolt_prover::ProverConfig;
use jolt_riscv::JoltTraceRow;
use jolt_witness::{JoltVmWitnessConfig, JoltVmWitnessInputs, TraceBackend};
use tracer::execution_backend::TracerBackend;

#[expect(
    dead_code,
    reason = "each evaluator reads the subset of the geometry it reports"
)]
pub struct BuiltWitness {
    pub witness: TraceBackend<OwnedTrace>,
    pub trace_rows: usize,
    pub padded_rows: usize,
    pub log_t: usize,
    pub log_k: usize,
    pub committed_chunk_bits: usize,
    pub lowest_address: u64,
}

/// Builds the production witness for `bench_name` on `input`, padded to at
/// most `max_trace_length` cycles. Panics on any guest, trace or
/// preprocessing failure: the evaluators have no fallback.
pub fn build_witness(bench_name: &str, input: &[u8], max_trace_length: usize) -> BuiltWitness {
    let mut program = Program::new(&format!("{bench_name}-guest"));
    let (_, sizing_trace, _, io_device) = program.trace(input, &[], &[]);
    assert!(
        sizing_trace.len().next_power_of_two() <= max_trace_length,
        "trace is longer than the requested padded domain"
    );
    drop(sizing_trace);
    let memory_layout = io_device.memory_layout.clone();
    let jolt_program = Arc::new(program.build_jolt_program().expect("build Jolt program"));
    let program_preprocessing = JoltProgramPreprocessing::new(
        jolt_program.expanded_bytecode.clone(),
        jolt_program.memory_init.clone(),
        memory_layout.clone(),
        jolt_program.entry_address,
        max_trace_length,
        program.instruction_profile(),
    )
    .expect("program preprocessing");
    let trace_output = trace_compact(
        &jolt_program,
        &memory_layout,
        &program_preprocessing.bytecode,
        input,
    );
    let trace_rows = trace_output.trace.len();
    let config = ProverConfig::derive_compact::<AkitaField>(
        trace_output.trace.as_slice(),
        &memory_layout,
        program_preprocessing.ram.min_bytecode_address,
        program_preprocessing.ram.bytecode_words.len(),
        max_trace_length,
    )
    .expect("derive prover config");
    let prover_preprocessing =
        preprocess_full(program_preprocessing, &config).expect("Akita preprocessing");
    let program_preprocessing = prover_preprocessing
        .program_arc()
        .expect("full program preprocessing");
    let log_t = config.trace_length.ilog2() as usize;
    let witness = TraceBackend::<OwnedTrace>::from_compact(
        JoltVmWitnessConfig::new(log_t, config.ram_K, config.one_hot_config),
        JoltVmWitnessInputs::new(&jolt_program, &program_preprocessing, trace_output),
    );
    BuiltWitness {
        witness,
        trace_rows,
        padded_rows: config.trace_length,
        log_t,
        log_k: config.ram_K.ilog2() as usize,
        committed_chunk_bits: config.one_hot_config.committed_chunk_bits(),
        lowest_address: memory_layout.get_lowest_address(),
    }
}

fn trace_compact(
    program: &JoltProgram,
    memory_layout: &common::jolt_device::MemoryLayout,
    bytecode: &BytecodePreprocessing,
    inputs: &[u8],
) -> TraceOutput<Arc<Vec<JoltTraceRow>>> {
    let memory_config = MemoryConfig {
        max_untrusted_advice_size: memory_layout.max_untrusted_advice_size,
        max_trusted_advice_size: memory_layout.max_trusted_advice_size,
        max_input_size: memory_layout.max_input_size,
        max_output_size: memory_layout.max_output_size,
        stack_size: memory_layout.stack_size,
        heap_size: memory_layout.heap_size,
        program_size: Some(memory_layout.program_size),
    };
    TracerBackend::new()
        .trace_compact(
            program,
            TraceInputs {
                inputs: inputs.to_vec(),
                untrusted_advice: Vec::new(),
                trusted_advice: Vec::new(),
                memory_config,
                advice_tape: None,
            },
            bytecode,
        )
        .expect("modular trace")
}
