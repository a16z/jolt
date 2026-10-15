#![expect(
    clippy::expect_used,
    reason = "fixture generation should fail loudly when guest construction breaks"
)]

use std::sync::Arc;

use common::jolt_device::MemoryConfig;
use jolt_host::{JoltProgramSource, Program};
use jolt_program::execution::{ExecutionBackend, JoltProgram, TraceInputs, TraceOutput};
use jolt_program::preprocess::JoltProgramPreprocessing;
use jolt_prover::ProverConfig;
#[cfg(feature = "field-inline")]
use jolt_riscv::RV64IMAC_JOLT_FIELD_INLINE;
use jolt_witness::{JoltVmWitnessConfig, JoltVmWitnessInputs, TraceBackend};

use tracer::execution_backend::TracerBackend;

pub struct PreparedGuest {
    pub program: Arc<JoltProgram>,
    pub program_preprocessing: JoltProgramPreprocessing,
    pub trace: TraceOutput,
}

pub fn prepare_guest(
    mut source: Program,
    inputs: &[u8],
    untrusted_advice: &[u8],
    trusted_advice: &[u8],
) -> PreparedGuest {
    #[cfg(feature = "field-inline")]
    source.set_instruction_profile(RV64IMAC_JOLT_FIELD_INLINE);
    let (_, sizing_trace, _, device) = source.trace(inputs, untrusted_advice, trusted_advice);
    assert!(sizing_trace.len().next_power_of_two() <= 1 << 16);
    let layout = device.memory_layout;
    let program = Arc::new(source.build_jolt_program().expect("build Jolt program"));
    let program_preprocessing = JoltProgramPreprocessing::new(
        program.expanded_bytecode.clone(),
        program.memory_init.clone(),
        layout.clone(),
        program.entry_address,
        1 << 16,
        source.instruction_profile(),
    )
    .expect("program preprocessing");
    let memory_config = MemoryConfig {
        max_untrusted_advice_size: layout.max_untrusted_advice_size,
        max_trusted_advice_size: layout.max_trusted_advice_size,
        max_input_size: layout.max_input_size,
        max_output_size: layout.max_output_size,
        stack_size: layout.stack_size,
        heap_size: layout.heap_size,
        program_size: Some(layout.program_size),
    };
    let inputs = TraceInputs::new(
        inputs.to_vec(),
        untrusted_advice.to_vec(),
        trusted_advice.to_vec(),
        memory_config,
    );
    let trace = TracerBackend::new()
        .trace(&program, inputs)
        .expect("modular trace");
    PreparedGuest {
        program,
        program_preprocessing,
        trace,
    }
}

pub fn fixture_witness(
    program: &Arc<JoltProgram>,
    preprocessing: &Arc<JoltProgramPreprocessing>,
    trace: TraceOutput,
    config: &ProverConfig,
    trusted_advice: bool,
) -> TraceBackend {
    let witness_config = JoltVmWitnessConfig::new(
        config.trace_length.ilog2() as usize,
        config.ram_K,
        config.one_hot_config,
    )
    .include_untrusted_advice(!trace.device.untrusted_advice.is_empty())
    .include_trusted_advice(trusted_advice);
    let witness = TraceBackend::try_new(
        witness_config,
        JoltVmWitnessInputs::new(program, preprocessing, trace),
    )
    .expect("trace witness");
    #[cfg(feature = "field-inline")]
    let witness = witness.with_field_inline().expect("field-inline witness");
    witness
}
