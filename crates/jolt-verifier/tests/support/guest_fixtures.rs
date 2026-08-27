#![expect(
    clippy::expect_used,
    reason = "fixture generation should fail loudly when guest construction breaks"
)]

use std::sync::Arc;

use jolt_host::{JoltProgramSource, Program};
use jolt_program::execution::{JoltProgram, TraceInputs, TraceOutput};
use jolt_program::preprocess::JoltProgramPreprocessing;
use jolt_riscv::JoltTraceRow;
use tracer::execution_backend::TracerBackend;

pub struct PreparedGuest {
    pub program: Arc<JoltProgram>,
    pub program_preprocessing: JoltProgramPreprocessing,
    pub trace: TraceOutput<Arc<Vec<JoltTraceRow>>>,
}

pub fn prepare_guest(
    mut source: Program,
    inputs: &[u8],
    untrusted_advice: &[u8],
    trusted_advice: &[u8],
) -> PreparedGuest {
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
    let memory_config = common::jolt_device::MemoryConfig {
        max_untrusted_advice_size: layout.max_untrusted_advice_size,
        max_trusted_advice_size: layout.max_trusted_advice_size,
        max_input_size: layout.max_input_size,
        max_output_size: layout.max_output_size,
        stack_size: layout.stack_size,
        heap_size: layout.heap_size,
        program_size: Some(layout.program_size),
    };
    let trace = TracerBackend::new()
        .trace_compact(
            &program,
            TraceInputs::new(
                inputs.to_vec(),
                untrusted_advice.to_vec(),
                trusted_advice.to_vec(),
                memory_config,
            ),
            &program_preprocessing.bytecode,
        )
        .expect("modular trace");
    PreparedGuest {
        program,
        program_preprocessing,
        trace,
    }
}
