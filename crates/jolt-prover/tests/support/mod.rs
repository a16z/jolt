//! Guest preparation shared by the end-to-end suites: build the guest through
//! the host toolchain, preprocess it, and trace it with the modular tracer.

#![expect(
    clippy::expect_used,
    reason = "fixture preparation fails loudly when guest construction breaks"
)]

use std::sync::Arc;
#[cfg(feature = "zk")]
use std::thread::Builder;

use common::jolt_device::{MemoryConfig, MemoryLayout};
use jolt_host::{JoltProgramSource, Program};
use jolt_program::execution::{JoltProgram, TraceInputs, TraceOutput};
use jolt_program::preprocess::JoltProgramPreprocessing;
use jolt_riscv::JoltTraceRow;
use tracer::execution_backend::TracerBackend;

/// One guest execution to prove: the example crate, its entry point, memory
/// overrides, the postcard-encoded inputs and advice, and the postcard-encoded
/// output the guest must produce.
#[derive(Clone)]
pub struct GuestCase {
    pub name: &'static str,
    pub func: Option<&'static str>,
    pub std: bool,
    pub stack_size: Option<u64>,
    pub inputs: Vec<u8>,
    pub untrusted_advice: Vec<u8>,
    pub trusted_advice: Vec<u8>,
    /// Checked against the guest's output buffer before proving, so a wrong
    /// result fails as a broken guest rather than proving a wrong claim.
    pub expected_output: Option<Vec<u8>>,
    /// Padded trace bound baked into preprocessing.
    pub max_padded_trace_length: usize,
}

impl GuestCase {
    pub fn new(name: &'static str) -> Self {
        Self {
            name,
            func: None,
            std: false,
            stack_size: None,
            inputs: Vec::new(),
            untrusted_advice: Vec::new(),
            trusted_advice: Vec::new(),
            expected_output: None,
            max_padded_trace_length: 1 << 16,
        }
    }
}

pub struct PreparedGuest {
    pub program: Arc<JoltProgram>,
    pub preprocessing: JoltProgramPreprocessing,
    pub trace: TraceOutput<Arc<Vec<JoltTraceRow>>>,
}

fn memory_config(layout: &MemoryLayout) -> MemoryConfig {
    MemoryConfig {
        max_untrusted_advice_size: layout.max_untrusted_advice_size,
        max_trusted_advice_size: layout.max_trusted_advice_size,
        max_input_size: layout.max_input_size,
        max_output_size: layout.max_output_size,
        stack_size: layout.stack_size,
        heap_size: layout.heap_size,
        program_size: Some(layout.program_size),
    }
}

/// Builds, sizes, preprocesses, and traces `case`. A guest panic, a wrong
/// output, or a trace over the padded bound is a broken case rather than a
/// prover result, so all three fail here before any proving starts.
pub fn prepare(case: &GuestCase) -> PreparedGuest {
    let mut source = Program::new(case.name);
    if let Some(func) = case.func {
        source.set_func(func);
    }
    source.set_std(case.std);
    if let Some(bytes) = case.stack_size {
        source.set_stack_size(bytes);
    }
    let (_, sizing_trace, _, device) =
        source.trace(&case.inputs, &case.untrusted_advice, &case.trusted_advice);
    assert!(!device.panic, "{} panicked during execution", case.name);
    if let Some(expected) = &case.expected_output {
        let (head, tail) = device
            .outputs
            .split_at(expected.len().min(device.outputs.len()));
        assert!(
            head == expected.as_slice() && tail.iter().all(|byte| *byte == 0),
            "{}: guest output {:?} does not match the expected {:?}",
            case.name,
            &device.outputs[..expected.len().min(device.outputs.len())],
            expected,
        );
    }
    // Same padding law as `ProverConfig::derive`: one row is reserved beyond
    // the executed trace.
    assert!(
        (sizing_trace.len() + 1).next_power_of_two() <= case.max_padded_trace_length,
        "{}: {} cycles exceed the padded trace bound {}",
        case.name,
        sizing_trace.len(),
        case.max_padded_trace_length,
    );
    let layout = device.memory_layout;
    let program = Arc::new(source.build_jolt_program().expect("build Jolt program"));
    let preprocessing = JoltProgramPreprocessing::new(
        program.expanded_bytecode.clone(),
        program.memory_init.clone(),
        layout.clone(),
        program.entry_address,
        case.max_padded_trace_length,
        source.instruction_profile(),
    )
    .expect("program preprocessing");
    let trace = TracerBackend::new()
        .trace_compact(
            &program,
            TraceInputs::new(
                case.inputs.clone(),
                case.untrusted_advice.clone(),
                case.trusted_advice.clone(),
                memory_config(&layout),
            ),
            &preprocessing.bytecode,
        )
        .expect("modular trace");
    PreparedGuest {
        program,
        preprocessing,
        trace,
    }
}

/// BlindFold verification recurses over a large folded R1CS, so ZK tests run
/// on a wide stack like the verifier's own ZK suites.
#[cfg(feature = "zk")]
pub fn with_zk_stack(body: impl FnOnce() + Send + 'static) {
    Builder::new()
        .stack_size(128 * 1024 * 1024)
        .spawn(body)
        .expect("spawn ZK test thread")
        .join()
        .expect("ZK test thread panicked");
}
