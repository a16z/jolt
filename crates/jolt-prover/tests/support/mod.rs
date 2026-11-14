#![expect(
    clippy::expect_used,
    reason = "fixture preparation fails loudly when guest construction breaks"
)]

use std::sync::Arc;
#[cfg(feature = "zk")]
use std::thread::Builder;

use common::jolt_device::{JoltDevice, MemoryConfig, MemoryLayout};
use jolt_field::Fr;
use jolt_host::{JoltProgramSource, Program};
use jolt_program::execution::{ExecutionBackend, JoltProgram, TraceData, TraceInputs, TraceOutput};
use jolt_program::preprocess::JoltProgramPreprocessing;
use jolt_prover::ProverConfig;
use jolt_riscv::{JoltInstructionKind, JoltTraceRow};
use tracer::execution_backend::TracerBackend;

#[cfg(feature = "field-inline")]
pub mod field_inline;

/// One guest execution to prove: the example crate, its entry point, memory
/// overrides, the postcard-encoded inputs and advice, and the postcard-encoded
/// output the guest must produce.
#[derive(Clone)]
pub struct GuestCase {
    pub name: &'static str,
    pub func: Option<&'static str>,
    pub std: bool,
    pub stack_size: Option<u64>,
    /// Must match the guest's `max_untrusted_advice_size` attribute.
    pub max_untrusted_advice_size: Option<u64>,
    pub inputs: Vec<u8>,
    pub untrusted_advice: Vec<u8>,
    pub trusted_advice: Vec<u8>,
    /// Checked against the guest's output buffer before proving, so a wrong
    /// result fails as a broken guest rather than proving a wrong claim.
    pub expected_output: Option<Vec<u8>>,
    /// Padded trace maximum baked into preprocessing. `None` bakes in the
    /// execution's own padded domain and re-executes the guest's terminal `j .`
    /// stall until the trace fills it, so the proof has no padding row and its
    /// length equals the maximum.
    pub max_padded_trace_length: Option<usize>,
    #[cfg(feature = "field-inline")]
    pub field_inline_active: bool,
}

impl GuestCase {
    pub fn new(name: &'static str) -> Self {
        Self {
            name,
            func: None,
            std: false,
            stack_size: None,
            max_untrusted_advice_size: None,
            inputs: Vec::new(),
            untrusted_advice: Vec::new(),
            trusted_advice: Vec::new(),
            expected_output: None,
            max_padded_trace_length: Some(1 << 16),
            #[cfg(feature = "field-inline")]
            field_inline_active: false,
        }
    }

    fn assert_output(&self, device: &JoltDevice) {
        assert!(!device.panic, "{} panicked during execution", self.name);
        if let Some(expected) = &self.expected_output {
            let (head, tail) = device
                .outputs
                .split_at(expected.len().min(device.outputs.len()));
            assert!(
                head == expected.as_slice() && tail.iter().all(|byte| *byte == 0),
                "{}: guest output {:?} does not match the expected {:?}",
                self.name,
                head,
                expected,
            );
        }
    }
}

pub struct PreparedGuest {
    pub program: Arc<JoltProgram>,
    pub preprocessing: JoltProgramPreprocessing,
    pub trace: TraceOutput,
}

pub fn memory_config(layout: &MemoryLayout) -> MemoryConfig {
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
    #[cfg(feature = "field-inline")]
    source.enable_field_inline();
    if let Some(func) = case.func {
        source.set_func(func);
    }
    source.set_std(case.std);
    if let Some(bytes) = case.stack_size {
        source.set_stack_size(bytes);
    }
    if let Some(bytes) = case.max_untrusted_advice_size {
        source.set_max_untrusted_advice_size(bytes);
    }
    let (_, sizing_trace, _, device) =
        source.trace(&case.inputs, &case.untrusted_advice, &case.trusted_advice);
    case.assert_output(&device);
    let padded_trace_length =
        ProverConfig::padded_trace_length::<Fr>(sizing_trace.len()).expect("padded trace domain");
    let max_padded_trace_length = match case.max_padded_trace_length {
        Some(max) => {
            assert!(
                padded_trace_length <= max,
                "{}: {} cycles exceed the padded trace bound {max}",
                case.name,
                sizing_trace.len(),
            );
            max
        }
        None => padded_trace_length,
    };
    let layout = device.memory_layout;
    let program = Arc::new(source.build_jolt_program().expect("build Jolt program"));
    let preprocessing = JoltProgramPreprocessing::new(
        program.expanded_bytecode.clone(),
        program.memory_init.clone(),
        layout.clone(),
        program.entry_address,
        max_padded_trace_length,
        source.instruction_profile(),
    )
    .expect("program preprocessing");
    let inputs = TraceInputs::new(
        case.inputs.clone(),
        case.untrusted_advice.clone(),
        case.trusted_advice.clone(),
        memory_config(&layout),
    );
    let mut trace = TracerBackend::new()
        .trace(&program, inputs)
        .expect("modular trace");
    case.assert_output(&trace.device);
    if case.max_padded_trace_length.is_none() {
        trace = fill_trace_domain(trace, max_padded_trace_length);
    }
    #[cfg(feature = "field-inline")]
    assert_eq!(
        !trace.trace.field_events().is_empty(),
        case.field_inline_active,
        "{}: unexpected field-register activity",
        case.name,
    );
    PreparedGuest {
        program,
        preprocessing,
        trace,
    }
}

/// Extends an execution by re-executing its final row, which must be the
/// guest's `j .` stall. That continues the execution and does not pad it: every
/// added row is a taken jump to itself, which R1CS and the shift relation check
/// like any other row. The tracer rewrites `j .`'s `rd = x0` to a virtual
/// register, which already holds the link value on every re-execution.
fn fill_trace_domain(trace: TraceOutput, rows: usize) -> TraceOutput {
    let mut filled = trace.trace.proof_rows().to_vec();
    assert!(
        filled.len() <= rows,
        "the execution exceeds its padded domain"
    );
    let stall = *filled
        .last()
        .expect("a traced guest ends in its stall jump");
    let instruction = stall.instruction();
    assert!(
        instruction.instruction_kind == JoltInstructionKind::JAL && instruction.operands.imm == 0,
        "full-domain trace must end in a JAL jump to its own PC"
    );
    let mut registers = stall.registers();
    if let Some(link) = registers.rd.as_mut() {
        link.pre_value = link.post_value;
    }
    let repeat = JoltTraceRow::new(
        instruction,
        registers,
        stall.ram_access(),
        u32::try_from(stall.pc()).expect("bytecode PC fits u32"),
    )
    .expect("re-executed stall row");
    filled.resize(rows, repeat);
    #[cfg(feature = "field-inline")]
    let data = TraceData::from_parts(filled, trace.trace.field_events().to_vec())
        .expect("field events index executed rows");
    #[cfg(not(feature = "field-inline"))]
    let data = TraceData::new(filled);
    TraceOutput::new(data, trace.device, trace.final_memory, trace.advice_tape)
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
