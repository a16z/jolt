use onnx_tracer::trace_types::{MemoryState, ONNXCycle, ONNXInstr};

pub mod add;
pub mod div;
pub mod mul;
pub mod rebase_scale;
pub mod sub;
pub mod virtual_advice;

pub trait VirtualInstructionSequence {
    const SEQUENCE_LENGTH: usize;
    fn virtual_sequence(instr: ONNXInstr) -> Vec<ONNXInstr> {
        let dummy_cycle = ONNXCycle {
            instr,
            memory_state: MemoryState::default(),
            advice_value: None,
        };
        Self::virtual_trace(dummy_cycle)
            .into_iter()
            .map(|cycle| cycle.instr)
            .collect()
    }
    fn virtual_trace(cycle: ONNXCycle) -> Vec<ONNXCycle>;
    fn sequence_output(x: u64, y: u64) -> u64;
}
