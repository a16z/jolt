use common::constants::virtual_register_index;
use tracer::{ELFInstruction, RVTraceRow, RegisterState, RV32IM};

use crate::jolt::instruction::VirtualInstructionSequence;
use crate::jolt::instruction::{
    add::ADDInstruction, beq::BEQInstruction, mul::MULInstruction,
    virtual_advice::ADVICEInstruction, virtual_assert_valid_div0::AssertValidDiv0Instruction,
    virtual_assert_valid_signed_remainder::AssertValidSignedRemainderInstruction, JoltInstruction,
};
use crate::jolt_onnx::common::onnx_trace::{LayerState, ONNXInstruction, ONNXTraceRow, Operator};
/// Return the maximum in a window.
pub struct MaxWindowInstruction;

// TODO: Need a modified for `VirtualInstructionSequence` for ONNX

impl MaxWindowInstruction {
    const SEQUENCE_LENGTH: usize = todo!();

    fn virtual_trace(trace_row: ONNXTraceRow) -> Vec<ONNXTraceRow> {
        assert_eq!(trace_row.instruction.opcode, Operator::MaxWindow);
        // MaxWindow source slots // TODO

        // Virtual registers used in sequence
        let v_0 = Some(virtual_register_index(0));

        // MaxWindow operands
        let x = trace_row
            .layer_state
            .input_vals
            .and_then(|i| i.get(0).cloned())
            .unwrap();

        let mut virtual_trace = vec![];
        for i in 0..x.len() {
            virtual_trace.push(ONNXTraceRow {
                instruction: ONNXInstruction {
                    opcode: Operator::Max,
                    attributes: None,
                    inputs: vec![],
                    outputs: vec![],
                },
                layer_state: LayerState {
                    input_vals: None,
                    output_vals: None,
                },
            })
        }
        virtual_trace
    }

    fn sequence_output(x: Vec<i8>, _: Vec<i8>) -> i8 {
        x.iter().max().cloned().unwrap()
    }
}

#[cfg(test)]
mod test {}
