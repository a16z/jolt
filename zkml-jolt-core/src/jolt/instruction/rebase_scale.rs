// TODO

// use common::constants::virtual_register_index;
// use tracer::{ELFInstruction, RVTraceRow, RegisterState, RV32IM};

// use super::VirtualInstructionSequence;
// use crate::jolt::instruction::{
//     add::ADDInstruction, beq::BEQInstruction, mul::MULInstruction,
//     virtual_advice::ADVICEInstruction, virtual_assert_valid_div0::AssertValidDiv0Instruction,
//     virtual_assert_valid_signed_remainder::AssertValidSignedRemainderInstruction, JoltInstruction,
// };

use onnx_tracer::trace_types::{ONNXCycle, ONNXOpcode};

use crate::jolt::instruction::VirtualInstructionSequence;

macro_rules! expect_rebase_scale {
    ($cycle:expr) => {
        match $cycle.instr.opcode {
            ONNXOpcode::RebaseScale(_) => {}
            _ => panic!("Expected ONNXOpcode::RebaseScale"),
        }
    };
}

/// Perform signed division and return the result
pub struct REBASEInstruction;

impl VirtualInstructionSequence for REBASEInstruction {
    const SEQUENCE_LENGTH: usize = 2;

    fn virtual_trace(cycle: ONNXCycle) -> Vec<ONNXCycle> {
        expect_rebase_scale!(cycle);
        let mut virtual_sequence = vec![];
        if let ONNXOpcode::RebaseScale(inner_opcode) = &cycle.instr.opcode {
            let inner_opcode = (**inner_opcode).clone();
            let mut instr = cycle.instr.clone();
            instr.opcode = inner_opcode;
            virtual_sequence.push(ONNXCycle {
                instr,
                memory_state: cycle.memory_state.clone(),
                advice_value: None,
            });
        };
        // TODO: Do expanded div sequence here
        let mut div_cycle = cycle;
        div_cycle.instr.opcode = ONNXOpcode::Div;
        virtual_sequence.push(div_cycle);
        virtual_sequence
    }

    fn sequence_output(_x: u64, _y: u64) -> u64 {
        todo!()
    }
}

// #[cfg(test)]
// mod test {
//     use super::*;
//     use crate::{jolt::instruction::JoltInstruction, jolt_virtual_sequence_test};

//     #[test]
//     fn div_virtual_sequence_32() {
//         jolt_virtual_sequence_test!(DIVInstruction::<32>, RV32IM::DIV);
//     }
// }
