// Implementation of the ECALL instruction.
// Reuse existing advice instruction (16x for each precompile word in the precompile output).
// Last instruction is a call to the precompile based on t0 register. 

use common::constants::virtual_register_index;
use tracer::{ELFInstruction, RVTraceRow, RegisterState, RV32IM};

use super::VirtualInstructionSequence;
use crate::jolt::instruction::{
    virtual_advice::ADVICEInstruction, JoltInstruction,
};
/// Call a precompile based on the value in to register,
/// fetch the input from the precompile input memory region,
/// and write the output to the precompile output memory region.
pub struct EcallInstruction;

impl VirtualInstructionSequence for EcallInstruction {
    const SEQUENCE_LENGTH: usize = 8; // needs change?

    fn virtual_trace(trace_row: RVTraceRow) -> Vec<RVTraceRow> {
        assert_eq!(trace_row.instruction.opcode, RV32IM::ECALL);
        // Ecall source registers
        let r_x = trace_row.instruction.rs1;
        let r_y = trace_row.instruction.rs2;
        // Virtual registers used in sequence
        let v_0 = Some(virtual_register_index(0));
        let v_q: Option<u64> = Some(virtual_register_index(1));
        let v_r: Option<u64> = Some(virtual_register_index(2));
        let v_qy = Some(virtual_register_index(3));
        // Precompile input
        let x = trace_row.register_state.rs1_val.unwrap();
        let y = trace_row.register_state.rs2_val.unwrap();

        let mut virtual_trace = vec![];

        let precompile_output = // compute precompile output based on t0 register

        let q = ADVICEInstruction::(precompile_output).lookup_entry();
        virtual_trace.push(RVTraceRow {
            instruction: ELFInstruction {
                address: trace_row.instruction.address,
                opcode: RV32IM::VIRTUAL_ADVICE,
                rs1: None,
                rs2: None,
                rd: v_q,
                imm: None,
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
            },
            register_state: RegisterState {
                rs1_val: None,
                rs2_val: None,
                rd_post_val: Some(q),
            },
            memory_state: None,
            advice_value: Some(precompile_output),
        });

        virtual_trace.push(RVTraceRow {
            instruction: ELFInstruction {
                address: trace_row.instruction.address,
                opcode: RV32IM::VIRTUAL_MOVE,
                rs1: v_q,
                rs2: None,
                rd: trace_row.instruction.rd,
                imm: None,
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
            },
            register_state: RegisterState {
                rs1_val: Some(q),
                rs2_val: None,
                rd_post_val: Some(q),
            },
            memory_state: None,
            advice_value: None,
        });

        virtual_trace
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{jolt::instruction::JoltInstruction, jolt_virtual_sequence_test};

    #[test]
    fn ecall_virtual_sequence_32() {
        jolt_virtual_sequence_test!(EcallInstruction::<32>, RV32IM::ECALL);
    }
}
