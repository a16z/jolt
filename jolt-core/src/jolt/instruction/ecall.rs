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
    const SEQUENCE_LENGTH: usize = 16; // 16 or 17?

    fn virtual_trace(trace_row: RVTraceRow) -> Vec<RVTraceRow> {
        assert_eq!(trace_row.instruction.opcode, RV32IM::ECALL);
        // Ecall source registers
        let r_x = trace_row.instruction.rs1;
        let r_y = trace_row.instruction.rs2;
        // Virtual registers used in sequence
        let v_0 = Some(virtual_register_index(0));
        let v_ao0: Option<u64> = Some(virtual_register_index(1));
        let v_ao1: Option<u64> = Some(virtual_register_index(2));
        let v_ao2: Option<u64> = Some(virtual_register_index(3));
        let v_ao3: Option<u64> = Some(virtual_register_index(4));
        let v_ao4: Option<u64> = Some(virtual_register_index(5));
        let v_ao5: Option<u64> = Some(virtual_register_index(6));
        let v_ao6: Option<u64> = Some(virtual_register_index(7));
        let v_ao7: Option<u64> = Some(virtual_register_index(8));
        let v_ao8: Option<u64> = Some(virtual_register_index(9));
        let v_ao9: Option<u64> = Some(virtual_register_index(10));
        let v_ao10: Option<u64> = Some(virtual_register_index(11));
        let v_ao11: Option<u64> = Some(virtual_register_index(12));
        let v_ao12: Option<u64> = Some(virtual_register_index(13));
        let v_ao13: Option<u64> = Some(virtual_register_index(14));
        let v_ao14: Option<u64> = Some(virtual_register_index(15));
        let v_ao15: Option<u64> = Some(virtual_register_index(16));

        let mut virtual_trace = vec![];

        let precompile_output: &[u32; 16] = &[0u32; 16];  // compute precompile output based on t0 register

        let  ao0 = ADVICEInstruction::(precompile_output[0]).lookup_entry();
        virtual_trace.push(RVTraceRow {
            instruction: ELFInstruction {
                address: trace_row.instruction.address,
                opcode: RV32IM::VIRTUAL_ADVICE,
                rs1: None,
                rs2: None,
                rd: v_ao0,
                imm: None,
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
            },
            register_state: RegisterState {
                rs1_val: None,
                rs2_val: None,
                rd_post_val: Some(ao0),
            },
            memory_state: None,
            advice_value: Some(precompile_output[0]),
        });

        let  ao1 = ADVICEInstruction::(precompile_output[1]).lookup_entry();
        virtual_trace.push(RVTraceRow {
            instruction: ELFInstruction {
                address: trace_row.instruction.address,
                opcode: RV32IM::VIRTUAL_ADVICE,
                rs1: None,
                rs2: None,
                rd: v_ao1,
                imm: None,
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
            },
            register_state: RegisterState {
                rs1_val: None,
                rs2_val: None,
                rd_post_val: Some(ao1),
            },
            memory_state: None,
            advice_value: Some(precompile_output[1]),
        });

        let  ao2 = ADVICEInstruction::(precompile_output[2]).lookup_entry();
        virtual_trace.push(RVTraceRow {
            instruction: ELFInstruction {
                address: trace_row.instruction.address,
                opcode: RV32IM::VIRTUAL_ADVICE,
                rs1: None,
                rs2: None,
                rd: v_ao2,
                imm: None,
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
            },
            register_state: RegisterState {
                rs1_val: None,
                rs2_val: None,
                rd_post_val: Some(ao2),
            },
            memory_state: None,
            advice_value: Some(precompile_output[2]),
        });

        let  ao3 = ADVICEInstruction::(precompile_output[3]).lookup_entry();
        virtual_trace.push(RVTraceRow {
            instruction: ELFInstruction {
                address: trace_row.instruction.address,
                opcode: RV32IM::VIRTUAL_ADVICE,
                rs1: None,
                rs2: None,
                rd: v_ao3,
                imm: None,
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
            },
            register_state: RegisterState {
                rs1_val: None,
                rs2_val: None,
                rd_post_val: Some(ao3),
            },
            memory_state: None,
            advice_value: Some(precompile_output[3]),
        });

        let  ao4 = ADVICEInstruction::(precompile_output[4]).lookup_entry();
        virtual_trace.push(RVTraceRow {
            instruction: ELFInstruction {
                address: trace_row.instruction.address,
                opcode: RV32IM::VIRTUAL_ADVICE,
                rs1: None,
                rs2: None,
                rd: v_ao4,
                imm: None,
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
            },
            register_state: RegisterState {
                rs1_val: None,
                rs2_val: None,
                rd_post_val: Some(ao4),
            },
            memory_state: None,
            advice_value: Some(precompile_output[4]),
        });

        let  ao5 = ADVICEInstruction::(precompile_output[5]).lookup_entry();
        virtual_trace.push(RVTraceRow {
            instruction: ELFInstruction {
                address: trace_row.instruction.address,
                opcode: RV32IM::VIRTUAL_ADVICE,
                rs1: None,
                rs2: None,
                rd: v_ao5,
                imm: None,
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
            },
            register_state: RegisterState {
                rs1_val: None,
                rs2_val: None,
                rd_post_val: Some(ao5),
            },
            memory_state: None,
            advice_value: Some(precompile_output[5]),
        });

        let  ao6 = ADVICEInstruction::(precompile_output[6]).lookup_entry();
        virtual_trace.push(RVTraceRow {
            instruction: ELFInstruction {
                address: trace_row.instruction.address,
                opcode: RV32IM::VIRTUAL_ADVICE,
                rs1: None,
                rs2: None,
                rd: v_ao6,
                imm: None,
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
            },
            register_state: RegisterState {
                rs1_val: None,
                rs2_val: None,
                rd_post_val: Some(ao6),
            },
            memory_state: None,
            advice_value: Some(precompile_output[6]),
        });

        let  ao7 = ADVICEInstruction::(precompile_output[7]).lookup_entry();
        virtual_trace.push(RVTraceRow {
            instruction: ELFInstruction {
                address: trace_row.instruction.address,
                opcode: RV32IM::VIRTUAL_ADVICE,
                rs1: None,
                rs2: None,
                rd: v_ao7,
                imm: None,
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
            },
            register_state: RegisterState {
                rs1_val: None,
                rs2_val: None,
                rd_post_val: Some(ao7),
            },
            memory_state: None,
            advice_value: Some(precompile_output[7]),
        });

        let  ao8 = ADVICEInstruction::(precompile_output[8]).lookup_entry();
        virtual_trace.push(RVTraceRow {
            instruction: ELFInstruction {
                address: trace_row.instruction.address,
                opcode: RV32IM::VIRTUAL_ADVICE,
                rs1: None,
                rs2: None,
                rd: v_ao8,
                imm: None,
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
            },
            register_state: RegisterState {
                rs1_val: None,
                rs2_val: None,
                rd_post_val: Some(ao8),
            },
            memory_state: None,
            advice_value: Some(precompile_output[8]),
        });

        let  ao9 = ADVICEInstruction::(precompile_output[9]).lookup_entry();
        virtual_trace.push(RVTraceRow {
            instruction: ELFInstruction {
                address: trace_row.instruction.address,
                opcode: RV32IM::VIRTUAL_ADVICE,
                rs1: None,
                rs2: None,
                rd: v_ao9,
                imm: None,
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
            },
            register_state: RegisterState {
                rs1_val: None,
                rs2_val: None,
                rd_post_val: Some(ao9),
            },
            memory_state: None,
            advice_value: Some(precompile_output[9]),
        });

        let  ao10 = ADVICEInstruction::(precompile_output[10]).lookup_entry();
        virtual_trace.push(RVTraceRow {
            instruction: ELFInstruction {
                address: trace_row.instruction.address,
                opcode: RV32IM::VIRTUAL_ADVICE,
                rs1: None,
                rs2: None,
                rd: v_ao10,
                imm: None,
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
            },
            register_state: RegisterState {
                rs1_val: None,
                rs2_val: None,
                rd_post_val: Some(a10),
            },
            memory_state: None,
            advice_value: Some(precompile_output[10]),
        });

        let  ao11 = ADVICEInstruction::(precompile_output[11]).lookup_entry();
        virtual_trace.push(RVTraceRow {
            instruction: ELFInstruction {
                address: trace_row.instruction.address,
                opcode: RV32IM::VIRTUAL_ADVICE,
                rs1: None,
                rs2: None,
                rd: v_ao11,
                imm: None,
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
            },
            register_state: RegisterState {
                rs1_val: None,
                rs2_val: None,
                rd_post_val: Some(ao11),
            },
            memory_state: None,
            advice_value: Some(precompile_output[11]),
        });

        let  ao12 = ADVICEInstruction::(precompile_output[12]).lookup_entry();
        virtual_trace.push(RVTraceRow {
            instruction: ELFInstruction {
                address: trace_row.instruction.address,
                opcode: RV32IM::VIRTUAL_ADVICE,
                rs1: None,
                rs2: None,
                rd: v_a012,
                imm: None,
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
            },
            register_state: RegisterState {
                rs1_val: None,
                rs2_val: None,
                rd_post_val: Some(ao12),
            },
            memory_state: None,
            advice_value: Some(precompile_output[12]),
        });

        let  ao13 = ADVICEInstruction::(precompile_output[13]).lookup_entry();
        virtual_trace.push(RVTraceRow {
            instruction: ELFInstruction {
                address: trace_row.instruction.address,
                opcode: RV32IM::VIRTUAL_ADVICE,
                rs1: None,
                rs2: None,
                rd: v_ao13,
                imm: None,
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
            },
            register_state: RegisterState {
                rs1_val: None,
                rs2_val: None,
                rd_post_val: Some(ao13),
            },
            memory_state: None,
            advice_value: Some(precompile_output[13]),
        });

        let  ao14 = ADVICEInstruction::(precompile_output[14]).lookup_entry();
        virtual_trace.push(RVTraceRow {
            instruction: ELFInstruction {
                address: trace_row.instruction.address,
                opcode: RV32IM::VIRTUAL_ADVICE,
                rs1: None,
                rs2: None,
                rd: v_ao14,
                imm: None,
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
            },
            register_state: RegisterState {
                rs1_val: None,
                rs2_val: None,
                rd_post_val: Some(ao14),
            },
            memory_state: None,
            advice_value: Some(precompile_output[14]),
        });

        let  ao15 = ADVICEInstruction::(precompile_output[15]).lookup_entry();
        virtual_trace.push(RVTraceRow {
            instruction: ELFInstruction {
                address: trace_row.instruction.address,
                opcode: RV32IM::VIRTUAL_ADVICE,
                rs1: None,
                rs2: None,
                rd: v_ao15,
                imm: None,
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
            },
            register_state: RegisterState {
                rs1_val: None,
                rs2_val: None,
                rd_post_val: Some(ao15),
            },
            memory_state: None,
            advice_value: Some(precompile_output[15]),
        });

        // Precompile instruction.  How do we set the precompile circuit flag to 1?
        let precompile_instruction = PRECOMPILEInstructionInstruction::(trace_row.register_state.t0).lookup_entry();
        virtual_trace.push(RVTraceRow {
            instruction: ELFInstruction {
                address: trace_row.instruction.address,
                opcode: RV32IM::PRECOMPILE,
                rs1: None,
                rs2: None,
                rd: None,
                imm: None,
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
            },
            register_state: RegisterState {
                rs1_val: None,
                rs2_val: None,
                rd_post_val: None,
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
        jolt_virtual_sequence_test!(EcallInstruction, RV32IM::ECALL);
    }
}
