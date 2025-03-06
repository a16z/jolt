// Implementation of the ECALL instruction.
// Reuse existing advice instruction (16x for each precompile word in the precompile output).
// Last instruction is a call to the precompile based on t0 register. 

use common::constants::virtual_register_index;
use common::precompiles::Precompile;
use tracer::{ELFInstruction, RVTraceRow, RegisterState, RV32IM};

use super::VirtualInstructionSequence;
use crate::jolt::instruction::{
    virtual_advice::ADVICEInstruction, JoltInstruction, precompile::PRECOMPILEInstruction,
};
/// Call a precompile based on the value in to register,
/// fetch the input from the precompile input memory region,
/// and write the output to the precompile output memory region.
pub struct EcallInstruction<const WORD_SIZE: usize>;

impl<const WORD_SIZE: usize> VirtualInstructionSequence for EcallInstruction<WORD_SIZE> {
    const SEQUENCE_LENGTH: usize = 16; // 16 or 17?

    fn virtual_trace(trace_row: RVTraceRow) -> Vec<RVTraceRow> {
        assert_eq!(trace_row.instruction.opcode, RV32IM::ECALL);
        // Ecall source registers
        let r_t0 = trace_row.instruction.rs1;

        let mut virtual_trace = vec![];

        // Precompile input is in the memory region reserved for the precompile input.
        let precompile_input = trace_row.precompile_input.unwrap();
        let precompile_output: [u32; 16] = Precompile::from_u64(trace_row.register_state.rs1_val.unwrap()).unwrap().execute(precompile_input);
        let precompile_output_address = trace_row.precompile_output_address.unwrap();

        let  ao0 = ADVICEInstruction::<WORD_SIZE>(precompile_output[0]).lookup_entry();
        virtual_trace.push(RVTraceRow {
            instruction: ELFInstruction {
                address: trace_row.instruction.address,
                opcode: RV32IM::VIRTUAL_ADVICE,
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
            memory_state: Some(MemoryState::Write {
                address: precompile_output_address,
                pre_value: 0_64,
                post_value: precompile_output[0] as u64,
            }),
            advice_value: None,
            precompile_input: None,
            precompile_output_address: None,
        });

        let  ao1 = ADVICEInstruction::<WORD_SIZE>(precompile_output[1]).lookup_entry();
        virtual_trace.push(RVTraceRow {
            instruction: ELFInstruction {
                address: trace_row.instruction.address,
                opcode: RV32IM::VIRTUAL_ADVICE,
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
            memory_state: Some(MemoryState::Write {
                address: precompile_output_address + 1_u64,
                pre_value: 0_64,
                post_value: precompile_output[1] as u64,
            }),
            advice_value: None,
            precompile_input: None,
            precompile_output_address: None,
        });

        let  ao2 = ADVICEInstruction::<WORD_SIZE>(precompile_output[2]).lookup_entry();
        virtual_trace.push(RVTraceRow {
            instruction: ELFInstruction {
                address: trace_row.instruction.address,
                opcode: RV32IM::VIRTUAL_ADVICE,
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
            memory_state: Some(MemoryState::Write {
                address: precompile_output_address + 2_u64,
                pre_value: 0_64,
                post_value: precompile_output[0] as u64,
            }),
            advice_value: None,
            precompile_input: None,
            precompile_output_address: None,
        });

        let  ao3 = ADVICEInstruction::<WORD_SIZE>(precompile_output[3]).lookup_entry();
        virtual_trace.push(RVTraceRow {
            instruction: ELFInstruction {
                address: trace_row.instruction.address,
                opcode: RV32IM::VIRTUAL_ADVICE,
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
            memory_state: Some(MemoryState::Write {
                address: precompile_output_address + 3_u64,
                pre_value: 0_64,
                post_value: precompile_output[3] as u64,
            }),
            advice_value: None,
            precompile_input: None,
            precompile_output_address: None,
        });

        let  ao4 = ADVICEInstruction::<WORD_SIZE>(precompile_output[4]).lookup_entry();
        virtual_trace.push(RVTraceRow {
            instruction: ELFInstruction {
                address: trace_row.instruction.address,
                opcode: RV32IM::VIRTUAL_ADVICE,
                rs1: None,
                rs2: None,
                rd: None,
                imm: None,
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
            },
            register_state: RegisterState {
                rs1_val: None,
                rs2_val: None,
                rd_post_val: Some(ao4),
            },
            memory_state: Some(MemoryState::Write {
                address: precompile_output_address + 4_u64,
                pre_value: 0_64,
                post_value: precompile_output[0] as u64,
            }),
            advice_value: None,
            precompile_input: None,
            precompile_output_address: None,
        });

        let  ao5 = ADVICEInstruction::<WORD_SIZE>(precompile_output[5]).lookup_entry();
        virtual_trace.push(RVTraceRow {
            instruction: ELFInstruction {
                address: trace_row.instruction.address,
                opcode: RV32IM::VIRTUAL_ADVICE,
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
            memory_state: Some(MemoryState::Write {
                address: precompile_output_address + + 5_u64,
                pre_value: 0_64,
                post_value: precompile_output[0] as u64,
            }),
            advice_value: None,
            precompile_input: None,
            precompile_output_address: None,
        });

        let  ao6 = ADVICEInstruction::<WORD_SIZE>(precompile_output[6]).lookup_entry();
        virtual_trace.push(RVTraceRow {
            instruction: ELFInstruction {
                address: trace_row.instruction.address,
                opcode: RV32IM::VIRTUAL_ADVICE,
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
            memory_state: Some(MemoryState::Write {
                address: precompile_output_address + 6_u64,
                pre_value: 0_64,
                post_value: precompile_output[0] as u64,
            }),
            advice_value: None,
            precompile_input: None,
            precompile_output_address: None,
        });

        let  ao7 = ADVICEInstruction::<WORD_SIZE>(precompile_output[7]).lookup_entry();
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
                rd_post_val: None,
            },
            memory_state: Some(MemoryState::Write {
                address: precompile_output_address + 7_u64,
                pre_value: 0_64,
                post_value: precompile_output[0] as u64,
            }),
            advice_value: None,
            precompile_input: None,
            precompile_output_address: None,
        });

        let  ao8 = ADVICEInstruction::<WORD_SIZE>(precompile_output[8]).lookup_entry();
        virtual_trace.push(RVTraceRow {
            instruction: ELFInstruction {
                address: trace_row.instruction.address,
                opcode: RV32IM::VIRTUAL_ADVICE,
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
            memory_state: Some(MemoryState::Write {
                address: precompile_output_address + 8_u64,
                pre_value: 0_64,
                post_value: precompile_output[0] as u64,
            }),
            advice_value: None,
            precompile_input: None,
            precompile_output_address: None,
        });

        let  ao9 = ADVICEInstruction::<WORD_SIZE>(precompile_output[9]).lookup_entry();
        virtual_trace.push(RVTraceRow {
            instruction: ELFInstruction {
                address: trace_row.instruction.address,
                opcode: RV32IM::VIRTUAL_ADVICE,
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
            memory_state: Some(MemoryState::Write {
                address: precompile_output_address + 9_u64,
                pre_value: 0_64,
                post_value: precompile_output[0] as u64,
            }),
            advice_value: None,
            precompile_input: None,
            precompile_output_address: None,
        });

        let  ao10 = ADVICEInstruction::<WORD_SIZE>(precompile_output[10]).lookup_entry();
        virtual_trace.push(RVTraceRow {
            instruction: ELFInstruction {
                address: trace_row.instruction.address,
                opcode: RV32IM::VIRTUAL_ADVICE,
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
            memory_state: Some(MemoryState::Write {
                address: precompile_output_address + 10_u64,
                pre_value: 0_64,
                post_value: precompile_output[10] as u64,
            }),
            advice_value: None,
            precompile_input: None,
            precompile_output_address: None,
        });

        let  ao11 = ADVICEInstruction::<WORD_SIZE>(precompile_output[11]).lookup_entry();
        virtual_trace.push(RVTraceRow {
            instruction: ELFInstruction {
                address: trace_row.instruction.address,
                opcode: RV32IM::VIRTUAL_ADVICE,
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
            memory_state: Some(MemoryState::Write {
                address: precompile_output_address + 11_u64,
                pre_value: 0_64,
                post_value: precompile_output[11] as u64,
            }),
            advice_value: None,
            precompile_input: None,
            precompile_output_address: None,
        });

        let  ao12 = ADVICEInstruction::<WORD_SIZE>(precompile_output[12]).lookup_entry();
        virtual_trace.push(RVTraceRow {
            instruction: ELFInstruction {
                address: trace_row.instruction.address,
                opcode: RV32IM::VIRTUAL_ADVICE,
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
            memory_state: Some(MemoryState::Write {
                address: precompile_output_address + 12_u64,
                pre_value: 0_64,
                post_value: precompile_output[12] as u64,
            }),
            advice_value: None,
            precompile_input: None,
            precompile_output_address: None,
        });

        let  ao13 = ADVICEInstruction::<WORD_SIZE>(precompile_output[13]).lookup_entry();
        virtual_trace.push(RVTraceRow {
            instruction: ELFInstruction {
                address: trace_row.instruction.address,
                opcode: RV32IM::VIRTUAL_ADVICE,
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
            memory_state: Some(MemoryState::Write {
                address: precompile_output_address + 13_u64,
                pre_value: 0_64,
                post_value: precompile_output[13] as u64,
            }),
            advice_value: None,
            precompile_input: None,
            precompile_output_address: None,
        });

        let  ao14 = ADVICEInstruction::<WORD_SIZE>(precompile_output[14]).lookup_entry();
        virtual_trace.push(RVTraceRow {
            instruction: ELFInstruction {
                address: trace_row.instruction.address,
                opcode: RV32IM::VIRTUAL_ADVICE,
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
            memory_state: Some(MemoryState::Write {
                address: precompile_output_address + 14_u64,
                pre_value: 0_64,
                post_value: precompile_output[14] as u64,
            }),
            advice_value: None,
            precompile_input: None,
            precompile_output_address: None,
        });

        let  ao15 = ADVICEInstruction::<WORD_SIZE>(precompile_output[15]).lookup_entry();
        virtual_trace.push(RVTraceRow {
            instruction: ELFInstruction {
                address: trace_row.instruction.address,
                opcode: RV32IM::VIRTUAL_ADVICE,
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
            memory_state: Some(MemoryState::Write {
                address: precompile_output_address + 15_u64,
                pre_value: 0_64,
                post_value: precompile_output[15] as u64,
            }),
            advice_value: None,
            precompile_input: None,
            precompile_output_address: None,
        });

        let precompile_instruction = PRECOMPILEInstruction::<WORD_SIZE>(trace_row.register_state.rs1_val.unwrap()).lookup_entry();
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
            precompile_input: Some(precompile_input),
            precompile_output_address: Some(precompile_output_address),
        });

        virtual_trace
    }

    fn sequence_output(x: u64, y: u64) -> u64 {
        unimplemented!()
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
