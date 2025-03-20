use common::constants::virtual_register_index;
use tracer::{ELFInstruction, RVTraceRow, RegisterState, RV32IM};

use super::VirtualInstructionSequence;

pub struct SRAVirtualSequence<const WORD_SIZE: usize>;

impl<const WORD_SIZE: usize> VirtualInstructionSequence for SRAVirtualSequence<WORD_SIZE> {
    const SEQUENCE_LENGTH: usize = 15;

    fn virtual_trace(trace_row: RVTraceRow) -> Vec<RVTraceRow> {
        let mut virtual_trace = vec![];
        let v0 = Some(virtual_register_index(0));
        let v1 = Some(virtual_register_index(1));
        let v2 = Some(virtual_register_index(2));
        let v3 = Some(virtual_register_index(3));
        let v4 = Some(virtual_register_index(4));
        let v_result = Some(virtual_register_index(5));

        match trace_row.instruction.opcode {
            RV32IM::SRA => {
                let x = trace_row.register_state.rs1_val.unwrap();
                let y = trace_row.register_state.rs2_val.unwrap();
                let shift = y as usize % WORD_SIZE;
                let result = Self::sequence_output(x, y);

                let bitmask_shift = y ^ (WORD_SIZE - 1) as u64;
                virtual_trace.push(RVTraceRow {
                    instruction: ELFInstruction {
                        address: trace_row.instruction.address,
                        opcode: RV32IM::XORI,
                        rs1: trace_row.instruction.rs2,
                        rs2: None,
                        rd: v0,
                        imm: Some(WORD_SIZE as i64 - 1),
                        virtual_sequence_remaining: Some(
                            Self::SEQUENCE_LENGTH - virtual_trace.len() - 1,
                        ),
                    },
                    register_state: RegisterState {
                        rs1_val: trace_row.register_state.rs2_val,
                        rs2_val: None,
                        rd_post_val: Some(bitmask_shift as u64),
                    },
                    memory_state: None,
                    advice_value: None,
                });

                let shift_pow2: u64 = 1 << shift;
                virtual_trace.push(RVTraceRow {
                    instruction: ELFInstruction {
                        address: trace_row.instruction.address,
                        opcode: RV32IM::VIRTUAL_POW2,
                        rs1: trace_row.instruction.rs2,
                        rs2: None,
                        rd: v1,
                        imm: None,
                        virtual_sequence_remaining: Some(
                            Self::SEQUENCE_LENGTH - virtual_trace.len() - 1,
                        ),
                    },
                    register_state: RegisterState {
                        rs1_val: trace_row.register_state.rs2_val,
                        rs2_val: None,
                        rd_post_val: Some(shift_pow2),
                    },
                    memory_state: None,
                    advice_value: None,
                });

                virtual_trace.push(RVTraceRow {
                    instruction: ELFInstruction {
                        address: trace_row.instruction.address,
                        opcode: RV32IM::VIRTUAL_ADVICE,
                        rs1: None,
                        rs2: None,
                        rd: v_result,
                        imm: None,
                        virtual_sequence_remaining: Some(
                            Self::SEQUENCE_LENGTH - virtual_trace.len() - 1,
                        ),
                    },
                    register_state: RegisterState {
                        rs1_val: None,
                        rs2_val: None,
                        rd_post_val: Some(result),
                    },
                    memory_state: None,
                    advice_value: Some(result),
                });

                let bitmask_shift_pow2 = 1 << (bitmask_shift % WORD_SIZE as u64);
                virtual_trace.push(RVTraceRow {
                    instruction: ELFInstruction {
                        address: trace_row.instruction.address,
                        opcode: RV32IM::VIRTUAL_POW2,
                        rs1: v0,
                        rs2: None,
                        rd: v2,
                        imm: None,
                        virtual_sequence_remaining: Some(
                            Self::SEQUENCE_LENGTH - virtual_trace.len() - 1,
                        ),
                    },
                    register_state: RegisterState {
                        rs1_val: Some(bitmask_shift as u64),
                        rs2_val: None,
                        rd_post_val: Some(bitmask_shift_pow2),
                    },
                    memory_state: None,
                    advice_value: None,
                });

                let mut bitmask = (1 << WORD_SIZE) - 1;
                virtual_trace.push(RVTraceRow {
                    instruction: ELFInstruction {
                        address: trace_row.instruction.address,
                        opcode: RV32IM::LUI,
                        rs1: None,
                        rs2: None,
                        rd: v3,
                        imm: Some(bitmask as i64),
                        virtual_sequence_remaining: Some(
                            Self::SEQUENCE_LENGTH - virtual_trace.len() - 1,
                        ),
                    },
                    register_state: RegisterState {
                        rs1_val: None,
                        rs2_val: None,
                        rd_post_val: Some(bitmask),
                    },
                    memory_state: None,
                    advice_value: None,
                });

                let bitmask_shifted =
                    (bitmask << (bitmask_shift % WORD_SIZE as u64)) % (1 << WORD_SIZE);
                virtual_trace.push(RVTraceRow {
                    instruction: ELFInstruction {
                        address: trace_row.instruction.address,
                        opcode: RV32IM::MUL,
                        rs1: v3,
                        rs2: v2,
                        rd: v3,
                        imm: None,
                        virtual_sequence_remaining: Some(
                            Self::SEQUENCE_LENGTH - virtual_trace.len() - 1,
                        ),
                    },
                    register_state: RegisterState {
                        rs1_val: Some(bitmask),
                        rs2_val: Some(bitmask_shift_pow2),
                        rd_post_val: Some(bitmask_shifted),
                    },
                    memory_state: None,
                    advice_value: None,
                });
                bitmask = bitmask_shifted;

                virtual_trace.push(RVTraceRow {
                    instruction: ELFInstruction {
                        address: trace_row.instruction.address,
                        opcode: RV32IM::ADD,
                        rs1: v3,
                        rs2: v3,
                        rd: v3,
                        imm: None,
                        virtual_sequence_remaining: Some(
                            Self::SEQUENCE_LENGTH - virtual_trace.len() - 1,
                        ),
                    },
                    register_state: RegisterState {
                        rs1_val: Some(bitmask),
                        rs2_val: Some(bitmask),
                        rd_post_val: Some((bitmask + bitmask) % (1 << WORD_SIZE)),
                    },
                    memory_state: None,
                    advice_value: None,
                });
                bitmask = (bitmask + bitmask) % (1 << WORD_SIZE);

                let masked_advice = bitmask & result;
                virtual_trace.push(RVTraceRow {
                    instruction: ELFInstruction {
                        address: trace_row.instruction.address,
                        opcode: RV32IM::AND,
                        rs1: v3,
                        rs2: v_result,
                        rd: v2,
                        imm: None,
                        virtual_sequence_remaining: Some(
                            Self::SEQUENCE_LENGTH - virtual_trace.len() - 1,
                        ),
                    },
                    register_state: RegisterState {
                        rs1_val: Some(bitmask),
                        rs2_val: Some(result),
                        rd_post_val: Some(masked_advice),
                    },
                    memory_state: None,
                    advice_value: None,
                });

                let is_negative = x >> (WORD_SIZE - 1);
                virtual_trace.push(RVTraceRow {
                    instruction: ELFInstruction {
                        address: trace_row.instruction.address,
                        opcode: RV32IM::SLT,
                        rs1: trace_row.instruction.rs1,
                        rs2: Some(0), // zero register
                        rd: v4,
                        imm: None,
                        virtual_sequence_remaining: Some(
                            Self::SEQUENCE_LENGTH - virtual_trace.len() - 1,
                        ),
                    },
                    register_state: RegisterState {
                        rs1_val: trace_row.register_state.rs1_val,
                        rs2_val: Some(0),
                        rd_post_val: Some(is_negative),
                    },
                    memory_state: None,
                    advice_value: None,
                });

                virtual_trace.push(RVTraceRow {
                    instruction: ELFInstruction {
                        address: trace_row.instruction.address,
                        opcode: RV32IM::MUL,
                        rs1: v3,
                        rs2: v4,
                        rd: v3,
                        imm: None,
                        virtual_sequence_remaining: Some(
                            Self::SEQUENCE_LENGTH - virtual_trace.len() - 1,
                        ),
                    },
                    register_state: RegisterState {
                        rs1_val: Some(bitmask),
                        rs2_val: Some(is_negative),
                        rd_post_val: Some(is_negative * bitmask),
                    },
                    memory_state: None,
                    advice_value: None,
                });

                virtual_trace.push(RVTraceRow {
                    instruction: ELFInstruction {
                        address: trace_row.instruction.address,
                        opcode: RV32IM::VIRTUAL_ASSERT_EQ,
                        rs1: v2,
                        rs2: v3,
                        rd: None,
                        imm: None,
                        virtual_sequence_remaining: Some(
                            Self::SEQUENCE_LENGTH - virtual_trace.len() - 1,
                        ),
                    },
                    register_state: RegisterState {
                        rs1_val: Some(masked_advice),
                        rs2_val: Some(is_negative * bitmask),
                        rd_post_val: None,
                    },
                    memory_state: None,
                    advice_value: None,
                });

                let shifted_advice = (result * shift_pow2) % (1 << WORD_SIZE);
                virtual_trace.push(RVTraceRow {
                    instruction: ELFInstruction {
                        address: trace_row.instruction.address,
                        opcode: RV32IM::MUL,
                        rs1: v_result,
                        rs2: v1,
                        rd: v3,
                        imm: None,
                        virtual_sequence_remaining: Some(
                            Self::SEQUENCE_LENGTH - virtual_trace.len() - 1,
                        ),
                    },
                    register_state: RegisterState {
                        rs1_val: Some(result),
                        rs2_val: Some(shift_pow2),
                        rd_post_val: Some(shifted_advice),
                    },
                    memory_state: None,
                    advice_value: None,
                });

                virtual_trace.push(RVTraceRow {
                    instruction: ELFInstruction {
                        address: trace_row.instruction.address,
                        opcode: RV32IM::AND,
                        rs1: trace_row.instruction.rs1,
                        rs2: v3,
                        rd: v4,
                        imm: None,
                        virtual_sequence_remaining: Some(
                            Self::SEQUENCE_LENGTH - virtual_trace.len() - 1,
                        ),
                    },
                    register_state: RegisterState {
                        rs1_val: Some(x),
                        rs2_val: Some(shifted_advice),
                        rd_post_val: Some(x & shifted_advice),
                    },
                    memory_state: None,
                    advice_value: None,
                });

                virtual_trace.push(RVTraceRow {
                    instruction: ELFInstruction {
                        address: trace_row.instruction.address,
                        opcode: RV32IM::VIRTUAL_ASSERT_EQ,
                        rs1: v3,
                        rs2: v4,
                        rd: None,
                        imm: None,
                        virtual_sequence_remaining: Some(
                            Self::SEQUENCE_LENGTH - virtual_trace.len() - 1,
                        ),
                    },
                    register_state: RegisterState {
                        rs1_val: Some(shifted_advice),
                        rs2_val: Some(x & shifted_advice),
                        rd_post_val: None,
                    },
                    memory_state: None,
                    advice_value: None,
                });

                virtual_trace.push(RVTraceRow {
                    instruction: ELFInstruction {
                        address: trace_row.instruction.address,
                        opcode: RV32IM::VIRTUAL_MOVE,
                        rs1: v_result,
                        rs2: None,
                        rd: trace_row.instruction.rd,
                        imm: None,
                        virtual_sequence_remaining: Some(
                            Self::SEQUENCE_LENGTH - virtual_trace.len() - 1,
                        ),
                    },
                    register_state: RegisterState {
                        rs1_val: Some(result),
                        rs2_val: None,
                        rd_post_val: Some(result),
                    },
                    memory_state: None,
                    advice_value: None,
                });
            }
            RV32IM::SRAI => {
                let x = trace_row.register_state.rs1_val.unwrap();
                let imm = trace_row.instruction.imm.unwrap() as u64;
                let shift = imm as usize % WORD_SIZE;

                let result = Self::sequence_output(x, imm);
                todo!()
            }
            _ => panic!("Unexpected opcode {:?}", trace_row.instruction.opcode),
        };

        virtual_trace
    }

    fn sequence_output(x: u64, y: u64) -> u64 {
        match WORD_SIZE {
            8 => ((x as i8).wrapping_shr(y as u32 % 8)) as u8 as u64,
            32 => ((x as i32).wrapping_shr(y as u32 % 32)) as u32 as u64,
            _ => panic!("{WORD_SIZE}-bit word size is unsupported"),
        }
    }
}

#[cfg(test)]
mod test {
    use crate::jolt::instruction::test::jolt_virtual_sequence_test;

    use super::*;

    #[test]
    fn sra_virtual_sequence_32() {
        jolt_virtual_sequence_test::<SRAVirtualSequence<32>>(RV32IM::SRA);
    }
}
