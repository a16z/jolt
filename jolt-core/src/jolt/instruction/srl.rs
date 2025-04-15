use common::constants::virtual_register_index;
use tracer::{ELFInstruction, RVTraceRow, RegisterState, RV32IM};

use super::{
    virtual_shift_right_bitmask::ShiftRightBitmaskInstruction, virtual_srl::VirtualSRLInstruction,
    JoltInstruction, VirtualInstructionSequence,
};

/// Performs a logical right shift as a division by a power of 2.
pub struct SRLVirtualSequence<const WORD_SIZE: usize>;

impl<const WORD_SIZE: usize> VirtualInstructionSequence for SRLVirtualSequence<WORD_SIZE> {
    const SEQUENCE_LENGTH: usize = 2;

    fn virtual_trace(trace_row: RVTraceRow) -> Vec<RVTraceRow> {
        let mut virtual_trace = vec![];
        let v0 = Some(virtual_register_index(0));

        let (x, bitmask) = match trace_row.instruction.opcode {
            RV32IM::SRL => {
                let x = trace_row.register_state.rs1_val.unwrap();
                let y = trace_row.register_state.rs2_val.unwrap();

                let bitmask = ShiftRightBitmaskInstruction::<WORD_SIZE>(y).lookup_entry();
                virtual_trace.push(RVTraceRow {
                    instruction: ELFInstruction {
                        address: trace_row.instruction.address,
                        opcode: RV32IM::VIRTUAL_SHIFT_RIGHT_BITMASK,
                        rs1: trace_row.instruction.rs2,
                        rs2: None,
                        rd: v0,
                        imm: None,
                        virtual_sequence_remaining: Some(
                            Self::SEQUENCE_LENGTH - virtual_trace.len() - 1,
                        ),
                    },
                    register_state: RegisterState {
                        rs1_val: trace_row.register_state.rs2_val,
                        rs2_val: None,
                        rd_post_val: Some(bitmask),
                    },
                    memory_state: None,
                    advice_value: None,
                    precompile_input: None,
                    precompile_output_address: None,
                });

                (x, bitmask)
            }
            RV32IM::SRLI => {
                let x = trace_row.register_state.rs1_val.unwrap();
                let imm = trace_row.instruction.imm.unwrap() as u64;

                let bitmask = ShiftRightBitmaskInstruction::<WORD_SIZE>(imm).lookup_entry();
                virtual_trace.push(RVTraceRow {
                    instruction: ELFInstruction {
                        address: trace_row.instruction.address,
                        opcode: RV32IM::VIRTUAL_SHIFT_RIGHT_BITMASKI,
                        rs1: None,
                        rs2: None,
                        rd: v0,
                        imm: trace_row.instruction.imm,
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
                    precompile_input: None,
                    precompile_output_address: None,
                });

                (x, bitmask)
            }
            _ => panic!("Unexpected opcode {:?}", trace_row.instruction.opcode),
        };

        let result = VirtualSRLInstruction::<WORD_SIZE>(x, bitmask).lookup_entry();
        virtual_trace.push(RVTraceRow {
            instruction: ELFInstruction {
                address: trace_row.instruction.address,
                opcode: RV32IM::VIRTUAL_SRL,
                rs1: trace_row.instruction.rs1,
                rs2: v0,
                rd: trace_row.instruction.rd,
                imm: None,
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
            },
            register_state: RegisterState {
                rs1_val: trace_row.register_state.rs1_val,
                rs2_val: Some(bitmask),
                rd_post_val: Some(result),
            },
            memory_state: None,
            advice_value: None,
            precompile_input: None,
            precompile_output_address: None,
        });

        virtual_trace
    }

    fn sequence_output(x: u64, y: u64) -> u64 {
        x >> (y % WORD_SIZE as u64)
    }
}

#[cfg(test)]
mod test {
    use crate::jolt::instruction::test::jolt_virtual_sequence_test;

    use super::*;

    #[test]
    fn srl_virtual_sequence_32() {
        jolt_virtual_sequence_test::<SRLVirtualSequence<32>>(RV32IM::SRL);
    }
}
