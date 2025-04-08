use common::constants::virtual_register_index;
use tracer::{ELFInstruction, RVTraceRow, RegisterState, RV32IM};

use super::{divu::DIVUInstruction, VirtualInstructionSequence};

/// Performs a logical right shift as a division by a power of 2.
pub struct SRLVirtualSequence<const WORD_SIZE: usize>;

impl<const WORD_SIZE: usize> VirtualInstructionSequence for SRLVirtualSequence<WORD_SIZE> {
    const SEQUENCE_LENGTH: usize = 1 + DIVUInstruction::<WORD_SIZE>::SEQUENCE_LENGTH;

    fn virtual_trace(trace_row: RVTraceRow) -> Vec<RVTraceRow> {
        let mut virtual_trace = vec![];
        // DIVU sequence uses virtual registers 0-3
        let v_pow2 = Some(virtual_register_index(4));

        let (pow2, result) = match trace_row.instruction.opcode {
            RV32IM::SRL => {
                let x = trace_row.register_state.rs1_val.unwrap();
                let y = trace_row.register_state.rs2_val.unwrap();
                let shift = y as usize % WORD_SIZE;

                let pow2: u64 = 1 << shift;
                virtual_trace.push(RVTraceRow {
                    instruction: ELFInstruction {
                        address: trace_row.instruction.address,
                        opcode: RV32IM::VIRTUAL_POW2,
                        rs1: trace_row.instruction.rs2,
                        rs2: None,
                        rd: v_pow2,
                        imm: None,
                        virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - 1),
                    },
                    register_state: RegisterState {
                        rs1_val: trace_row.register_state.rs2_val,
                        rs2_val: None,
                        rd_post_val: Some(pow2),
                    },
                    memory_state: None,
                    advice_value: None,
                    precompile_input: None,
                    precompile_output_address: None,
                });

                let result = x >> shift;
                (pow2, result)
            }
            RV32IM::SRLI => {
                let x = trace_row.register_state.rs1_val.unwrap();
                let shift = trace_row.instruction.imm.unwrap() as u64 as usize % WORD_SIZE;

                let pow2: u64 = 1 << shift;
                virtual_trace.push(RVTraceRow {
                    instruction: ELFInstruction {
                        address: trace_row.instruction.address,
                        opcode: RV32IM::VIRTUAL_POW2I,
                        rs1: None,
                        rs2: None,
                        rd: v_pow2,
                        imm: trace_row.instruction.imm,
                        virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - 1),
                    },
                    register_state: RegisterState {
                        rs1_val: None,
                        rs2_val: None,
                        rd_post_val: Some(pow2),
                    },
                    memory_state: None,
                    advice_value: None,
                    precompile_input: None,
                    precompile_output_address: None,
                });

                let result = x >> shift;
                (pow2, result)
            }
            _ => panic!("Unexpected opcode {:?}", trace_row.instruction.opcode),
        };

        let divu = RVTraceRow {
            instruction: ELFInstruction {
                address: trace_row.instruction.address,
                opcode: RV32IM::DIVU,
                rs1: trace_row.instruction.rs1,
                rs2: v_pow2,
                rd: trace_row.instruction.rd,
                imm: None,
                virtual_sequence_remaining: None,
            },
            register_state: RegisterState {
                rs1_val: trace_row.register_state.rs1_val,
                rs2_val: Some(pow2),
                rd_post_val: Some(result),
            },
            memory_state: None,
            advice_value: None,
            precompile_input: None,
            precompile_output_address: None,
        };
        let divu_sequence = DIVUInstruction::<WORD_SIZE>::virtual_trace(divu);
        virtual_trace.extend(divu_sequence);

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
