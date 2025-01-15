use common::constants::virtual_register_index;
use tracer::{ELFInstruction, MemoryState, RVTraceRow, RegisterState, RV32IM};

use super::VirtualInstructionSequence;
use crate::jolt::instruction::{
    add::ADDInstruction, and::ANDInstruction, sll::SLLInstruction, sra::SRAInstruction,
    virtual_assert_aligned_memory_access::AssertAlignedMemoryAccessInstruction,
    xor::XORInstruction, JoltInstruction,
};
/// Loads a halfword from memory and sign-extends it
pub struct LHInstruction<const WORD_SIZE: usize>;

impl<const WORD_SIZE: usize> VirtualInstructionSequence for LHInstruction<WORD_SIZE> {
    const SEQUENCE_LENGTH: usize = 8;

    fn virtual_trace(trace_row: RVTraceRow) -> Vec<RVTraceRow> {
        assert_eq!(trace_row.instruction.opcode, RV32IM::LH);
        let expected_rd_post_val = trace_row.register_state.rd_post_val.unwrap();
        // LH source registers
        let rs1 = trace_row.instruction.rs1;
        let rd = trace_row.instruction.rd;
        // Virtual registers used in sequence
        let v_address = Some(virtual_register_index(0));
        let v_word_address = Some(virtual_register_index(1));
        let v_word = Some(virtual_register_index(2));
        let v_shift = Some(virtual_register_index(3));
        // LH operands
        let rs1_val = trace_row.register_state.rs1_val.unwrap();
        let offset = trace_row.instruction.imm.unwrap();

        let mut virtual_trace = vec![];

        let offset_unsigned = match WORD_SIZE {
            32 => (offset & u32::MAX as i64) as u64,
            64 => offset as u64,
            _ => panic!("Unsupported WORD_SIZE: {}", WORD_SIZE),
        };

        let is_aligned =
            AssertAlignedMemoryAccessInstruction::<WORD_SIZE, 2>(rs1_val, offset_unsigned)
                .lookup_entry();
        debug_assert_eq!(is_aligned, 1);
        virtual_trace.push(RVTraceRow {
            instruction: ELFInstruction {
                address: trace_row.instruction.address,
                opcode: RV32IM::VIRTUAL_ASSERT_HALFWORD_ALIGNMENT,
                rs1,
                rs2: None,
                rd: None,
                imm: Some(offset),
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
            },
            register_state: RegisterState {
                rs1_val: Some(rs1_val),
                rs2_val: None,
                rd_post_val: None,
            },
            memory_state: None,
            advice_value: None,
        });

        let ram_address = ADDInstruction::<WORD_SIZE>(rs1_val, offset_unsigned).lookup_entry();
        virtual_trace.push(RVTraceRow {
            instruction: ELFInstruction {
                address: trace_row.instruction.address,
                opcode: RV32IM::ADDI,
                rs1,
                rs2: None,
                rd: v_address,
                imm: Some(offset),
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
            },
            register_state: RegisterState {
                rs1_val: Some(rs1_val),
                rs2_val: None,
                rd_post_val: Some(ram_address),
            },
            memory_state: None,
            advice_value: None,
        });

        let word_address_bitmask = ((1u128 << WORD_SIZE) - 4) as u64;
        let word_address =
            ANDInstruction::<WORD_SIZE>(ram_address, word_address_bitmask).lookup_entry();
        virtual_trace.push(RVTraceRow {
            instruction: ELFInstruction {
                address: trace_row.instruction.address,
                opcode: RV32IM::ANDI,
                rs1: v_address,
                rs2: None,
                rd: v_word_address,
                imm: Some(word_address_bitmask as i64),
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
            },
            register_state: RegisterState {
                rs1_val: Some(ram_address),
                rs2_val: None,
                rd_post_val: Some(word_address),
            },
            memory_state: None,
            advice_value: None,
        });

        let word = match trace_row.memory_state.unwrap() {
            MemoryState::Read { address, value } => {
                if address != 0 {
                    // HACK: Don't check this if `virtual_trace`
                    // is being invoked by `virtual_sequence`, which
                    // passes in a dummy `trace_row`
                    assert_eq!(address, word_address);
                }
                value
            }
            MemoryState::Write {
                address: _,
                pre_value: _,
                post_value: _,
            } => panic!("Unexpected Write"),
        };
        virtual_trace.push(RVTraceRow {
            instruction: ELFInstruction {
                address: trace_row.instruction.address,
                opcode: RV32IM::LW,
                rs1: v_word_address,
                rs2: None,
                rd: v_word,
                imm: Some(0),
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
            },
            register_state: RegisterState {
                rs1_val: Some(word_address),
                rs2_val: None,
                rd_post_val: Some(word),
            },
            memory_state: Some(MemoryState::Read {
                address: word_address,
                value: word,
            }),
            advice_value: None,
        });

        let byte_shift = XORInstruction::<WORD_SIZE>(ram_address, 0b10).lookup_entry();
        virtual_trace.push(RVTraceRow {
            instruction: ELFInstruction {
                address: trace_row.instruction.address,
                opcode: RV32IM::XORI,
                rs1: v_address,
                rs2: None,
                rd: v_shift,
                imm: Some(0b10),
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
            },
            register_state: RegisterState {
                rs1_val: Some(ram_address),
                rs2_val: None,
                rd_post_val: Some(byte_shift),
            },
            memory_state: None,
            advice_value: None,
        });

        let bit_shift = SLLInstruction::<WORD_SIZE>(byte_shift, 3).lookup_entry();
        virtual_trace.push(RVTraceRow {
            instruction: ELFInstruction {
                address: trace_row.instruction.address,
                opcode: RV32IM::SLLI,
                rs1: v_shift,
                rs2: None,
                rd: v_shift,
                imm: Some(3),
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
            },
            register_state: RegisterState {
                rs1_val: Some(byte_shift),
                rs2_val: None,
                rd_post_val: Some(bit_shift),
            },
            memory_state: None,
            advice_value: None,
        });

        let left_aligned_halfword = SLLInstruction::<WORD_SIZE>(word, bit_shift).lookup_entry();
        virtual_trace.push(RVTraceRow {
            instruction: ELFInstruction {
                address: trace_row.instruction.address,
                opcode: RV32IM::SLL,
                rs1: v_word,
                rs2: v_shift,
                rd,
                imm: None,
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
            },
            register_state: RegisterState {
                rs1_val: Some(word),
                rs2_val: Some(bit_shift),
                rd_post_val: Some(left_aligned_halfword),
            },
            memory_state: None,
            advice_value: None,
        });

        let sign_extended_halfword =
            SRAInstruction::<WORD_SIZE>(left_aligned_halfword, 16).lookup_entry();
        assert_eq!(sign_extended_halfword, expected_rd_post_val);
        virtual_trace.push(RVTraceRow {
            instruction: ELFInstruction {
                address: trace_row.instruction.address,
                opcode: RV32IM::SRAI,
                rs1: rd,
                rs2: None,
                rd,
                imm: Some(16),
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
            },
            register_state: RegisterState {
                rs1_val: Some(left_aligned_halfword),
                rs2_val: None,
                rd_post_val: Some(sign_extended_halfword),
            },
            memory_state: None,
            advice_value: None,
        });

        virtual_trace
    }

    fn sequence_output(_: u64, _: u64) -> u64 {
        unimplemented!("")
    }

    fn virtual_sequence(instruction: ELFInstruction) -> Vec<ELFInstruction> {
        let dummy_trace_row = RVTraceRow {
            instruction,
            register_state: RegisterState {
                rs1_val: Some(0),
                rs2_val: Some(0),
                rd_post_val: Some(0),
            },
            memory_state: Some(MemoryState::Read {
                address: 0,
                value: 0,
            }),
            advice_value: None,
        };
        Self::virtual_trace(dummy_trace_row)
            .into_iter()
            .map(|trace_row| trace_row.instruction)
            .collect()
    }
}

#[cfg(test)]
mod test {
    use ark_std::test_rng;
    use rand_core::RngCore;

    use super::*;

    #[test]
    fn lh_virtual_sequence_32() {
        let mut rng = test_rng();
        for _ in 0..256 {
            let rs1 = rng.next_u64() % 32;
            let rd = rng.next_u64() % 32;

            let mut rs1_val = rng.next_u32() as u64;
            let mut imm = rng.next_u64() as i64 % (1 << 12);

            // Reroll rs1_val and imm until dest is aligned to a halfword
            while (rs1_val as i64 + imm as i64) % 2 != 0 || (rs1_val as i64 + imm as i64) < 0 {
                rs1_val = rng.next_u32() as u64;
                imm = rng.next_u64() as i64 % (1 << 12);
            }
            let address = (rs1_val as i64 + imm as i64) as u64;

            let word_address = (address >> 2) << 2;
            let word = rng.next_u32() as u64;

            let halfword = match address % 4 {
                0 => word & 0x0000ffff,
                2 => (word & 0xffff0000) >> 16,
                _ => unreachable!(),
            } as u16 as i16 as i32 as u32 as u64; // sign-extend

            let lh_trace_row = RVTraceRow {
                instruction: ELFInstruction {
                    address: rng.next_u64(),
                    opcode: RV32IM::LH,
                    rs1: Some(rs1),
                    rs2: None,
                    rd: Some(rd),
                    imm: Some(imm),
                    virtual_sequence_remaining: None,
                },
                register_state: RegisterState {
                    rs1_val: Some(rs1_val),
                    rs2_val: None,
                    rd_post_val: Some(halfword),
                },
                memory_state: Some(MemoryState::Read {
                    address: word_address,
                    value: word,
                }),
                advice_value: None,
            };

            let trace = LHInstruction::<32>::virtual_trace(lh_trace_row);
            assert_eq!(trace.len(), LHInstruction::<32>::SEQUENCE_LENGTH);
        }
    }
}
