use common::constants::virtual_register_index;
use tracer::{ELFInstruction, MemoryState, RVTraceRow, RegisterState, RV32IM};

use super::VirtualInstructionSequence;
use crate::jolt::instruction::{
    add::ADDInstruction, and::ANDInstruction, sll::SLLInstruction,
    virtual_assert_aligned_memory_access::AssertAlignedMemoryAccessInstruction,
    xor::XORInstruction, JoltInstruction,
};
/// Stores a halfword in memory
pub struct SHInstruction<const WORD_SIZE: usize>;

impl<const WORD_SIZE: usize> VirtualInstructionSequence for SHInstruction<WORD_SIZE> {
    const SEQUENCE_LENGTH: usize = 12;

    fn virtual_trace(trace_row: RVTraceRow) -> Vec<RVTraceRow> {
        assert_eq!(trace_row.instruction.opcode, RV32IM::SH);
        // SH source registers
        let r_dest = trace_row.instruction.rs1;
        let r_value = trace_row.instruction.rs2;
        // Virtual registers used in sequence
        let v_address = Some(virtual_register_index(0));
        let v_word_address = Some(virtual_register_index(1));
        let v_word = Some(virtual_register_index(2));
        let v_shift = Some(virtual_register_index(3));
        let v_mask = Some(virtual_register_index(4));
        let v_halfword = Some(virtual_register_index(5));
        // SH operands
        let dest = trace_row.register_state.rs1_val.unwrap();
        let value = trace_row.register_state.rs2_val.unwrap();
        let offset = trace_row.instruction.imm.unwrap();

        let mut virtual_trace = vec![];

        let offset_unsigned = match WORD_SIZE {
            32 => (offset & u32::MAX as i64) as u64,
            64 => offset as u64,
            _ => panic!("Unsupported WORD_SIZE: {}", WORD_SIZE),
        };

        let is_aligned =
            AssertAlignedMemoryAccessInstruction::<WORD_SIZE, 2>(dest, offset_unsigned)
                .lookup_entry();
        debug_assert_eq!(is_aligned, 1);
        virtual_trace.push(RVTraceRow {
            instruction: ELFInstruction {
                address: trace_row.instruction.address,
                opcode: RV32IM::VIRTUAL_ASSERT_HALFWORD_ALIGNMENT,
                rs1: r_dest,
                rs2: None,
                rd: None,
                imm: Some(offset),
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
            },
            register_state: RegisterState {
                rs1_val: Some(dest),
                rs2_val: None,
                rd_post_val: None,
            },
            memory_state: None,
            advice_value: None,
        });

        let ram_address = ADDInstruction::<WORD_SIZE>(dest, offset_unsigned).lookup_entry();
        assert!(ram_address % 2 == 0);
        virtual_trace.push(RVTraceRow {
            instruction: ELFInstruction {
                address: trace_row.instruction.address,
                opcode: RV32IM::ADDI,
                rs1: r_dest,
                rs2: None,
                rd: v_address,
                imm: Some(offset),
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
            },
            register_state: RegisterState {
                rs1_val: Some(dest),
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

        let (word_loaded, word_stored) = match trace_row.memory_state.unwrap() {
            MemoryState::Read {
                address: _,
                value: _,
            } => panic!("Unexpected Read"),
            MemoryState::Write {
                address,
                pre_value,
                post_value,
            } => {
                if address != 0 {
                    // HACK: Don't check this if `virtual_trace`
                    // is being invoked by `virtual_sequence`, which
                    // passes in a dummy `trace_row`
                    assert_eq!(address, word_address);
                }
                (pre_value, post_value)
            }
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
                rd_post_val: Some(word_loaded),
            },
            memory_state: Some(MemoryState::Read {
                address: word_address,
                value: word_loaded,
            }),
            advice_value: None,
        });

        let bit_shift = SLLInstruction::<WORD_SIZE>(ram_address, 3).lookup_entry();
        virtual_trace.push(RVTraceRow {
            instruction: ELFInstruction {
                address: trace_row.instruction.address,
                opcode: RV32IM::SLLI,
                rs1: v_address,
                rs2: None,
                rd: v_shift,
                imm: Some(3),
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
            },
            register_state: RegisterState {
                rs1_val: Some(ram_address),
                rs2_val: None,
                rd_post_val: Some(bit_shift),
            },
            memory_state: None,
            advice_value: None,
        });

        // Technically such a LUI instruction isn't valid RISC-V, since the lower
        // 12 bits of the immediate should be 0s, but this shouldn't impact soundness.
        let halfword_mask = 0xffff;
        virtual_trace.push(RVTraceRow {
            instruction: ELFInstruction {
                address: trace_row.instruction.address,
                opcode: RV32IM::LUI,
                rs1: None,
                rs2: None,
                rd: v_mask,
                imm: Some(halfword_mask),
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
            },
            register_state: RegisterState {
                rs1_val: None,
                rs2_val: None,
                rd_post_val: Some(halfword_mask as u64),
            },
            memory_state: None,
            advice_value: None,
        });

        let shifted_mask =
            SLLInstruction::<WORD_SIZE>(halfword_mask as u64, bit_shift).lookup_entry();
        debug_assert!(shifted_mask == 0xffff0000 || shifted_mask == 0x0000ffff);
        virtual_trace.push(RVTraceRow {
            instruction: ELFInstruction {
                address: trace_row.instruction.address,
                opcode: RV32IM::SLL,
                rs1: v_mask,
                rs2: v_shift,
                rd: v_mask,
                imm: None,
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
            },
            register_state: RegisterState {
                rs1_val: Some(halfword_mask as u64),
                rs2_val: Some(bit_shift),
                rd_post_val: Some(shifted_mask),
            },
            memory_state: None,
            advice_value: None,
        });

        let shifted_value = SLLInstruction::<WORD_SIZE>(value, bit_shift).lookup_entry();
        virtual_trace.push(RVTraceRow {
            instruction: ELFInstruction {
                address: trace_row.instruction.address,
                opcode: RV32IM::SLL,
                rs1: r_value,
                rs2: v_shift,
                rd: v_halfword,
                imm: None,
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
            },
            register_state: RegisterState {
                rs1_val: Some(value),
                rs2_val: Some(bit_shift),
                rd_post_val: Some(shifted_value),
            },
            memory_state: None,
            advice_value: None,
        });

        // The next three instructions (XOR, AND, XOR) splices the halfword into
        // the word using the mask.
        // https://graphics.stanford.edu/~seander/bithacks.html#MaskedMerge
        let word_xor_halfword =
            XORInstruction::<WORD_SIZE>(word_loaded, shifted_value).lookup_entry();
        virtual_trace.push(RVTraceRow {
            instruction: ELFInstruction {
                address: trace_row.instruction.address,
                opcode: RV32IM::XOR,
                rs1: v_word,
                rs2: v_halfword,
                rd: v_halfword,
                imm: None,
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
            },
            register_state: RegisterState {
                rs1_val: Some(word_loaded),
                rs2_val: Some(shifted_value),
                rd_post_val: Some(word_xor_halfword),
            },
            memory_state: None,
            advice_value: None,
        });

        let masked = ANDInstruction::<WORD_SIZE>(word_xor_halfword, shifted_mask).lookup_entry();
        virtual_trace.push(RVTraceRow {
            instruction: ELFInstruction {
                address: trace_row.instruction.address,
                opcode: RV32IM::AND,
                rs1: v_halfword,
                rs2: v_mask,
                rd: v_halfword,
                imm: None,
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
            },
            register_state: RegisterState {
                rs1_val: Some(word_xor_halfword),
                rs2_val: Some(shifted_mask),
                rd_post_val: Some(masked),
            },
            memory_state: None,
            advice_value: None,
        });

        let result = XORInstruction::<WORD_SIZE>(word_loaded, masked).lookup_entry();
        assert_eq!(result, word_stored);
        virtual_trace.push(RVTraceRow {
            instruction: ELFInstruction {
                address: trace_row.instruction.address,
                opcode: RV32IM::XOR,
                rs1: v_word,
                rs2: v_halfword,
                rd: v_word,
                imm: None,
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
            },
            register_state: RegisterState {
                rs1_val: Some(word_loaded),
                rs2_val: Some(masked),
                rd_post_val: Some(result),
            },
            memory_state: None,
            advice_value: None,
        });

        virtual_trace.push(RVTraceRow {
            instruction: ELFInstruction {
                address: trace_row.instruction.address,
                opcode: RV32IM::SW,
                rs1: v_word_address,
                rs2: v_word,
                rd: None,
                imm: Some(0),
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
            },
            register_state: RegisterState {
                rs1_val: Some(word_address),
                rs2_val: Some(result), // Lookup query
                rd_post_val: None,
            },
            memory_state: Some(MemoryState::Write {
                address: word_address,
                pre_value: word_loaded,
                post_value: result,
            }),
            advice_value: None,
        });

        virtual_trace
    }

    fn sequence_output(_: u64, _: u64) -> u64 {
        unimplemented!("SH does not write to a destination register")
    }

    fn virtual_sequence(instruction: ELFInstruction) -> Vec<ELFInstruction> {
        let dummy_trace_row = RVTraceRow {
            instruction,
            register_state: RegisterState {
                rs1_val: Some(0),
                rs2_val: Some(0),
                rd_post_val: Some(0),
            },
            memory_state: Some(MemoryState::Write {
                address: 0,
                pre_value: 0,
                post_value: 0,
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
    fn sh_virtual_sequence_32() {
        let mut rng = test_rng();
        for _ in 0..256 {
            let rs1 = rng.next_u64() % 32;
            let rs2 = rng.next_u64() % 32;

            let mut rs1_val = rng.next_u32() as u64;
            let mut imm = rng.next_u64() as i64 % (1 << 12);

            // Reroll rs1_val and imm until dest is aligned to a halfword
            while (rs1_val as i64 + imm as i64) % 2 != 0 || (rs1_val as i64 + imm as i64) < 0 {
                rs1_val = rng.next_u32() as u64;
                imm = rng.next_u64() as i64 % (1 << 12);
            }
            let dest = (rs1_val as i64 + imm as i64) as u64;

            let rs2_val = rng.next_u32() as u64;
            let halfword = rs2_val & 0xffff;

            let word_address = (dest >> 2) << 2;
            let word_before = rng.next_u32() as u64;

            let word_after = if dest % 4 == 2 {
                (halfword << 16) | (word_before & 0xffff)
            } else {
                halfword | (word_before & 0xffff0000)
            };

            let sh_trace_row = RVTraceRow {
                instruction: ELFInstruction {
                    address: rng.next_u64(),
                    opcode: RV32IM::SH,
                    rs1: Some(rs1),
                    rs2: Some(rs2),
                    rd: None,
                    imm: Some(imm),
                    virtual_sequence_remaining: None,
                },
                register_state: RegisterState {
                    rs1_val: Some(rs1_val),
                    rs2_val: Some(rs2_val),
                    rd_post_val: None,
                },
                memory_state: Some(MemoryState::Write {
                    address: word_address,
                    pre_value: word_before,
                    post_value: word_after,
                }),
                advice_value: None,
            };

            let trace = SHInstruction::<32>::virtual_trace(sh_trace_row);
            assert_eq!(trace.len(), SHInstruction::<32>::SEQUENCE_LENGTH);
        }
    }
}
