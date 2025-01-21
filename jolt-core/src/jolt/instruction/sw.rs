use tracer::{ELFInstruction, MemoryState, RVTraceRow, RegisterState, RV32IM};

use super::VirtualInstructionSequence;
use crate::jolt::instruction::{
    virtual_assert_aligned_memory_access::AssertAlignedMemoryAccessInstruction, JoltInstruction,
};
/// Stores a word to memory
pub struct SWInstruction<const WORD_SIZE: usize>;

impl<const WORD_SIZE: usize> VirtualInstructionSequence for SWInstruction<WORD_SIZE> {
    const SEQUENCE_LENGTH: usize = 2;

    fn virtual_trace(mut trace_row: RVTraceRow) -> Vec<RVTraceRow> {
        assert_eq!(trace_row.instruction.opcode, RV32IM::SW);
        // SW source registers
        let rs1 = trace_row.instruction.rs1;
        // SW operands
        let dest = trace_row.register_state.rs1_val.unwrap();
        let offset = trace_row.instruction.imm.unwrap();

        let mut virtual_trace = vec![];

        let offset_unsigned = match WORD_SIZE {
            32 => (offset & u32::MAX as i64) as u64,
            64 => offset as u64,
            _ => panic!("Unsupported WORD_SIZE: {}", WORD_SIZE),
        };

        let is_aligned =
            AssertAlignedMemoryAccessInstruction::<WORD_SIZE, 4>(dest, offset_unsigned)
                .lookup_entry();
        debug_assert_eq!(is_aligned, 1);
        virtual_trace.push(RVTraceRow {
            instruction: ELFInstruction {
                address: trace_row.instruction.address,
                opcode: RV32IM::VIRTUAL_ASSERT_WORD_ALIGNMENT,
                rs1,
                rs2: None,
                rd: None,
                imm: Some(offset_unsigned as i64),
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

        trace_row.instruction.virtual_sequence_remaining = Some(0);
        virtual_trace.push(trace_row);

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
    fn sw_virtual_sequence_32() {
        let mut rng = test_rng();
        for _ in 0..256 {
            let rs1 = rng.next_u64() % 32;
            let rs2 = rng.next_u64() % 32;

            let mut rs1_val = rng.next_u32() as u64;
            let mut imm = rng.next_u64() as i64 % (1 << 12);

            // Reroll rs1_val and imm until dest is aligned to a word
            while (rs1_val as i64 + imm as i64) % 4 != 0 || (rs1_val as i64 + imm as i64) < 0 {
                rs1_val = rng.next_u32() as u64;
                imm = rng.next_u64() as i64 % (1 << 12);
            }

            let dest = (rs1_val as i64 + imm as i64) as u64;

            let word_before = rng.next_u32() as u64;
            let word_after = rng.next_u32() as u64;

            let sw_trace_row = RVTraceRow {
                instruction: ELFInstruction {
                    address: rng.next_u64(),
                    opcode: RV32IM::SW,
                    rs1: Some(rs1),
                    rs2: Some(rs2),
                    rd: None,
                    imm: Some(imm),
                    virtual_sequence_remaining: None,
                },
                register_state: RegisterState {
                    rs1_val: Some(rs1_val),
                    rs2_val: Some(word_after),
                    rd_post_val: None,
                },
                memory_state: Some(MemoryState::Write {
                    address: dest,
                    pre_value: word_before,
                    post_value: word_after,
                }),
                advice_value: None,
            };

            let trace = SWInstruction::<32>::virtual_trace(sw_trace_row);
            assert_eq!(trace.len(), SWInstruction::<32>::SEQUENCE_LENGTH);
        }
    }
}
