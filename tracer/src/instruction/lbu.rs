use serde::{Deserialize, Serialize};

use crate::emulator::cpu::Cpu;

use super::MemoryRead;

use super::{
    format::{FormatI, InstructionFormat},
    RISCVInstruction,
};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LBU<const WORD_SIZE: usize> {
    pub address: u64,
    pub operands: FormatI,
    /// If this instruction is part of a "virtual sequence" (see Section 6.2 of the
    /// Jolt paper), then this contains the number of virtual instructions after this
    /// one in the sequence. I.e. if this is the last instruction in the sequence,
    /// `virtual_sequence_remaining` will be Some(0); if this is the penultimate instruction
    /// in the sequence, `virtual_sequence_remaining` will be Some(1); etc.
    pub virtual_sequence_remaining: Option<usize>,
}

impl<const WORD_SIZE: usize> RISCVInstruction for LBU<WORD_SIZE> {
    const MASK: u32 = 0x0000707f;
    const MATCH: u32 = 0x00004003;

    type Format = FormatI;
    type RAMAccess = MemoryRead;

    fn operands(&self) -> &Self::Format {
        &self.operands
    }

    fn new(word: u32, address: u64) -> Self {
        debug_assert_eq!(word & Self::MASK, Self::MATCH);

        Self {
            address,
            operands: FormatI::parse(word),
            virtual_sequence_remaining: None,
        }
    }

    fn execute(&self, cpu: &mut Cpu, memory_state: &mut Self::RAMAccess) {
        cpu.x[self.operands.rd] = match cpu
            .mmu
            .load(cpu.x[self.operands.rs1].wrapping_add(self.operands.imm) as u64)
        {
            Ok((byte, memory_read)) => {
                *memory_state = memory_read;
                byte as i64
            }
            Err(_) => panic!("MMU load error"),
        };
    }
}
