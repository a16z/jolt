use serde::{Deserialize, Serialize};

use crate::emulator::cpu::Cpu;

use super::MemoryWrite;

use super::{
    format::{FormatS, InstructionFormat},
    RISCVInstruction,
};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SB<const WORD_SIZE: usize> {
    pub address: u64,
    pub operands: FormatS,
    /// If this instruction is part of a "virtual sequence" (see Section 6.2 of the
    /// Jolt paper), then this contains the number of virtual instructions after this
    /// one in the sequence. I.e. if this is the last instruction in the sequence,
    /// `virtual_sequence_remaining` will be Some(0); if this is the penultimate instruction
    /// in the sequence, `virtual_sequence_remaining` will be Some(1); etc.
    pub virtual_sequence_remaining: Option<usize>,
}

impl<const WORD_SIZE: usize> RISCVInstruction for SB<WORD_SIZE> {
    const MASK: u32 = 0x0000707f;
    const MATCH: u32 = 0x00000023;

    type Format = FormatS;
    type RAMAccess = MemoryWrite;

    fn operands(&self) -> &Self::Format {
        &self.operands
    }

    fn new(word: u32, address: u64) -> Self {
        Self {
            address,
            operands: FormatS::parse(word),
            virtual_sequence_remaining: None,
        }
    }

    fn execute(&self, cpu: &mut Cpu, memory_state: &mut Self::RAMAccess) {
        *memory_state = cpu
            .mmu
            .store(
                cpu.x[self.operands.rs1].wrapping_add(self.operands.imm) as u64,
                cpu.x[self.operands.rs2] as u8,
            )
            .ok()
            .unwrap();
    }
}
