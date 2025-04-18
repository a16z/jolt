use serde::{Deserialize, Serialize};

use crate::emulator::cpu::Cpu;

use super::{
    format::{FormatI, InstructionFormat},
    RISCVInstruction,
};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct JALR<const WORD_SIZE: usize> {
    pub address: u64,
    pub operands: FormatI,
    /// If this instruction is part of a "virtual sequence" (see Section 6.2 of the
    /// Jolt paper), then this contains the number of virtual instructions after this
    /// one in the sequence. I.e. if this is the last instruction in the sequence,
    /// `virtual_sequence_remaining` will be Some(0); if this is the penultimate instruction
    /// in the sequence, `virtual_sequence_remaining` will be Some(1); etc.
    pub virtual_sequence_remaining: Option<usize>,
}

impl<const WORD_SIZE: usize> RISCVInstruction for JALR<WORD_SIZE> {
    const MASK: u32 = 0x0000707f;
    const MATCH: u32 = 0x00000067;

    type Format = FormatI;
    type RAMAccess = ();

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

    fn execute(&self, cpu: &mut Cpu, _: &mut Self::RAMAccess) {
        let tmp = cpu.sign_extend(cpu.pc as i64);
        cpu.pc = (cpu.x[self.operands.rs1] as u64).wrapping_add(self.operands.imm as u64);
        cpu.x[self.operands.rd] = tmp;
    }
}
