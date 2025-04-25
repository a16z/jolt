use serde::{Deserialize, Serialize};

use crate::emulator::cpu::Cpu;

use super::{
    format::{format_i::FormatI, normalize_imm, InstructionFormat},
    RISCVInstruction, RISCVTrace,
};

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct SLTI {
    pub address: u64,
    pub operands: FormatI,
    /// If this instruction is part of a "virtual sequence" (see Section 6.2 of the
    /// Jolt paper), then this contains the number of virtual instructions after this
    /// one in the sequence. I.e. if this is the last instruction in the sequence,
    /// `virtual_sequence_remaining` will be Some(0); if this is the penultimate instruction
    /// in the sequence, `virtual_sequence_remaining` will be Some(1); etc.
    pub virtual_sequence_remaining: Option<usize>,
}

impl RISCVInstruction for SLTI {
    const MASK: u32 = 0x0000707f;
    const MATCH: u32 = 0x00002013;

    type Format = FormatI;
    type RAMAccess = ();

    fn operands(&self) -> &Self::Format {
        &self.operands
    }

    fn new(word: u32, address: u64, validate: bool) -> Self {
        if validate {
            debug_assert_eq!(word & Self::MASK, Self::MATCH);
        }

        Self {
            address,
            operands: FormatI::parse(word),
            virtual_sequence_remaining: None,
        }
    }

    fn execute(&self, cpu: &mut Cpu, _: &mut Self::RAMAccess) {
        cpu.x[self.operands.rd] = match cpu.x[self.operands.rs1] < normalize_imm(self.operands.imm)
        {
            true => 1,
            false => 0,
        };
    }
}

impl RISCVTrace for SLTI {}
