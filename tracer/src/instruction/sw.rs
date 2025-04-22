use serde::{Deserialize, Serialize};

use crate::emulator::cpu::Cpu;

use super::RAMWrite;

use super::{
    format::{format_s::FormatS, InstructionFormat},
    RISCVInstruction, RISCVTrace,
};

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct SW {
    pub address: u64,
    pub operands: FormatS,
    /// If this instruction is part of a "virtual sequence" (see Section 6.2 of the
    /// Jolt paper), then this contains the number of virtual instructions after this
    /// one in the sequence. I.e. if this is the last instruction in the sequence,
    /// `virtual_sequence_remaining` will be Some(0); if this is the penultimate instruction
    /// in the sequence, `virtual_sequence_remaining` will be Some(1); etc.
    pub virtual_sequence_remaining: Option<usize>,
}

impl RISCVInstruction for SW {
    const MASK: u32 = 0x0000707f;
    const MATCH: u32 = 0x00002023;

    type Format = FormatS;
    type RAMAccess = RAMWrite;

    fn operands(&self) -> &Self::Format {
        &self.operands
    }

    fn new(word: u32, address: u64, validate: bool) -> Self {
        if validate {
            assert_eq!(word & Self::MASK, Self::MATCH);
        }

        Self {
            address,
            operands: FormatS::parse(word),
            virtual_sequence_remaining: None,
        }
    }

    fn execute(&self, cpu: &mut Cpu, ram_access: &mut Self::RAMAccess) {
        *ram_access = cpu
            .mmu
            .store_word(
                cpu.x[self.operands.rs1].wrapping_add(self.operands.imm) as u64,
                cpu.x[self.operands.rs2] as u32,
            )
            .ok()
            .unwrap();
    }
}

impl RISCVTrace for SW {}
