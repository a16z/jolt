use serde::{Deserialize, Serialize};

use crate::emulator::cpu::{Cpu, Xlen};

use super::{
    format::{FormatR, InstructionFormat},
    RISCVInstruction,
};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MULH<const WORD_SIZE: usize> {
    pub address: u64,
    pub operands: FormatR,
    /// If this instruction is part of a "virtual sequence" (see Section 6.2 of the
    /// Jolt paper), then this contains the number of virtual instructions after this
    /// one in the sequence. I.e. if this is the last instruction in the sequence,
    /// `virtual_sequence_remaining` will be Some(0); if this is the penultimate instruction
    /// in the sequence, `virtual_sequence_remaining` will be Some(1); etc.
    pub virtual_sequence_remaining: Option<usize>,
}

impl<const WORD_SIZE: usize> RISCVInstruction for MULH<WORD_SIZE> {
    const MASK: u32 = 0xfe00707f;
    const MATCH: u32 = 0x02001033;

    type Format = FormatR;
    type RAMAccess = ();

    fn operands(&self) -> &Self::Format {
        &self.operands
    }

    fn new(word: u32, address: u64) -> Self {
        Self {
            address,
            operands: FormatR::parse(word),
            virtual_sequence_remaining: None,
        }
    }

    fn execute(&self, cpu: &mut Cpu, _: &mut Self::RAMAccess) {
        cpu.x[self.operands.rd] = match cpu.xlen {
            Xlen::Bit32 => {
                cpu.sign_extend((cpu.x[self.operands.rs1] * cpu.x[self.operands.rs2]) >> 32)
            }
            Xlen::Bit64 => {
                (((cpu.x[self.operands.rs1] as i128) * (cpu.x[self.operands.rs2] as i128)) >> 64)
                    as i64
            }
        };
    }
}
