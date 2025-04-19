use serde::{Deserialize, Serialize};

use crate::emulator::cpu::{Cpu, Xlen};

use super::{format::FormatJ, RISCVInstruction};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct VirtualShiftRightBitmaskI<const WORD_SIZE: usize> {
    pub address: u64,
    pub operands: FormatJ,
    /// If this instruction is part of a "virtual sequence" (see Section 6.2 of the
    /// Jolt paper), then this contains the number of virtual instructions after this
    /// one in the sequence. I.e. if this is the last instruction in the sequence,
    /// `virtual_sequence_remaining` will be Some(0); if this is the penultimate instruction
    /// in the sequence, `virtual_sequence_remaining` will be Some(1); etc.
    pub virtual_sequence_remaining: Option<usize>,
}

impl<const WORD_SIZE: usize> RISCVInstruction for VirtualShiftRightBitmaskI<WORD_SIZE> {
    const MASK: u32 = 0; // Virtual
    const MATCH: u32 = 0; // Virtual

    type Format = FormatJ;
    type RAMAccess = ();

    fn operands(&self) -> &Self::Format {
        &self.operands
    }

    fn new(_: u32, _: u64) -> Self {
        unimplemented!("virtual instruction")
    }

    fn execute(&self, cpu: &mut Cpu, _: &mut Self::RAMAccess) {
        match cpu.xlen {
            Xlen::Bit32 => {
                let shift = self.operands.imm as u64 % 32;
                let ones = (1u64 << (32 - shift)) - 1;
                cpu.x[self.operands.rd] = (ones << shift) as i64;
            }
            Xlen::Bit64 => {
                let shift = self.operands.imm as u64 % 64;
                let ones = (1u128 << (64 - shift)) - 1;
                cpu.x[self.operands.rd] = (ones << shift) as i64;
            }
        }
    }
}
