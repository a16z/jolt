use serde::{Deserialize, Serialize};

use crate::emulator::cpu::{Cpu, Xlen};

use super::{format::FormatI, RISCVInstruction};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct VirtualMovsign<const WORD_SIZE: usize> {
    pub address: u64,
    pub operands: FormatI,
    /// If this instruction is part of a "virtual sequence" (see Section 6.2 of the
    /// Jolt paper), then this contains the number of virtual instructions after this
    /// one in the sequence. I.e. if this is the last instruction in the sequence,
    /// `virtual_sequence_remaining` will be Some(0); if this is the penultimate instruction
    /// in the sequence, `virtual_sequence_remaining` will be Some(1); etc.
    pub virtual_sequence_remaining: Option<usize>,
}

// Constants for 32-bit and 64-bit word sizes
const ALL_ONES_32: u64 = 0xFFFF_FFFF;
const ALL_ONES_64: u64 = 0xFFFF_FFFF_FFFF_FFFF;
const SIGN_BIT_32: u64 = 0x8000_0000;
const SIGN_BIT_64: u64 = 0x8000_0000_0000_0000;

impl<const WORD_SIZE: usize> RISCVInstruction for VirtualMovsign<WORD_SIZE> {
    const MASK: u32 = 0; // Virtual
    const MATCH: u32 = 0; // Virtual

    type Format = FormatI;
    type RAMAccess = ();

    fn operands(&self) -> &Self::Format {
        &self.operands
    }

    fn new(_: u32, _: u64) -> Self {
        unimplemented!("virtual instruction")
    }

    fn execute(&self, cpu: &mut Cpu, _: &mut Self::RAMAccess) {
        let val = cpu.x[self.operands.rs1] as u64;
        cpu.x[self.operands.rd] = match cpu.xlen {
            Xlen::Bit32 => {
                if val & SIGN_BIT_32 != 0 {
                    // Should this be ALL_ONES_64?
                    ALL_ONES_32 as i64
                } else {
                    0
                }
            }
            Xlen::Bit64 => {
                if val & SIGN_BIT_64 != 0 {
                    ALL_ONES_64 as i64
                } else {
                    0
                }
            }
        };
        cpu.x[self.operands.rd] = cpu.x[self.operands.rs1];
    }
}
