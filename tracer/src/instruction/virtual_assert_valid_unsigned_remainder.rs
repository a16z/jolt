use serde::{Deserialize, Serialize};

use crate::emulator::cpu::{Cpu, Xlen};

use super::{format::FormatB, RISCVInstruction, RISCVTrace};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct VirtualAssertValidUnsignedRemainder {
    pub address: u64,
    pub operands: FormatB,
    /// If this instruction is part of a "virtual sequence" (see Section 6.2 of the
    /// Jolt paper), then this contains the number of virtual instructions after this
    /// one in the sequence. I.e. if this is the last instruction in the sequence,
    /// `virtual_sequence_remaining` will be Some(0); if this is the penultimate instruction
    /// in the sequence, `virtual_sequence_remaining` will be Some(1); etc.
    pub virtual_sequence_remaining: Option<usize>,
}

impl RISCVInstruction for VirtualAssertValidUnsignedRemainder {
    const MASK: u32 = 0; // Virtual
    const MATCH: u32 = 0; // Virtual

    type Format = FormatB;
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
                let remainder = cpu.x[self.operands.rs1] as i32 as u32;
                let divisor = cpu.x[self.operands.rs2] as i32 as u32;
                assert!(divisor == 0 || remainder < divisor);
            }
            Xlen::Bit64 => {
                let remainder = cpu.x[self.operands.rs1] as u64;
                let divisor = cpu.x[self.operands.rs2] as u64;
                assert!(divisor == 0 || remainder < divisor);
            }
        }
    }
}

impl RISCVTrace for VirtualAssertValidUnsignedRemainder {}
