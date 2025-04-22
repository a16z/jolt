use rand::{rngs::StdRng, RngCore};
use serde::{Deserialize, Serialize};

use crate::emulator::cpu::{Cpu, Xlen};

use super::{
    format::{format_b::FormatB, InstructionFormat},
    RISCVInstruction, RISCVTrace,
};

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct VirtualAssertValidSignedRemainder {
    pub address: u64,
    pub operands: FormatB,
    /// If this instruction is part of a "virtual sequence" (see Section 6.2 of the
    /// Jolt paper), then this contains the number of virtual instructions after this
    /// one in the sequence. I.e. if this is the last instruction in the sequence,
    /// `virtual_sequence_remaining` will be Some(0); if this is the penultimate instruction
    /// in the sequence, `virtual_sequence_remaining` will be Some(1); etc.
    pub virtual_sequence_remaining: Option<usize>,
}

impl RISCVInstruction for VirtualAssertValidSignedRemainder {
    const MASK: u32 = 0; // Virtual
    const MATCH: u32 = 0; // Virtual

    type Format = FormatB;
    type RAMAccess = ();

    fn operands(&self) -> &Self::Format {
        &self.operands
    }

    fn new(_: u32, _: u64, _: bool) -> Self {
        unimplemented!("virtual instruction")
    }

    fn random(rng: &mut StdRng) -> Self {
        Self {
            address: rng.next_u64(),
            operands: FormatB::random(rng),
            virtual_sequence_remaining: None,
        }
    }

    fn execute(&self, cpu: &mut Cpu, _: &mut Self::RAMAccess) {
        match cpu.xlen {
            Xlen::Bit32 => {
                let remainder = cpu.x[self.operands.rs1] as i32;
                let divisor = cpu.x[self.operands.rs2] as i32;
                if remainder != 0 && divisor != 0 {
                    let remainder_sign = remainder >> 31;
                    let divisor_sign = divisor >> 31;
                    assert!(
                        remainder.unsigned_abs() < divisor.unsigned_abs()
                            && remainder_sign == divisor_sign
                    );
                }
            }
            Xlen::Bit64 => {
                let remainder = cpu.x[self.operands.rs1];
                let divisor = cpu.x[self.operands.rs2];
                if remainder != 0 && divisor != 0 {
                    let remainder_sign = remainder >> 63;
                    let divisor_sign = divisor >> 63;
                    assert!(
                        remainder.unsigned_abs() < divisor.unsigned_abs()
                            && remainder_sign == divisor_sign
                    );
                }
            }
        }
    }
}

impl RISCVTrace for VirtualAssertValidSignedRemainder {}
