use rand::{rngs::StdRng, RngCore};
use serde::{Deserialize, Serialize};

use crate::emulator::cpu::Cpu;

use super::{
    format::{format_j::FormatJ, InstructionFormat, NormalizedOperands},
    RISCVInstruction, RISCVTrace,
};

// Special case for VirtualAdvice as it has an extra 'advice' field
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq)]
pub struct VirtualAdvice {
    pub address: u64,
    pub operands: FormatJ,
    /// If this instruction is part of a "virtual sequence" (see Section 6.2 of the
    /// Jolt paper), then this contains the number of virtual instructions after this
    /// one in the sequence. I.e. if this is the last instruction in the sequence,
    /// `virtual_sequence_remaining` will be Some(0); if this is the penultimate instruction
    /// in the sequence, `virtual_sequence_remaining` will be Some(1); etc.
    pub virtual_sequence_remaining: Option<u16>,
    pub advice: u64,
    pub is_compressed: bool,
}

impl RISCVInstruction for VirtualAdvice {
    const MASK: u32 = 0; // Virtual
    const MATCH: u32 = 0; // Virtual

    type Format = FormatJ;
    type RAMAccess = ();

    fn operands(&self) -> &Self::Format {
        &self.operands
    }

    fn new(_: u32, _: u64, _: bool, _: bool) -> Self {
        panic!("virtual instruction `VirtualAdvice` cannot be built from a machine word");
    }

    fn from_normalized(operands: NormalizedOperands, address: u64, is_compressed: bool) -> Self {
        Self {
            address,
            operands: FormatJ::from_normalized(operands),
            advice: 0,
            virtual_sequence_remaining: None,
            is_compressed,
        }
    }

    fn random(rng: &mut StdRng) -> Self {
        Self {
            address: rng.next_u64(),
            operands: FormatJ::random(rng),
            advice: rng.next_u64(),
            virtual_sequence_remaining: None,
            is_compressed: false,
        }
    }

    fn execute(&self, cpu: &mut Cpu, _: &mut Self::RAMAccess) {
        cpu.x[self.operands.rd as usize] = self.advice as i64;
    }
}

impl RISCVTrace for VirtualAdvice {}
