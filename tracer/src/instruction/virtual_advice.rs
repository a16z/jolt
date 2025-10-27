use serde::{Deserialize, Serialize};

use super::{format::format_j::FormatJ, RISCVInstruction, RISCVTrace};
use crate::{emulator::cpu::Cpu, instruction::NormalizedInstruction};

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
    pub is_first_in_sequence: bool,
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

    #[cfg(any(feature = "test-utils", test))]
    fn random(rng: &mut rand::rngs::StdRng) -> Self {
        use rand::RngCore;

        use crate::instruction::format::InstructionFormat;
        Self {
            address: rng.next_u64(),
            operands: FormatJ::random(rng),
            advice: rng.next_u64(),
            virtual_sequence_remaining: None,
            is_first_in_sequence: false,
            is_compressed: false,
        }
    }

    fn execute(&self, cpu: &mut Cpu, _: &mut Self::RAMAccess) {
        cpu.x[self.operands.rd as usize] = self.advice as i64;
    }
}

impl From<NormalizedInstruction> for VirtualAdvice {
    fn from(ni: NormalizedInstruction) -> Self {
        Self {
            address: ni.address as u64,
            operands: ni.operands.into(),
            advice: 0,
            virtual_sequence_remaining: ni.virtual_sequence_remaining,
            is_first_in_sequence: ni.is_first_in_sequence,
            is_compressed: ni.is_compressed,
        }
    }
}

impl From<VirtualAdvice> for NormalizedInstruction {
    fn from(val: VirtualAdvice) -> Self {
        NormalizedInstruction {
            address: val.address as usize,
            operands: val.operands.into(),
            is_compressed: val.is_compressed,
            is_first_in_sequence: val.is_first_in_sequence,
            virtual_sequence_remaining: val.virtual_sequence_remaining,
        }
    }
}

impl RISCVTrace for VirtualAdvice {}
