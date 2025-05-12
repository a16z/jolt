use rand::{rngs::StdRng, RngCore};
use serde::{Deserialize, Serialize};

use crate::{emulator::cpu::Cpu, instruction::format::format_virtual_i::FormatVirtualI};

use super::{format::InstructionFormat, RISCVInstruction, RISCVTrace};

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct VirtualSRAI {
    pub address: u64,
    pub operands: FormatVirtualI,
    /// If this instruction is part of a "virtual sequence" (see Section 6.2 of the
    /// Jolt paper), then this contains the number of virtual instructions after this
    /// one in the sequence. I.e. if this is the last instruction in the sequence,
    /// `virtual_sequence_remaining` will be Some(0); if this is the penultimate instruction
    /// in the sequence, `virtual_sequence_remaining` will be Some(1); etc.
    pub virtual_sequence_remaining: Option<usize>,
}

impl RISCVInstruction for VirtualSRAI {
    const MASK: u32 = 0; // Virtual
    const MATCH: u32 = 0; // Virtual

    type Format = FormatVirtualI;
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
            operands: FormatVirtualI::random(rng),
            virtual_sequence_remaining: None,
        }
    }

    fn execute(&self, cpu: &mut Cpu, _: &mut Self::RAMAccess) {
        let shift = self.operands.imm.trailing_zeros();
        cpu.x[self.operands.rd] = cpu.sign_extend(cpu.x[self.operands.rs1].wrapping_shr(shift));
    }
}

impl RISCVTrace for VirtualSRAI {}
