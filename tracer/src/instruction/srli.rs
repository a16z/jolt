use serde::{Deserialize, Serialize};

use crate::{
    emulator::cpu::{Cpu, Xlen},
    instruction::{format::format_virtual_i::FormatVirtualI, virtual_srli::VirtualSRLI},
};

use super::{
    format::{format_i::FormatI, InstructionFormat},
    RISCVInstruction, RISCVTrace, RV32IMInstruction, VirtualInstructionSequence,
};

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct SRLI {
    pub address: u64,
    pub operands: FormatI,
    /// If this instruction is part of a "virtual sequence" (see Section 6.2 of the
    /// Jolt paper), then this contains the number of virtual instructions after this
    /// one in the sequence. I.e. if this is the last instruction in the sequence,
    /// `virtual_sequence_remaining` will be Some(0); if this is the penultimate instruction
    /// in the sequence, `virtual_sequence_remaining` will be Some(1); etc.
    pub virtual_sequence_remaining: Option<usize>,
}

impl RISCVInstruction for SRLI {
    const MASK: u32 = 0xfc00707f;
    const MATCH: u32 = 0x00005013;

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
        let mask = match cpu.xlen {
            Xlen::Bit32 => 0x1f,
            Xlen::Bit64 => 0x3f,
        };
        cpu.x[self.operands.rd] = cpu.sign_extend(
            cpu.unsigned_data(cpu.x[self.operands.rs1])
                .wrapping_shr(self.operands.imm as u32 & mask) as i64,
        );
    }
}

impl RISCVTrace for SRLI {
    fn trace(&self, cpu: &mut Cpu) {
        let virtual_sequence = self.virtual_sequence();
        for instr in virtual_sequence {
            instr.trace(cpu);
        }
    }
}

impl VirtualInstructionSequence for SRLI {
    fn virtual_sequence(&self) -> Vec<RV32IMInstruction> {
        let virtual_sequence_remaining = self.virtual_sequence_remaining.unwrap_or(0);
        let mut sequence = vec![];

        // TODO: this only works for Xlen = 32
        let shift = self.operands.imm % 32;
        let ones = (1u64 << (32 - shift)) - 1;
        let bitmask = ones << shift;

        let srl = VirtualSRLI {
            address: self.address,
            operands: FormatVirtualI {
                rd: self.operands.rd,
                rs1: self.operands.rs1,
                imm: bitmask,
            },
            virtual_sequence_remaining: Some(virtual_sequence_remaining),
        };
        sequence.push(srl.into());

        sequence
    }
}
