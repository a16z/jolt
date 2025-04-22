use common::constants::virtual_register_index;
use serde::{Deserialize, Serialize};

use crate::emulator::cpu::{Cpu, Xlen};

use super::{
    format::{FormatI, FormatJ, FormatR, InstructionFormat},
    mul::MUL,
    virtual_pow2i::VirtualPow2I,
    RISCVInstruction, RISCVTrace, RV32IMInstruction, VirtualInstructionSequence,
};

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct SLLI {
    pub address: u64,
    pub operands: FormatI,
    /// If this instruction is part of a "virtual sequence" (see Section 6.2 of the
    /// Jolt paper), then this contains the number of virtual instructions after this
    /// one in the sequence. I.e. if this is the last instruction in the sequence,
    /// `virtual_sequence_remaining` will be Some(0); if this is the penultimate instruction
    /// in the sequence, `virtual_sequence_remaining` will be Some(1); etc.
    pub virtual_sequence_remaining: Option<usize>,
}

impl RISCVInstruction for SLLI {
    const MASK: u32 = 0xfc00707f;
    const MATCH: u32 = 0x00001013;

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
        cpu.x[self.operands.rd] =
            cpu.sign_extend(cpu.x[self.operands.rs1].wrapping_shl(self.operands.imm as u32 & mask));
    }
}

impl RISCVTrace for SLLI {
    fn trace(&self, cpu: &mut Cpu) {
        let virtual_sequence = self.virtual_sequence();
        for instr in virtual_sequence {
            instr.trace(cpu);
        }
    }
}

impl VirtualInstructionSequence for SLLI {
    fn virtual_sequence(&self) -> Vec<RV32IMInstruction> {
        // Virtual registers used in sequence
        let v_pow2 = virtual_register_index(6) as usize;

        let mut virtual_sequence_remaining = self.virtual_sequence_remaining.unwrap_or(1);
        let mut sequence = vec![];

        let pow2 = RV32IMInstruction::Pow2I(VirtualPow2I {
            address: self.address,
            operands: FormatJ {
                rd: v_pow2,
                imm: self.operands.imm,
            },
            virtual_sequence_remaining: Some(virtual_sequence_remaining),
        });
        sequence.push(pow2);
        virtual_sequence_remaining -= 1;

        let mul = RV32IMInstruction::MUL(MUL {
            address: self.address,
            operands: FormatR {
                rd: self.operands.rd,
                rs1: self.operands.rs1,
                rs2: v_pow2,
            },
            virtual_sequence_remaining: Some(virtual_sequence_remaining),
        });
        sequence.push(mul);

        sequence
    }
}
