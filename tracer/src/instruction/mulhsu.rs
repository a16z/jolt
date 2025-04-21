use common::constants::virtual_register_index;
use serde::{Deserialize, Serialize};

use crate::emulator::cpu::{Cpu, Xlen};

use super::{
    add::ADD,
    format::{FormatI, FormatR, InstructionFormat},
    mulhu::MULHU,
    virtual_movsign::VirtualMovsign,
    RISCVInstruction, RISCVTrace, RV32IMInstruction, VirtualInstructionSequence,
};

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct MULHSU {
    pub address: u64,
    pub operands: FormatR,
    /// If this instruction is part of a "virtual sequence" (see Section 6.2 of the
    /// Jolt paper), then this contains the number of virtual instructions after this
    /// one in the sequence. I.e. if this is the last instruction in the sequence,
    /// `virtual_sequence_remaining` will be Some(0); if this is the penultimate instruction
    /// in the sequence, `virtual_sequence_remaining` will be Some(1); etc.
    pub virtual_sequence_remaining: Option<usize>,
}

impl RISCVInstruction for MULHSU {
    const MASK: u32 = 0xfe00707f;
    const MATCH: u32 = 0x02002033;

    type Format = FormatR;
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
            operands: FormatR::parse(word),
            virtual_sequence_remaining: None,
        }
    }

    fn execute(&self, cpu: &mut Cpu, _: &mut Self::RAMAccess) {
        cpu.x[self.operands.rd] = match cpu.xlen {
            Xlen::Bit32 => cpu.sign_extend(
                ((cpu.x[self.operands.rs1] as i64)
                    .wrapping_mul(cpu.x[self.operands.rs2] as u32 as i64)
                    >> 32) as i64,
            ),
            Xlen::Bit64 => {
                ((cpu.x[self.operands.rs1] as u128)
                    .wrapping_mul(cpu.x[self.operands.rs2] as u64 as u128)
                    >> 64) as i64
            }
        };
    }
}

impl RISCVTrace for MULHSU {
    fn trace(&self, cpu: &mut Cpu) {
        let virtual_sequence = self.virtual_sequence();
        for instr in virtual_sequence {
            instr.trace(cpu);
        }
    }
}

impl VirtualInstructionSequence for MULHSU {
    fn virtual_sequence(&self) -> Vec<RV32IMInstruction> {
        // Virtual registers used in sequence
        let v_sx = virtual_register_index(0) as usize;
        let v_1 = virtual_register_index(1) as usize;
        let v_2 = virtual_register_index(2) as usize;

        let mut sequence = vec![];

        let movsign = VirtualMovsign {
            address: self.address,
            operands: FormatI {
                rd: v_sx,
                rs1: self.operands.rs1,
                imm: 0,
            },
            virtual_sequence_remaining: Some(3),
        };
        sequence.push(movsign.into());

        let mulhu = MULHU {
            address: self.address,
            operands: FormatR {
                rd: v_1,
                rs1: self.operands.rs1,
                rs2: self.operands.rs2,
            },
            virtual_sequence_remaining: Some(2),
        };
        sequence.push(mulhu.into());

        let mulu = MULHU {
            address: self.address,
            operands: FormatR {
                rd: v_2,
                rs1: v_sx,
                rs2: self.operands.rs2,
            },
            virtual_sequence_remaining: Some(1),
        };
        sequence.push(mulu.into());

        let add = ADD {
            address: self.address,
            operands: FormatR {
                rd: self.operands.rd,
                rs1: v_1,
                rs2: v_2,
            },
            virtual_sequence_remaining: Some(0),
        };
        sequence.push(add.into());

        sequence
    }
}
