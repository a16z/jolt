use serde::{Deserialize, Serialize};

use crate::emulator::cpu::Cpu;

use super::andi::ANDI;
use super::format::FormatR;
use super::lw::LW;
use super::sll::SLL;
use super::slli::SLLI;
use super::srai::SRAI;
use super::virtual_assert_halfword_alignment::{
    HalfwordAlignFormat, VirtualAssertHalfwordAlignment,
};
use super::xori::XORI;
use super::{addi::ADDI, RV32IMInstruction};
use super::{RAMRead, VirtualInstructionSequence};
use common::constants::virtual_register_index;

use super::{
    format::{FormatI, InstructionFormat},
    RISCVInstruction, RISCVTrace,
};

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct LH {
    pub address: u64,
    pub operands: FormatI,
    /// If this instruction is part of a "virtual sequence" (see Section 6.2 of the
    /// Jolt paper), then this contains the number of virtual instructions after this
    /// one in the sequence. I.e. if this is the last instruction in the sequence,
    /// `virtual_sequence_remaining` will be Some(0); if this is the penultimate instruction
    /// in the sequence, `virtual_sequence_remaining` will be Some(1); etc.
    pub virtual_sequence_remaining: Option<usize>,
}

impl RISCVInstruction for LH {
    const MASK: u32 = 0x0000707f;
    const MATCH: u32 = 0x00001003;

    type Format = FormatI;
    type RAMAccess = RAMRead;

    fn operands(&self) -> &Self::Format {
        &self.operands
    }

    fn new(word: u32, address: u64, validate: bool) -> Self {
        if validate {
            assert_eq!(word & Self::MASK, Self::MATCH);
        }

        Self {
            address,
            operands: FormatI::parse(word),
            virtual_sequence_remaining: None,
        }
    }

    fn execute(&self, cpu: &mut Cpu, ram_access: &mut Self::RAMAccess) {
        cpu.x[self.operands.rd] = match cpu
            .mmu
            .load_halfword(cpu.x[self.operands.rs1].wrapping_add(self.operands.imm) as u64)
        {
            Ok((halfword, memory_read)) => {
                *ram_access = memory_read;
                halfword as i16 as i64
            }
            Err(_) => panic!("MMU load error"),
        };
    }
}

impl RISCVTrace for LH {
    fn trace(&self, cpu: &mut Cpu) {
        let virtual_sequence = self.virtual_sequence();
        for instr in virtual_sequence {
            instr.trace(cpu);
        }
    }
}

impl VirtualInstructionSequence for LH {
    fn virtual_sequence(&self) -> Vec<RV32IMInstruction> {
        // Virtual registers used in sequence
        let v_address = virtual_register_index(0) as usize;
        let v_word_address = virtual_register_index(1) as usize;
        let v_word = virtual_register_index(2) as usize;
        let v_shift = virtual_register_index(3) as usize;

        let mut sequence = vec![];

        let alignment_check = VirtualAssertHalfwordAlignment {
            address: self.address,
            operands: HalfwordAlignFormat {
                rs1: self.operands.rs1,
                imm: self.operands.imm,
            },
            virtual_sequence_remaining: Some(10),
        };
        sequence.push(alignment_check.into());

        let add = ADDI {
            address: self.address,
            operands: FormatI {
                rd: v_address,
                rs1: self.operands.rs1,
                imm: self.operands.imm,
            },
            virtual_sequence_remaining: Some(9),
        };
        sequence.push(add.into());

        let andi = ANDI {
            address: self.address,
            operands: FormatI {
                rd: v_word_address,
                rs1: v_address,
                imm: -4,
            },
            virtual_sequence_remaining: Some(8),
        };
        sequence.push(andi.into());

        let lw = LW {
            address: self.address,
            operands: FormatI {
                rd: v_word,
                rs1: v_word_address,
                imm: 0,
            },
            virtual_sequence_remaining: Some(7),
        };
        sequence.push(lw.into());

        let xori = XORI {
            address: self.address,
            operands: FormatI {
                rd: v_shift,
                rs1: v_address,
                imm: 2,
            },
            virtual_sequence_remaining: Some(6),
        };
        sequence.push(xori.into());

        let slli = SLLI {
            address: self.address,
            operands: FormatI {
                rd: v_shift,
                rs1: v_shift,
                imm: 3,
            },
            virtual_sequence_remaining: Some(5),
        };
        sequence.extend(slli.virtual_sequence().into_iter());

        let sll = SLL {
            address: self.address,
            operands: FormatR {
                rd: self.operands.rd,
                rs1: v_word,
                rs2: v_shift,
            },
            virtual_sequence_remaining: Some(3),
        };
        sequence.extend(sll.virtual_sequence().into_iter());

        let srai = SRAI {
            address: self.address,
            operands: FormatI {
                rd: self.operands.rd,
                rs1: self.operands.rd,
                imm: 16,
            },
            virtual_sequence_remaining: Some(1),
        };
        sequence.extend(srai.virtual_sequence().into_iter());

        sequence
    }
}
