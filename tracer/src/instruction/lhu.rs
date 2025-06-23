use serde::{Deserialize, Serialize};

use crate::declare_riscv_instr;
use crate::emulator::cpu::Cpu;

use super::andi::ANDI;
use super::format::format_load::FormatLoad;
use super::format::format_r::FormatR;
use super::format::format_virtual_halfword_alignment::HalfwordAlignFormat;
use super::lw::LW;
use super::sll::SLL;
use super::slli::SLLI;
use super::srli::SRLI;
use super::virtual_assert_halfword_alignment::VirtualAssertHalfwordAlignment;
use super::xori::XORI;
use super::{addi::ADDI, RV32IMInstruction};
use super::{RAMRead, VirtualInstructionSequence};
use common::constants::virtual_register_index;

use super::{
    format::{format_i::FormatI, InstructionFormat},
    RISCVInstruction, RISCVTrace,
};

declare_riscv_instr!(
    name   = LHU,
    mask   = 0x0000707f,
    match  = 0x00005003,
    format = FormatLoad,
    ram    = RAMRead
);

impl LHU {
    fn exec(&self, cpu: &mut Cpu, ram_access: &mut <LHU as RISCVInstruction>::RAMAccess) {
        cpu.x[self.operands.rd] = match cpu
            .mmu
            .load_halfword(cpu.x[self.operands.rs1].wrapping_add(self.operands.imm) as u64)
        {
            Ok((halfword, memory_read)) => {
                *ram_access = memory_read;
                halfword as i64
            }
            Err(_) => panic!("MMU load error"),
        };
    }
}

impl RISCVTrace for LHU {
    fn trace(&self, cpu: &mut Cpu) {
        let virtual_sequence = self.virtual_sequence();
        for instr in virtual_sequence {
            instr.trace(cpu);
        }
    }
}

impl VirtualInstructionSequence for LHU {
    fn virtual_sequence(&self) -> Vec<RV32IMInstruction> {
        // Virtual registers used in sequence
        let v_address = virtual_register_index(0) as usize;
        let v_word_address = virtual_register_index(1) as usize;
        let v_word = virtual_register_index(2) as usize;
        let v_shift = virtual_register_index(3) as usize;

        let mut sequence = vec![];

        let assert_alignment = VirtualAssertHalfwordAlignment {
            address: self.address,
            operands: HalfwordAlignFormat {
                rs1: self.operands.rs1,
                imm: self.operands.imm,
            },
            virtual_sequence_remaining: Some(8),
        };
        sequence.push(assert_alignment.into());

        let add = ADDI {
            address: self.address,
            operands: FormatI {
                rd: v_address,
                rs1: self.operands.rs1,
                imm: self.operands.imm as u32 as u64, // TODO(moodlezoup): this only works for Xlen = 32
            },
            virtual_sequence_remaining: Some(7),
        };
        sequence.push(add.into());

        let andi = ANDI {
            address: self.address,
            operands: FormatI {
                rd: v_word_address,
                rs1: v_address,
                imm: -4i64 as u32 as u64, // TODO(moodlezoup): this only works for Xlen = 32
            },
            virtual_sequence_remaining: Some(6),
        };
        sequence.push(andi.into());

        let lw = LW {
            address: self.address,
            operands: FormatLoad {
                rd: v_word,
                rs1: v_word_address,
                imm: 0,
            },
            virtual_sequence_remaining: Some(5),
        };
        sequence.push(lw.into());

        let xori = XORI {
            address: self.address,
            operands: FormatI {
                rd: v_shift,
                rs1: v_address,
                imm: 2,
            },
            virtual_sequence_remaining: Some(4),
        };
        sequence.push(xori.into());

        let slli = SLLI {
            address: self.address,
            operands: FormatI {
                rd: v_shift,
                rs1: v_shift,
                imm: 3,
            },
            virtual_sequence_remaining: Some(3),
        };
        sequence.extend(slli.virtual_sequence());

        let sll = SLL {
            address: self.address,
            operands: FormatR {
                rd: self.operands.rd,
                rs1: v_word,
                rs2: v_shift,
            },
            virtual_sequence_remaining: Some(2),
        };
        sequence.extend(sll.virtual_sequence());

        let srli = SRLI {
            address: self.address,
            operands: FormatI {
                rd: self.operands.rd,
                rs1: self.operands.rd,
                imm: 16,
            },
            virtual_sequence_remaining: Some(0),
        };
        sequence.extend(srli.virtual_sequence());

        sequence
    }
}
