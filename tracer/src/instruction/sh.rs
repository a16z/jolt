use serde::{Deserialize, Serialize};

use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::addi::ADDI;
use super::and::AND;
use super::andi::ANDI;
use super::format::format_i::FormatI;
use super::format::format_load::FormatLoad;
use super::format::format_r::FormatR;
use super::format::format_u::FormatU;
use super::format::format_virtual_halfword_alignment::HalfwordAlignFormat;
use super::lui::LUI;
use super::lw::LW;
use super::sll::SLL;
use super::slli::SLLI;
use super::sw::SW;
use super::virtual_assert_halfword_alignment::VirtualAssertHalfwordAlignment;
use super::xor::XOR;
use super::RAMWrite;
use super::RV32IMInstruction;
use super::VirtualInstructionSequence;
use common::constants::virtual_register_index;

use super::{
    format::{format_s::FormatS, InstructionFormat},
    RISCVInstruction, RISCVTrace,
};

declare_riscv_instr!(
    name   = SH,
    mask   = 0x0000707f,
    match  = 0x00001023,
    format = FormatS,
    ram    = RAMWrite
);

impl SH {
    fn exec(&self, cpu: &mut Cpu, ram_access: &mut <SH as RISCVInstruction>::RAMAccess) {
        *ram_access = cpu
            .mmu
            .store_halfword(
                cpu.x[self.operands.rs1].wrapping_add(self.operands.imm) as u64,
                cpu.x[self.operands.rs2] as u16,
            )
            .ok()
            .unwrap();
    }
}

impl RISCVTrace for SH {
    fn trace(&self, cpu: &mut Cpu) {
        let virtual_sequence = self.virtual_sequence();
        for instr in virtual_sequence {
            instr.trace(cpu);
        }
    }
}

impl VirtualInstructionSequence for SH {
    fn virtual_sequence(&self) -> Vec<RV32IMInstruction> {
        // Virtual registers used in sequence
        let v_address = virtual_register_index(0) as usize;
        let v_word_address = virtual_register_index(1) as usize;
        let v_word = virtual_register_index(2) as usize;
        let v_shift = virtual_register_index(3) as usize;
        let v_mask = virtual_register_index(4) as usize;
        let v_halfword = virtual_register_index(5) as usize;

        let mut sequence = vec![];

        let align_check = VirtualAssertHalfwordAlignment {
            address: self.address,
            operands: HalfwordAlignFormat {
                rs1: self.operands.rs1,
                imm: self.operands.imm,
            },
            virtual_sequence_remaining: Some(13),
        };
        sequence.push(align_check.into());

        let add = ADDI {
            address: self.address,
            operands: FormatI {
                rd: v_address,
                rs1: self.operands.rs1,
                imm: self.operands.imm as u32 as u64, // TODO(moodlezoup): this only works for Xlen = 32
            },
            virtual_sequence_remaining: Some(12),
        };
        sequence.push(add.into());

        let andi = ANDI {
            address: self.address,
            operands: FormatI {
                rd: v_word_address,
                rs1: v_address,
                imm: -4i64 as u32 as u64, // TODO(moodlezoup): this only works for Xlen = 32
            },
            virtual_sequence_remaining: Some(11),
        };
        sequence.push(andi.into());

        let lw = LW {
            address: self.address,
            operands: FormatLoad {
                rd: v_word,
                rs1: v_word_address,
                imm: 0,
            },
            virtual_sequence_remaining: Some(10),
        };
        sequence.push(lw.into());

        let slli = SLLI {
            address: self.address,
            operands: FormatI {
                rd: v_shift,
                rs1: v_address,
                imm: 3,
            },
            virtual_sequence_remaining: Some(9),
        };
        sequence.extend(slli.virtual_sequence());

        let lui = LUI {
            address: self.address,
            operands: FormatU {
                rd: v_mask,
                imm: 0xffff,
            },
            virtual_sequence_remaining: Some(8),
        };
        sequence.push(lui.into());

        let sll_mask = SLL {
            address: self.address,
            operands: FormatR {
                rd: v_mask,
                rs1: v_mask,
                rs2: v_shift,
            },
            virtual_sequence_remaining: Some(7),
        };
        sequence.extend(sll_mask.virtual_sequence());

        let sll_value = SLL {
            address: self.address,
            operands: FormatR {
                rd: v_halfword,
                rs1: self.operands.rs2,
                rs2: v_shift,
            },
            virtual_sequence_remaining: Some(5),
        };
        sequence.extend(sll_value.virtual_sequence());

        let xor = XOR {
            address: self.address,
            operands: FormatR {
                rd: v_halfword,
                rs1: v_word,
                rs2: v_halfword,
            },
            virtual_sequence_remaining: Some(3),
        };
        sequence.push(xor.into());

        let and = AND {
            address: self.address,
            operands: FormatR {
                rd: v_halfword,
                rs1: v_halfword,
                rs2: v_mask,
            },
            virtual_sequence_remaining: Some(2),
        };
        sequence.push(and.into());

        let xor_final = XOR {
            address: self.address,
            operands: FormatR {
                rd: v_word,
                rs1: v_word,
                rs2: v_halfword,
            },
            virtual_sequence_remaining: Some(1),
        };
        sequence.push(xor_final.into());

        let sw = SW {
            address: self.address,
            operands: FormatS {
                rs1: v_word_address,
                rs2: v_word,
                imm: 0,
            },
            virtual_sequence_remaining: Some(0),
        };
        sequence.push(sw.into());

        sequence
    }
}
