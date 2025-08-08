use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
    instruction::format::format_load::FormatLoad,
};

use super::addi::ADDI;
use super::and::AND;
use super::andi::ANDI;
use super::format::format_i::FormatI;
use super::format::format_r::FormatR;
use super::format::format_u::FormatU;
use super::format::format_virtual_halfword_alignment::HalfwordAlignFormat;
use super::ld::LD;
use super::lui::LUI;
use super::sd::SD;
use super::sll::SLL;
use super::slli::SLLI;
use super::virtual_assert_halfword_alignment::VirtualAssertHalfwordAlignment;
use super::virtual_lw::VirtualLW;
use super::virtual_sw::VirtualSW;
use super::xor::XOR;
use super::RAMWrite;
use super::RV32IMInstruction;
use common::constants::virtual_register_index;

use super::{
    format::{format_s::FormatS, InstructionFormat},
    RISCVInstruction, RISCVTrace, RV32IMCycle,
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
                cpu.x[self.operands.rs1 as usize].wrapping_add(self.operands.imm) as u64,
                cpu.x[self.operands.rs2 as usize] as u16,
            )
            .ok()
            .unwrap();
    }
}

impl RISCVTrace for SH {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<RV32IMCycle>>) {
        let virtual_sequence = self.virtual_sequence(cpu.xlen);
        let mut trace = trace;
        for instr in virtual_sequence {
            // In each iteration, create a new Option containing a re-borrowed reference
            instr.trace(cpu, trace.as_deref_mut());
        }
    }

    fn virtual_sequence(&self, xlen: Xlen) -> Vec<RV32IMInstruction> {
        match xlen {
            Xlen::Bit32 => self.virtual_sequence_32(),
            Xlen::Bit64 => self.virtual_sequence_64(),
        }
    }
}

impl SH {
    fn virtual_sequence_32(&self) -> Vec<RV32IMInstruction> {
        // Virtual registers used in sequence
        let v_address = virtual_register_index(0);
        let v_word_address = virtual_register_index(1);
        let v_word = virtual_register_index(2);
        let v_shift = virtual_register_index(3);
        let v_mask = virtual_register_index(4);
        let v_halfword = virtual_register_index(5);

        let mut sequence = vec![];

        let align_check = VirtualAssertHalfwordAlignment {
            address: self.address,
            operands: HalfwordAlignFormat {
                rs1: self.operands.rs1,
                imm: self.operands.imm,
            },
            virtual_sequence_remaining: Some(13),
            is_compressed: self.is_compressed,
        };
        sequence.push(align_check.into());

        let add = ADDI {
            address: self.address,
            operands: FormatI {
                rd: v_address,
                rs1: self.operands.rs1,
                imm: self.operands.imm as u64,
            },
            virtual_sequence_remaining: Some(12),
            is_compressed: self.is_compressed,
        };
        sequence.push(add.into());

        let andi = ANDI {
            address: self.address,
            operands: FormatI {
                rd: v_word_address,
                rs1: v_address,
                imm: -4i64 as u64,
            },
            virtual_sequence_remaining: Some(11),
            is_compressed: self.is_compressed,
        };
        sequence.push(andi.into());

        let lw = VirtualLW {
            address: self.address,
            operands: FormatI {
                rd: v_word,
                rs1: v_word_address,
                imm: 0,
            },
            virtual_sequence_remaining: Some(10),
            is_compressed: self.is_compressed,
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
            is_compressed: self.is_compressed,
        };
        sequence.extend(slli.virtual_sequence(Xlen::Bit32));

        let lui = LUI {
            address: self.address,
            operands: FormatU {
                rd: v_mask,
                imm: 0xffff,
            },
            virtual_sequence_remaining: Some(8),
            is_compressed: self.is_compressed,
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
            is_compressed: self.is_compressed,
        };
        sequence.extend(sll_mask.virtual_sequence(Xlen::Bit32));

        let sll_value = SLL {
            address: self.address,
            operands: FormatR {
                rd: v_halfword,
                rs1: self.operands.rs2,
                rs2: v_shift,
            },
            virtual_sequence_remaining: Some(5),
            is_compressed: self.is_compressed,
        };
        sequence.extend(sll_value.virtual_sequence(Xlen::Bit32));

        let xor = XOR {
            address: self.address,
            operands: FormatR {
                rd: v_halfword,
                rs1: v_word,
                rs2: v_halfword,
            },
            virtual_sequence_remaining: Some(3),
            is_compressed: self.is_compressed,
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
            is_compressed: self.is_compressed,
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
            is_compressed: self.is_compressed,
        };
        sequence.push(xor_final.into());

        let sw = VirtualSW {
            address: self.address,
            operands: FormatS {
                rs1: v_word_address,
                rs2: v_word,
                imm: 0,
            },
            virtual_sequence_remaining: Some(0),
            is_compressed: self.is_compressed,
        };
        sequence.push(sw.into());

        sequence
    }

    fn virtual_sequence_64(&self) -> Vec<RV32IMInstruction> {
        // Virtual registers used in sequence
        let v_address = virtual_register_index(6);
        let v_dword_address = virtual_register_index(7);
        let v_dword = virtual_register_index(8);
        let v_shift = virtual_register_index(9);
        let v_mask = virtual_register_index(10);
        let v_halfword = virtual_register_index(11);

        let mut sequence = vec![];

        let align_check = VirtualAssertHalfwordAlignment {
            address: self.address,
            operands: HalfwordAlignFormat {
                rs1: self.operands.rs1,
                imm: self.operands.imm,
            },
            virtual_sequence_remaining: Some(13),
            is_compressed: self.is_compressed,
        };
        sequence.push(align_check.into());

        let add = ADDI {
            address: self.address,
            operands: FormatI {
                rd: v_address,
                rs1: self.operands.rs1,
                imm: self.operands.imm as u64,
            },
            virtual_sequence_remaining: Some(12),
            is_compressed: self.is_compressed,
        };
        sequence.push(add.into());

        let andi = ANDI {
            address: self.address,
            operands: FormatI {
                rd: v_dword_address,
                rs1: v_address,
                imm: -8i64 as u64,
            },
            virtual_sequence_remaining: Some(11),
            is_compressed: self.is_compressed,
        };
        sequence.push(andi.into());

        let ld = LD {
            address: self.address,
            operands: FormatLoad {
                rd: v_dword,
                rs1: v_dword_address,
                imm: 0,
            },
            virtual_sequence_remaining: Some(10),
            is_compressed: self.is_compressed,
        };
        sequence.push(ld.into());

        let slli = SLLI {
            address: self.address,
            operands: FormatI {
                rd: v_shift,
                rs1: v_address,
                imm: 3,
            },
            virtual_sequence_remaining: Some(9),
            is_compressed: self.is_compressed,
        };
        sequence.extend(slli.virtual_sequence(Xlen::Bit64));

        let lui = LUI {
            address: self.address,
            operands: FormatU {
                rd: v_mask,
                imm: 0xffff,
            },
            virtual_sequence_remaining: Some(8),
            is_compressed: self.is_compressed,
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
            is_compressed: self.is_compressed,
        };
        sequence.extend(sll_mask.virtual_sequence(Xlen::Bit64));

        let sll_value = SLL {
            address: self.address,
            operands: FormatR {
                rd: v_halfword,
                rs1: self.operands.rs2,
                rs2: v_shift,
            },
            virtual_sequence_remaining: Some(5),
            is_compressed: self.is_compressed,
        };
        sequence.extend(sll_value.virtual_sequence(Xlen::Bit64));

        let xor = XOR {
            address: self.address,
            operands: FormatR {
                rd: v_halfword,
                rs1: v_dword,
                rs2: v_halfword,
            },
            virtual_sequence_remaining: Some(3),
            is_compressed: self.is_compressed,
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
            is_compressed: self.is_compressed,
        };
        sequence.push(and.into());

        let xor_final = XOR {
            address: self.address,
            operands: FormatR {
                rd: v_dword,
                rs1: v_dword,
                rs2: v_halfword,
            },
            virtual_sequence_remaining: Some(1),
            is_compressed: self.is_compressed,
        };
        sequence.push(xor_final.into());

        let sd = SD {
            address: self.address,
            operands: FormatS {
                rs1: v_dword_address,
                rs2: v_dword,
                imm: 0,
            },
            virtual_sequence_remaining: Some(0),
            is_compressed: self.is_compressed,
        };
        sequence.push(sd.into());

        sequence
    }
}
