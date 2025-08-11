use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
    instruction::{format::format_load::FormatLoad, ori::ORI, srli::SRLI},
};

use super::addi::ADDI;
use super::and::AND;
use super::andi::ANDI;
use super::format::format_i::FormatI;
use super::format::format_r::FormatR;
use super::format::format_virtual_halfword_alignment::HalfwordAlignFormat;
use super::ld::LD;
use super::sd::SD;
use super::sll::SLL;
use super::slli::SLLI;
use super::virtual_assert_word_alignment::VirtualAssertWordAlignment;
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
    name   = SW,
    mask   = 0x0000707f,
    match  = 0x00002023,
    format = FormatS,
    ram    = RAMWrite
);

impl SW {
    fn exec(&self, cpu: &mut Cpu, ram_access: &mut <SW as RISCVInstruction>::RAMAccess) {
        *ram_access = cpu
            .mmu
            .store_word(
                cpu.x[self.operands.rs1 as usize].wrapping_add(self.operands.imm) as u64,
                cpu.x[self.operands.rs2 as usize] as u32,
            )
            .ok()
            .unwrap();
    }
}

impl RISCVTrace for SW {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<RV32IMCycle>>) {
        let inline_sequence = self.inline_sequence(cpu.xlen);
        let mut trace = trace;
        for instr in inline_sequence {
            // In each iteration, create a new Option containing a re-borrowed reference
            instr.trace(cpu, trace.as_deref_mut());
        }
    }

    fn inline_sequence(&self, xlen: Xlen) -> Vec<RV32IMInstruction> {
        match xlen {
            Xlen::Bit32 => self.inline_sequence_32(),
            Xlen::Bit64 => self.inline_sequence_64(),
        }
    }
}

impl SW {
    fn inline_sequence_32(&self) -> Vec<RV32IMInstruction> {
        let mut sequence = vec![];
        let sw = VirtualSW {
            address: self.address,
            operands: FormatS {
                rs1: self.operands.rs1,
                rs2: self.operands.rs2,
                imm: self.operands.imm,
            },
            inline_sequence_remaining: Some(0),
            is_compressed: self.is_compressed,
        };
        sequence.push(sw.into());

        sequence
    }

    fn inline_sequence_64(&self) -> Vec<RV32IMInstruction> {
        // Virtual registers used in sequence
        // Use high-numbered virtual registers as scratch to avoid clobbering
        // low indices that inline builders frequently use for program values.
        let v_address = virtual_register_index(90);
        let v_dword_address = virtual_register_index(91);
        let v_dword = virtual_register_index(92);
        let v_shift = virtual_register_index(93);
        let v_mask = virtual_register_index(94);
        let v_word = virtual_register_index(95);

        let mut sequence = vec![];
        let mut inline_sequence_remaining = self.inline_sequence_remaining.unwrap_or(14);

        let align_check = VirtualAssertWordAlignment {
            address: self.address,
            operands: HalfwordAlignFormat {
                rs1: self.operands.rs1,
                imm: self.operands.imm,
            },
            inline_sequence_remaining: Some(inline_sequence_remaining),
            is_compressed: self.is_compressed,
        };
        sequence.push(align_check.into());
        inline_sequence_remaining -= 1;

        let add = ADDI {
            address: self.address,
            operands: FormatI {
                rd: v_address,
                rs1: self.operands.rs1,
                imm: self.operands.imm as u64,
            },
            inline_sequence_remaining: Some(inline_sequence_remaining),
            is_compressed: self.is_compressed,
        };
        sequence.push(add.into());
        inline_sequence_remaining -= 1;

        let andi = ANDI {
            address: self.address,
            operands: FormatI {
                rd: v_dword_address,
                rs1: v_address,
                imm: -8i64 as u64,
            },
            inline_sequence_remaining: Some(inline_sequence_remaining),
            is_compressed: self.is_compressed,
        };
        sequence.push(andi.into());
        inline_sequence_remaining -= 1;

        let ld = LD {
            address: self.address,
            operands: FormatLoad {
                rd: v_dword,
                rs1: v_dword_address,
                imm: 0,
            },
            inline_sequence_remaining: Some(inline_sequence_remaining),
            is_compressed: self.is_compressed,
        };
        sequence.push(ld.into());
        inline_sequence_remaining -= 1;

        let slli = SLLI {
            address: self.address,
            operands: FormatI {
                rd: v_shift,
                rs1: v_address,
                imm: 3,
            },
            inline_sequence_remaining: Some(inline_sequence_remaining),
            is_compressed: self.is_compressed,
        };
        let slli_sequence = slli.inline_sequence(Xlen::Bit64);
        let slli_sequence_len = slli_sequence.len();
        sequence.extend(slli_sequence);
        inline_sequence_remaining -= slli_sequence_len as u16;

        let ori = ORI {
            address: self.address,
            operands: FormatI {
                rd: v_mask,
                rs1: 0,
                imm: -1i64 as u64,
            },
            inline_sequence_remaining: Some(inline_sequence_remaining),
            is_compressed: self.is_compressed,
        };
        sequence.push(ori.into()); // v_mask gets 0xFFFFFFFF_FFFFFFFF
        inline_sequence_remaining -= 1;

        let srli = SRLI {
            address: self.address,
            operands: FormatI {
                rd: v_mask,
                rs1: v_mask,
                imm: 32, // Logical right shift by 32 bits
            },
            inline_sequence_remaining: Some(inline_sequence_remaining),
            is_compressed: self.is_compressed,
        };
        sequence.extend(srli.inline_sequence(Xlen::Bit64)); // v_mask gets 0x00000000_FFFFFFFF
        inline_sequence_remaining -= 1;

        let sll_mask = SLL {
            address: self.address,
            operands: FormatR {
                rd: v_mask,
                rs1: v_mask,
                rs2: v_shift,
            },
            inline_sequence_remaining: Some(inline_sequence_remaining),
            is_compressed: self.is_compressed,
        };
        let sll_mask_sequence = sll_mask.inline_sequence(Xlen::Bit64);
        let sll_mask_sequence_len = sll_mask_sequence.len();
        sequence.extend(sll_mask_sequence);
        inline_sequence_remaining -= sll_mask_sequence_len as u16;

        let sll_value = SLL {
            address: self.address,
            operands: FormatR {
                rd: v_word,
                rs1: self.operands.rs2,
                rs2: v_shift,
            },
            inline_sequence_remaining: Some(inline_sequence_remaining),
            is_compressed: self.is_compressed,
        };
        let sll_value_sequence = sll_value.inline_sequence(Xlen::Bit64);
        let sll_value_sequence_len = sll_value_sequence.len();
        sequence.extend(sll_value_sequence);
        inline_sequence_remaining -= sll_value_sequence_len as u16;

        let xor = XOR {
            address: self.address,
            operands: FormatR {
                rd: v_word,
                rs1: v_dword,
                rs2: v_word,
            },
            inline_sequence_remaining: Some(inline_sequence_remaining),
            is_compressed: self.is_compressed,
        };
        sequence.push(xor.into());
        inline_sequence_remaining -= 1;

        let and = AND {
            address: self.address,
            operands: FormatR {
                rd: v_word,
                rs1: v_word,
                rs2: v_mask,
            },
            inline_sequence_remaining: Some(inline_sequence_remaining),
            is_compressed: self.is_compressed,
        };
        sequence.push(and.into());
        inline_sequence_remaining -= 1;

        let xor_final = XOR {
            address: self.address,
            operands: FormatR {
                rd: v_dword,
                rs1: v_dword,
                rs2: v_word,
            },
            inline_sequence_remaining: Some(inline_sequence_remaining),
            is_compressed: self.is_compressed,
        };
        sequence.push(xor_final.into());
        inline_sequence_remaining -= 1;

        let sd = SD {
            address: self.address,
            operands: FormatS {
                rs1: v_dword_address,
                rs2: v_dword,
                imm: 0,
            },
            inline_sequence_remaining: Some(inline_sequence_remaining),
            is_compressed: self.is_compressed,
        };
        sequence.push(sd.into());

        sequence
    }
}
