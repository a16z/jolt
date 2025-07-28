use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
    instruction::{ori::ORI, srli::SRLI},
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
use super::VirtualInstructionSequence;
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
                cpu.x[self.operands.rs1].wrapping_add(self.operands.imm) as u64,
                cpu.x[self.operands.rs2] as u32,
            )
            .ok()
            .unwrap();
    }
}

impl RISCVTrace for SW {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<RV32IMCycle>>) {
        let virtual_sequence = self.virtual_sequence(cpu.xlen);
        let mut trace = trace;
        for instr in virtual_sequence {
            // In each iteration, create a new Option containing a re-borrowed reference
            instr.trace(cpu, trace.as_deref_mut());
        }
    }
}

impl VirtualInstructionSequence for SW {
    fn virtual_sequence(&self, xlen: Xlen) -> Vec<RV32IMInstruction> {
        match xlen {
            Xlen::Bit32 => self.virtual_sequence_32(),
            Xlen::Bit64 => self.virtual_sequence_64(),
        }
    }
}

impl SW {
    fn virtual_sequence_32(&self) -> Vec<RV32IMInstruction> {
        let mut sequence = vec![];
        let sw = VirtualSW {
            address: self.address,
            operands: FormatS {
                rs1: self.operands.rs1,
                rs2: self.operands.rs2,
                imm: self.operands.imm,
            },
            virtual_sequence_remaining: Some(0),
        };
        sequence.push(sw.into());

        sequence
    }

    fn virtual_sequence_64(&self) -> Vec<RV32IMInstruction> {
        // Virtual registers used in sequence
        let v_address = virtual_register_index(6) as usize;
        let v_dword_address = virtual_register_index(7) as usize;
        let v_dword = virtual_register_index(8) as usize;
        let v_shift = virtual_register_index(9) as usize;
        let v_mask = virtual_register_index(10) as usize;
        let v_word = virtual_register_index(11) as usize;

        let mut sequence = vec![];
        let mut virtual_sequence_remaining = self.virtual_sequence_remaining.unwrap_or(14);

        let align_check = VirtualAssertWordAlignment {
            address: self.address,
            operands: HalfwordAlignFormat {
                rs1: self.operands.rs1,
                imm: self.operands.imm,
            },
            virtual_sequence_remaining: Some(virtual_sequence_remaining),
        };
        sequence.push(align_check.into());
        virtual_sequence_remaining -= 1;

        let add = ADDI {
            address: self.address,
            operands: FormatI {
                rd: v_address,
                rs1: self.operands.rs1,
                imm: self.operands.imm as u64,
            },
            virtual_sequence_remaining: Some(virtual_sequence_remaining),
        };
        sequence.push(add.into());
        virtual_sequence_remaining -= 1;

        let andi = ANDI {
            address: self.address,
            operands: FormatI {
                rd: v_dword_address,
                rs1: v_address,
                imm: -8i64 as u64,
            },
            virtual_sequence_remaining: Some(virtual_sequence_remaining),
        };
        sequence.push(andi.into());
        virtual_sequence_remaining -= 1;

        let ld = LD {
            address: self.address,
            operands: FormatI {
                rd: v_dword,
                rs1: v_dword_address,
                imm: 0,
            },
            virtual_sequence_remaining: Some(virtual_sequence_remaining),
        };
        sequence.push(ld.into());
        virtual_sequence_remaining -= 1;

        let slli = SLLI {
            address: self.address,
            operands: FormatI {
                rd: v_shift,
                rs1: v_address,
                imm: 3,
            },
            virtual_sequence_remaining: Some(virtual_sequence_remaining),
        };
        let slli_sequence = slli.virtual_sequence(Xlen::Bit64);
        let slli_sequence_len = slli_sequence.len();
        sequence.extend(slli_sequence);
        virtual_sequence_remaining -= slli_sequence_len;

        let ori = ORI {
            address: self.address,
            operands: FormatI {
                rd: v_mask,
                rs1: 0,
                imm: -1i64 as u64,
            },
            virtual_sequence_remaining: Some(virtual_sequence_remaining),
        };
        sequence.push(ori.into()); // v_mask gets 0xFFFFFFFF_FFFFFFFF
        virtual_sequence_remaining -= 1;

        let srli = SRLI {
            address: self.address,
            operands: FormatI {
                rd: v_mask,
                rs1: v_mask,
                imm: 32, // Logical right shift by 32 bits
            },
            virtual_sequence_remaining: Some(virtual_sequence_remaining),
        };
        sequence.push(srli.into()); // v_mask gets 0x00000000_FFFFFFFF
        virtual_sequence_remaining -= 1;

        let sll_mask = SLL {
            address: self.address,
            operands: FormatR {
                rd: v_mask,
                rs1: v_mask,
                rs2: v_shift,
            },
            virtual_sequence_remaining: Some(virtual_sequence_remaining),
        };
        let sll_mask_sequence = sll_mask.virtual_sequence(Xlen::Bit64);
        let sll_mask_sequence_len = sll_mask_sequence.len();
        sequence.extend(sll_mask_sequence);
        virtual_sequence_remaining -= sll_mask_sequence_len;

        let sll_value = SLL {
            address: self.address,
            operands: FormatR {
                rd: v_word,
                rs1: self.operands.rs2,
                rs2: v_shift,
            },
            virtual_sequence_remaining: Some(virtual_sequence_remaining),
        };
        let sll_value_sequence = sll_value.virtual_sequence(Xlen::Bit64);
        let sll_value_sequence_len = sll_value_sequence.len();
        sequence.extend(sll_value_sequence);
        virtual_sequence_remaining -= sll_value_sequence_len;

        let xor = XOR {
            address: self.address,
            operands: FormatR {
                rd: v_word,
                rs1: v_dword,
                rs2: v_word,
            },
            virtual_sequence_remaining: Some(virtual_sequence_remaining),
        };
        sequence.push(xor.into());
        virtual_sequence_remaining -= 1;

        let and = AND {
            address: self.address,
            operands: FormatR {
                rd: v_word,
                rs1: v_word,
                rs2: v_mask,
            },
            virtual_sequence_remaining: Some(virtual_sequence_remaining),
        };
        sequence.push(and.into());
        virtual_sequence_remaining -= 1;

        let xor_final = XOR {
            address: self.address,
            operands: FormatR {
                rd: v_dword,
                rs1: v_dword,
                rs2: v_word,
            },
            virtual_sequence_remaining: Some(virtual_sequence_remaining),
        };
        sequence.push(xor_final.into());
        virtual_sequence_remaining -= 1;

        let sd = SD {
            address: self.address,
            operands: FormatS {
                rs1: v_dword_address,
                rs2: v_dword,
                imm: 0,
            },
            virtual_sequence_remaining: Some(virtual_sequence_remaining),
        };
        sequence.push(sd.into());

        sequence
    }
}
