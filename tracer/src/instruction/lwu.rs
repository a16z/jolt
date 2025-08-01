use super::andi::ANDI;
use super::format::format_i::FormatI;
use super::format::format_r::FormatR;
use super::format::format_virtual_halfword_alignment::HalfwordAlignFormat;
use super::ld::LD;
use super::sll::SLL;
use super::slli::SLLI;
use super::srli::SRLI;
use super::virtual_assert_word_alignment::VirtualAssertWordAlignment;
use super::xori::XORI;
use super::VirtualInstructionSequence;
use super::{addi::ADDI, RV32IMInstruction};
use super::{
    format::{format_load::FormatLoad, InstructionFormat},
    RISCVInstruction, RISCVTrace, RV32IMCycle,
};
use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
};
use common::constants::virtual_register_index;
use serde::{Deserialize, Serialize};

declare_riscv_instr!(
    name   = LWU,
    mask   = 0x0000707f,
    match  = 0x00006003,
    format = FormatLoad,
    ram    = super::RAMRead
);

impl LWU {
    fn exec(&self, cpu: &mut Cpu, ram_access: &mut <LWU as RISCVInstruction>::RAMAccess) {
        // The LWU instruction, on the other hand, zero-extends the 32-bit value from memory for
        // RV64I.
        let address = cpu.x[self.operands.rs1].wrapping_add(self.operands.imm) as u64;
        let value = cpu.mmu.load_word(address);

        cpu.x[self.operands.rd] = match value {
            Ok((word, memory_read)) => {
                *ram_access = memory_read;
                // Zero extension for unsigned word load
                word as i64
            }
            Err(_) => panic!("MMU load error"),
        };
    }
}

impl RISCVTrace for LWU {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<RV32IMCycle>>) {
        let virtual_sequence = self.virtual_sequence(cpu.xlen);
        let mut trace = trace;
        for instr in virtual_sequence {
            // In each iteration, create a new Option containing a re-borrowed reference
            instr.trace(cpu, trace.as_deref_mut());
        }
    }
}

impl VirtualInstructionSequence for LWU {
    fn virtual_sequence(&self, xlen: Xlen) -> Vec<RV32IMInstruction> {
        match xlen {
            Xlen::Bit32 => panic!("LWU is invalid in 32b mode"),
            Xlen::Bit64 => self.virtual_sequence_64(xlen),
        }
    }
}

impl LWU {
    fn virtual_sequence_64(&self, xlen: Xlen) -> Vec<RV32IMInstruction> {
        // Virtual registers used in sequence
        let v_address = virtual_register_index(6) as usize;
        let v_dword_address = virtual_register_index(7) as usize;
        let v_dword = virtual_register_index(8) as usize;
        let v_shift = virtual_register_index(9) as usize;

        let mut sequence = vec![];

        let assert_alignment = VirtualAssertWordAlignment {
            address: self.address,
            operands: HalfwordAlignFormat {
                rs1: self.operands.rs1,
                imm: self.operands.imm,
            },
            virtual_sequence_remaining: Some(8),
            is_compressed: self.is_compressed,
        };
        sequence.push(assert_alignment.into());

        let add = ADDI {
            address: self.address,
            operands: FormatI {
                rd: v_address,
                rs1: self.operands.rs1,
                imm: self.operands.imm as u64,
            },
            virtual_sequence_remaining: Some(7),
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
            virtual_sequence_remaining: Some(6),
            is_compressed: self.is_compressed,
        };
        sequence.push(andi.into());

        let ld = LD {
            address: self.address,
            operands: FormatI {
                rd: v_dword,
                rs1: v_dword_address,
                imm: 0,
            },
            virtual_sequence_remaining: Some(5),
            is_compressed: self.is_compressed,
        };
        sequence.push(ld.into());

        let xori = XORI {
            address: self.address,
            operands: FormatI {
                rd: v_shift,
                rs1: v_address,
                imm: 4,
            },
            virtual_sequence_remaining: Some(4),
            is_compressed: self.is_compressed,
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
            is_compressed: self.is_compressed,
        };
        sequence.extend(slli.virtual_sequence(xlen));

        let sll = SLL {
            address: self.address,
            operands: FormatR {
                rd: self.operands.rd,
                rs1: v_dword,
                rs2: v_shift,
            },
            virtual_sequence_remaining: Some(2),
            is_compressed: self.is_compressed,
        };
        sequence.extend(sll.virtual_sequence(xlen));

        let srli = SRLI {
            address: self.address,
            operands: FormatI {
                rd: self.operands.rd,
                rs1: self.operands.rd,
                imm: 32,
            },
            virtual_sequence_remaining: Some(0),
            is_compressed: self.is_compressed,
        };
        sequence.extend(srli.virtual_sequence(xlen));

        sequence
    }
}
