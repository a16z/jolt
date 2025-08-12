use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
    instruction::ld::LD,
};

use super::andi::ANDI;
use super::format::format_i::FormatI;
use super::format::format_load::FormatLoad;
use super::format::format_r::FormatR;
use super::format::format_virtual_halfword_alignment::HalfwordAlignFormat;
use super::slli::SLLI;
use super::srl::SRL;
use super::virtual_assert_word_alignment::VirtualAssertWordAlignment;
use super::virtual_lw::VirtualLW;
use super::virtual_sign_extend::VirtualSignExtend;
use super::RAMRead;
use super::{addi::ADDI, RV32IMInstruction};
use crate::utils::virtual_registers::allocate_virtual_register;

use super::{format::InstructionFormat, RISCVInstruction, RISCVTrace, RV32IMCycle};

declare_riscv_instr!(
    name   = LW,
    mask   = 0x0000707f,
    match  = 0x00002003,
    format = FormatLoad,
    ram    = RAMRead
);

impl LW {
    fn exec(&self, cpu: &mut Cpu, ram_access: &mut <LW as RISCVInstruction>::RAMAccess) {
        cpu.x[self.operands.rd as usize] = match cpu
            .mmu
            .load_word(cpu.x[self.operands.rs1 as usize].wrapping_add(self.operands.imm) as u64)
        {
            Ok((word, memory_read)) => {
                *ram_access = memory_read;
                word as i32 as i64
            }
            Err(_) => panic!("MMU load error"),
        };
    }
}

impl RISCVTrace for LW {
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

impl LW {
    fn inline_sequence_32(&self) -> Vec<RV32IMInstruction> {
        let mut sequence = vec![];
        let lw = VirtualLW {
            address: self.address,
            operands: FormatI {
                rd: self.operands.rd,
                rs1: self.operands.rs1,
                imm: self.operands.imm as u64,
            },
            inline_sequence_remaining: Some(0),
            is_compressed: self.is_compressed,
        };
        sequence.push(lw.into());

        sequence
    }

    fn inline_sequence_64(&self) -> Vec<RV32IMInstruction> {
        // Virtual registers used in sequence
        let v_address = allocate_virtual_register();
        let v_dword_address = allocate_virtual_register();
        let v_dword = allocate_virtual_register();
        let v_shift = allocate_virtual_register();

        let mut sequence = vec![];

        let assert_alignment = VirtualAssertWordAlignment {
            address: self.address,
            operands: HalfwordAlignFormat {
                rs1: self.operands.rs1,
                imm: self.operands.imm,
            },
            inline_sequence_remaining: Some(7),
            is_compressed: self.is_compressed,
        };
        sequence.push(assert_alignment.into());

        let add = ADDI {
            address: self.address,
            operands: FormatI {
                rd: *v_address,
                rs1: self.operands.rs1,
                imm: self.operands.imm as u64,
            },
            inline_sequence_remaining: Some(6),
            is_compressed: self.is_compressed,
        };
        sequence.push(add.into());

        let andi = ANDI {
            address: self.address,
            operands: FormatI {
                rd: *v_dword_address,
                rs1: *v_address,
                imm: -8i64 as u64,
            },
            inline_sequence_remaining: Some(5),
            is_compressed: self.is_compressed,
        };
        sequence.push(andi.into());

        let ld = LD {
            address: self.address,
            operands: FormatLoad {
                rd: *v_dword,
                rs1: *v_dword_address,
                imm: 0,
            },
            inline_sequence_remaining: Some(4),
            is_compressed: self.is_compressed,
        };
        sequence.push(ld.into());

        let slli = SLLI {
            address: self.address,
            operands: FormatI {
                rd: *v_shift,
                rs1: *v_address,
                imm: 3,
            },
            inline_sequence_remaining: Some(3),
            is_compressed: self.is_compressed,
        };
        sequence.extend(slli.inline_sequence(Xlen::Bit64));

        let srl = SRL {
            address: self.address,
            operands: FormatR {
                rd: self.operands.rd,
                rs1: *v_dword,
                rs2: *v_shift,
            },
            inline_sequence_remaining: Some(2),
            is_compressed: self.is_compressed,
        };
        sequence.extend(srl.inline_sequence(Xlen::Bit64));

        let signext = VirtualSignExtend {
            address: self.address,
            operands: FormatI {
                rd: self.operands.rd,
                rs1: self.operands.rd,
                imm: 0,
            },
            inline_sequence_remaining: Some(0),
            is_compressed: self.is_compressed,
        };
        sequence.push(signext.into());

        sequence
    }
}
