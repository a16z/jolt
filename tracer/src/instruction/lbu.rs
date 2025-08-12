use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
};

use super::addi::ADDI;
use super::andi::ANDI;
use super::format::format_load::FormatLoad;
use super::format::format_r::FormatR;
use super::ld::LD;
use super::sll::SLL;
use super::slli::SLLI;
use super::srli::SRLI;
use super::virtual_lw::VirtualLW;
use super::xori::XORI;
use super::{RAMRead, RV32IMInstruction};
use crate::utils::virtual_registers::allocate_virtual_register;

use super::{
    format::{format_i::FormatI, InstructionFormat},
    RISCVInstruction, RISCVTrace, RV32IMCycle,
};

declare_riscv_instr!(
    name   = LBU,
    mask   = 0x0000707f,
    match  = 0x00004003,
    format = FormatLoad,
    ram    = RAMRead
);

impl LBU {
    fn exec(&self, cpu: &mut Cpu, ram_access: &mut <LBU as RISCVInstruction>::RAMAccess) {
        cpu.x[self.operands.rd as usize] = match cpu
            .mmu
            .load(cpu.x[self.operands.rs1 as usize].wrapping_add(self.operands.imm) as u64)
        {
            Ok((byte, memory_read)) => {
                *ram_access = memory_read;
                byte as i64
            }
            Err(_) => panic!("MMU load error"),
        };
    }
}

impl RISCVTrace for LBU {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<RV32IMCycle>>) {
        let inline_sequence = self.inline_sequence(cpu.xlen);
        let mut trace = trace;
        for instr in inline_sequence {
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

impl LBU {
    fn inline_sequence_32(&self) -> Vec<RV32IMInstruction> {
        // Virtual registers used in sequence
        let v_address = allocate_virtual_register();
        let v_word_address = allocate_virtual_register();
        let v_word = allocate_virtual_register();
        let v_shift = allocate_virtual_register();

        let mut sequence = vec![];

        let add = ADDI {
            address: self.address,
            operands: FormatI {
                rd: *v_address,
                rs1: self.operands.rs1,
                imm: self.operands.imm as u64,
            },
            inline_sequence_remaining: Some(7),
            is_compressed: self.is_compressed,
        };
        sequence.push(add.into());

        let andi = ANDI {
            address: self.address,
            operands: FormatI {
                rd: *v_word_address,
                rs1: *v_address,
                imm: -4i64 as u64,
            },
            inline_sequence_remaining: Some(6),
            is_compressed: self.is_compressed,
        };
        sequence.push(andi.into());

        let lw = VirtualLW {
            address: self.address,
            operands: FormatI {
                rd: *v_word,
                rs1: *v_word_address,
                imm: 0,
            },
            inline_sequence_remaining: Some(5),
            is_compressed: self.is_compressed,
        };
        sequence.push(lw.into());

        let xori = XORI {
            address: self.address,
            operands: FormatI {
                rd: *v_shift,
                rs1: *v_address,
                imm: 3,
            },
            inline_sequence_remaining: Some(4),
            is_compressed: self.is_compressed,
        };
        sequence.push(xori.into());

        let slli = SLLI {
            address: self.address,
            operands: FormatI {
                rd: *v_shift,
                rs1: *v_shift,
                imm: 3,
            },
            inline_sequence_remaining: Some(3),
            is_compressed: self.is_compressed,
        };
        sequence.extend(slli.inline_sequence(Xlen::Bit32));

        let sll = SLL {
            address: self.address,
            operands: FormatR {
                rd: self.operands.rd,
                rs1: *v_word,
                rs2: *v_shift,
            },
            inline_sequence_remaining: Some(2),
            is_compressed: self.is_compressed,
        };
        sequence.extend(sll.inline_sequence(Xlen::Bit32));

        let srli = SRLI {
            address: self.address,
            operands: FormatI {
                rd: self.operands.rd,
                rs1: self.operands.rd,
                imm: 24,
            },
            inline_sequence_remaining: Some(0),
            is_compressed: self.is_compressed,
        };
        sequence.extend(srli.inline_sequence(Xlen::Bit32));

        sequence
    }

    fn inline_sequence_64(&self) -> Vec<RV32IMInstruction> {
        // Virtual registers used in sequence
        let v_address = allocate_virtual_register();
        let v_dword_address = allocate_virtual_register();
        let v_dword = allocate_virtual_register();
        let v_shift = allocate_virtual_register();

        let mut sequence = vec![];

        let add = ADDI {
            address: self.address,
            operands: FormatI {
                rd: *v_address,
                rs1: self.operands.rs1,
                imm: self.operands.imm as u64,
            },
            inline_sequence_remaining: Some(7),
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
            inline_sequence_remaining: Some(6),
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
            inline_sequence_remaining: Some(5),
            is_compressed: self.is_compressed,
        };
        sequence.push(ld.into());

        let xori = XORI {
            address: self.address,
            operands: FormatI {
                rd: *v_shift,
                rs1: *v_address,
                imm: 7,
            },
            inline_sequence_remaining: Some(4),
            is_compressed: self.is_compressed,
        };
        sequence.push(xori.into());

        let slli = SLLI {
            address: self.address,
            operands: FormatI {
                rd: *v_shift,
                rs1: *v_shift,
                imm: 3,
            },
            inline_sequence_remaining: Some(3),
            is_compressed: self.is_compressed,
        };
        sequence.extend(slli.inline_sequence(Xlen::Bit64));

        let sll = SLL {
            address: self.address,
            operands: FormatR {
                rd: self.operands.rd,
                rs1: *v_dword,
                rs2: *v_shift,
            },
            inline_sequence_remaining: Some(2),
            is_compressed: self.is_compressed,
        };
        sequence.extend(sll.inline_sequence(Xlen::Bit64));

        let srli = SRLI {
            address: self.address,
            operands: FormatI {
                rd: self.operands.rd,
                rs1: self.operands.rd,
                imm: 56,
            },
            inline_sequence_remaining: Some(0),
            is_compressed: self.is_compressed,
        };
        sequence.extend(srli.inline_sequence(Xlen::Bit64));

        sequence
    }
}
