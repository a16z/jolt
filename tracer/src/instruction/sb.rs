use crate::utils::virtual_registers::allocate_virtual_register;
use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
    instruction::format::format_load::FormatLoad,
};

use super::addi::ADDI;
use super::and::AND;
use super::andi::ANDI;
use super::format::{format_i::FormatI, format_r::FormatR, format_u::FormatU};
use super::ld::LD;
use super::lui::LUI;
use super::sd::SD;
use super::sll::SLL;
use super::slli::SLLI;
use super::virtual_lw::VirtualLW;
use super::virtual_sw::VirtualSW;
use super::xor::XOR;
use super::{RAMWrite, RV32IMInstruction};

use super::{
    format::{format_s::FormatS, InstructionFormat},
    RISCVInstruction, RISCVTrace, RV32IMCycle,
};

declare_riscv_instr!(
    name   = SB,
    mask   = 0x0000707f,
    match  = 0x00000023,
    format = FormatS,
    ram    = RAMWrite
);

impl SB {
    fn exec(&self, cpu: &mut Cpu, ram_access: &mut <SB as RISCVInstruction>::RAMAccess) {
        *ram_access = cpu
            .mmu
            .store(
                cpu.x[self.operands.rs1 as usize].wrapping_add(self.operands.imm) as u64,
                cpu.x[self.operands.rs2 as usize] as u8,
            )
            .ok()
            .unwrap();
    }
}

impl RISCVTrace for SB {
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

impl SB {
    fn inline_sequence_32(&self) -> Vec<RV32IMInstruction> {
        // Virtual registers used in sequence
        let v_address = allocate_virtual_register();
        let v_word_address = allocate_virtual_register();
        let v_word = allocate_virtual_register();
        let v_shift = allocate_virtual_register();
        let v_mask = allocate_virtual_register();
        let v_byte = allocate_virtual_register();

        let mut sequence = vec![];

        let add = ADDI {
            address: self.address,
            operands: FormatI {
                rd: *v_address,
                rs1: self.operands.rs1,
                imm: self.operands.imm as u64,
            },
            inline_sequence_remaining: Some(12),
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
            inline_sequence_remaining: Some(11),
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
            inline_sequence_remaining: Some(10),
            is_compressed: self.is_compressed,
        };
        sequence.push(lw.into());

        let slli = SLLI {
            address: self.address,
            operands: FormatI {
                rd: *v_shift,
                rs1: *v_address,
                imm: 3,
            },
            inline_sequence_remaining: Some(9),
            is_compressed: self.is_compressed,
        };
        sequence.extend(slli.inline_sequence(Xlen::Bit32));

        let lui = LUI {
            address: self.address,
            operands: FormatU {
                rd: *v_mask,
                imm: 0xff,
            },
            inline_sequence_remaining: Some(8),
            is_compressed: self.is_compressed,
        };
        sequence.push(lui.into());

        let sll = SLL {
            address: self.address,
            operands: FormatR {
                rd: *v_mask,
                rs1: *v_mask,
                rs2: *v_shift,
            },
            inline_sequence_remaining: Some(7),
            is_compressed: self.is_compressed,
        };
        sequence.extend(sll.inline_sequence(Xlen::Bit32));

        let sll = SLL {
            address: self.address,
            operands: FormatR {
                rd: *v_byte,
                rs1: self.operands.rs2,
                rs2: *v_shift,
            },
            inline_sequence_remaining: Some(5),
            is_compressed: self.is_compressed,
        };
        sequence.extend(sll.inline_sequence(Xlen::Bit32));

        let xor = XOR {
            address: self.address,
            operands: FormatR {
                rd: *v_byte,
                rs1: *v_word,
                rs2: *v_byte,
            },
            inline_sequence_remaining: Some(3),
            is_compressed: self.is_compressed,
        };
        sequence.push(xor.into());

        let and = AND {
            address: self.address,
            operands: FormatR {
                rd: *v_byte,
                rs1: *v_byte,
                rs2: *v_mask,
            },
            inline_sequence_remaining: Some(2),
            is_compressed: self.is_compressed,
        };
        sequence.push(and.into());

        let xor = XOR {
            address: self.address,
            operands: FormatR {
                rd: *v_word,
                rs1: *v_word,
                rs2: *v_byte,
            },
            inline_sequence_remaining: Some(1),
            is_compressed: self.is_compressed,
        };
        sequence.push(xor.into());

        let sw = VirtualSW {
            address: self.address,
            operands: FormatS {
                rs1: *v_word_address,
                rs2: *v_word,
                imm: 0,
            },
            inline_sequence_remaining: Some(0),
            is_compressed: self.is_compressed,
        };
        sequence.push(sw.into());

        sequence
    }

    fn inline_sequence_64(&self) -> Vec<RV32IMInstruction> {
        // Virtual registers used in sequence
        let v_address = allocate_virtual_register();
        let v_dword_address = allocate_virtual_register();
        let v_dword = allocate_virtual_register();
        let v_shift = allocate_virtual_register();
        let v_mask = allocate_virtual_register();
        let v_byte = allocate_virtual_register();

        let mut sequence = vec![];

        let add = ADDI {
            address: self.address,
            operands: FormatI {
                rd: *v_address,
                rs1: self.operands.rs1,
                imm: self.operands.imm as u64,
            },
            inline_sequence_remaining: Some(12),
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
            inline_sequence_remaining: Some(11),
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
            inline_sequence_remaining: Some(10),
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
            inline_sequence_remaining: Some(9),
            is_compressed: self.is_compressed,
        };
        sequence.extend(slli.inline_sequence(Xlen::Bit64));

        let lui = LUI {
            address: self.address,
            operands: FormatU {
                rd: *v_mask,
                imm: 0xff,
            },
            inline_sequence_remaining: Some(8),
            is_compressed: self.is_compressed,
        };
        sequence.push(lui.into());

        let sll = SLL {
            address: self.address,
            operands: FormatR {
                rd: *v_mask,
                rs1: *v_mask,
                rs2: *v_shift,
            },
            inline_sequence_remaining: Some(7),
            is_compressed: self.is_compressed,
        };
        sequence.extend(sll.inline_sequence(Xlen::Bit64));

        let sll = SLL {
            address: self.address,
            operands: FormatR {
                rd: *v_byte,
                rs1: self.operands.rs2,
                rs2: *v_shift,
            },
            inline_sequence_remaining: Some(5),
            is_compressed: self.is_compressed,
        };
        sequence.extend(sll.inline_sequence(Xlen::Bit64));

        let xor = XOR {
            address: self.address,
            operands: FormatR {
                rd: *v_byte,
                rs1: *v_dword,
                rs2: *v_byte,
            },
            inline_sequence_remaining: Some(3),
            is_compressed: self.is_compressed,
        };
        sequence.push(xor.into());

        let and = AND {
            address: self.address,
            operands: FormatR {
                rd: *v_byte,
                rs1: *v_byte,
                rs2: *v_mask,
            },
            inline_sequence_remaining: Some(2),
            is_compressed: self.is_compressed,
        };
        sequence.push(and.into());

        let xor = XOR {
            address: self.address,
            operands: FormatR {
                rd: *v_dword,
                rs1: *v_dword,
                rs2: *v_byte,
            },
            inline_sequence_remaining: Some(1),
            is_compressed: self.is_compressed,
        };
        sequence.push(xor.into());

        let sd = SD {
            address: self.address,
            operands: FormatS {
                rs1: *v_dword_address,
                rs2: *v_dword,
                imm: 0,
            },
            inline_sequence_remaining: Some(0),
            is_compressed: self.is_compressed,
        };
        sequence.push(sd.into());

        sequence
    }
}
