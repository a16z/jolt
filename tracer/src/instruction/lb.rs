use common::constants::virtual_register_index;
use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
};

use super::addi::ADDI;
use super::andi::ANDI;
use super::format::format_i::FormatI;
use super::format::format_r::FormatR;
use super::ld::LD;
use super::sll::SLL;
use super::slli::SLLI;
use super::srai::SRAI;
use super::virtual_lw::VirtualLW;
use super::virtual_muli::VirtualMULI;
use super::xori::XORI;
use super::{RAMRead, RV32IMInstruction, VirtualInstructionSequence};

use super::{
    format::{format_load::FormatLoad, InstructionFormat},
    RISCVInstruction, RISCVTrace, RV32IMCycle,
};

declare_riscv_instr!(
    name   = LB,
    mask   = 0x0000707f,
    match  = 0x00000003,
    format = FormatLoad,
    ram    = RAMRead
);

impl LB {
    fn exec(&self, cpu: &mut Cpu, ram_access: &mut <LB as RISCVInstruction>::RAMAccess) {
        cpu.x[self.operands.rd] = match cpu
            .mmu
            .load(cpu.x[self.operands.rs1].wrapping_add(self.operands.imm) as u64)
        {
            Ok((byte, memory_read)) => {
                *ram_access = memory_read;
                byte as i8 as i64
            }
            Err(_) => panic!("MMU load error"),
        };
    }
}

impl RISCVTrace for LB {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<RV32IMCycle>>) {
        let virtual_sequence = self.virtual_sequence(cpu.xlen == Xlen::Bit32);
        let mut trace = trace;
        for instr in virtual_sequence {
            // In each iteration, create a new Option containing a re-borrowed reference
            instr.trace(cpu, trace.as_deref_mut());
        }
    }
}

impl VirtualInstructionSequence for LB {
    fn virtual_sequence(&self, is_32: bool) -> Vec<RV32IMInstruction> {
        if is_32 {
            self.virtual_sequence_32()
        } else {
            self.virtual_sequence_64()
        }
    }
}

impl LB {
    fn virtual_sequence_32(&self) -> Vec<RV32IMInstruction> {
        // Virtual registers used in sequence
        let v_address = virtual_register_index(0) as usize;
        let v_word_address = virtual_register_index(1) as usize;
        let v_word = virtual_register_index(2) as usize;
        let v_shift = virtual_register_index(3) as usize;

        let mut sequence = vec![];

        let add = ADDI {
            address: self.address,
            operands: FormatI {
                rd: v_address,
                rs1: self.operands.rs1,
                imm: self.operands.imm as u64,
            },
            virtual_sequence_remaining: Some(7),
        };
        sequence.push(add.into());

        let andi = ANDI {
            address: self.address,
            operands: FormatI {
                rd: v_word_address,
                rs1: v_address,
                imm: -4i64 as u64,
            },
            virtual_sequence_remaining: Some(6),
        };
        sequence.push(andi.into());

        let lw = VirtualLW {
            address: self.address,
            operands: FormatI {
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
                imm: 3,
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
        sequence.extend(slli.virtual_sequence(true));

        let sll = SLL {
            address: self.address,
            operands: FormatR {
                rd: self.operands.rd,
                rs1: v_word,
                rs2: v_shift,
            },
            virtual_sequence_remaining: Some(2),
        };
        sequence.extend(sll.virtual_sequence(true));

        let srai = SRAI {
            address: self.address,
            operands: FormatI {
                rd: self.operands.rd,
                rs1: self.operands.rd,
                imm: 24,
            },
            virtual_sequence_remaining: Some(0),
        };
        sequence.extend(srai.virtual_sequence(true));

        sequence
    }

    fn virtual_sequence_64(&self) -> Vec<RV32IMInstruction> {
        // Virtual registers used in sequence
        let v_address = virtual_register_index(6) as usize;
        let v_dword_address = virtual_register_index(7) as usize;
        let v_dword = virtual_register_index(8) as usize;
        let v_shift = virtual_register_index(9) as usize;

        let mut sequence = vec![];

        let add = ADDI {
            address: self.address,
            operands: FormatI {
                rd: v_address,
                rs1: self.operands.rs1,
                imm: self.operands.imm as u64,
            },
            virtual_sequence_remaining: Some(7),
        };
        sequence.push(add.into());

        let andi = ANDI {
            address: self.address,
            operands: FormatI {
                rd: v_dword_address,
                rs1: v_address,
                imm: -4i64 as u64,
            },
            virtual_sequence_remaining: Some(6),
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
        };
        sequence.push(ld.into());

        let xori = XORI {
            address: self.address,
            operands: FormatI {
                rd: v_shift,
                rs1: v_address,
                imm: 7,
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
        sequence.extend(slli.virtual_sequence(false));

        let sll = SLL {
            address: self.address,
            operands: FormatR {
                rd: self.operands.rd,
                rs1: v_dword,
                rs2: v_shift,
            },
            virtual_sequence_remaining: Some(2),
        };
        sequence.extend(sll.virtual_sequence(false));

        let srai = SRAI {
            address: self.address,
            operands: FormatI {
                rd: self.operands.rd,
                rs1: self.operands.rd,
                imm: 56,
            },
            virtual_sequence_remaining: Some(0),
        };
        sequence.extend(srai.virtual_sequence(false));

        sequence
    }
}
