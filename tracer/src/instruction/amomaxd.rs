use serde::{Deserialize, Serialize};

use super::add::ADD;
use super::format::format_i::FormatI;
use super::format::format_s::FormatS;
use super::ld::LD;
use super::mul::MUL;
use super::sd::SD;
use super::slt::SLT;
use super::virtual_move::VirtualMove;
use super::xori::XORI;
use super::RV32IMInstruction;
use super::VirtualInstructionSequence;
use crate::instruction::format::format_load::FormatLoad;
use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
};
use common::constants::virtual_register_index;

use super::{
    format::{format_r::FormatR, InstructionFormat},
    RAMAtomic, RISCVInstruction, RISCVTrace, RV32IMCycle,
};

declare_riscv_instr!(
    name   = AMOMAXD,
    mask   = 0xf800707f,
    match  = 0xa000302f,
    format = FormatR,
    ram    = RAMAtomic
);

impl AMOMAXD {
    fn exec(&self, cpu: &mut Cpu, ram_access: &mut <AMOMAXD as RISCVInstruction>::RAMAccess) {
        let address = cpu.x[self.operands.rs1] as u64;
        let compare_value = cpu.x[self.operands.rs2];

        // Load the original doubleword from memory
        let load_result = cpu.mmu.load_doubleword(address);
        let original_value = match load_result {
            Ok((doubleword, memory_read)) => {
                // Store the read access
                ram_access.read = memory_read;
                doubleword as i64
            }
            Err(_) => panic!("MMU load error"),
        };

        // Find the maximum and store back to memory
        let new_value = if original_value >= compare_value {
            original_value
        } else {
            compare_value
        };
        let store_result = cpu.mmu.store_doubleword(address, new_value as u64);
        match store_result {
            Ok(memory_write) => {
                // Store the write access
                ram_access.write = memory_write;
            }
            Err(_) => panic!("MMU store error"),
        }

        // Return the original value
        cpu.x[self.operands.rd] = original_value;
    }
}

impl RISCVTrace for AMOMAXD {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<RV32IMCycle>>) {
        let virtual_sequence = self.virtual_sequence(cpu.xlen);
        let mut trace = trace;
        for instr in virtual_sequence {
            // In each iteration, create a new Option containing a re-borrowed reference
            instr.trace(cpu, trace.as_deref_mut());
        }
    }
}

impl VirtualInstructionSequence for AMOMAXD {
    fn virtual_sequence(&self, _xlen: Xlen) -> Vec<RV32IMInstruction> {
        let v_rs2 = virtual_register_index(6) as usize;
        let v_rd = virtual_register_index(7) as usize;
        let v_sel_rs2 = virtual_register_index(8) as usize;
        let v_sel_rd = virtual_register_index(9) as usize;
        let v_tmp = virtual_register_index(10) as usize;

        let mut sequence = vec![];
        let mut virtual_sequence_remaining = self.virtual_sequence_remaining.unwrap_or(7);

        let ld = LD {
            address: self.address,
            operands: FormatLoad {
                rd: v_rd,
                rs1: self.operands.rs1,
                imm: 0,
            },
            virtual_sequence_remaining: Some(virtual_sequence_remaining),
            is_compressed: self.is_compressed,
        };
        sequence.push(ld.into());
        virtual_sequence_remaining -= 1;

        let slt = SLT {
            address: self.address,
            operands: FormatR {
                rd: v_sel_rs2,
                rs1: v_rd,
                rs2: self.operands.rs2,
            },
            virtual_sequence_remaining: Some(virtual_sequence_remaining),
            is_compressed: self.is_compressed,
        };
        sequence.push(slt.into());
        virtual_sequence_remaining -= 1;

        let xori = XORI {
            address: self.address,
            operands: FormatI {
                rd: v_sel_rd,
                rs1: v_sel_rs2,
                imm: 1,
            },
            virtual_sequence_remaining: Some(virtual_sequence_remaining),
            is_compressed: self.is_compressed,
        };
        sequence.push(xori.into());
        virtual_sequence_remaining -= 1;

        let mul = MUL {
            address: self.address,
            operands: FormatR {
                rd: v_rs2,
                rs1: v_sel_rs2,
                rs2: self.operands.rs2,
            },
            virtual_sequence_remaining: Some(virtual_sequence_remaining),
            is_compressed: self.is_compressed,
        };
        sequence.push(mul.into());
        virtual_sequence_remaining -= 1;

        let mul = MUL {
            address: self.address,
            operands: FormatR {
                rd: v_tmp,
                rs1: v_sel_rd,
                rs2: v_rd,
            },
            virtual_sequence_remaining: Some(virtual_sequence_remaining),
            is_compressed: self.is_compressed,
        };
        sequence.push(mul.into());
        virtual_sequence_remaining -= 1;

        let add = ADD {
            address: self.address,
            operands: FormatR {
                rd: v_rs2,
                rs1: v_tmp,
                rs2: v_rs2,
            },
            virtual_sequence_remaining: Some(virtual_sequence_remaining),
            is_compressed: self.is_compressed,
        };
        sequence.push(add.into());
        virtual_sequence_remaining -= 1;

        let sd = SD {
            address: self.address,
            operands: FormatS {
                rs1: self.operands.rs1,
                rs2: v_rs2,
                imm: 0,
            },
            virtual_sequence_remaining: Some(virtual_sequence_remaining),
            is_compressed: self.is_compressed,
        };
        sequence.push(sd.into());
        virtual_sequence_remaining -= 1;

        let vmove = VirtualMove {
            address: self.address,
            operands: FormatI {
                rd: self.operands.rd,
                rs1: v_rd,
                imm: 0,
            },
            virtual_sequence_remaining: Some(virtual_sequence_remaining),
            is_compressed: self.is_compressed,
        };
        sequence.push(vmove.into());

        sequence
    }
}
