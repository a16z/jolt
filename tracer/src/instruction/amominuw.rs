use serde::{Deserialize, Serialize};

use super::add::ADD;
use super::amoswapw::{amo_post, amo_pre};
use super::mul::MUL;
use super::sltu::SLTU;
use super::virtual_extend::VirtualExtend;
use super::xor::XOR;
use super::xori::XORI;
use super::RV32IMInstruction;
use super::VirtualInstructionSequence;
use crate::{declare_riscv_instr, emulator::cpu::Cpu};
use common::constants::virtual_register_index;

use super::{
    format::{format_i::FormatI, format_r::FormatR, InstructionFormat},
    RAMAtomic, RISCVInstruction, RISCVTrace, RV32IMCycle,
};

declare_riscv_instr!(
    name   = AMOMINUW,
    mask   = 0xf800707f,
    match  = 0xc000202f,
    format = FormatR,
    ram    = RAMAtomic
);

impl AMOMINUW {
    fn exec(&self, cpu: &mut Cpu, ram_access: &mut <AMOMINUW as RISCVInstruction>::RAMAccess) {
        let address = cpu.x[self.operands.rs1] as u64;
        let compare_value = cpu.x[self.operands.rs2] as u32;

        // Load the original word from memory
        let load_result = cpu.mmu.load_word(address);
        let original_value = match load_result {
            Ok((word, memory_read)) => {
                // Store the read access
                ram_access.read = memory_read;
                word as i32 as i64
            }
            Err(_) => panic!("MMU load error"),
        };

        // Find the minimum (unsigned comparison) and store back to memory
        let new_value = if (original_value as u32) <= compare_value {
            original_value as u32
        } else {
            compare_value
        };
        let store_result = cpu.mmu.store_word(address, new_value);
        match store_result {
            Ok(memory_write) => {
                // Store the write access
                ram_access.write = memory_write;
            }
            Err(_) => panic!("MMU store error"),
        }

        // Return the original value (sign extended)
        cpu.x[self.operands.rd] = original_value;
    }
}

impl RISCVTrace for AMOMINUW {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<RV32IMCycle>>) {
        let virtual_sequence = self.virtual_sequence(cpu);
        let mut trace = trace;
        for instr in virtual_sequence {
            // In each iteration, create a new Option containing a re-borrowed reference
            instr.trace(cpu, trace.as_deref_mut());
        }
    }
}

impl VirtualInstructionSequence for AMOMINUW {
    fn virtual_sequence(&self, cpu: &Cpu) -> Vec<RV32IMInstruction> {
        // Virtual registers used in sequence
        let v_mask = virtual_register_index(10) as usize;
        let v_dword_address = virtual_register_index(11) as usize;
        let v_dword = virtual_register_index(12) as usize;
        let v_word = virtual_register_index(13) as usize;
        let v_shift = virtual_register_index(14) as usize;
        let v_rd = virtual_register_index(15) as usize;
        let v_rs2 = virtual_register_index(16) as usize;
        let v_sel_rs2 = virtual_register_index(17) as usize;
        let v_sel_rd = virtual_register_index(18) as usize;
        let v_tmp = virtual_register_index(19) as usize;

        let mut sequence = vec![];
        let mut remaining = 22;
        remaining = amo_pre(
            cpu,
            &mut sequence,
            self.address,
            self.operands.rs1,
            v_rd,
            v_dword_address,
            v_dword,
            v_shift,
            remaining,
        );

        let ext = VirtualExtend {
            address: self.address,
            operands: FormatI {
                rd: v_rs2,
                rs1: self.operands.rs2,
                imm: 0,
            },
            virtual_sequence_remaining: Some(remaining),
        };
        sequence.push(ext.into());
        remaining -= 1;

        let ext = VirtualExtend {
            address: self.address,
            operands: FormatI {
                rd: v_tmp,
                rs1: v_rd,
                imm: 0,
            },
            virtual_sequence_remaining: Some(remaining),
        };
        sequence.push(ext.into());
        remaining -= 1;

        let sltu = SLTU {
            address: self.address,
            operands: FormatR {
                rd: v_sel_rs2,
                rs1: v_rs2,
                rs2: v_tmp,
            },
            virtual_sequence_remaining: Some(remaining),
        };
        sequence.push(sltu.into());
        remaining -= 1;

        let xori = XORI {
            address: self.address,
            operands: FormatI {
                rd: v_sel_rd,
                rs1: v_sel_rs2,
                imm: 1,
            },
            virtual_sequence_remaining: Some(remaining),
        };
        sequence.push(xori.into());
        remaining -= 1;

        let mul = MUL {
            address: self.address,
            operands: FormatR {
                rd: v_rs2,
                rs1: v_sel_rs2,
                rs2: self.operands.rs2,
            },
            virtual_sequence_remaining: Some(remaining),
        };
        sequence.push(mul.into());
        remaining -= 1;

        let mul = MUL {
            address: self.address,
            operands: FormatR {
                rd: v_tmp,
                rs1: v_sel_rd,
                rs2: v_rd,
            },
            virtual_sequence_remaining: Some(remaining),
        };
        sequence.push(mul.into());
        remaining -= 1;

        let add = ADD {
            address: self.address,
            operands: FormatR {
                rd: v_rs2,
                rs1: v_tmp,
                rs2: v_rs2,
            },
            virtual_sequence_remaining: Some(1),
        };
        sequence.push(add.into());
        remaining -= 1;

        amo_post(
            cpu,
            &mut sequence,
            self.address,
            v_rs2,
            v_dword_address,
            v_dword,
            v_shift,
            v_mask,
            v_word,
            self.operands.rd,
            v_rd,
            remaining,
        );

        sequence
    }
}
