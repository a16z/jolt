use serde::{Deserialize, Serialize};

use super::add::ADD;
use super::amoswapw::{amo_post, amo_pre};
use super::RV32IMInstruction;
use super::VirtualInstructionSequence;
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
    name   = AMOADDW,
    mask   = 0xf800707f,
    match  = 0x0000202f,
    format = FormatR,
    ram    = RAMAtomic
);

impl AMOADDW {
    fn exec(&self, cpu: &mut Cpu, ram_access: &mut <AMOADDW as RISCVInstruction>::RAMAccess) {
        let address = cpu.x[self.operands.rs1] as u64;
        let add_value = cpu.x[self.operands.rs2] as i32;

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

        // Add the values and store back to memory
        let new_value = (original_value as i32).wrapping_add(add_value) as u32;
        let store_result = cpu.mmu.store_word(address, new_value);
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

impl RISCVTrace for AMOADDW {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<RV32IMCycle>>) {
        let virtual_sequence = self.virtual_sequence(cpu.xlen == Xlen::Bit32);
        let mut trace = trace;
        for instr in virtual_sequence {
            // In each iteration, create a new Option containing a re-borrowed reference
            instr.trace(cpu, trace.as_deref_mut());
        }
    }
}

impl VirtualInstructionSequence for AMOADDW {
    fn virtual_sequence(&self, is_32: bool) -> Vec<RV32IMInstruction> {
        // Virtual registers used in sequence
        let v_mask = virtual_register_index(10) as usize;
        let v_dword_address = virtual_register_index(11) as usize;
        let v_dword = virtual_register_index(12) as usize;
        let v_word = virtual_register_index(13) as usize;
        let v_shift = virtual_register_index(14) as usize;
        let v_rd = virtual_register_index(15) as usize;
        let v_rs2 = virtual_register_index(16) as usize;

        let mut sequence = vec![];
        let mut remaining = 17;
        remaining = amo_pre(
            &mut sequence,
            self.address,
            self.operands.rs1,
            v_rd,
            v_dword_address,
            v_dword,
            v_shift,
            remaining,
        );
        let add = ADD {
            address: self.address,
            operands: FormatR {
                rd: v_rs2,
                rs1: v_rd,
                rs2: self.operands.rs2,
            },
            virtual_sequence_remaining: Some(remaining),
        };
        sequence.push(add.into());
        remaining -= 1;
        amo_post(
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
