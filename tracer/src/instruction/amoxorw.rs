use serde::{Deserialize, Serialize};

use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::{
    format::{format_r::FormatR, InstructionFormat},
    RAMAtomic, RISCVInstruction, RISCVTrace,
};

declare_riscv_instr!(
    name   = AMOXORW,
    mask   = 0xf800707f,
    match  = 0x2000202f,
    format = FormatR,
    ram    = RAMAtomic
);

impl AMOXORW {
    fn exec(&self, cpu: &mut Cpu, ram_access: &mut <AMOXORW as RISCVInstruction>::RAMAccess) {
        let address = cpu.x[self.operands.rs1] as u64;
        let xor_value = cpu.x[self.operands.rs2] as u32;

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

        // XOR the values and store back to memory
        let new_value = (original_value as u32) ^ xor_value;
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

impl RISCVTrace for AMOXORW {}
