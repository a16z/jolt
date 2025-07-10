use serde::{Deserialize, Serialize};

use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::{
    format::{format_r::FormatR, InstructionFormat},
    RAMAtomic, RISCVInstruction, RISCVTrace,
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

impl RISCVTrace for AMOADDW {}
