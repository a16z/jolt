use serde::{Deserialize, Serialize};

use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::{
    format::{format_r::FormatR, InstructionFormat},
    RAMAtomic, RISCVInstruction, RISCVTrace,
};

declare_riscv_instr!(
    name   = AMOANDD,
    mask   = 0xf800707f,
    match  = 0x6000302f,
    format = FormatR,
    ram    = RAMAtomic
);

impl AMOANDD {
    fn exec(&self, cpu: &mut Cpu, ram_access: &mut <AMOANDD as RISCVInstruction>::RAMAccess) {
        let address = cpu.x[self.operands.rs1] as u64;
        let and_value = cpu.x[self.operands.rs2] as u64;

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

        // AND the values and store back to memory
        let new_value = (original_value as u64) & and_value;
        let store_result = cpu.mmu.store_doubleword(address, new_value);
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

impl RISCVTrace for AMOANDD {}
