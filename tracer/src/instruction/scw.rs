use serde::{Deserialize, Serialize};

use super::{format::format_r::FormatR, RAMWrite, RISCVInstruction, RISCVTrace};
use crate::{declare_riscv_instr, emulator::cpu::Cpu};

declare_riscv_instr!(
    name   = SCW,
    mask   = 0xf800707f,
    match  = 0x1800202f,
    format = FormatR,
    ram    = RAMWrite
);

impl SCW {
    fn exec(&self, cpu: &mut Cpu, ram_access: &mut <SCW as RISCVInstruction>::RAMAccess) {
        let address = cpu.x[self.operands.rs1 as usize] as u64;
        let value = cpu.x[self.operands.rs2 as usize] as u32;

        // Check if reservation is set and matches the address
        if cpu.has_reservation(address) {
            // Store the word to memory
            let result = cpu.mmu.store_word(address, value);

            match result {
                Ok(memory_write) => {
                    *ram_access = memory_write;
                    // Clear the reservation
                    cpu.clear_reservation();
                    // Return 0 to indicate success
                    cpu.x[self.operands.rd as usize] = 0;
                }
                Err(_) => panic!("MMU store error"),
            }
        } else {
            // Reservation failed, return 1 to indicate failure
            cpu.x[self.operands.rd as usize] = 1;
        }
    }
}

impl RISCVTrace for SCW {}
