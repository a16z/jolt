use serde::{Deserialize, Serialize};

use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::{
    format::{format_r::FormatR, InstructionFormat},
    RAMRead, RISCVInstruction, RISCVTrace,
};

declare_riscv_instr!(
    name   = LRD,
    mask   = 0xf9f0707f,
    match  = 0x1000302f,
    format = FormatR,
    ram    = RAMRead
);

impl LRD {
    fn exec(&self, cpu: &mut Cpu, ram_access: &mut <LRD as RISCVInstruction>::RAMAccess) {
        if cpu.is_reservation_set() {
            println!("LRD: Reservation is already set");
        }

        let address = cpu.x[self.operands.rs1] as u64;

        // Load the doubleword from memory
        let value = cpu.mmu.load_doubleword(address);

        cpu.x[self.operands.rd] = match value {
            Ok((doubleword, memory_read)) => {
                *ram_access = memory_read;

                // Set reservation for this address
                cpu.set_reservation(address);

                // Return the 64-bit value
                doubleword as i64
            }
            Err(_) => panic!("MMU load error"),
        };
    }
}

impl RISCVTrace for LRD {}
