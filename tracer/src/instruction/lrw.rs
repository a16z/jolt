use serde::{Deserialize, Serialize};

use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::{
    format::{format_r::FormatR, InstructionFormat},
    RAMRead, RISCVInstruction, RISCVTrace,
};

declare_riscv_instr!(
    name   = LRW,
    mask   = 0xf9f0707f,
    match  = 0x1000202f,
    format = FormatR,
    ram    = RAMRead
);

impl LRW {
    fn exec(&self, cpu: &mut Cpu, ram_access: &mut <LRW as RISCVInstruction>::RAMAccess) {
        if cpu.is_reservation_set() {
            println!("LRW: Reservation is already set");
        }

        let address = cpu.x[self.operands.rs1] as u64;

        // Load the word from memory
        let value = cpu.mmu.load_word(address);

        cpu.x[self.operands.rd] = match value {
            Ok((word, memory_read)) => {
                *ram_access = memory_read;

                // Set reservation for this address
                cpu.set_reservation(address);

                // Sign extend the 32-bit value
                word as i32 as i64
            }
            Err(_) => panic!("MMU load error"),
        };
    }
}

impl RISCVTrace for LRW {}
