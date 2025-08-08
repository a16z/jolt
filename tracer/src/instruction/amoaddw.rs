use serde::{Deserialize, Serialize};

use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::{
    format::{format_r::FormatR, InstructionFormat},
    RISCVInstruction, RISCVTrace,
};

declare_riscv_instr!(
    name   = AMOADDW,
    mask   = 0xf800707f,
    match  = 0x0000202f,
    format = FormatR,
    ram    = ()
);

impl AMOADDW {
    fn exec(&self, cpu: &mut Cpu, _: &mut <AMOADDW as RISCVInstruction>::RAMAccess) {
        let address = cpu.x[self.operands.rs1 as usize] as u64;
        let add_value = cpu.x[self.operands.rs2 as usize] as i32;

        // Load the original word from memory
        let load_result = cpu.mmu.load_word(address);
        let original_value = match load_result {
            Ok((word, _)) => word as i32 as i64,
            Err(_) => panic!("MMU load error"),
        };

        // Add the values and store back to memory
        let new_value = (original_value as i32).wrapping_add(add_value) as u32;
        cpu.mmu
            .store_word(address, new_value)
            .expect("MMU store error");

        // Return the original value
        cpu.x[self.operands.rd as usize] = original_value;
    }
}

impl RISCVTrace for AMOADDW {}
