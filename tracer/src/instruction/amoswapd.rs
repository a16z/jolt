use serde::{Deserialize, Serialize};

use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::{
    format::{format_r::FormatR, InstructionFormat},
    RISCVInstruction, RISCVTrace,
};

declare_riscv_instr!(
    name   = AMOSWAPD,
    mask   = 0xf800707f,
    match  = 0x0800302f,
    format = FormatR,
    ram    = ()
);

impl AMOSWAPD {
    fn exec(&self, cpu: &mut Cpu, _: &mut <AMOSWAPD as RISCVInstruction>::RAMAccess) {
        let address = cpu.x[self.operands.rs1 as usize] as u64;
        let new_value = cpu.x[self.operands.rs2 as usize] as u64;

        // Load the original doubleword from memory
        let load_result = cpu.mmu.load_doubleword(address);
        let original_value = match load_result {
            Ok((doubleword, _)) => doubleword as i64,
            Err(_) => panic!("MMU load error"),
        };

        // Store the new value to memory
        cpu.mmu
            .store_doubleword(address, new_value)
            .expect("MMU store error");

        // Return the original value
        cpu.x[self.operands.rd as usize] = original_value;
    }
}

impl RISCVTrace for AMOSWAPD {}
