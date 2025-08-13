use serde::{Deserialize, Serialize};

use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::{
    format::{format_r::FormatR, InstructionFormat},
    RISCVInstruction, RISCVTrace,
};

declare_riscv_instr!(
    name   = AMOMINUW,
    mask   = 0xf800707f,
    match  = 0xc000202f,
    format = FormatR,
    ram    = ()
);

impl AMOMINUW {
    fn exec(&self, cpu: &mut Cpu, _: &mut <AMOMINUW as RISCVInstruction>::RAMAccess) {
        let address = cpu.x[self.operands.rs1 as usize] as u64;
        let compare_value = cpu.x[self.operands.rs2 as usize] as u32;

        // Load the original word from memory
        let load_result = cpu.mmu.load_word(address);
        let original_value = match load_result {
            Ok((word, _)) => word as i32 as i64,
            Err(_) => panic!("MMU load error"),
        };

        // Find the minimum (unsigned comparison) and store back to memory
        let new_value = if (original_value as u32) <= compare_value {
            original_value as u32
        } else {
            compare_value
        };
        cpu.mmu
            .store_word(address, new_value)
            .expect("MMU store error");

        // Return the original value (sign extended)
        cpu.x[self.operands.rd as usize] = original_value;
    }
}

impl RISCVTrace for AMOMINUW {}
