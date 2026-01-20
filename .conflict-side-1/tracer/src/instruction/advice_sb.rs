use serde::{Deserialize, Serialize};

use crate::{declare_riscv_instr, emulator::cpu::{Cpu, advice_tape_read}};

use super::{format::format_s::FormatS, RAMWrite, RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name   = AdviceSB,
    mask   = 0,
    match  = 0,
    format = FormatS,
    ram    = RAMWrite
);

impl AdviceSB {
    fn exec(&self, cpu: &mut Cpu, ram_access: &mut <AdviceSB as RISCVInstruction>::RAMAccess) {
        // Read 1 byte from the advice tape
        let advice_value = advice_tape_read(1).expect("Failed to read from advice tape");

        // Store the advice value to memory at address rs1 + imm
        *ram_access = cpu
            .mmu
            .store(
                cpu.x[self.operands.rs1 as usize].wrapping_add(self.operands.imm) as u64,
                advice_value as u8,
            )
            .ok()
            .unwrap();
    }
}

impl RISCVTrace for AdviceSB {}
