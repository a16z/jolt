use serde::{Deserialize, Serialize};

use super::{format::format_advice_s::FormatAdviceS, RISCVInstruction, RISCVTrace};
use crate::{declare_riscv_instr, emulator::cpu::{Cpu, advice_tape_read}};

declare_riscv_instr!(
    name = VirtualAdviceSB,
    mask = 0,
    match = 0,
    format = FormatAdviceS,
    ram    = super::RAMWrite
);

impl VirtualAdviceSB {
    fn exec(&self, cpu: &mut Cpu, ram_access: &mut <VirtualAdviceSB as RISCVInstruction>::RAMAccess) {
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

impl RISCVTrace for VirtualAdviceSB {}
