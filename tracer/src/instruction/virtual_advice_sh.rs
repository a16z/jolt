use serde::{Deserialize, Serialize};

use super::{format::format_advice_s::FormatAdviceS, RISCVInstruction, RISCVTrace};
use crate::{declare_riscv_instr, emulator::cpu::{Cpu, advice_tape_read}};

declare_riscv_instr!(
    name = VirtualAdviceSH,
    mask = 0,
    match = 0,
    format = FormatAdviceS,
    ram    = super::RAMWrite
);

impl VirtualAdviceSH {
    fn exec(&self, cpu: &mut Cpu, ram_access: &mut <VirtualAdviceSH as RISCVInstruction>::RAMAccess) {
        // Read 2 bytes from the advice tape
        let advice_value = advice_tape_read(2).expect("Failed to read from advice tape");

        // Store the advice value to memory at address rs1 + imm
        *ram_access = cpu
            .mmu
            .store_halfword(
                cpu.x[self.operands.rs1 as usize].wrapping_add(self.operands.imm) as u64,
                advice_value as u16,
            )
            .ok()
            .unwrap();
    }
}

impl RISCVTrace for VirtualAdviceSH {}
