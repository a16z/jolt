use serde::{Deserialize, Serialize};

use super::{
    format::{format_s::FormatS, InstructionFormat},
    RISCVInstruction, RISCVTrace,
};
use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
};

declare_riscv_instr!(
    name = VirtualSW,
    mask = 0,
    match = 0,
    format = FormatS,
    ram    = super::RAMWrite,
    is_virtual = true
);

impl VirtualSW {
    fn exec(&self, cpu: &mut Cpu, ram_access: &mut <VirtualSW as RISCVInstruction>::RAMAccess) {
        *ram_access = cpu
            .mmu
            .store_word(
                cpu.x[self.operands.rs1].wrapping_add(self.operands.imm) as u64,
                cpu.x[self.operands.rs2] as u32,
            )
            .ok()
            .unwrap();
    }
}

impl RISCVTrace for VirtualSW {}
