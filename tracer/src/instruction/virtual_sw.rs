use serde::{Deserialize, Serialize};

use super::{format::format_s::FormatS, RISCVInstruction, RISCVTrace};
use crate::declare_riscv_instr;
use crate::emulator::cpu::{Cpu, Xlen};

declare_riscv_instr!(
    name = VirtualSW,
    mask = 0,
    match = 0,
    format = FormatS,
    ram    = super::RAMWrite
);

impl VirtualSW {
    fn exec(&self, cpu: &mut Cpu, ram_access: &mut <VirtualSW as RISCVInstruction>::RAMAccess) {
        // virtual lw is only supported on bit32. On bit64 LW doesn't use this instruction
        assert_eq!(cpu.xlen, Xlen::Bit32);
        *ram_access = cpu
            .mmu
            .store_word(
                cpu.x[self.operands.rs1 as usize].wrapping_add(self.operands.imm) as u64,
                cpu.x[self.operands.rs2 as usize] as u32,
            )
            .ok()
            .unwrap();
    }
}

impl RISCVTrace for VirtualSW {}
