use serde::{Deserialize, Serialize};

use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::{format::format_i::FormatI, RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name = VirtualShiftRightBitmaskW,
    mask = 0,
    match = 0,
    format = FormatI,
    ram = ()
);

impl VirtualShiftRightBitmaskW {
    fn exec(
        &self,
        cpu: &mut Cpu,
        _: &mut <VirtualShiftRightBitmaskW as RISCVInstruction>::RAMAccess,
    ) {
        let shift = cpu.x[self.operands.rs1 as usize] as u64 & 0x1f;
        let bitmask = (1u64 << 32) - (1u64 << shift);
        cpu.write_register(self.operands.rd as usize, bitmask as i64);
    }
}

impl RISCVTrace for VirtualShiftRightBitmaskW {}
