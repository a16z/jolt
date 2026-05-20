use serde::{Deserialize, Serialize};

use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::{format::format_i::FormatI, RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name = VirtualShiftRightBitmask,
    mask = 0,
    match = 0,
    format = FormatI,
    ram = ()
);

impl VirtualShiftRightBitmask {
    fn exec(
        &self,
        cpu: &mut Cpu,
        _: &mut <VirtualShiftRightBitmask as RISCVInstruction>::RAMAccess,
    ) {
        let shift = cpu.x[self.operands.rs1 as usize] as u64 & 0x3F;
        let ones = (1u128 << (64 - shift)) - 1;
        cpu.write_register(self.operands.rd as usize, (ones << shift) as i64);
    }
}

impl RISCVTrace for VirtualShiftRightBitmask {}
