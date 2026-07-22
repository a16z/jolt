use serde::{Deserialize, Serialize};

use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::{format::format_j::FormatJ, RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name = VirtualShiftRightBitmaskI,
    mask = 0,
    match = 0,
    format = FormatJ,
    ram = ()
);

impl VirtualShiftRightBitmaskI {
    fn exec(
        &self,
        cpu: &mut Cpu,
        _: &mut <VirtualShiftRightBitmaskI as RISCVInstruction>::RAMAccess,
    ) {
        let shift = self.operands.imm % 64;
        let ones = (1u128 << (64 - shift)) - 1;
        cpu.write_register(self.operands.rd as usize, (ones << shift) as i64);
    }
}

impl RISCVTrace for VirtualShiftRightBitmaskI {}
