use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::cpu::Cpu,
    instruction::format::format_virtual_right_shift_i::FormatVirtualRightShiftI,
};

use super::{RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name = VirtualSRAI,
    mask = 0,
    match = 0,
    format = FormatVirtualRightShiftI,
    ram = ()
);

impl VirtualSRAI {
    fn exec(
        &self,
        cpu: &mut Cpu,
        _: &mut <VirtualSRAI as RISCVInstruction>::RAMAccess,
    ) {
        let shift = self.operands.imm.trailing_zeros();
        cpu.x[self.operands.rd as usize] =
            cpu.sign_extend(cpu.x[self.operands.rs1 as usize].wrapping_shr(shift));
    }
}

impl RISCVTrace for VirtualSRAI {}
