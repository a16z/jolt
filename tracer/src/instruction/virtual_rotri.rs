use serde::{Deserialize, Serialize};

use crate::instruction::format::format_virtual_right_shift_i::FormatVirtualRightShiftI;
use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::{RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name = VirtualROTRI,
    mask = 0,
    match = 0,
    format = FormatVirtualRightShiftI,
    ram = ()
);

impl VirtualROTRI {
    fn exec(&self, cpu: &mut Cpu, _: &mut <VirtualROTRI as RISCVInstruction>::RAMAccess) {
        // Extract rotation amount from bitmask: trailing zeros = rotation amount
        let shift = self.operands.imm.trailing_zeros();

        let rotated = cpu.x[self.operands.rs1 as usize].rotate_right(shift);

        cpu.write_register(self.operands.rd as usize, cpu.sign_extend(rotated));
    }
}

impl RISCVTrace for VirtualROTRI {}
