use serde::{Deserialize, Serialize};

use crate::instruction::format::format_virtual_right_shift_i::FormatVirtualRightShiftI;
use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::{format::InstructionFormat, RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name = VirtualROTRI,
    mask = 0,
    match = 0,
    format = FormatVirtualRightShiftI,
    ram = (),
    is_virtual = true
);

impl VirtualROTRI {
    fn exec(&self, cpu: &mut Cpu, _: &mut <VirtualROTRI as RISCVInstruction>::RAMAccess) {
        // Extract rotation amount from bitmask: trailing zeros = rotation amount
        let shift = self.operands.imm.trailing_zeros();
        // Extract 32-bit value and rotate only within 32 bits
        let val_32 = cpu.x[self.operands.rs1] as u32;
        let rotated_32 = val_32.rotate_right(shift);
        cpu.x[self.operands.rd] = cpu.sign_extend(rotated_32 as i64);
    }
}

impl RISCVTrace for VirtualROTRI {}
