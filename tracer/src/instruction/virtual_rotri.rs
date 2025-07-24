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
        let val = cpu.x[self.operands.rs1];
        let rotated = val.rotate_right(shift);
        cpu.x[self.operands.rd] = cpu.sign_extend(rotated);
    }
}

impl RISCVTrace for VirtualROTRI {}

#[cfg(test)]
mod small_test {
    #[test]
    fn check_rotation() {
        let shift = 12;
        let val: i64 =
            0b0001_0101_1001_1001_0011_0000_1111_1110_1001_0001_1111_0010_1010_0111_1111_1001;
        let result = val.rotate_right(shift);
        assert_eq!(
            result,
            0b0111_1111_1001_0001_0101_1001_1001_0011_0000_1111_1110_1001_0001_1111_0010_1010
        )
    }
}
