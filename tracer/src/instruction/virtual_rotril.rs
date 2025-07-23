use serde::{Deserialize, Serialize};

use crate::instruction::format::format_virtual_right_shift_i::FormatVirtualRightShiftI;
use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::{format::InstructionFormat, RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name = VirtualROTRIL,
    mask = 0,
    match = 0,
    format = FormatVirtualRightShiftI,
    ram = (),
    is_virtual = true
);

impl VirtualROTRIL {
    fn exec(&self, cpu: &mut Cpu, _: &mut <VirtualROTRIL as RISCVInstruction>::RAMAccess) {
        // Extract rotation amount from bitmask: trailing zeros = rotation amount
        let shift = self.operands.imm.trailing_zeros();
        // Extract only the lower 32-bit value
        let val_32 = cpu.x[self.operands.rs1] as u32;
        // Rotate only within the 32 bits
        let rotated_32 = val_32.rotate_right(shift);
        // Set result with upper 32 bits as 0
        cpu.x[self.operands.rd] = rotated_32 as u64 as i64;
    }
}

impl RISCVTrace for VirtualROTRIL {}

#[cfg(test)]
mod small_test {
    #[test]
    fn check_rotation_32bit_only() {
        let shift = 12;
        // Test with a 64-bit value that has bits in upper 32 bits
        let val: i64 = 0x0000_0000_1234_5678;
        let val_32 = val as u32; // 0x1234_5678
        let result = val_32.rotate_right(shift); // Rotate only lower 32 bits
        let expected = 0x6781234_5; // 0x1234_5678 rotated right by 12
        assert_eq!(result, expected);
        
        // Verify upper bits are cleared
        let final_result = result as u64 as i64;
        assert_eq!(final_result >> 32, 0x0000_0000);
    }
}
