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
mod tests {
    use crate::emulator::cpu::Cpu;
    use crate::emulator::default_terminal::DefaultTerminal;
    use crate::emulator::mmu::DRAM_BASE;
    use crate::instruction::format::format_virtual_right_shift_i::FormatVirtualRightShiftI;
    use crate::instruction::virtual_rotri::VirtualROTRI;

    #[test]
    fn test_manual_rotation() {
        let shift = 12;
        let val: i64 =
            0b0001_0101_1001_1001_0011_0000_1111_1110_1001_0001_1111_0010_1010_0111_1111_1001;
        let result = val.rotate_right(shift);
        assert_eq!(
            result,
            0b0111_1111_1001_0001_0101_1001_1001_0011_0000_1111_1110_1001_0001_1111_0010_1010
        )
    }

    /// Helper function to execute a VirtualROTRI instruction and return the result.
    fn execute_and_read_rotri(cpu: &mut Cpu, rs1_val: i64, imm: u64, rs1: usize, rd: usize) -> i64 {
        cpu.x[rs1] = rs1_val;
        let instruction = VirtualROTRI {
            address: DRAM_BASE,
            operands: FormatVirtualRightShiftI { rd, rs1, imm },
            virtual_sequence_remaining: Some(0),
        };
        instruction.exec(cpu, &mut ());
        cpu.x[rd]
    }

    #[test]
    fn test_virtual_rotri() {
        let mut cpu = Cpu::new(Box::new(DefaultTerminal::new()));

        let test_cases = [
            // Test Case 1: Simple rotation by 1 bit
            // imm = 0b10 has 1 trailing zero, so rotate right by 1.
            // 0x12345678 rotated right by 1 = 0x091A2B3C
            (0x12345678, 0b10, 10, 12, 0x091A2B3C, "Rotation by 1"),
            // Test Case 2: No rotation (shift = 0)
            // imm = 0b1 has 0 trailing zeros, so no rotation.
            (0x12345678, 0b1, 10, 12, 0x12345678, "No rotation"),
            // Test Case 3: Rotation by 4 bits
            // imm = 0b10000 has 4 trailing zeros, so rotate right by 4.
            // 0x12345678 rotated right by 4 = 0x81234567
            (0x12345678, 0b10000, 10, 12, 0x81234567, "Rotation by 4"),
            // Test Case 4: Rotation by 16 bits (half word)
            // imm = 1 << 16 has 16 trailing zeros, so rotate right by 16.
            // 0x12345678 rotated right by 16 = 0x56781234
            (0x12345678, 1u64 << 16, 10, 12, 0x56781234, "Rotation by 16"),
            // Test Case 5: Rotation by 31 bits (almost full rotation)
            // imm = 1 << 31 has 31 trailing zeros, so rotate right by 31.
            // 0x80000001 rotated right by 31 = 0x00000003
            (
                0x80000001u32 as i32 as i64,
                1u64 << 31,
                10,
                12,
                0x00000003,
                "Rotation by 31",
            ),
            // Test Case 6: Full rotation (32 bits) - should be same as original
            // imm = 1 << 32 has 32 trailing zeros, so rotate right by 32.
            (
                0x12345678,
                1u64 << 32,
                10,
                12,
                0x12345678,
                "Full rotation (32 bits)",
            ),
            // Test Case 7: Sign extension behavior
            // 0x00000001 rotated right by 1 = 0x80000000
            (0x00000001, 0b10, 10, 12, 0x80000000, "Sign extension"),
            // Test Case 8: Register aliasing (destination equals source)
            // imm = 0b1000 has 3 trailing zeros, so rotate right by 3.
            // 0xAAAAAAAA rotated right by 3 = 0x55555555
            (
                0xAAAAAAAAu32 as i32 as i64,
                0b1000,
                10,
                10,
                0x55555555,
                "Register aliasing",
            ),
        ];

        for (i, &(rs1_val, imm, rs1, rd, expected, msg)) in test_cases.iter().enumerate() {
            let result = execute_and_read_rotri(&mut cpu, rs1_val, imm, rs1, rd);
            assert_eq!(result, expected, "Test case {} failed: {}", i, msg);
        }
    }
}
