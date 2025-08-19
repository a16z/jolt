use serde::{Deserialize, Serialize};

use crate::instruction::format::format_virtual_right_shift_i::FormatVirtualRightShiftI;
use crate::{declare_riscv_instr, emulator::cpu::Cpu, emulator::cpu::Xlen};

use super::{RISCVInstruction, RISCVTrace};

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

        // Rotate right by `shift` respecting current XLEN width (matches ROTRI semantics)
        let rotated = match cpu.xlen {
            Xlen::Bit32 => {
                let val_32 = cpu.x[self.operands.rs1 as usize] as u32;
                val_32.rotate_right(shift) as i64
            }
            Xlen::Bit64 => {
                let val = cpu.x[self.operands.rs1 as usize];
                val.rotate_right(shift)
            }
        };

        cpu.x[self.operands.rd as usize] = cpu.sign_extend(rotated);
    }
}

impl RISCVTrace for VirtualROTRI {}

#[cfg(test)]
mod tests {
    use crate::emulator::cpu::Cpu;
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

    fn get_rotri(cpu: &mut Cpu, rs1_val: i64, imm: u64, rs1: u8, rd: u8) -> VirtualROTRI {
        cpu.x[rs1 as usize] = rs1_val;
        VirtualROTRI {
            address: DRAM_BASE,
            operands: FormatVirtualRightShiftI { rd, rs1, imm },
            inline_sequence_remaining: Some(0),
            is_compressed: false,
        }
    }

    fn trace_and_read_rotri(cpu: &mut Cpu, rs1_val: i64, imm: u64, rs1: u8, rd: u8) -> i64 {
        let instruction = get_rotri(cpu, rs1_val, imm, rs1, rd);
        let mut dummy: Vec<crate::instruction::RV32IMCycle> = Vec::new();
        instruction.trace(cpu, Some(&mut dummy));
        cpu.x[rd as usize]
    }

    /// Helper function to execute a VirtualROTRI instruction and return the result.
    fn exec_and_read_rotri(cpu: &mut Cpu, rs1_val: i64, imm: u64, rs1: u8, rd: u8) -> i64 {
        let instruction = get_rotri(cpu, rs1_val, imm, rs1, rd);
        instruction.exec(cpu, &mut ());
        cpu.x[rd as usize]
    }

    #[test]
    fn test_virtual_rotri() {
        let test_cases: [(i64, u64, u8, u8, i64, &'static str); 8] = [
            // Test Case 1: Simple rotation by 1 bit
            // imm = 0b10 has 1 trailing zero, so rotate right by 1.
            // 0x12345678 rotated right by 1 = 0x091A2B3C
            (0x12345678, 0b10, 10, 12, 0x091A2B3C, "Rotation by 1"),
            // Test Case 2: No rotation (shift = 0)
            // imm = 0b1 has 0 trailing zeros, so no rotation.
            (0x12345678, 0b1, 10, 12, 0x12345678, "No rotation"),
            // Test Case 3: Rotation by 4 bits
            // imm = 0b10000 has 4 trailing zeros, so rotate right by 4.
            // 0x12345678 rotated right by 4 = 0x81234567 (sign-extended)
            (
                0x12345678,
                0b10000,
                10,
                12,
                0x81234567u32 as i32 as i64,
                "Rotation by 4",
            ),
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
            // 0x00000001 rotated right by 1 = 0x80000000 (sign-extended)
            (
                0x00000001,
                0b10,
                10,
                12,
                0x80000000u32 as i32 as i64,
                "Sign extension",
            ),
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

        for xlen in [Xlen::Bit32, Xlen::Bit64] {
            let mut cpu = CpuTestHarness::new().cpu;
            cpu.update_xlen(xlen);

            for (i, &(rs1_val, imm, rs1, rd, expected_rv32, msg)) in test_cases.iter().enumerate() {
                let result = exec_and_read_rotri(&mut cpu, rs1_val, imm, rs1, rd);
                let shift = imm.trailing_zeros();
                let expected = match xlen {
                    Xlen::Bit32 => expected_rv32,
                    Xlen::Bit64 => (rs1_val as u64).rotate_right(shift) as i64,
                };

                assert_eq!(
                    result, expected,
                    "Test case {i} failed: {msg} (xlen={xlen:?})"
                );
            }
        }
    }

    use crate::emulator::cpu::Xlen;
    use crate::emulator::test_harness::CpuTestHarness;
    use crate::instruction::RISCVTrace;
    use rand::{rngs::StdRng, Rng, SeedableRng};

    /// Helper that returns a VirtualROTRI with random operands respecting XLEN
    fn random_rotri(rng: &mut StdRng, xlen_64: bool) -> (VirtualROTRI, i64) {
        let rs1 = 5; // arbitrary non-zero register
        let rd = 6; // different destination

        // random input value depending on XLEN
        let val = if xlen_64 {
            rng.gen::<i64>()
        } else {
            rng.gen::<u32>() as i32 as i64
        };

        // choose a random rotation amount within word size
        let shift = if xlen_64 {
            rng.gen_range(0..64)
        } else {
            rng.gen_range(0..32)
        } as u32;
        let word_size = if xlen_64 { 64 } else { 32 };

        // Build bitmask whose number of trailing zeros equals `shift`.
        // Formula: imm = ((1 << (word_size - shift)) - 1) << shift
        let ones: u64 = if shift == 0 {
            u64::MAX >> (64 - word_size) // all ones in word_size bits
        } else {
            (1u64 << (word_size - shift)) - 1
        };
        let imm = ones << shift;

        let instr = VirtualROTRI {
            address: 0x1000,
            operands: FormatVirtualRightShiftI { rd, rs1, imm },
            inline_sequence_remaining: Some(0),
            is_compressed: false,
        };

        (instr, val)
    }

    #[test]
    fn test_exec_vs_trace() {
        for xlen in [Xlen::Bit32, Xlen::Bit64] {
            let mut harness_exec = CpuTestHarness::new();
            let mut harness_trace = CpuTestHarness::new();

            // default is Bit64
            let mut rng = StdRng::seed_from_u64(42);
            for _ in 0..100 {
                let (rotri, val) = random_rotri(&mut rng, xlen == Xlen::Bit64);

                let result_exec = exec_and_read_rotri(
                    &mut harness_exec.cpu,
                    val,
                    rotri.operands.imm,
                    rotri.operands.rs1,
                    rotri.operands.rd,
                );
                let result_trace = trace_and_read_rotri(
                    &mut harness_trace.cpu,
                    val,
                    rotri.operands.imm,
                    rotri.operands.rs1,
                    rotri.operands.rd,
                );

                assert_eq!(
                    result_exec, result_trace,
                    "ROTRI(xlen={xlen:?}, val={val:?}): Mismatch between exec and trace",
                );
            }
        }
    }
}
