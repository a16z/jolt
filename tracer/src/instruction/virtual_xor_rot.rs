/// Virtual XOR-Rotate Instructions
///
/// This module implements virtual instructions that perform: (a XOR b) rotate_right(constant)
/// In general, this requires three inputs of the instruction. As rotations are constant and limited,
/// we define a new instruction for each constant value and keep working on two inputs.
///
use super::{format::format_r::FormatR, RISCVInstruction, RISCVTrace};
use crate::{declare_riscv_instr, emulator::cpu::Cpu};
use serde::{Deserialize, Serialize};

// Macro to generate virtual XOR-rotate instructions for 64-bit values
macro_rules! declare_virtual_xor_rot_64 {
    ($name:ident, $shift:expr) => {
        declare_riscv_instr!(
            name = $name,
            mask = 0,
            match = 0,
            format = FormatR,
            ram = (),
            is_virtual = true
        );

        impl $name {
            fn exec(&self, cpu: &mut Cpu, _: &mut <$name as RISCVInstruction>::RAMAccess) {
                let xor_result = cpu.x[self.operands.rs1 as usize] ^ cpu.x[self.operands.rs2 as usize];
                let rotated = (xor_result).rotate_right($shift);
                cpu.x[self.operands.rd as usize] = rotated as i64;
            }
        }

        impl RISCVTrace for $name {}
    };
}

// Generate the 4 virtual XOR-rotate instructions for 64-bit values
declare_virtual_xor_rot_64!(VirtualXORROT32, 32);
declare_virtual_xor_rot_64!(VirtualXORROT24, 24);
declare_virtual_xor_rot_64!(VirtualXORROT16, 16);
declare_virtual_xor_rot_64!(VirtualXORROT63, 63);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::emulator::{cpu::Cpu, default_terminal::DefaultTerminal};
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    // Macro to generate tests for 64-bit rotation variations with random inputs
    macro_rules! test_virtual_xor_rot_64 {
        ($test_name:ident, $instr_type:ident, $shift:expr) => {
            #[test]
            fn $test_name() {
                // Use a fixed seed for reproducibility
                let mut rng = StdRng::seed_from_u64(42);

                // Test with 100 random inputs
                for i in 0..100 {
                    let mut cpu = Cpu::new(Box::new(DefaultTerminal::default()));

                    // Generate random 64-bit values
                    let val1: u64 = rng.gen();
                    let val2: u64 = rng.gen();

                    cpu.x[1] = val1 as i64;
                    cpu.x[2] = val2 as i64;

                    let instr = $instr_type {
                        address: 0,
                        operands: FormatR {
                            rd: 3,
                            rs1: 1,
                            rs2: 2,
                        },
                        inline_sequence_remaining: None,
                        is_compressed: false,
                    };

                    instr.exec(&mut cpu, &mut ());

                    // Expected: full 64-bit XOR, then rotate_right($shift)
                    let expected_xor = val1 ^ val2;
                    let expected_result = expected_xor.rotate_right($shift);
                    assert_eq!(
                        cpu.x[3] as u64, expected_result,
                        "Test failed at iteration {} with inputs: val1={:#x}, val2={:#x}",
                        i, val1, val2
                    );
                }
            }
        };
    }

    // Generate tests for all 4 64-bit rotation variations
    test_virtual_xor_rot_64!(test_virtual_xor_rot32, VirtualXORROT32, 32);
    test_virtual_xor_rot_64!(test_virtual_xor_rot24, VirtualXORROT24, 24);
    test_virtual_xor_rot_64!(test_virtual_xor_rot16_64, VirtualXORROT16, 16);
    test_virtual_xor_rot_64!(test_virtual_xor_rot63, VirtualXORROT63, 63);

    // Macro to generate edge case tests for 64-bit rotation variations
    macro_rules! test_virtual_xor_rot_64_edge_cases {
        ($test_name:ident, $instr_type:ident, $shift:expr) => {
            #[test]
            fn $test_name() {
                let mut rng = StdRng::seed_from_u64(1337);

                // Helper function to test a single case
                let test_case = |val1: u64, val2: u64, case_name: &str| {
                    let mut cpu = Cpu::new(Box::new(DefaultTerminal::default()));
                    cpu.x[1] = val1 as i64;
                    cpu.x[2] = val2 as i64;

                    let instr = $instr_type {
                        address: 0,
                        operands: FormatR {
                            rd: 3,
                            rs1: 1,
                            rs2: 2,
                        },
                        inline_sequence_remaining: None,
                        is_compressed: false,
                    };

                    instr.exec(&mut cpu, &mut ());

                    let expected_xor = val1 ^ val2;
                    let expected_result = expected_xor.rotate_right($shift);
                    assert_eq!(
                        cpu.x[3] as u64, expected_result,
                        "Edge case '{}' failed with inputs: val1={:#x}, val2={:#x}",
                        case_name, val1, val2
                    );
                };

                // Test case 1: Both zero
                test_case(0u64, 0u64, "both zero");

                // Test case 2: Both max
                test_case(u64::MAX, u64::MAX, "both max");

                // Test case 3: One zero, one random
                test_case(0u64, rng.gen(), "zero and random");

                // Test case 4: One random, one zero
                test_case(rng.gen(), 0u64, "random and zero");

                // Test case 5: One max, one random
                test_case(u64::MAX, rng.gen(), "max and random");

                // Test case 6: One random, one max
                test_case(rng.gen(), u64::MAX, "random and max");

                // Test case 7: Alternating bit patterns
                let pattern = rng.gen::<u8>() as u64;
                test_case(
                    pattern.wrapping_mul(0x0101010101010101),
                    pattern.wrapping_mul(0x0101010101010101),
                    "alternating bit patterns",
                );

                // Test case 8: Complementary values
                let val: u64 = rng.gen();
                test_case(val, !val, "complementary values");

                // Test case 9: Powers of 2
                let shift1 = rng.gen_range(0..64);
                let shift2 = rng.gen_range(0..64);
                test_case(1u64 << shift1, 1u64 << shift2, "powers of 2");

                // Test case 10: Random values
                test_case(rng.gen(), rng.gen(), "random values");
            }
        };
    }

    // Generate edge case tests for all 64-bit rotation variations
    test_virtual_xor_rot_64_edge_cases!(test_virtual_xor_rot32_edge, VirtualXORROT32, 32);
    test_virtual_xor_rot_64_edge_cases!(test_virtual_xor_rot24_edge, VirtualXORROT24, 24);
    test_virtual_xor_rot_64_edge_cases!(test_virtual_xor_rot16_64_edge, VirtualXORROT16, 16);
    test_virtual_xor_rot_64_edge_cases!(test_virtual_xor_rot63_edge, VirtualXORROT63, 63);
}
