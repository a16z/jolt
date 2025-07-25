use serde::{Deserialize, Serialize};

use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::{
    format::{format_r::FormatR, InstructionFormat},
    RISCVInstruction, RISCVTrace,
};

// Macro to generate virtual XOR-rotate instructions
macro_rules! declare_virtual_rot_xor_lower {
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
                // XOR the two registers
                let xor_result = cpu.x[self.operands.rs1] ^ cpu.x[self.operands.rs2];
                // Extract only the lower 32-bit value
                let xor_result_32 = xor_result as u32;
                // Rotate only within the 32 bits
                let rotated_32 = xor_result_32.rotate_right($shift);
                // Store result with upper 32 bits cleared (no sign extension needed)
                cpu.x[self.operands.rd] = rotated_32 as u64 as i64;
            }
        }

        impl RISCVTrace for $name {}
    };
}

// Generate the 4 virtual XOR-rotate instructions
declare_virtual_rot_xor_lower!(VirtualROTXOR16L, 16);
declare_virtual_rot_xor_lower!(VirtualROTXOR12L, 12);
declare_virtual_rot_xor_lower!(VirtualROTXOR8L, 8);
declare_virtual_rot_xor_lower!(VirtualROTXOR7L, 7);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::emulator::{cpu::{Cpu}, default_terminal::DefaultTerminal};

    // Macro to generate tests for different rotation variations
    macro_rules! test_virtual_rot_xor_l {
        ($test_name:ident, $instr_type:ident, $shift:expr) => {
            #[test]
            fn $test_name() {
                let mut cpu = Cpu::new(Box::new(DefaultTerminal::new()));
                cpu.x[1] = 0x12345678;
                cpu.x[2] = 0xabcdef00;
                
                let instr = $instr_type {
                    address: 0,
                    operands: FormatR { rd: 3, rs1: 1, rs2: 2 },
                    virtual_sequence_remaining: None,
                };
                
                instr.exec(&mut cpu, &mut ());
                
                // Expected: (0x12345678 ^ 0xabcdef00) as u32, then rotate_right($shift)
                let expected_xor = (0x12345678u64 ^ 0xabcdef00u64) as u32;
                let expected_result = expected_xor.rotate_right($shift) as u64;
                assert_eq!(cpu.x[3] as u64, expected_result);
                // Verify upper 32 bits are cleared
                assert_eq!(cpu.x[3] >> 32, 0);
            }
        };
    }

    // Generate tests for all 4 rotation variations
    test_virtual_rot_xor_l!(test_virtual_rot_xor16l, VirtualROTXOR16L, 16);
    test_virtual_rot_xor_l!(test_virtual_rot_xor12l, VirtualROTXOR12L, 12);
    test_virtual_rot_xor_l!(test_virtual_rot_xor8l, VirtualROTXOR8L, 8);
    test_virtual_rot_xor_l!(test_virtual_rot_xor7l, VirtualROTXOR7L, 7);

    // Edge case tests
    #[test]
    fn test_virtual_rot_xor_edge_cases() {
        let mut cpu = Cpu::new(Box::new(DefaultTerminal::new()));
        
        // Test with zero values
        cpu.x[1] = 0;
        cpu.x[2] = 0;
        let instr = VirtualROTXOR16L {
            address: 0,
            operands: FormatR { rd: 3, rs1: 1, rs2: 2 },
            virtual_sequence_remaining: None,
        };
        instr.exec(&mut cpu, &mut ());
        assert_eq!(cpu.x[3], 0);
        
        // Test with max u32 values
        cpu.x[1] = 0xFFFFFFFF;
        cpu.x[2] = 0xFFFFFFFF;
        instr.exec(&mut cpu, &mut ());
        assert_eq!(cpu.x[3], 0); // XOR of same values is 0
        
        // Test with one register being zero
        cpu.x[1] = 0x12345678;
        cpu.x[2] = 0;
        instr.exec(&mut cpu, &mut ());
        let expected = (0x12345678u32).rotate_right(16) as u64;
        assert_eq!(cpu.x[3] as u64, expected);
        // Verify upper 32 bits are cleared
        assert_eq!(cpu.x[3] >> 32, 0);
        
        // Test with alternating bit patterns
        cpu.x[1] = 0xAAAAAAAA; // 10101010...
        cpu.x[2] = 0x55555555; // 01010101...
        instr.exec(&mut cpu, &mut ());
        let expected_xor = (0xAAAAAAAAu64 ^ 0x55555555u64) as u32; // Should be all 1s
        let expected_result = expected_xor.rotate_right(16) as u64;
        assert_eq!(cpu.x[3] as u64, expected_result);
        assert_eq!(cpu.x[3] >> 32, 0);
        
        // Test with high bits set in 64-bit registers (should be ignored)
        cpu.x[1] = 0xFFFFFFFF12345678u64 as i64;
        cpu.x[2] = 0xFFFFFFFFabcdef00u64 as i64;
        instr.exec(&mut cpu, &mut ());
        // Should only use lower 32 bits: 0x12345678 ^ 0xabcdef00
        let expected_xor_high = (0x12345678u64 ^ 0xabcdef00u64) as u32;
        let expected_result_high = expected_xor_high.rotate_right(16) as u64;
        assert_eq!(cpu.x[3] as u64, expected_result_high);
        assert_eq!(cpu.x[3] >> 32, 0);
    }
}
