use serde::{Deserialize, Serialize};

use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::{
    format::{format_r::FormatR, InstructionFormat},
    RISCVInstruction, RISCVTrace,
};

// Macro to generate virtual XOR-rotate instructions
macro_rules! declare_virtual_rot_xor {
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
declare_virtual_rot_xor!(VirtualROTXOR16, 16);
declare_virtual_rot_xor!(VirtualROTXOR12, 12);
declare_virtual_rot_xor!(VirtualROTXOR8, 8);
declare_virtual_rot_xor!(VirtualROTXOR7, 7);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::emulator::{cpu::{Cpu}, default_terminal::DefaultTerminal};

    #[test]
    fn test_virtual_rot_xor16() {
        let mut cpu = Cpu::new(Box::new(DefaultTerminal::new()));
        cpu.x[1] = 0x12345678;
        cpu.x[2] = 0xabcdef00;
        
        let instr = VirtualROTXOR16 {
            address: 0,
            operands: FormatR { rd: 3, rs1: 1, rs2: 2 },
            virtual_sequence_remaining: None,
        };
        
        instr.exec(&mut cpu, &mut ());
        
        // Expected: (0x12345678 ^ 0xabcdef00) as u32, then rotate_right(16)
        let expected_xor = (0x12345678u64 ^ 0xabcdef00u64) as u32;
        let expected_result = expected_xor.rotate_right(16) as u64;
        assert_eq!(cpu.x[3] as u64, expected_result);
        // Verify upper 32 bits are cleared
        assert_eq!(cpu.x[3] >> 32, 0);
    }
}
