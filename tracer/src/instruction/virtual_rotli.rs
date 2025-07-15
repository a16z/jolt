use serde::{Deserialize, Serialize};

// We reuse FormatVirtualRightShiftI here because the operand structure (rd, rs1, imm)
// is identical for both left and right virtual rotations. The actual rotation
// direction is determined by the `exec` method of the instruction itself.
use crate::instruction::format::format_virtual_right_shift_i::{
    FormatVirtualRightShiftI, RegisterStateFormatVirtualI,
};
use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::{format::InstructionFormat, RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name = VirtualROTLI,
    mask = 0,
    match = 0,
    format = FormatVirtualRightShiftI,
    ram = (),
    is_virtual = true
);

impl VirtualROTLI {
    fn exec(&self, cpu: &mut Cpu, _: &mut <VirtualROTLI as RISCVInstruction>::RAMAccess) {
        // Extract rotation amount from bitmask: trailing zeros = rotation amount
        let shift = self.operands.imm.trailing_zeros();
        // Perform a full 64-bit rotation, as required by Keccak which uses 64-bit lanes.
        let val_64 = cpu.x[self.operands.rs1] as u64;
        let rotated_64 = val_64.rotate_left(shift);
        cpu.x[self.operands.rd] = rotated_64 as i64;
    }
}

impl RISCVTrace for VirtualROTLI {}

// TODO: Implement and debug properly.
// #[cfg(test)]
// mod test {
//     use super::*;
//     use crate::emulator::cpu::Cpu;
//     use crate::instruction::format::format_virtual_right_shift_i::FormatVirtualRightShiftI;

//     #[test]
//     fn test_virtual_rotli() {
//         // 1. Setup
//         let mut cpu = Cpu::new(0);
//         let mut ram_access = (); // Virtual instructions don't access RAM directly

//         // Test case: Rotate 0xAAAAAAAAAAAAAAAA by 1, expecting 0x5555555555555555
//         let initial_value: i64 = 0xAAAAAAAAAAAAAAAA_i64;
//         let rotation_amount = 1;
//         let expected_result = initial_value.rotate_left(rotation_amount);

//         // Create the instruction bitmask. The convention is that the rotation amount
//         // is the number of trailing zeros in the immediate.
//         let bitmask = (1u64 << (64 - rotation_amount)) - 1;
//         let instr = VirtualROTLI {
//             operands: FormatVirtualRightShiftI {
//                 rd: 1,
//                 rs1: 2,
//                 imm: bitmask.wrapping_shl(rotation_amount),
//             },
//             address: 0,
//             virtual_sequence_remaining: None,
//         };

//         // Set initial register state
//         cpu.x[2] = initial_value;
//         cpu.x[1] = 0; // Clear destination register

//         // 2. Execute
//         instr.exec(&mut cpu, &mut ram_access);

//         // 3. Assert
//         assert_eq!(
//             cpu.x[1], expected_result,
//             "Rotation failed: expected {:#x}, got {:#x}",
//             expected_result, cpu.x[1]
//         );

//         // --- Add another test case for good measure ---
//         // Test case: Rotate a different value by 12 bits
//         let initial_value_2: i64 = 0x123456789ABCDEF0;
//         let rotation_amount_2 = 12;
//         let expected_result_2 = initial_value_2.rotate_left(rotation_amount_2);

//         let bitmask_2 = (1u64 << (64 - rotation_amount_2)) - 1;
//         let instr_2 = VirtualROTLI {
//             operands: FormatVirtualRightShiftI {
//                 rd: 3,
//                 rs1: 4,
//                 imm: bitmask_2.wrapping_shl(rotation_amount_2),
//             },
//             address: 0,
//             virtual_sequence_remaining: None,
//         };

//         cpu.x[4] = initial_value_2;
//         cpu.x[3] = 0;

//         instr_2.exec(&mut cpu, &mut ram_access);

//         assert_eq!(
//             cpu.x[3], expected_result_2,
//             "Second rotation failed: expected {:#x}, got {:#x}",
//             expected_result_2, cpu.x[3]
//         );
//     }
// }
