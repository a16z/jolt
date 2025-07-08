use serde::{Deserialize, Serialize};

use crate::instruction::format::format_virtual_left_shift_i::FormatVirtualLeftShiftI;
use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::{format::InstructionFormat, RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name = VirtualROTLI,
    mask = 0,
    match = 0,
    format = FormatVirtualLeftShiftI,
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
