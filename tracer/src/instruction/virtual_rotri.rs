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
