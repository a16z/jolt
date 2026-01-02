use serde::{Deserialize, Serialize};

use crate::emulator::cpu::GeneralizedCpu;
use crate::emulator::memory::MemoryData;
use crate::instruction::format::format_virtual_right_shift_i::FormatVirtualRightShiftI;
use crate::{declare_riscv_instr, emulator::cpu::Xlen};

use super::{RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name = VirtualROTRI,
    mask = 0,
    match = 0,
    format = FormatVirtualRightShiftI,
    ram = ()
);

impl VirtualROTRI {
    fn exec<D: MemoryData>(
        &self,
        cpu: &mut GeneralizedCpu<D>,
        _: &mut <VirtualROTRI as RISCVInstruction>::RAMAccess,
    ) {
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
