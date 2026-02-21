use common::constants::XLEN;
use serde::{Deserialize, Serialize};

use crate::instruction::format::format_virtual_right_shift_i::FormatVirtualRightShiftI;
use crate::{declare_riscv_instr, emulator::cpu::Cpu, emulator::cpu::Xlen};

use super::{RISCVInstruction, RISCVTrace};

// Note, unlike ROTIW from Zbb extension of RiscV, ROTRIW does not sign extend the result
declare_riscv_instr!(
    name = VirtualROTRIW,
    mask = 0,
    match = 0,
    format = FormatVirtualRightShiftI,
    ram = ()
);

impl VirtualROTRIW {
    fn exec(&self, cpu: &mut Cpu, _: &mut <VirtualROTRIW as RISCVInstruction>::RAMAccess) {
        // Extract rotation amount from bitmask: trailing zeros = rotation amount
        let shift = self.operands.imm.trailing_zeros().min(XLEN as u32 / 2);

        // Rotate right by `shift` in lower 32bits width (matches ROTRI semantics)
        let rotated = match cpu.xlen {
            Xlen::Bit32 => {
                panic!("ROTRIW is not supported in 32-bit mode");
            }
            Xlen::Bit64 => {
                let val = cpu.x[self.operands.rs1 as usize] as u64 as u32;
                val.rotate_right(shift)
            }
        };

        cpu.write_register(self.operands.rd as usize, rotated as i64);
    }
}

impl RISCVTrace for VirtualROTRIW {}
