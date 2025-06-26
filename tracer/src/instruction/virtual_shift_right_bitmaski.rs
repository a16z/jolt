use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
};

use super::{
    format::{format_j::FormatJ, InstructionFormat},
    RISCVInstruction, RISCVTrace,
};

declare_riscv_instr!(
    name = VirtualShiftRightBitmaskI,
    mask = 0,
    match = 0,
    format = FormatJ,
    ram = (),
    is_virtual = true
);

impl VirtualShiftRightBitmaskI {
    fn exec(
        &self,
        cpu: &mut Cpu,
        _: &mut <VirtualShiftRightBitmaskI as RISCVInstruction>::RAMAccess,
    ) {
        match cpu.xlen {
            Xlen::Bit32 => {
                let shift = self.operands.imm % 32;
                let ones = (1u64 << (32 - shift)) - 1;
                cpu.x[self.operands.rd] = (ones << shift) as i64;
            }
            Xlen::Bit64 => {
                let shift = self.operands.imm % 64;
                let ones = (1u128 << (64 - shift)) - 1;
                cpu.x[self.operands.rd] = (ones << shift) as i64;
            }
        }
    }
}

impl RISCVTrace for VirtualShiftRightBitmaskI {}
