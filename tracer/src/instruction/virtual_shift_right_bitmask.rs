use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
};

use super::{
    format::{format_i::FormatI, InstructionFormat},
    RISCVInstruction, RISCVTrace,
};

declare_riscv_instr!(
    name = VirtualShiftRightBitmask,
    mask = 0,
    match = 0,
    format = FormatI,
    ram = (),
    is_virtual = true
);

impl VirtualShiftRightBitmask {
    fn exec(
        &self,
        cpu: &mut Cpu,
        _: &mut <VirtualShiftRightBitmask as RISCVInstruction>::RAMAccess,
    ) {
        match cpu.xlen {
            Xlen::Bit32 => {
                let shift = cpu.x[self.operands.rs1] as u64 & 0x1F;
                let ones = (1u64 << (32 - shift)) - 1;
                cpu.x[self.operands.rd] = (ones << shift) as i64;
            }
            Xlen::Bit64 => {
                let shift = cpu.x[self.operands.rs1] as u64 & 0x3F;
                let ones = (1u128 << (64 - shift)) - 1;
                cpu.x[self.operands.rd] = (ones << shift) as i64;
            }
        }
    }
}

impl RISCVTrace for VirtualShiftRightBitmask {}
