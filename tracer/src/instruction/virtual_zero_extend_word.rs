use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
};

use super::{format::format_i::FormatI, RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name = VirtualZeroExtendWord,
    mask = 0,
    match = 0,
    format = FormatI,
    ram = ()
);

impl VirtualZeroExtendWord {
    fn exec(&self, cpu: &mut Cpu, _: &mut <VirtualZeroExtendWord as RISCVInstruction>::RAMAccess) {
        match cpu.xlen {
            Xlen::Bit32 => panic!("VirtualExtend is not supported for 32-bit mode"),
            Xlen::Bit64 => {
                cpu.x[self.operands.rd as usize] = cpu.x[self.operands.rs1 as usize] & 0xFFFFFFFF
            }
        }
    }
}

impl RISCVTrace for VirtualZeroExtendWord {}
