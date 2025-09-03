use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
};

use super::{format::format_i::FormatI, RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name = VirtualSignExtendWord,
    mask = 0,
    match = 0,
    format = FormatI,
    ram = (),
    is_virtual = true
);

impl VirtualSignExtendWord {
    fn exec(&self, cpu: &mut Cpu, _: &mut <VirtualSignExtendWord as RISCVInstruction>::RAMAccess) {
        match cpu.xlen {
            Xlen::Bit32 => panic!("VirtualSignExtend is not supported for 32-bit mode"),
            Xlen::Bit64 => {
                cpu.x[self.operands.rd as usize] = (cpu.x[self.operands.rs1 as usize] << 32) >> 32
            }
        }
    }
}

impl RISCVTrace for VirtualSignExtendWord {}
