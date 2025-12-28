use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::{
        cpu::{GeneralizedCpu, Xlen},
        memory::MemoryData,
    },
};

use super::{format::format_i::FormatI, RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name = VirtualPow2,
    mask = 0,
    match = 0,
    format = FormatI,
    ram = ()
);

impl VirtualPow2 {
    fn exec<D: MemoryData>(
        &self,
        cpu: &mut GeneralizedCpu<D>,
        _: &mut <VirtualPow2 as RISCVInstruction>::RAMAccess,
    ) {
        match cpu.xlen {
            Xlen::Bit32 => {
                cpu.x[self.operands.rd as usize] =
                    1 << (cpu.x[self.operands.rs1 as usize] as u64 % 32)
            }
            Xlen::Bit64 => {
                cpu.x[self.operands.rd as usize] =
                    1 << (cpu.x[self.operands.rs1 as usize] as u64 % 64)
            }
        }
    }
}

impl RISCVTrace for VirtualPow2 {}
