use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::{
        cpu::{GeneralizedCpu, Xlen},
        memory::MemoryData,
    },
};

use super::{format::format_j::FormatJ, RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name = VirtualPow2IW,
    mask = 0,
    match = 0,
    format = FormatJ,
    ram = ()
);

impl VirtualPow2IW {
    fn exec<D: MemoryData>(
        &self,
        cpu: &mut GeneralizedCpu<D>,
        _: &mut <VirtualPow2IW as RISCVInstruction>::RAMAccess,
    ) {
        match cpu.xlen {
            Xlen::Bit32 => panic!("VirtualPow2IW is invalid in 32b mode"),
            Xlen::Bit64 => cpu.x[self.operands.rd as usize] = 1 << (self.operands.imm % 32),
        }
    }
}

impl RISCVTrace for VirtualPow2IW {}
