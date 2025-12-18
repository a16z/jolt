use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::{cpu::GeneralizedCpu, memory::MemoryData},
};

use super::{format::format_r::FormatR, RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name   = AND,
    mask   = 0xfe00707f,
    match  = 0x00007033,
    format = FormatR,
    ram    = ()
);

impl AND {
    fn exec<D: MemoryData>(
        &self,
        cpu: &mut GeneralizedCpu<D>,
        _: &mut <AND as RISCVInstruction>::RAMAccess,
    ) {
        cpu.x[self.operands.rd as usize] =
            cpu.sign_extend(cpu.x[self.operands.rs1 as usize] & cpu.x[self.operands.rs2 as usize]);
    }
}

impl RISCVTrace for AND {}
