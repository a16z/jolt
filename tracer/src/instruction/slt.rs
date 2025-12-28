use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::{cpu::GeneralizedCpu, memory::MemoryData},
};

use super::{format::format_r::FormatR, RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name   = SLT,
    mask   = 0xfe00707f,
    match  = 0x00002033,
    format = FormatR,
    ram    = ()
);

impl SLT {
    fn exec<D: MemoryData>(
        &self,
        cpu: &mut GeneralizedCpu<D>,
        _: &mut <SLT as RISCVInstruction>::RAMAccess,
    ) {
        cpu.x[self.operands.rd as usize] =
            match cpu.x[self.operands.rs1 as usize] < cpu.x[self.operands.rs2 as usize] {
                true => 1,
                false => 0,
            };
    }
}

impl RISCVTrace for SLT {}
