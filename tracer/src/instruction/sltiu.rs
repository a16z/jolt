use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::{cpu::GeneralizedCpu, memory::MemoryData},
};

use super::{
    format::{format_i::FormatI, normalize_imm},
    RISCVInstruction, RISCVTrace,
};

declare_riscv_instr!(
    name   = SLTIU,
    mask   = 0x0000707f,
    match  = 0x00003013,
    format = FormatI,
    ram    = ()
);

impl SLTIU {
    fn exec<D: MemoryData>(
        &self,
        cpu: &mut GeneralizedCpu<D>,
        _: &mut <SLTIU as RISCVInstruction>::RAMAccess,
    ) {
        cpu.x[self.operands.rd as usize] = match cpu
            .unsigned_data(cpu.x[self.operands.rs1 as usize])
            < cpu.unsigned_data(normalize_imm(self.operands.imm, &cpu.xlen))
        {
            true => 1,
            false => 0,
        };
    }
}

impl RISCVTrace for SLTIU {}
