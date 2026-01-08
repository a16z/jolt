use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::cpu::Cpu,
};

use super::{format::format_r::FormatR, RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name   = SLTU,
    mask   = 0xfe00707f,
    match  = 0x00003033,
    format = FormatR,
    ram    = ()
);

impl SLTU {
    fn exec(
        &self,
        cpu: &mut Cpu,
        _: &mut <SLTU as RISCVInstruction>::RAMAccess,
    ) {
        cpu.x[self.operands.rd as usize] = match cpu
            .unsigned_data(cpu.x[self.operands.rs1 as usize])
            < cpu.unsigned_data(cpu.x[self.operands.rs2 as usize])
        {
            true => 1,
            false => 0,
        };
    }
}

impl RISCVTrace for SLTU {}
