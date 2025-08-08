use serde::{Deserialize, Serialize};

use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::{
    format::{format_b::FormatB, InstructionFormat},
    RISCVInstruction, RISCVTrace,
};

declare_riscv_instr!(
    name   = BLTU,
    mask   = 0x0000707f,
    match  = 0x00006063,
    format = FormatB,
    ram    = ()
);

impl BLTU {
    fn exec(&self, cpu: &mut Cpu, _: &mut <BLTU as RISCVInstruction>::RAMAccess) {
        if cpu.unsigned_data(cpu.x[self.operands.rs1 as usize])
            < cpu.unsigned_data(cpu.x[self.operands.rs2 as usize])
        {
            cpu.pc = (self.address as i64 + self.operands.imm as i64) as u64;
        }
    }
}

impl RISCVTrace for BLTU {}
