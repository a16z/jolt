use serde::{Deserialize, Serialize};

use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::{
    format::{format_i::FormatI, InstructionFormat},
    RISCVInstruction, RISCVTrace,
};

declare_riscv_instr!(
    name   = SLTI,
    mask   = 0x0000707f,
    match  = 0x00002013,
    format = FormatI,
    ram    = ()
);

impl SLTI {
    fn exec(&self, cpu: &mut Cpu, _: &mut <SLTI as RISCVInstruction>::RAMAccess) {
        cpu.x[self.operands.rd] = match cpu.x[self.operands.rs1] < self.operands.imm as i64 {
            true => 1,
            false => 0,
        };
    }
}

impl RISCVTrace for SLTI {}
