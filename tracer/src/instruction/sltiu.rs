use serde::{Deserialize, Serialize};

use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::{
    format::{format_i::FormatI, normalize_imm, InstructionFormat},
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
    fn exec(&self, cpu: &mut Cpu, _: &mut <SLTIU as RISCVInstruction>::RAMAccess) {
        cpu.x[self.operands.rd] = match cpu.unsigned_data(cpu.x[self.operands.rs1])
            < cpu.unsigned_data(normalize_imm(self.operands.imm))
        {
            true => 1,
            false => 0,
        };
    }
}

impl RISCVTrace for SLTIU {}
