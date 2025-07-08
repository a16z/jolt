use serde::{Deserialize, Serialize};

use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::{
    format::{format_u::FormatU, InstructionFormat},
    RISCVInstruction, RISCVTrace,
};

declare_riscv_instr!(
    name   = LUI,
    mask   = 0x0000007f,
    match  = 0x00000037,
    format = FormatU,
    ram    = ()
);

impl LUI {
    fn exec(&self, cpu: &mut Cpu, _: &mut <LUI as RISCVInstruction>::RAMAccess) {
        cpu.x[self.operands.rd] = self.operands.imm as i64;
    }
}

impl RISCVTrace for LUI {}
