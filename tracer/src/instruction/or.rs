use serde::{Deserialize, Serialize};

use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::{
    format::{format_r::FormatR, InstructionFormat},
    RISCVInstruction, RISCVTrace,
};

declare_riscv_instr!(
    name   = OR,
    mask   = 0xfe00707f,
    match  = 0x00006033,
    format = FormatR,
    ram    = ()
);

impl OR {
    fn exec(&self, cpu: &mut Cpu, _: &mut <OR as RISCVInstruction>::RAMAccess) {
        cpu.x[self.operands.rd as usize] =
            cpu.sign_extend(cpu.x[self.operands.rs1 as usize] | cpu.x[self.operands.rs2 as usize]);
    }
}

impl RISCVTrace for OR {}
