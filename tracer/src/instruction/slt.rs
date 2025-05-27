use serde::{Deserialize, Serialize};

use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::{
    format::{format_r::FormatR, InstructionFormat},
    RISCVInstruction, RISCVTrace,
};

declare_riscv_instr!(
    name   = SLT,
    mask   = 0xfe00707f,
    match  = 0x00002033,
    format = FormatR,
    ram    = ()
);

impl SLT {
    fn exec(&self, cpu: &mut Cpu, _: &mut <SLT as RISCVInstruction>::RAMAccess) {
        cpu.x[self.operands.rd] = match cpu.x[self.operands.rs1] < cpu.x[self.operands.rs2] {
            true => 1,
            false => 0,
        };
    }
}

impl RISCVTrace for SLT {}
