use serde::{Deserialize, Serialize};

use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::{
    format::{format_r::FormatR, InstructionFormat},
    RISCVInstruction, RISCVTrace,
};

declare_riscv_instr!(
    name   = ADD,
    mask   = 0xfe00707f,
    match  = 0x00000033,
    format = FormatR,
    ram    = ()
);

impl ADD {
    fn exec(&self, cpu: &mut Cpu, _: &mut <ADD as RISCVInstruction>::RAMAccess) {
        cpu.x[self.operands.rd] =
            cpu.sign_extend(cpu.x[self.operands.rs1].wrapping_add(cpu.x[self.operands.rs2]));
    }
}

impl RISCVTrace for ADD {}
