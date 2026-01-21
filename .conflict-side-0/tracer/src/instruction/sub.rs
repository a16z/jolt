use serde::{Deserialize, Serialize};

use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::{format::format_r::FormatR, RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name   = SUB,
    mask   = 0xfe00707f,
    match  = 0x40000033,
    format = FormatR,
    ram    = ()
);

impl SUB {
    fn exec(&self, cpu: &mut Cpu, _: &mut <SUB as RISCVInstruction>::RAMAccess) {
        cpu.x[self.operands.rd as usize] = cpu.sign_extend(
            cpu.x[self.operands.rs1 as usize].wrapping_sub(cpu.x[self.operands.rs2 as usize]),
        );
    }
}

impl RISCVTrace for SUB {}
