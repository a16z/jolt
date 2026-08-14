use serde::{Deserialize, Serialize};

use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::{format::format_r::FormatR, RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name   = ADDW,
    mask   = 0xfe00707f,
    match  = 0x0000003b,
    format = FormatR,
    ram    = ()
);

impl ADDW {
    fn exec(&self, cpu: &mut Cpu, _: &mut <ADDW as RISCVInstruction>::RAMAccess) {
        cpu.write_register(
            self.operands.rd as usize,
            cpu.x[self.operands.rs1 as usize].wrapping_add(cpu.x[self.operands.rs2 as usize]) as i32
                as i64,
        );
    }
}

impl RISCVTrace for ADDW {}
