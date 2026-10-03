use crate::instruction::registers::r::RegisterStateR;
use crate::{declare_riscv_instr, emulator::cpu::Cpu};
use serde::{Deserialize, Serialize};

use super::{format::format_r::FormatR, RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name   = SUBW,
    mask   = 0xfe00707f,
    match  = 0x4000003b,
    format = FormatR,
    registers = RegisterStateR,
    ram    = ()
);

impl SUBW {
    fn exec(&self, cpu: &mut Cpu, _: &mut <SUBW as RISCVInstruction>::RAMAccess) {
        cpu.write_register(
            self.operands.rd as usize,
            (cpu.x[self.operands.rs1 as usize].wrapping_sub(cpu.x[self.operands.rs2 as usize])
                as i32) as i64,
        );
    }
}

impl RISCVTrace for SUBW {}
