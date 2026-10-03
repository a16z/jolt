use crate::instruction::registers::r::RegisterStateR;
use serde::{Deserialize, Serialize};

use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::{format::format_r::FormatR, RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name   = MULW,
    mask   = 0xfe00707f,
    match  = 0x0200003b,
    format = FormatR,
    registers = RegisterStateR,
    ram    = ()
);

impl MULW {
    fn exec(&self, cpu: &mut Cpu, _: &mut <MULW as RISCVInstruction>::RAMAccess) {
        let a = cpu.x[self.operands.rs1 as usize] as i32;
        let b = cpu.x[self.operands.rs2 as usize] as i32;
        cpu.write_register(self.operands.rd as usize, a.wrapping_mul(b) as i64);
    }
}

impl RISCVTrace for MULW {}
