use super::{
    format::{format_r::FormatR, InstructionFormat},
    RISCVInstruction, RISCVTrace,
};
use crate::{declare_riscv_instr, emulator::cpu::Cpu};
use serde::{Deserialize, Serialize};

declare_riscv_instr!(
    name   = MULW,
    mask   = 0xfe00707f,
    match  = 0x0200003b,
    format = FormatR,
    ram    = ()
);

impl MULW {
    fn exec(&self, cpu: &mut Cpu, _: &mut <MULW as RISCVInstruction>::RAMAccess) {
        let a = cpu.x[self.operands.rs1] as i32 as i64;
        let b = cpu.x[self.operands.rs2] as i32 as i64;
        cpu.x[self.operands.rd] = a.wrapping_mul(b) as i32 as i64;
    }
}
impl RISCVTrace for MULW {}
