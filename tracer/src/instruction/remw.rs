use super::{
    format::{format_r::FormatR, InstructionFormat},
    RISCVInstruction, RISCVTrace,
};
use crate::{declare_riscv_instr, emulator::cpu::Cpu};
use serde::{Deserialize, Serialize};

declare_riscv_instr!(
    name   = REMW,
    mask   = 0xfe00707f,
    match  = 0x1e00003b,
    format = FormatR,
    ram    = ()
);

impl REMW {
    fn exec(&self, cpu: &mut Cpu, _: &mut <REMW as RISCVInstruction>::RAMAccess) {
        let dividend = cpu.x[self.operands.rs1] as i32;
        let divisor = cpu.x[self.operands.rs2] as i32;
        cpu.x[self.operands.rd] = if divisor == 0 {
            dividend as i64
        } else if dividend == i32::MIN && divisor == -1 {
            0
        } else {
            (dividend.wrapping_rem(divisor)) as i64
        };
    }
}
impl RISCVTrace for REMW {}
