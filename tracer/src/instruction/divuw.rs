use super::{
    format::{format_r::FormatR, InstructionFormat},
    RISCVInstruction, RISCVTrace,
};
use crate::{declare_riscv_instr, emulator::cpu::Cpu};
use serde::{Deserialize, Serialize};

declare_riscv_instr!(
    name   = DIVUW,
    mask   = 0xfe00707f,
    match  = 0x1b00003b,
    format = FormatR,
    ram    = ()
);

impl DIVUW {
    fn exec(&self, cpu: &mut Cpu, _: &mut <DIVUW as RISCVInstruction>::RAMAccess) {
        let dividend = cpu.x[self.operands.rs1] as u32;
        let divisor = cpu.x[self.operands.rs2] as u32;
        cpu.x[self.operands.rd] = (if divisor == 0 {
            u32::MAX
        } else {
            dividend.wrapping_div(divisor)
        }) as i32 as i64;
    }
}
impl RISCVTrace for DIVUW {}
