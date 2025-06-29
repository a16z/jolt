use super::{
    format::{format_r::FormatR, InstructionFormat},
    RISCVInstruction, RISCVTrace,
};
use crate::{declare_riscv_instr, emulator::cpu::Cpu};
use serde::{Deserialize, Serialize};

declare_riscv_instr!(
    name   = REMUW,
    mask   = 0xfe00707f,
    match  = 0x1f00003b,
    format = FormatR,
    ram    = ()
);

impl REMUW {
    fn exec(&self, cpu: &mut Cpu, _: &mut <REMUW as RISCVInstruction>::RAMAccess) {
        // REMW and REMUW are RV64 instructions that provide the corresponding signed and unsigned
        // remainder operations. Both REMW and REMUW always sign-extend the 32-bit result to 64 bits,
        // including on a divide by zero.
        let dividend = cpu.x[self.operands.rs1] as u32;
        let divisor = cpu.x[self.operands.rs2] as u32;
        cpu.x[self.operands.rd] = (if divisor == 0 {
            dividend
        } else {
            dividend.wrapping_rem(divisor)
        }) as i32 as i64;
    }
}
impl RISCVTrace for REMUW {}
