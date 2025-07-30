use super::{
    format::{format_r::FormatR, InstructionFormat},
    RISCVInstruction, RISCVTrace,
};
use crate::{declare_riscv_instr, emulator::cpu::Cpu};
use serde::{Deserialize, Serialize};

declare_riscv_instr!(
    name   = DIVW,
    mask   = 0xfe00707f,
    match  = 0x1a00003b,
    format = FormatR,
    ram    = ()
);

impl DIVW {
    fn exec(&self, cpu: &mut Cpu, _: &mut <DIVW as RISCVInstruction>::RAMAccess) {
        // DIVW and DIVUW are RV64 instructions that divide the lower 32 bits of rs1 by the lower
        // 32 bits of rs2, treating them as signed and unsigned integers, placing the 32-bit
        // quotient in rd, sign-extended to 64 bits.
        let dividend = cpu.x[self.operands.rs1] as i32;
        let divisor = cpu.x[self.operands.rs2] as i32;
        cpu.x[self.operands.rd] = (if divisor == 0 {
            -1i32
        } else if dividend == i32::MIN && divisor == -1 {
            dividend
        } else {
            dividend.wrapping_div(divisor)
        }) as i64;
    }
}
impl RISCVTrace for DIVW {}
