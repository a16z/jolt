use super::{
    format::{format_i::FormatI, InstructionFormat},
    RISCVInstruction, RISCVTrace,
};
use crate::{declare_riscv_instr, emulator::cpu::Cpu};
use serde::{Deserialize, Serialize};

declare_riscv_instr!(
    name   = SRAIW,
    mask   = 0xfc00707f,
    match  = 0x4000501b,
    format = FormatI,
    ram    = ()
);

impl SRAIW {
    fn exec(&self, cpu: &mut Cpu, _: &mut <SRAIW as RISCVInstruction>::RAMAccess) {
        cpu.x[self.operands.rd] =
            ((cpu.x[self.operands.rs1] as i32) >> (self.operands.imm & 0x1f)) as i64;
    }
}
impl RISCVTrace for SRAIW {}
