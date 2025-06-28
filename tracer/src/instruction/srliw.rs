use super::{
    format::{format_r::FormatR, InstructionFormat},
    RISCVInstruction, RISCVTrace,
};
use crate::{declare_riscv_instr, emulator::cpu::Cpu};
use serde::{Deserialize, Serialize};

declare_riscv_instr!(
    name   = SRLIW,
    mask   = 0xfc00707f,
    match  = 0x0000501b,
    format = FormatR,
    ram    = ()
);

impl SRLIW {
    fn exec(&self, cpu: &mut Cpu, _: &mut <SRLIW as RISCVInstruction>::RAMAccess) {
        cpu.x[self.operands.rd] =
            ((cpu.x[self.operands.rs1] as u32) >> self.operands.rs2) as i32 as i64;
    }
}
impl RISCVTrace for SRLIW {}
