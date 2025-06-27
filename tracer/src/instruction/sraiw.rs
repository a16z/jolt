use super::{
    format::{format_r::FormatR, InstructionFormat},
    RISCVInstruction, RISCVTrace,
};
use crate::{declare_riscv_instr, emulator::cpu::Cpu};
use serde::{Deserialize, Serialize};

declare_riscv_instr!(
    name   = SRAIW,
    mask   = 0xfc00707f,
    match  = 0x4000501b,
    format = FormatR,
    ram    = ()
);

impl SRAIW {
    fn exec(&self, cpu: &mut Cpu, _: &mut <SRAIW as RISCVInstruction>::RAMAccess) {
        let shamt = (cpu.x[self.operands.rs2] & 0x1f) as u32;
        cpu.x[self.operands.rd] =
            cpu.sign_extend(((cpu.x[self.operands.rs1] as i32) >> shamt) as i64);
    }
}
impl RISCVTrace for SRAIW {}
