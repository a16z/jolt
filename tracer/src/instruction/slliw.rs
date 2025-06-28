use super::{
    format::{format_r::FormatR, InstructionFormat},
    RISCVInstruction, RISCVTrace,
};
use crate::{declare_riscv_instr, emulator::cpu::Cpu};
use serde::{Deserialize, Serialize};

declare_riscv_instr!(
    name   = SLLIW,
    mask   = 0xfc00707f,
    match  = 0x0000101b,
    format = FormatR,
    ram    = ()
);

impl SLLIW {
    fn exec(&self, cpu: &mut Cpu, _: &mut <SLLIW as RISCVInstruction>::RAMAccess) {
        cpu.x[self.operands.rd] = (cpu.x[self.operands.rs1] << self.operands.rs2) as i32 as i64;
    }
}
impl RISCVTrace for SLLIW {}
