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
        let shamt = (cpu.x[self.operands.rs2] & 0x1f) as u32;
        cpu.x[self.operands.rd] =
            cpu.sign_extend(((cpu.x[self.operands.rs1] as u32) << shamt) as i32 as i64);
    }
}
impl RISCVTrace for SLLIW {}
