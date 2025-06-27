use super::{
    format::{format_r::FormatR, InstructionFormat},
    RISCVInstruction, RISCVTrace,
};
use crate::{declare_riscv_instr, emulator::cpu::Cpu};
use serde::{Deserialize, Serialize};

declare_riscv_instr!(
    name   = SUBW,
    mask   = 0xfe00707f,
    match  = 0x4000003b,
    format = FormatR,
    ram    = ()
);

impl SUBW {
    fn exec(&self, cpu: &mut Cpu, _: &mut <SUBW as RISCVInstruction>::RAMAccess) {
        cpu.x[self.operands.rd] = cpu.sign_extend(
            (cpu.x[self.operands.rs1].wrapping_sub(cpu.x[self.operands.rs2]) as i32) as i64,
        );
    }
}
impl RISCVTrace for SUBW {}
