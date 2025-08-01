use super::{
    format::{format_r::FormatR, InstructionFormat},
    RISCVInstruction, RISCVTrace,
};
use crate::{declare_riscv_instr, emulator::cpu::Cpu};
use serde::{Deserialize, Serialize};

declare_riscv_instr!(
    name   = SLLW,
    mask   = 0xfe00707f,
    match  = 0x0000003b | (0b001 << 12),
    format = FormatR,
    ram    = ()
);

impl SLLW {
    fn exec(&self, cpu: &mut Cpu, _: &mut <SLLW as RISCVInstruction>::RAMAccess) {
        // SLLW, SRLW, and SRAW are RV64I-only instructions that are analogously defined but operate
        // on 32-bit values and sign-extend their 32-bit results to 64 bits. The shift amount is
        // given by rs2[4:0].
        let shamt = (cpu.x[self.operands.rs2] & 0x1f) as u32;
        cpu.x[self.operands.rd] = ((cpu.x[self.operands.rs1] as u32) << shamt) as i32 as i64;
    }
}
impl RISCVTrace for SLLW {}
