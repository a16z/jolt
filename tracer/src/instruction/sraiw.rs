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
        // SLLIW, SRLIW, and SRAIW are RV64I-only instructions that are analogously defined but
        // operate on 32-bit values and sign-extend their 32-bit results to 64 bits. SLLIW, SRLIW,
        // and SRAIW encodings with imm[5] ≠ 0 are reserved.
        let shamt = (self.operands.imm & 0x1f) as u32;
        cpu.x[self.operands.rd] = ((cpu.x[self.operands.rs1] as i32) >> shamt) as i64;
    }
}
impl RISCVTrace for SRAIW {}
