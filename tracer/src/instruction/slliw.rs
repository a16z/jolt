use super::{
    format::{format_i::FormatI, InstructionFormat},
    RISCVInstruction, RISCVTrace,
};
use crate::{declare_riscv_instr, emulator::cpu::Cpu};
use serde::{Deserialize, Serialize};

declare_riscv_instr!(
    name   = SLLIW,
    mask   = 0xfc00707f,
    match  = 0x0000101b,
    format = FormatI,
    ram    = ()
);

impl SLLIW {
    fn exec(&self, cpu: &mut Cpu, _: &mut <SLLIW as RISCVInstruction>::RAMAccess) {
        // SLLIW, SRLIW, and SRAIW are RV64I-only instructions that are analogously defined but
        // operate on 32-bit values and sign-extend their 32-bit results to 64 bits. SLLIW, SRLIW,
        // and SRAIW encodings with imm[5] ≠ 0 are reserved.
        let shamt = (self.operands.imm & 0x1f) as u32;
        cpu.x[self.operands.rd] = ((cpu.x[self.operands.rs1] as u32) << shamt) as i32 as i64;
    }
}
impl RISCVTrace for SLLIW {}
