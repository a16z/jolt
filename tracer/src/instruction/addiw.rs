use serde::{Deserialize, Serialize};

use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::{
    format::{format_i::FormatI, normalize_imm, InstructionFormat},
    RISCVInstruction, RISCVTrace,
};

declare_riscv_instr!(
    name   = ADDIW,
    mask   = 0x0000707f,
    match  = 0x0000001b,
    format = FormatI,
    ram    = ()
);

impl ADDIW {
    fn exec(&self, cpu: &mut Cpu, _: &mut <ADDIW as RISCVInstruction>::RAMAccess) {
        cpu.x[self.operands.rd] =
            cpu.x[self.operands.rs1].wrapping_add(normalize_imm(self.operands.imm)) as i32 as i64;
    }
}

impl RISCVTrace for ADDIW {}
