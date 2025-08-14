use serde::{Deserialize, Serialize};

use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::{
    format::{format_i::FormatI, normalize_imm, InstructionFormat},
    RISCVInstruction, RISCVTrace,
};

declare_riscv_instr!(
    name   = SLTI,
    mask   = 0x0000707f,
    match  = 0x00002013,
    format = FormatI,
    ram    = ()
);

impl SLTI {
    fn exec(&self, cpu: &mut Cpu, _: &mut <SLTI as RISCVInstruction>::RAMAccess) {
        cpu.x[self.operands.rd as usize] =
            match cpu.x[self.operands.rs1 as usize] < normalize_imm(self.operands.imm) {
                true => 1,
                false => 0,
            };
    }
}

impl RISCVTrace for SLTI {}
