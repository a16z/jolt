use serde::{Deserialize, Serialize};

use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::{
    format::{format_i::FormatI, normalize_imm, InstructionFormat},
    RISCVInstruction, RISCVTrace,
};

declare_riscv_instr!(
    name   = ADDI,
    mask   = 0x0000707f,
    match  = 0x00000013,
    format = FormatI,
    ram    = ()
);

impl ADDI {
    fn exec(&self, cpu: &mut Cpu, _: &mut <ADDI as RISCVInstruction>::RAMAccess) {
        cpu.x[self.operands.rd] = cpu
            .sign_extend(cpu.x[self.operands.rs1].wrapping_add(normalize_imm(self.operands.imm)));
    }
}

impl RISCVTrace for ADDI {}
