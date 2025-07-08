use serde::{Deserialize, Serialize};

use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::{
    format::{format_j::FormatJ, InstructionFormat},
    RISCVInstruction, RISCVTrace,
};

declare_riscv_instr!(
    name   = JAL,
    mask   = 0x0000_007f,
    match  = 0x0000_006f,
    format = FormatJ,
    ram    = ()
);

impl JAL {
    fn exec(&self, cpu: &mut Cpu, _: &mut <JAL as RISCVInstruction>::RAMAccess) {
        if self.operands.rd != 0 {
            cpu.x[self.operands.rd] = cpu.sign_extend(cpu.pc as i64);
        }
        cpu.pc = (self.address as i64 + self.operands.imm as i64) as u64;
    }
}

impl RISCVTrace for JAL {}
