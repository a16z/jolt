use serde::{Deserialize, Serialize};

use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::{
    format::{format_i::FormatI, InstructionFormat},
    RISCVInstruction, RISCVTrace,
};

declare_riscv_instr!(
    name   = JALR,
    mask   = 0x0000707f,
    match  = 0x00000067,
    format = FormatI,
    ram    = ()
);

impl JALR {
    fn exec(&self, cpu: &mut Cpu, _: &mut <JALR as RISCVInstruction>::RAMAccess) {
        let tmp = cpu.sign_extend(cpu.pc as i64);
        cpu.pc = (cpu.x[self.operands.rs1] as u64).wrapping_add(self.operands.imm);
        if self.operands.rd != 0 {
            cpu.x[self.operands.rd] = tmp;
        }
    }
}

impl RISCVTrace for JALR {}
