use serde::{Deserialize, Serialize};

use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::{
    format::{format_b::FormatB, InstructionFormat},
    RISCVInstruction, RISCVTrace,
};

declare_riscv_instr!(
    name   = BGE,
    mask   = 0x0000707f,
    match  = 0x00005063,
    format = FormatB,
    ram    = ()
);

impl BGE {
    fn exec(&self, cpu: &mut Cpu, _: &mut <BGE as RISCVInstruction>::RAMAccess) {
        if cpu.sign_extend(cpu.x[self.operands.rs1 as usize])
            >= cpu.sign_extend(cpu.x[self.operands.rs2 as usize])
        {
            cpu.pc = (self.address as i64 + self.operands.imm as i64) as u64;
        }
    }
}

impl RISCVTrace for BGE {}
