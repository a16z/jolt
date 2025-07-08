use serde::{Deserialize, Serialize};

use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::{
    format::{format_i::FormatI, InstructionFormat},
    RISCVInstruction, RISCVTrace,
};

declare_riscv_instr!(
    name   = ANDI,
    mask   = 0x0000707f,
    match  = 0x00007013,
    format = FormatI,
    ram    = ()
);

impl ANDI {
    fn exec(&self, cpu: &mut Cpu, _: &mut <ANDI as RISCVInstruction>::RAMAccess) {
        cpu.x[self.operands.rd] =
            cpu.sign_extend(cpu.x[self.operands.rs1] & self.operands.imm as i64);
    }
}

impl RISCVTrace for ANDI {}
