use serde::{Deserialize, Serialize};

use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::{
    format::{format_u::FormatU, InstructionFormat},
    RISCVInstruction, RISCVTrace,
};

declare_riscv_instr!(
    name   = AUIPC,
    mask   = 0x0000007f,
    match  = 0x00000017,
    format = FormatU,
    ram    = ()
);

impl AUIPC {
    fn exec(&self, cpu: &mut Cpu, _: &mut <AUIPC as RISCVInstruction>::RAMAccess) {
        cpu.x[self.operands.rd] = cpu.sign_extend(self.address as i64 + self.operands.imm as i64);
    }
}

impl RISCVTrace for AUIPC {}
