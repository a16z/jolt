use serde::{Deserialize, Serialize};

use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::{
    format::{format_u::FormatU, normalize_imm, InstructionFormat},
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
        cpu.x[self.operands.rd] =
            cpu.sign_extend(self.address as i64 + normalize_imm(self.operands.imm, &cpu.xlen));
    }
}

impl RISCVTrace for AUIPC {}
