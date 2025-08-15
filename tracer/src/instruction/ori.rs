use serde::{Deserialize, Serialize};

use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::{
    format::{format_i::FormatI, normalize_imm, InstructionFormat},
    RISCVInstruction, RISCVTrace,
};

declare_riscv_instr!(
    name   = ORI,
    mask   = 0x0000707f,
    match  = 0x00006013,
    format = FormatI,
    ram    = ()
);

impl ORI {
    fn exec(&self, cpu: &mut Cpu, _: &mut <ORI as RISCVInstruction>::RAMAccess) {
        cpu.x[self.operands.rd as usize] = cpu.sign_extend(
            cpu.x[self.operands.rs1 as usize] | normalize_imm(self.operands.imm, &cpu.xlen),
        );
    }
}

impl RISCVTrace for ORI {}
