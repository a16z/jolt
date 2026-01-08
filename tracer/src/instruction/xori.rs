use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::cpu::Cpu,
};

use super::{
    format::{format_i::FormatI, normalize_imm},
    RISCVInstruction, RISCVTrace,
};

declare_riscv_instr!(
    name   = XORI,
    mask   = 0x0000707f,
    match  = 0x00004013,
    format = FormatI,
    ram    = ()
);

impl XORI {
    fn exec(
        &self,
        cpu: &mut Cpu,
        _: &mut <XORI as RISCVInstruction>::RAMAccess,
    ) {
        cpu.x[self.operands.rd as usize] = cpu.sign_extend(
            cpu.x[self.operands.rs1 as usize] ^ normalize_imm(self.operands.imm, &cpu.xlen),
        );
    }
}

impl RISCVTrace for XORI {}
