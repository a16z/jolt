use serde::{Deserialize, Serialize};

use super::{format::format_i::FormatI, RISCVInstruction, RISCVTrace};
use crate::instruction::format::{normalize_imm, NormalizedOperands};
use crate::{declare_riscv_instr, emulator::cpu::Cpu};

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
        cpu.pc = ((cpu.x[self.operands.rs1 as usize] as u64)
            .wrapping_add(self.operands.imm as i32 as u64))
            & !1;
        if self.operands.rd != 0 {
            // Skip returns (rd=0) and non-standard link registers
            if self.operands.rd == 1 {
                cpu.track_call(self.address, NormalizedOperands::from(self.operands));
            }
            cpu.x[self.operands.rd as usize] = tmp;
        }
    }
}

impl RISCVTrace for JALR {}
