use serde::{Deserialize, Serialize};

use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::{format::format_i::FormatI, Cycle, Instruction, RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name   = SRLI,
    mask   = 0xfc00707f,
    match  = 0x00005013,
    format = FormatI,
    ram    = ()
);

impl SRLI {
    fn exec(&self, cpu: &mut Cpu, _: &mut <SRLI as RISCVInstruction>::RAMAccess) {
        let mask = 0x3f;
        cpu.write_register(
            self.operands.rd as usize,
            cpu.sign_extend(
                cpu.unsigned_data(cpu.x[self.operands.rs1 as usize])
                    .wrapping_shr(self.operands.imm as u32 & mask) as i64,
            ),
        );
    }
}

impl RISCVTrace for SRLI {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        let inline_sequence = Instruction::from(*self).inline_sequence(&cpu.vr_allocator);
        let mut trace = trace;
        for instr in inline_sequence {
            instr.trace(cpu, trace.as_deref_mut());
        }
    }
}
