use serde::{Deserialize, Serialize};

use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::{format::format_r::FormatR, Cycle, Instruction, RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name   = SRA,
    mask   = 0xfe00707f,
    match  = 0x40005033,
    format = FormatR,
    ram    = ()
);

impl SRA {
    fn exec(&self, cpu: &mut Cpu, _: &mut <SRA as RISCVInstruction>::RAMAccess) {
        let mask = 0x3f;
        cpu.write_register(
            self.operands.rd as usize,
            cpu.sign_extend(
                cpu.x[self.operands.rs1 as usize]
                    .wrapping_shr(cpu.x[self.operands.rs2 as usize] as u32 & mask),
            ),
        );
    }
}

impl RISCVTrace for SRA {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        let inline_sequence = Instruction::from(*self).inline_sequence(&cpu.vr_allocator, cpu.xlen);
        let mut trace = trace;
        for instr in inline_sequence {
            instr.trace(cpu, trace.as_deref_mut());
        }
    }
}
