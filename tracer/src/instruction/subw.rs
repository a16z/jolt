use crate::{declare_riscv_instr, emulator::cpu::Cpu};
use serde::{Deserialize, Serialize};

use super::Instruction;

use super::{format::format_r::FormatR, Cycle, RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name   = SUBW,
    mask   = 0xfe00707f,
    match  = 0x4000003b,
    format = FormatR,
    ram    = ()
);

impl SUBW {
    fn exec(&self, cpu: &mut Cpu, _: &mut <SUBW as RISCVInstruction>::RAMAccess) {
        // ADDW and SUBW are RV64I-only instructions that are defined analogously to ADD and SUB
        // but operate on 32-bit values and produce signed 32-bit results. Overflows are ignored,
        // and the low 32-bits of the result is sign-extended to 64-bits and written to the
        // destination register.
        cpu.write_register(
            self.operands.rd as usize,
            (cpu.x[self.operands.rs1 as usize].wrapping_sub(cpu.x[self.operands.rs2 as usize])
                as i32) as i64,
        );
    }
}

impl RISCVTrace for SUBW {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        let inline_sequence = Instruction::from(*self).inline_sequence(&cpu.vr_allocator, cpu.xlen);
        let mut trace = trace;
        for instr in inline_sequence {
            instr.trace(cpu, trace.as_deref_mut());
        }
    }
}
