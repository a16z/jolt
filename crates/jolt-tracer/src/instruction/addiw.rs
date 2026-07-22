use serde::{Deserialize, Serialize};

use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::Instruction;

use super::{
    format::{format_i::FormatI, normalize_imm},
    Cycle, RISCVInstruction, RISCVTrace,
};

declare_riscv_instr!(
    name   = ADDIW,
    mask   = 0x0000707f,
    match  = 0x0000001b,
    format = FormatI,
    ram    = ()
);

impl ADDIW {
    fn exec(&self, cpu: &mut Cpu, _: &mut <ADDIW as RISCVInstruction>::RAMAccess) {
        cpu.write_register(
            self.operands.rd as usize,
            cpu.x[self.operands.rs1 as usize].wrapping_add(normalize_imm(self.operands.imm)) as i32
                as i64,
        );
    }
}

impl RISCVTrace for ADDIW {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        let inline_sequence = Instruction::from(*self).inline_sequence(&cpu.vr_allocator);
        let mut trace = trace;
        for instr in inline_sequence {
            instr.trace(cpu, trace.as_deref_mut());
        }
    }
}
