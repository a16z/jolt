use crate::instruction::registers::r::RegisterStateR;
use serde::{Deserialize, Serialize};

use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::{format::format_r::FormatR, Cycle, Instruction, RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name   = SRLW,
    mask   = 0xfe00707f,
    match  = 0x0000003b | (0b101 << 12),
    format = FormatR,
    registers = RegisterStateR,
    ram    = ()
);

impl SRLW {
    fn exec(&self, cpu: &mut Cpu, _: &mut <SRLW as RISCVInstruction>::RAMAccess) {
        let shamt = (cpu.x[self.operands.rs2 as usize] & 0x1f) as u32;
        cpu.write_register(
            self.operands.rd as usize,
            ((cpu.x[self.operands.rs1 as usize] as u32) >> shamt) as i32 as i64,
        );
    }
}

impl RISCVTrace for SRLW {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        super::trace_inline_sequence(&Instruction::from(*self), cpu, trace);
    }
}
