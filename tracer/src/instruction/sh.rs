use serde::{Deserialize, Serialize};

use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::Instruction;
use super::RAMWrite;

use super::{format::format_s::FormatS, Cycle, RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name   = SH,
    mask   = 0x0000707f,
    match  = 0x00001023,
    format = FormatS,
    ram    = RAMWrite
);

impl SH {
    fn exec(&self, cpu: &mut Cpu, ram_access: &mut <SH as RISCVInstruction>::RAMAccess) {
        *ram_access = cpu
            .mmu
            .store_halfword(
                cpu.x[self.operands.rs1 as usize].wrapping_add(self.operands.imm) as u64,
                cpu.x[self.operands.rs2 as usize] as u16,
            )
            .ok()
            .unwrap();
    }
}

impl RISCVTrace for SH {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        let inline_sequence = Instruction::from(*self).inline_sequence(&cpu.vr_allocator);
        let mut trace = trace;
        for instr in inline_sequence {
            instr.trace(cpu, trace.as_deref_mut());
        }
    }
}

impl SH {}
