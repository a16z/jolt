use serde::{Deserialize, Serialize};

use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::{Instruction, RAMWrite};

use super::{format::format_s::FormatS, Cycle, RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name   = SB,
    mask   = 0x0000707f,
    match  = 0x00000023,
    format = FormatS,
    ram    = RAMWrite
);

impl SB {
    fn exec(&self, cpu: &mut Cpu, ram_access: &mut <SB as RISCVInstruction>::RAMAccess) {
        *ram_access = cpu
            .mmu
            .store(
                cpu.x[self.operands.rs1 as usize].wrapping_add(self.operands.imm) as u64,
                cpu.x[self.operands.rs2 as usize] as u8,
            )
            .ok()
            .unwrap();
    }
}

impl RISCVTrace for SB {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        let inline_sequence = Instruction::from(*self).inline_sequence(&cpu.vr_allocator);
        let mut trace = trace;
        for instr in inline_sequence {
            instr.trace(cpu, trace.as_deref_mut());
        }
    }
}

impl SB {}
