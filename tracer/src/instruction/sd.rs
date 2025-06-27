use serde::{Deserialize, Serialize};

use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::RAMWrite;

use super::{
    format::{format_s::FormatS, InstructionFormat},
    RISCVInstruction, RISCVTrace,
};

declare_riscv_instr!(
    name   = SD,
    mask   = 0x0000707f,
    match  = 0x00003023,
    format = FormatS,
    ram    = RAMWrite
);

impl SD {
    fn exec(&self, cpu: &mut Cpu, ram_access: &mut <SD as RISCVInstruction>::RAMAccess) {
        *ram_access = cpu
            .mmu
            .store_doubleword(
                cpu.x[self.operands.rs1].wrapping_add(self.operands.imm) as u64,
                cpu.x[self.operands.rs2] as u64,
            )
            .ok()
            .unwrap();
    }
}

impl RISCVTrace for SD {}
