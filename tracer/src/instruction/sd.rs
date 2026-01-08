use serde::{Deserialize, Serialize};

use crate::declare_riscv_instr;
use crate::emulator::cpu::Cpu;

use super::RAMWrite;

use super::{format::format_s::FormatS, RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name   = SD,
    mask   = 0x0000707f,
    match  = 0x00003023,
    format = FormatS,
    ram    = RAMWrite
);

impl SD {
    fn exec(
        &self,
        cpu: &mut Cpu,
        ram_access: &mut <SD as RISCVInstruction>::RAMAccess,
    ) {
        // The SD, SW, SH, and SB instructions store 64-bit, 32-bit, 16-bit, and 8-bit values from
        // the low bits of register rs2 to memory respectively.
        *ram_access = cpu
            .mmu
            .store_doubleword(
                cpu.x[self.operands.rs1 as usize].wrapping_add(self.operands.imm) as u64,
                cpu.x[self.operands.rs2 as usize] as u64,
            )
            .ok()
            .unwrap();
    }
}

impl RISCVTrace for SD {}
