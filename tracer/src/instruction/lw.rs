use serde::{Deserialize, Serialize};

use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::RAMRead;

use super::{
    format::{format_load::FormatLoad, InstructionFormat},
    RISCVInstruction, RISCVTrace,
};

declare_riscv_instr!(
    name   = LW,
    mask   = 0x0000707f,
    match  = 0x00002003,
    format = FormatLoad,
    ram    = RAMRead
);

impl LW {
    fn exec(&self, cpu: &mut Cpu, ram_access: &mut <LW as RISCVInstruction>::RAMAccess) {
        cpu.x[self.operands.rd] = match cpu
            .mmu
            .load_word(cpu.x[self.operands.rs1].wrapping_add(self.operands.imm) as u64)
        {
            Ok((word, memory_read)) => {
                *ram_access = memory_read;
                word as i32 as i64
            }
            Err(_) => panic!("MMU load error"),
        };
    }
}

impl RISCVTrace for LW {}
