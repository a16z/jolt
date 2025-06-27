use super::{
    format::{format_load::FormatLoad, InstructionFormat},
    RISCVInstruction, RISCVTrace,
};
use crate::{declare_riscv_instr, emulator::cpu::Cpu};
use serde::{Deserialize, Serialize};

declare_riscv_instr!(
    name   = LWU,
    mask   = 0x0000707f,
    match  = 0x00006003,
    format = FormatLoad,
    ram    = super::RAMRead
);

impl LWU {
    fn exec(&self, cpu: &mut Cpu, ram_access: &mut <LWU as RISCVInstruction>::RAMAccess) {
        cpu.x[self.operands.rd] = match cpu
            .mmu
            .load_word(cpu.x[self.operands.rs1].wrapping_add(self.operands.imm) as u64)
        {
            Ok((word, memory_read)) => {
                *ram_access = memory_read;
                // Zero extension for unsigned word load
                word as i64
            }
            Err(_) => panic!("MMU load error"),
        };
    }
}
impl RISCVTrace for LWU {}
