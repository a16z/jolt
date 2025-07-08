use super::{
    format::{format_i::FormatI, InstructionFormat},
    RISCVInstruction, RISCVTrace,
};
use crate::{declare_riscv_instr, emulator::cpu::Cpu};
use serde::{Deserialize, Serialize};

declare_riscv_instr!(
    name   = LD,
    mask   = 0x0000707f,
    match  = 0x00003003,
    format = FormatI,
    ram    = super::RAMRead
);

impl LD {
    fn exec(&self, cpu: &mut Cpu, ram_access: &mut <LD as RISCVInstruction>::RAMAccess) {
        // The LD instruction loads a 64-bit value from memory into register rd for RV64I.
        let address = (cpu.x[self.operands.rs1] as u64).wrapping_add(self.operands.imm);
        let value = cpu.get_mut_mmu().load_doubleword(address);
        cpu.x[self.operands.rd] = match value {
            Ok((value, memory_read)) => {
                *ram_access = memory_read;
                value as i64
            }
            Err(_) => panic!("MMU load error"),
        };
    }
}
impl RISCVTrace for LD {}
