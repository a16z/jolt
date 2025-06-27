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
        let address =
            (cpu.x[self.operands.rs1] as u64).wrapping_add(self.operands.imm as i32 as u64);
        let value = cpu.get_mut_mmu().load_doubleword(address).unwrap();
        cpu.x[self.operands.rd] = value as i64;
        *ram_access = super::RAMRead { address, value };
    }
}
impl RISCVTrace for LD {}
