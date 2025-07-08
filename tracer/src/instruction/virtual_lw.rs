use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
};

use super::{
    format::{format_i::FormatI, InstructionFormat},
    RISCVInstruction, RISCVTrace,
};

declare_riscv_instr!(
    name = VirtualLW,
    mask = 0,
    match = 0,
    format = FormatI,
    ram    = super::RAMRead,
    is_virtual = true
);

impl VirtualLW {
    fn exec(&self, cpu: &mut Cpu, ram_access: &mut <VirtualLW as RISCVInstruction>::RAMAccess) {
        let address =
            (cpu.x[self.operands.rs1] as u64).wrapping_add(self.operands.imm as i32 as u64);
        let value = cpu.get_mut_mmu().load_word(address);
        cpu.x[self.operands.rd] = match value {
            Ok((value, memory_read)) => {
                *ram_access = memory_read;
                value as i32 as i64
            }
            Err(_) => panic!("MMU load error"),
        };
    }
}

impl RISCVTrace for VirtualLW {}
