use serde::{Deserialize, Serialize};

use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::{
    format::{format_i::FormatI, normalize_imm},
    RISCVInstruction, RISCVTrace,
};

declare_riscv_instr!(
    name = VirtualAlignAddr,
    mask = 0,
    match = 0,
    format = FormatI,
    ram = ()
);

impl VirtualAlignAddr {
    fn exec(&self, cpu: &mut Cpu, _: &mut <VirtualAlignAddr as RISCVInstruction>::RAMAccess) {
        // Address of the doubleword containing `rs1 + imm`: the fused
        // ADDI + ANDI(-8) of the sub-word memory sequences.
        let ea =
            cpu.x[self.operands.rs1 as usize].wrapping_add(normalize_imm(self.operands.imm)) as u64;
        cpu.write_register(self.operands.rd as usize, (ea & !7) as i64);
    }
}

impl RISCVTrace for VirtualAlignAddr {}
