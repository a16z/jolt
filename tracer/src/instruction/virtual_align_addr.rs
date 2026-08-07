use serde::{Deserialize, Serialize};

use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::{format::format_i::FormatI, RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name = VirtualAlignAddr,
    mask = 0,
    match = 0,
    format = FormatI,
    ram = ()
);

impl VirtualAlignAddr {
    fn exec(&self, cpu: &mut Cpu, _: &mut <Self as RISCVInstruction>::RAMAccess) {
        let address = cpu.x[self.operands.rs1 as usize].wrapping_add(self.operands.imm as i64);
        cpu.write_register(self.operands.rd as usize, address & !7);
    }
}

impl RISCVTrace for VirtualAlignAddr {}
