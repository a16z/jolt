use serde::{Deserialize, Serialize};

use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::{format::format_j::FormatJ, RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name = VirtualPow2I,
    mask = 0,
    match = 0,
    format = FormatJ,
    ram = ()
);

impl VirtualPow2I {
    fn exec(&self, cpu: &mut Cpu, _: &mut <VirtualPow2I as RISCVInstruction>::RAMAccess) {
        cpu.write_register(self.operands.rd as usize, 1 << (self.operands.imm % 64))
    }
}

impl RISCVTrace for VirtualPow2I {}
