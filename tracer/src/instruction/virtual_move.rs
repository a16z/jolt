use serde::{Deserialize, Serialize};

use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::{format::format_i::FormatI, RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name = VirtualMove,
    mask = 0,
    match = 0,
    format = FormatI,
    ram = (),
    is_virtual = true
);

impl VirtualMove {
    fn exec(&self, cpu: &mut Cpu, _: &mut <VirtualMove as RISCVInstruction>::RAMAccess) {
        cpu.x[self.operands.rd as usize] = cpu.x[self.operands.rs1 as usize];
    }
}

impl RISCVTrace for VirtualMove {}
