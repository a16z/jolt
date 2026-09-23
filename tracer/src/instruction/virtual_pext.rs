use serde::{Deserialize, Serialize};

use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::virtual_pext_signed::pext;
use super::{format::format_r::FormatR, RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name = VirtualPext,
    mask = 0,
    match = 0,
    format = FormatR,
    ram = ()
);

impl VirtualPext {
    fn exec(&self, cpu: &mut Cpu, _: &mut <VirtualPext as RISCVInstruction>::RAMAccess) {
        let x = cpu.x[self.operands.rs1 as usize] as u64;
        let y = cpu.x[self.operands.rs2 as usize] as u64;
        cpu.write_register(self.operands.rd as usize, pext(x, y) as i64);
    }
}

impl RISCVTrace for VirtualPext {}
