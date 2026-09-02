use serde::{Deserialize, Serialize};

use super::{RISCVInstruction, RISCVTrace};
use crate::instruction::format::format_r::FormatR;
use crate::{declare_riscv_instr, emulator::cpu::Cpu};

declare_riscv_instr!(
    name = VirtualXORROTL1,
    mask = 0,
    match = 0,
    format = FormatR,
    ram = ()
);

impl VirtualXORROTL1 {
    fn exec(&self, cpu: &mut Cpu, _: &mut <VirtualXORROTL1 as RISCVInstruction>::RAMAccess) {
        let result =
            cpu.x[self.operands.rs1 as usize] ^ cpu.x[self.operands.rs2 as usize].rotate_left(1);
        cpu.write_register(self.operands.rd as usize, result);
    }
}

impl RISCVTrace for VirtualXORROTL1 {}
