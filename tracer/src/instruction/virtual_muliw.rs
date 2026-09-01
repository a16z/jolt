use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::cpu::Cpu,
    instruction::{format::format_i::FormatI, RISCVInstruction, RISCVTrace},
};

declare_riscv_instr!(
    name = VirtualMULIW,
    mask = 0,
    match = 0,
    format = FormatI,
    ram = ()
);

impl VirtualMULIW {
    fn exec(&self, cpu: &mut Cpu, _: &mut <VirtualMULIW as RISCVInstruction>::RAMAccess) {
        let product = cpu.x[self.operands.rs1 as usize].wrapping_mul(self.operands.imm as i64);
        cpu.write_register(self.operands.rd as usize, product as u32 as i32 as i64);
    }
}

impl RISCVTrace for VirtualMULIW {}
