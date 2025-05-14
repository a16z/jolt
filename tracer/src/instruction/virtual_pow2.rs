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
    name = VirtualPow2,
    mask = 0,
    match = 0,
    format = FormatI,
    ram = (),
    is_virtual = true
);

impl VirtualPow2 {
    fn exec(&self, cpu: &mut Cpu, _: &mut <VirtualPow2 as RISCVInstruction>::RAMAccess) {
        match cpu.xlen {
            Xlen::Bit32 => cpu.x[self.operands.rd] = 1 << (cpu.x[self.operands.rs1] as u64 % 32),
            Xlen::Bit64 => cpu.x[self.operands.rd] = 1 << (cpu.x[self.operands.rs1] as u64 % 64),
        }
    }
}

impl RISCVTrace for VirtualPow2 {}
