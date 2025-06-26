use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
};

use super::{
    format::{format_j::FormatJ, InstructionFormat},
    RISCVInstruction, RISCVTrace,
};

declare_riscv_instr!(
    name = VirtualPow2I,
    mask = 0,
    match = 0,
    format = FormatJ,
    ram = (),
    is_virtual = true
);

impl VirtualPow2I {
    fn exec(&self, cpu: &mut Cpu, _: &mut <VirtualPow2I as RISCVInstruction>::RAMAccess) {
        match cpu.xlen {
            Xlen::Bit32 => cpu.x[self.operands.rd] = 1 << (self.operands.imm % 32),
            Xlen::Bit64 => cpu.x[self.operands.rd] = 1 << (self.operands.imm % 64),
        }
    }
}

impl RISCVTrace for VirtualPow2I {}
