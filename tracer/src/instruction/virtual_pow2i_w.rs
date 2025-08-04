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
    name = VirtualPow2IW,
    mask = 0,
    match = 0,
    format = FormatJ,
    ram = (),
    is_virtual = true
);

impl VirtualPow2IW {
    fn exec(&self, cpu: &mut Cpu, _: &mut <VirtualPow2IW as RISCVInstruction>::RAMAccess) {
        match cpu.xlen {
            Xlen::Bit32 => panic!("VirtualPow2IW is invalid in 32b mode"),
            Xlen::Bit64 => cpu.x[self.operands.rd] = 1 << (self.operands.imm % 32),
        }
    }
}

impl RISCVTrace for VirtualPow2IW {}
