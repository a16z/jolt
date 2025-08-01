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
    name = VirtualPow2W,
    mask = 0,
    match = 0,
    format = FormatI,
    ram = (),
    is_virtual = true
);

impl VirtualPow2W {
    fn exec(&self, cpu: &mut Cpu, _: &mut <VirtualPow2W as RISCVInstruction>::RAMAccess) {
        match cpu.xlen {
            Xlen::Bit32 => panic!("VirtualPow2W is invalid in 32b mode"),
            Xlen::Bit64 => cpu.x[self.operands.rd] = 1 << (cpu.x[self.operands.rs1] as u64 % 32),
        }
    }
}

impl RISCVTrace for VirtualPow2W {}
