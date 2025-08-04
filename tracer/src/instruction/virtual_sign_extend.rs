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
    name = VirtualSignExtend,
    mask = 0,
    match = 0,
    format = FormatI,
    ram = (),
    is_virtual = true
);

impl VirtualSignExtend {
    fn exec(&self, cpu: &mut Cpu, _: &mut <VirtualSignExtend as RISCVInstruction>::RAMAccess) {
        match cpu.xlen {
            Xlen::Bit32 => panic!("VirtualSignExtend is not supported for 32-bit mode"),
            Xlen::Bit64 => cpu.x[self.operands.rd] = (cpu.x[self.operands.rs1] << 32) >> 32,
        }
    }
}

impl RISCVTrace for VirtualSignExtend {}
