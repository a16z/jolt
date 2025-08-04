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
    name = VirtualExtend,
    mask = 0,
    match = 0,
    format = FormatI,
    ram = (),
    is_virtual = true
);

impl VirtualExtend {
    fn exec(&self, cpu: &mut Cpu, _: &mut <VirtualExtend as RISCVInstruction>::RAMAccess) {
        match cpu.xlen {
            Xlen::Bit32 => panic!("VirtualExtend is not supported for 32-bit mode"),
            Xlen::Bit64 => cpu.x[self.operands.rd] = cpu.x[self.operands.rs1] & 0xFFFFFFFF,
        }
    }
}

impl RISCVTrace for VirtualExtend {}
