use crate::instruction::registers::load::RegisterStateLoad;
use serde::{Deserialize, Serialize};

use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::{
    floor_access::{self, AlignmentClass},
    format::format_load::FormatLoad,
    RISCVInstruction, RISCVTrace, RamRead,
};

declare_riscv_instr!(
    name = VirtualLoadW,
    mask = 0,
    match = 0,
    format = FormatLoad,
    registers = RegisterStateLoad,
    ram = RamRead
);

impl VirtualLoadW {
    fn exec(&self, cpu: &mut Cpu, ram_access: &mut <VirtualLoadW as RISCVInstruction>::RamAccess) {
        *ram_access = floor_access::load(cpu, &self.operands, AlignmentClass::Word);
    }
}

impl RISCVTrace for VirtualLoadW {}
