use crate::instruction::registers::load::RegisterStateLoad;
use serde::{Deserialize, Serialize};

use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::{
    floor_access::{self, AlignmentClass},
    format::format_load::FormatLoad,
    RISCVInstruction, RISCVTrace, RamRead,
};

declare_riscv_instr!(
    name = VirtualLoadB,
    mask = 0,
    match = 0,
    format = FormatLoad,
    registers = RegisterStateLoad,
    ram = RamRead
);

impl VirtualLoadB {
    fn exec(&self, cpu: &mut Cpu, ram_access: &mut <VirtualLoadB as RISCVInstruction>::RamAccess) {
        *ram_access = floor_access::load(cpu, &self.operands, AlignmentClass::Byte);
    }
}

impl RISCVTrace for VirtualLoadB {}
