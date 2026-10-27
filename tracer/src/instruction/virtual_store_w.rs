use crate::instruction::registers::s::RegisterStateS;
use serde::{Deserialize, Serialize};

use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::{
    floor_access::{self, AlignmentClass},
    format::format_s::FormatS,
    RISCVInstruction, RISCVTrace, RamWrite,
};

declare_riscv_instr!(
    name = VirtualStoreW,
    mask = 0,
    match = 0,
    format = FormatS,
    registers = RegisterStateS,
    ram = RamWrite
);

impl VirtualStoreW {
    fn exec(&self, cpu: &mut Cpu, ram_access: &mut <VirtualStoreW as RISCVInstruction>::RamAccess) {
        *ram_access = floor_access::store(cpu, &self.operands, AlignmentClass::Word);
    }
}

impl RISCVTrace for VirtualStoreW {}
