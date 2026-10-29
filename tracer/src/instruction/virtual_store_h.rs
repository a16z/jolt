use crate::instruction::registers::s::RegisterStateS;
use serde::{Deserialize, Serialize};

use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::{
    floor_access::{self, AlignmentClass},
    format::format_s::FormatS,
    RISCVInstruction, RISCVTrace, RamWrite,
};

declare_riscv_instr!(
    name = VirtualStoreH,
    mask = 0,
    match = 0,
    format = FormatS,
    registers = RegisterStateS,
    ram = RamWrite
);

impl VirtualStoreH {
    fn exec(&self, cpu: &mut Cpu, ram_access: &mut <VirtualStoreH as RISCVInstruction>::RamAccess) {
        *ram_access = floor_access::store(cpu, &self.operands, AlignmentClass::Halfword);
    }
}

impl RISCVTrace for VirtualStoreH {}
