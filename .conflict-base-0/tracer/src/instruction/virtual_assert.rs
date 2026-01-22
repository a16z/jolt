use serde::{Deserialize, Serialize};

use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::{format::format_i::FormatI, RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name = VirtualAssert,
    mask = 0,
    match = 0,
    format = FormatI,
    ram = ()
);

impl VirtualAssert {
    fn exec(&self, cpu: &mut Cpu, _: &mut <VirtualAssert as RISCVInstruction>::RAMAccess) {
        // Check that rs1 is non-zero (the condition being asserted)
        // If rs1 is zero, panic to indicate assertion failure
        assert_ne!(
            cpu.x[self.operands.rs1 as usize], 0,
            "Advice assertion failed at PC: 0x{:x}",
            self.address
        );
    }
}

impl RISCVTrace for VirtualAssert {}
