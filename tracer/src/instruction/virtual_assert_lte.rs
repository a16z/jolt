use serde::{Deserialize, Serialize};

use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::{
    format::{format_b::FormatB, InstructionFormat},
    RISCVInstruction, RISCVTrace,
};

declare_riscv_instr!(
    name = VirtualAssertLTE,
    mask = 0,
    match = 0,
    format = FormatB,
    ram = (),
    is_virtual = true
);

impl VirtualAssertLTE {
    fn exec(&self, cpu: &mut Cpu, _: &mut <VirtualAssertLTE as RISCVInstruction>::RAMAccess) {
        assert!(cpu.x[self.operands.rs1] as u64 <= cpu.x[self.operands.rs2] as u64);
    }
}

impl RISCVTrace for VirtualAssertLTE {}
