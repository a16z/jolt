use serde::{Deserialize, Serialize};

use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::{
    format::{format_b::FormatB, InstructionFormat},
    RISCVInstruction, RISCVTrace,
};

declare_riscv_instr!(
    name = VirtualAssertEQ,
    mask = 0,
    match = 0,
    format = FormatB,
    ram = (),
    is_virtual = true
);

impl VirtualAssertEQ {
    fn exec(&self, cpu: &mut Cpu, _: &mut <VirtualAssertEQ as RISCVInstruction>::RAMAccess) {
        assert_eq!(cpu.x[self.operands.rs1], cpu.x[self.operands.rs2]);
    }
}

impl RISCVTrace for VirtualAssertEQ {}
