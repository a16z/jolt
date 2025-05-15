use serde::{Deserialize, Serialize};

use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::{
    format::{format_virtual_halfword_alignment::HalfwordAlignFormat, InstructionFormat},
    RISCVInstruction, RISCVTrace,
};

declare_riscv_instr!(
    name = VirtualAssertHalfwordAlignment,
    mask = 0,
    match = 0,
    format = HalfwordAlignFormat,
    ram = (),
    is_virtual = true
);

impl VirtualAssertHalfwordAlignment {
    fn exec(
        &self,
        cpu: &mut Cpu,
        _: &mut <VirtualAssertHalfwordAlignment as RISCVInstruction>::RAMAccess,
    ) {
        let address = cpu.x[self.operands.rs1] + self.operands.imm;
        assert!(
            address & 1 == 0,
            "RAM access (LH or LHU) is not halfword aligned: {address:x}"
        );
    }
}

impl RISCVTrace for VirtualAssertHalfwordAlignment {}
