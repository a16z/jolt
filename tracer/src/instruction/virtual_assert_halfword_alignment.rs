use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::{cpu::GeneralizedCpu, memory::MemoryData},
};

use super::{format::format_assert_align::AssertAlignFormat, RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name = VirtualAssertHalfwordAlignment,
    mask = 0,
    match = 0,
    format = AssertAlignFormat,
    ram = ()
);

impl VirtualAssertHalfwordAlignment {
    fn exec<D: MemoryData>(
        &self,
        cpu: &mut GeneralizedCpu<D>,
        _: &mut <VirtualAssertHalfwordAlignment as RISCVInstruction>::RAMAccess,
    ) {
        let address = cpu.x[self.operands.rs1 as usize] + self.operands.imm;
        assert!(
            address & 1 == 0,
            "RAM access (LH or LHU) is not halfword aligned: {address:x}"
        );
    }
}

impl RISCVTrace for VirtualAssertHalfwordAlignment {}
