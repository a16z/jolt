use serde::{Deserialize, Serialize};

use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::{format::format_assert_align::FormatAssert, RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name = VirtualAssertWordAlignment,
    mask = 0,
    match = 0,
    format = FormatAssert,
    ram = ()
);

impl VirtualAssertWordAlignment {
    fn exec(
        &self,
        cpu: &mut Cpu,
        _: &mut <VirtualAssertWordAlignment as RISCVInstruction>::RAMAccess,
    ) {
        let address = cpu.x[self.operands.rs1 as usize] + self.operands.imm;
        assert!(
            address & 3 == 0,
            "RAM access (LW or LWU) is not word aligned: {address:x}"
        );
    }
}

impl RISCVTrace for VirtualAssertWordAlignment {}
