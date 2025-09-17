use serde::{Deserialize, Serialize};

use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::{format::format_b::FormatB, RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name = VirtualAssertSignedMulNoOverflow,
    mask = 0,
    match = 0,
    format = FormatB,
    ram = (),
    is_virtual = true
);

impl VirtualAssertSignedMulNoOverflow {
    fn exec(
        &self,
        cpu: &mut Cpu,
        _: &mut <VirtualAssertSignedMulNoOverflow as RISCVInstruction>::RAMAccess,
    ) {
        let rs1_val = cpu.x[self.operands.rs1 as usize];
        let rs2_val = cpu.x[self.operands.rs2 as usize];
        assert!(rs1_val.checked_mul(rs2_val).is_some());
    }
}

impl RISCVTrace for VirtualAssertSignedMulNoOverflow {}
